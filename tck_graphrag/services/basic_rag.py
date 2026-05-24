"""
Basic RAG Query Service

Klasik RAG pipeline: soru → embedding → pgvector semantic search → LLM cevap

GraphRAG ile aynı arayüz (query() -> dict) — karşılaştırma için.

LLM Desteği:
  - OpenAI (GPT-4o, GPT-4o-mini, vb.)
  - Anthropic (Claude Sonnet, Claude Haiku, vb.)
  
  Model değişikliği için .env'de LLM_PROVIDER ve LLM_MODEL ayarlayın
  veya BasicRAGService(provider="anthropic", model="claude-sonnet-4-20250514") kullanın.
"""

from __future__ import annotations

from typing import List, Optional, Literal

import psycopg2
from langchain_core.prompts import ChatPromptTemplate

from tck_graphrag.core.config import get_settings
from tck_graphrag.core.llm_factory import get_llm, get_embeddings
from tck_graphrag.prompts.query_prompts import BASIC_RAG_ANSWER_PROMPT

from tck_graphrag._paths import load_project_dotenv
load_project_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 10


class BasicRAGService:
    """
    Basic RAG: pgvector semantic search + LLM answer generation.

    GraphRAG QueryService ile aynı query() arayüzünü kullanır.
    Tek fark: retrieval mekanizması (vector search vs graph traversal).
    
    LLM Desteği: OpenAI (GPT) ve Anthropic (Claude)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[Literal["openai", "anthropic"]] = None,
    ):
        """
        BasicRAGService başlatır.
        
        Args:
            model: LLM model ismi. None ise config'den alınır.
            provider: "openai" veya "anthropic". None ise config'den alınır.
        """
        settings = get_settings()
        
        # LLM oluştur (factory kullan)
        self.llm = get_llm(provider=provider, model=model, temperature=0.1)
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model
        
        # Embedding her zaman OpenAI (fair comparison için sabit)
        self.embeddings = get_embeddings(model=EMBEDDING_MODEL)
        
        self._settings = settings
        self.conn = self._create_connection()
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", BASIC_RAG_ANSWER_PROMPT),
            ("human", "{question}"),
        ])
        print(f"BasicRAGService başlatıldı (Provider: {self.provider}, Model: {self.model})")

    def _create_connection(self):
        return psycopg2.connect(
            host=self._settings.postgres_host,
            port=int(self._settings.postgres_port),
            dbname=self._settings.postgres_db,
            user=self._settings.postgres_user,
            password=self._settings.postgres_password,
        )

    def close(self):
        """PostgreSQL bağlantısını kapat."""
        if self.conn and not self.conn.closed:
            self.conn.close()

    def __del__(self):
        self.close()

    def _fetch_similar_maddes(self, q_vector: list[float], limit: int) -> List[dict]:
        if self.conn.closed:
            self.conn = self._create_connection()

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT madde_no, baslik, icerik,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM tck_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (q_vector, q_vector, limit),
            )
            return [
                {
                    "madde_no": row[0],
                    "baslik": row[1],
                    "icerik": row[2],
                    "similarity": round(float(row[3]), 4),
                }
                for row in cur.fetchall()
            ]

    def _filter_adaptive(self, rows: List[dict], k_default: int) -> List[dict]:
        """min_similarity + dynamic threshold; en az k_min madde."""
        if not rows:
            return []

        s = self._settings
        min_sim = float(s.basic_rag_min_similarity)
        k_min = int(s.basic_rag_top_k_min)

        if s.basic_rag_dynamic_k:
            top_sim = rows[0]["similarity"]
            thresh = max(min_sim, top_sim - float(s.basic_rag_dynamic_delta))
        else:
            thresh = min_sim

        selected = [r for r in rows if r["similarity"] >= thresh][:k_default]
        if len(selected) >= k_min:
            return selected
        return rows[:k_min]

    def _retrieve(self, question: str, top_k: int = TOP_K) -> List[dict]:
        """
        Soruyu embedding'le, pgvector'da en yakın maddeleri getir.

        adaptive_retrieval=True ise: geniş havuz (top_k_max) → eşik → en fazla top_k.
        """
        q_vector = self.embeddings.embed_query(question)
        s = self._settings

        if not s.basic_rag_adaptive_retrieval:
            return self._fetch_similar_maddes(q_vector, top_k)

        k_max = int(s.basic_rag_top_k_max)
        k_default = min(top_k, int(s.basic_rag_top_k_default))
        rows = self._fetch_similar_maddes(q_vector, k_max)
        return self._filter_adaptive(rows, k_default)

    def _build_context(self, chunks: List[dict]) -> str:
        """Getirilen chunk'lardan context oluştur."""
        lines = ["=== TCK MADDE METİNLERİ ==="]
        for c in chunks:
            lines.append(f"\n[Madde {c['madde_no']}] {c['baslik']}\n{c['icerik']}")
        return "\n".join(lines)

    def query(self, question: str) -> dict:
        """
        Ana sorgu — GraphRAG QueryService.query() ile aynı çıktı formatı.

        Pipeline: soru → embedding → top-k semantic search → context → LLM cevap
        """
        # 1. Retrieve
        chunks = self._retrieve(question)

        # 2. Context
        context = self._build_context(chunks)

        # 3. LLM Answer
        chain = self.answer_prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": question,
        })

        madde_sources = [c["madde_no"] for c in chunks]

        return {
            "question": question,
            "answer": response.content,
            "keywords": [],
            "sources": [f"Madde {c['madde_no']}" for c in chunks[:5]],
            "madde_sources": madde_sources,
            "context": context,
        }


# Test
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BasicRAGService Test")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider: openai, anthropic")
    parser.add_argument("--model", type=str, default=None, help="Model ismi")
    args = parser.parse_args()
    
    print("=" * 60)
    print("BasicRAGService Test")
    print("=" * 60)

    service = BasicRAGService(provider=args.provider, model=args.model)

    test_questions = [
        "Hırsızlık suçunun cezası nedir?",
        "Madde 81 ne diyor?",
        "Kasten öldürme ile taksirle öldürme arasındaki fark nedir?",
    ]

    for q in test_questions:
        print(f"\nSoru: {q}")
        result = service.query(q)
        print(f"Madde kaynakları: {result['madde_sources']}")
        print(f"Yanıt:\n{result['answer'][:300]}...")
        print("-" * 60)
