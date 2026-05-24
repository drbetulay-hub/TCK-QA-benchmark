"""
TCK GraphRAG Query Service

Knowledge Graph'in gucunu maksimize eden retrieval pipeline.

Pipeline:
  Soru -> Query Analysis (LLM + Regex)
       -> Entity Retrieval (exact + contains)
       -> Graph Traversal (tam zincir, sinirlamasiz)
       -> Context Reranking (LLM ile filtreleme)
       -> Answer Generation (zengin context ile)

LLM Desteği:
  - OpenAI (GPT-4o, GPT-4o-mini, vb.)
  - Anthropic (Claude Sonnet, Claude Haiku, vb.)
  
  Model değişikliği için .env'de LLM_PROVIDER ve LLM_MODEL ayarlayın
  veya QueryService(provider="anthropic", model="claude-sonnet-4-20250514") kullanın.
"""

import os
import re
import json
from typing import Optional, Literal

import tiktoken
from langchain_core.prompts import ChatPromptTemplate

from tck_graphrag.core.database import get_neo4j
from tck_graphrag.core.config import get_settings
from tck_graphrag.core.llm_factory import get_llm
from tck_graphrag.prompts.query_prompts import (
    QUERY_ANALYSIS_PROMPT,
    RERANK_PROMPT,
    ANSWER_SYSTEM_PROMPT,
)

from tck_graphrag._paths import load_project_dotenv
load_project_dotenv()

# Turkce karakter -> ASCII donusum tablosu (arama icin)
_TR_TO_ASCII = str.maketrans(
    "cCgGiIoOsSuUaAiIuU",
    "cCgGiIoOsSuUaAiIuU",
)
# Gercek tablo:
_TR_TO_ASCII = str.maketrans(
    "\u00e7\u00c7\u011f\u011e\u0131\u0130\u00f6\u00d6\u015f\u015e\u00fc\u00dc\u00e2\u00c2\u00ee\u00ce\u00fb\u00db",
    "cCgGiIoOsSuUaAiIuU",
)

_ENCODING = tiktoken.encoding_for_model("gpt-4o")


def _to_ascii(text: str) -> str:
    """Turkce karakterleri ASCII karsiligina cevirir (arama icin)."""
    return text.translate(_TR_TO_ASCII).lower()


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


class QueryService:
    """
    TCK GraphRAG Query Service

    Knowledge Graph'in yapisal gucunu maksimize eden pipeline:
    1. Query Analysis  - LLM her zaman calisir, regex/sozluk tamamlar
    2. Entity Retrieval - graph'ta entity bul
    3. Graph Traversal  - tum iliskileri sinirlamasiz gez, madde metinleri topla
    4. Context Rerank   - LLM ile toplanan maddelerden en ilgilileri sec
    5. Context Build    - zengin, yapilandirilmis context olustur
    6. Answer Generate  - LLM ile yanit uret
    
    LLM Desteği: OpenAI (GPT) ve Anthropic (Claude)
    """

    # Context token limiti — GPT-4o 128K, Claude 200K destekler
    MAX_CONTEXT_TOKENS = 15000
    # Traversal'da toplanacak max madde sayisi
    MAX_MADDE_COLLECT = 20

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[Literal["openai", "anthropic"]] = None,
    ):
        """
        QueryService başlatır.
        
        Args:
            model: LLM model ismi. None ise config'den alınır.
            provider: "openai" veya "anthropic". None ise config'den alınır.
        """
        settings = get_settings()
        
        # LLM oluştur (factory kullan)
        self.llm = get_llm(provider=provider, model=model, temperature=0.1)
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model
        
        self.neo4j = get_neo4j()
        if not self.neo4j.driver:
            self.neo4j.connect()

        self._setup_prompts()

        # Startup: bilinen entity isimlerini cache'le
        self._known_entities: dict[str, str] = {}  # ascii -> original
        self._load_known_entities()

        print(f"QueryService baslatildi (Provider: {self.provider}, Model: {self.model})")

    def _setup_prompts(self):
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", QUERY_ANALYSIS_PROMPT),
            ("human", "{question}"),
        ])
        self.rerank_prompt = ChatPromptTemplate.from_messages([
            ("human", RERANK_PROMPT),
        ])
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

    def _load_known_entities(self):
        """Graph'tan bilinen entity isimlerini yukle (startup'ta 1 kez)."""
        try:
            rows = self.neo4j.run_query("""
                MATCH (e:Entity)
                WHERE e.type IN ['SUC', 'KAVRAM', 'TANIM', 'CEZA', 'KOSUL', 'MADDE']
                RETURN e.name AS name, e.type AS type
            """)
            for r in rows:
                self._known_entities[_to_ascii(r["name"])] = r["name"]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 1. QUERY ANALYSIS — LLM birincil, regex/sozluk tamamlayici
    # ------------------------------------------------------------------

    def _analyze_query(self, question: str) -> dict:
        """
        LLM ile soruyu analiz eder, regex/sozluk ile tamamlar.
        LLM dolayli ifadeleri de anlar:
          "birini bicakla yaralasamne olur?" -> Kasten Yaralama, Silahla Islenmesi
        """
        q_ascii = _to_ascii(question)

        # --- LLM Analysis (birincil) ---
        llm_entities: list[str] = []
        llm_madde_nos: list[int] = []
        intent = "GENEL"

        llm_synonyms: dict[str, list[str]] = {}
        try:
            chain = self.analysis_prompt | self.llm
            response = chain.invoke({"question": question})
            content = response.content.strip()
            # JSON parse
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.split("```")[0]
            data = json.loads(content)
            llm_entities = data.get("entities", [])
            llm_madde_nos = data.get("madde_nos", [])
            llm_synonyms = data.get("synonyms", {})
            intent = data.get("intent", "GENEL")
        except Exception:
            pass  # LLM basti, regex/sozluk devam eder

        # --- Regex: Madde numarasi tespiti (LLM kacirabilir) ---
        regex_madde_nos = []
        for m in re.finditer(r'[Mm]adde\s+(\d+)', question):
            regex_madde_nos.append(int(m.group(1)))

        # --- Birlestir ---
        all_madde_nos = list(set(llm_madde_nos + regex_madde_nos))

        # LLM entity'lerini graph'taki isimlerle esle
        resolved_entities: list[str] = []
        unresolved_keywords: list[str] = []

        for ent in llm_entities:
            # Entity + synonym'lerini tek listede dene
            candidates = [ent] + llm_synonyms.get(ent, [])
            resolved = False

            for candidate in candidates:
                cand_ascii = _to_ascii(candidate)

                # Adim 1: Exact ASCII match (en guvenilir)
                if cand_ascii in self._known_entities:
                    resolved_entities.append(self._known_entities[cand_ascii])
                    resolved = True
                    break

                # Adim 2: Partial ASCII match — en uzun eslesen kazanir
                best_match = None
                best_len = 0
                for known_ascii, known_original in self._known_entities.items():
                    if len(known_ascii) < 3:
                        continue
                    if cand_ascii in known_ascii or known_ascii in cand_ascii:
                        if len(known_ascii) > best_len:
                            best_match = known_original
                            best_len = len(known_ascii)
                if best_match:
                    resolved_entities.append(best_match)
                    resolved = True
                    break

            if not resolved:
                # Tum candidate'lar cozumlenemedi — keyword olarak sakla
                unresolved_keywords.append(ent)
                # Synonym'leri de keyword'lere ekle
                for syn in llm_synonyms.get(ent, []):
                    if syn not in unresolved_keywords:
                        unresolved_keywords.append(syn)

        # Sozluk: Soruda gecen bilinen entity isimlerini de ekle (LLM kacirmis olabilir)
        for known_ascii, known_original in self._known_entities.items():
            if len(known_ascii) > 4 and known_ascii in q_ascii:
                if known_original not in resolved_entities:
                    resolved_entities.append(known_original)

        return {
            "madde_nos": all_madde_nos,
            "entity_names": list(set(resolved_entities)),
            "keywords": unresolved_keywords,
            "intent": intent,
        }

    # ------------------------------------------------------------------
    # 2. ENTITY RETRIEVAL
    # ------------------------------------------------------------------

    def _find_entities(self, analysis: dict) -> list[dict]:
        """Graph'ta entity'leri bulur."""
        found: dict[str, dict] = {}

        # 2a. Madde numaralari ile Madde node
        for mno in analysis["madde_nos"]:
            rows = self.neo4j.run_query("""
                MATCH (m:Entity:Madde)
                WHERE m.madde_no = $mno
                RETURN m.name AS name, m.type AS type,
                       m.description AS description,
                       m.madde_no AS madde_no,
                       m.full_text AS full_text,
                       m.madde_no_list AS madde_no_list
                LIMIT 1
            """, {"mno": mno})
            for r in rows:
                found[r["name"]] = r

        # 2b. Entity isimleri ile exact match
        for ename in analysis["entity_names"]:
            if ename in found:
                continue
            rows = self.neo4j.run_query("""
                MATCH (e:Entity {name: $name})
                RETURN e.name AS name, e.type AS type,
                       e.description AS description,
                       e.madde_no AS madde_no,
                       e.full_text AS full_text,
                       e.madde_no_list AS madde_no_list
                LIMIT 1
            """, {"name": ename})
            for r in rows:
                found[r["name"]] = r

        # 2c. Cozumlenmemis keyword'ler ile contains arama
        for kw in analysis["keywords"][:5]:
            if len(kw) < 2:
                continue
            rows = self.neo4j.run_query("""
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($kw)
                   OR toLower(e.description) CONTAINS toLower($kw)
                RETURN e.name AS name, e.type AS type,
                       e.description AS description,
                       e.madde_no AS madde_no,
                       e.full_text AS full_text,
                       e.madde_no_list AS madde_no_list
                ORDER BY
                    CASE WHEN toLower(e.name) = toLower($kw) THEN 0 ELSE 1 END,
                    CASE e.type WHEN 'SUC' THEN 0 WHEN 'MADDE' THEN 1 ELSE 2 END
                LIMIT 5
            """, {"kw": kw})
            for r in rows:
                if r["name"] not in found:
                    found[r["name"]] = r

        # 2d. Bulunan entity'lerin 1-hop SUC/MADDE komsularini genislet
        # Hub entity filtresi: komsu sayisi > threshold olan entity'lerde expansion yapma
        HUB_DEGREE_THRESHOLD = 15
        initial_names = list(found.keys())
        if initial_names:
            # Sadece hub olmayan entity'lerden genislet
            non_hub_rows = self.neo4j.run_query("""
                MATCH (e:Entity)
                WHERE e.name IN $names
                WITH e, size([(e)-[]-() | 1]) AS degree
                WHERE degree <= $threshold
                RETURN e.name AS name
            """, {"names": initial_names, "threshold": HUB_DEGREE_THRESHOLD})
            non_hub_names = [r["name"] for r in non_hub_rows]

            if non_hub_names:
                neighbor_rows = self.neo4j.run_query("""
                    MATCH (e:Entity)-[r]-(n:Entity)
                    WHERE e.name IN $names
                      AND n.type IN ['SUC', 'MADDE']
                      AND NOT n.name IN $names
                    RETURN DISTINCT n.name AS name, n.type AS type,
                           n.description AS description,
                           n.madde_no AS madde_no,
                           n.full_text AS full_text,
                           n.madde_no_list AS madde_no_list
                    LIMIT 10
                """, {"names": non_hub_names})
                for r in neighbor_rows:
                    if r["name"] not in found:
                        found[r["name"]] = r

        # 2e. Ardisik madde fallback: sadece dogrudan bulunan entity'lerin
        # (2a-2c adimlari) madde_no'larina yakin maddeleri getir.
        # Neighbor expansion (2d) sonrasi gelen entity'ler icin YAPMA — cok noise uretir.
        core_mnos = set()
        for name in initial_names:  # 2d oncesi bulunan entity'ler
            ent = found.get(name)
            if not ent:
                continue
            if ent.get("madde_no"):
                core_mnos.add(ent["madde_no"])
            for mno in (ent.get("madde_no_list") or []):
                core_mnos.add(mno)

        all_found_mnos = set()
        for ent in found.values():
            if ent.get("madde_no"):
                all_found_mnos.add(ent["madde_no"])

        adjacent_mnos = set()
        for mno in core_mnos:
            adjacent_mnos.add(mno - 1)
            adjacent_mnos.add(mno + 1)
        # Sadece henuz bulunmamis komsu maddeleri getir
        adjacent_mnos -= all_found_mnos
        adjacent_mnos.discard(0)

        if adjacent_mnos:
            adj_rows = self.neo4j.run_query("""
                MATCH (m:Entity:Madde)
                WHERE m.madde_no IN $mnos
                RETURN m.name AS name, m.type AS type,
                       m.description AS description,
                       m.madde_no AS madde_no,
                       m.full_text AS full_text,
                       m.madde_no_list AS madde_no_list
            """, {"mnos": list(adjacent_mnos)})
            for r in adj_rows:
                if r["name"] not in found:
                    found[r["name"]] = r

        return list(found.values())

    # ------------------------------------------------------------------
    # 3. GRAPH TRAVERSAL — sinirlamasiz, tam zincir
    # ------------------------------------------------------------------

    def _traverse_graph(self, entities: list[dict]) -> dict:
        """
        Bulunan entity'ler etrafinda SINIRLAMASIZ graph traversal.
        Knowledge Graph'in tum gucunu kullanir:
        - Suc->Ceza->Sure zincirleri
        - Agirlastirici/Hafifletici kosullar
        - Nitelikli suc baglantilari
        - Cross-reference (REFERANS_VERIR)
        - madde_no_list ile ilgili tum maddelerin full_text'leri
        """
        madde_texts: dict[int, dict] = {}  # mno -> {name, full_text}
        relationships: list[dict] = []
        visited_names: set[str] = set()

        for entity in entities:
            ename = entity["name"]
            etype = entity.get("type", "")
            visited_names.add(ename)

            # --- Madde node ise full_text'i topla ---
            if etype == "MADDE" and entity.get("madde_no") and entity.get("full_text"):
                madde_texts[entity["madde_no"]] = {
                    "name": ename,
                    "full_text": entity["full_text"],
                }

            # --- madde_no_list ile ilgili TUM maddelerin full_text'lerini topla ---
            mno_list = entity.get("madde_no_list") or []
            if mno_list:
                self._collect_madde_texts(mno_list, madde_texts)

            # --- Suc -> Ceza -> Sure zinciri ---
            chain_rows = self.neo4j.run_query("""
                MATCH (suc:Entity {name: $name})-[r1:CEZA_OLARAK]->(ceza:Entity)
                OPTIONAL MATCH (ceza)-[r2:SURE_OLARAK]->(sure:Entity)
                RETURN suc.name AS suc, ceza.name AS ceza, sure.name AS sure,
                       ceza.madde_no_list AS ceza_mnos
            """, {"name": ename})

            for row in chain_rows:
                if row.get("ceza"):
                    relationships.append({
                        "source": row["suc"],
                        "rel_type": "CEZA_OLARAK",
                        "target": row["ceza"],
                    })
                if row.get("sure"):
                    relationships.append({
                        "source": row["ceza"],
                        "rel_type": "SURE_OLARAK",
                        "target": row["sure"],
                    })

            # --- Agirlastirici / Hafifletici kosullar ---
            cond_rows = self.neo4j.run_query("""
                MATCH (k:Entity)-[r:AGIRLAŞTIRIR|HAFIFLETIR]->(e:Entity {name: $name})
                RETURN k.name AS kosul, type(r) AS rel_type,
                       k.description AS kosul_desc,
                       k.madde_no_list AS kosul_mnos
            """, {"name": ename})

            for row in cond_rows:
                relationships.append({
                    "source": row["kosul"],
                    "rel_type": row["rel_type"],
                    "target": ename,
                    "description": row.get("kosul_desc", ""),
                })
                self._collect_madde_texts(row.get("kosul_mnos") or [], madde_texts)

            # --- Nitelikli suc <-> Temel suc ---
            nit_rows = self.neo4j.run_query("""
                MATCH (n:Entity)-[r:NITELIKLISI]->(e:Entity {name: $name})
                RETURN n.name AS nitelikli, n.madde_no_list AS mnos
                UNION
                MATCH (e:Entity {name: $name})-[r:NITELIKLISI]->(t:Entity)
                RETURN t.name AS nitelikli, t.madde_no_list AS mnos
            """, {"name": ename})

            for row in nit_rows:
                if row["nitelikli"] != ename:
                    relationships.append({
                        "source": row["nitelikli"],
                        "rel_type": "NITELIKLISI",
                        "target": ename,
                    })
                    self._collect_madde_texts(row.get("mnos") or [], madde_texts)

            # --- Giden iliskiler (TANIMLAR, REFERANS_VERIR, vb.) ---
            out_rows = self.neo4j.run_query("""
                MATCH (e:Entity {name: $name})-[r]->(t:Entity)
                RETURN t.name AS target, t.type AS target_type,
                       type(r) AS rel_type,
                       t.madde_no_list AS target_mnos
                LIMIT 10
            """, {"name": ename})

            for row in out_rows:
                relationships.append({
                    "source": ename,
                    "rel_type": row["rel_type"],
                    "target": row["target"],
                })
                # MADDE ve SUC tipindeki target'larin madde'lerini topla
                # CEZA ve SURE tipleri cok fazla maddeye bagli (200+), bunlari TOPLAMA
                if row.get("target_type") in ("MADDE", "SUC", "KOSUL"):
                    self._collect_madde_texts(
                        row.get("target_mnos") or [], madde_texts
                    )

            # --- Gelen iliskiler ---
            in_rows = self.neo4j.run_query("""
                MATCH (s:Entity)-[r]->(e:Entity {name: $name})
                WHERE NONE(t IN ['AGIRLAŞTIRIR', 'HAFIFLETIR', 'NITELIKLISI'] WHERE type(r) = t)
                RETURN s.name AS source, s.type AS source_type,
                       type(r) AS rel_type
                LIMIT 10
            """, {"name": ename})

            for row in in_rows:
                relationships.append({
                    "source": row["source"],
                    "rel_type": row["rel_type"],
                    "target": ename,
                })

            # --- Cross-reference (REFERANS_VERIR) ---
            if etype == "MADDE":
                ref_rows = self.neo4j.run_query("""
                    MATCH (m:Entity {name: $name})-[:REFERANS_VERIR]->(ref:Entity:Madde)
                    RETURN ref.madde_no AS ref_mno, ref.name AS ref_name,
                           ref.full_text AS ref_text
                """, {"name": ename})

                for row in ref_rows:
                    relationships.append({
                        "source": ename,
                        "rel_type": "REFERANS_VERIR",
                        "target": row["ref_name"],
                    })
                    if row.get("ref_mno") and row.get("ref_text"):
                        madde_texts[row["ref_mno"]] = {
                            "name": row["ref_name"],
                            "full_text": row["ref_text"],
                        }

        # --- 2. TUR: KOSUL entity'lerinden bagli SUC'lari bul ---
        # "Gece Vakti Islenmesi" bulunduysa -> hangi suclara bagli? -> o suclarin maddelerini topla
        kosul_names = [e["name"] for e in entities if e.get("type") == "KOSUL"]
        for kname in kosul_names:
            linked_rows = self.neo4j.run_query("""
                MATCH (k:Entity {name: $name})-[r:AGIRLAŞTIRIR|HAFIFLETIR]->(suc:Entity)
                RETURN suc.name AS suc, suc.type AS suc_type,
                       type(r) AS rel_type,
                       suc.madde_no_list AS suc_mnos
            """, {"name": kname})

            for row in linked_rows:
                relationships.append({
                    "source": kname,
                    "rel_type": row["rel_type"],
                    "target": row["suc"],
                })
                self._collect_madde_texts(row.get("suc_mnos") or [], madde_texts)

        return {
            "madde_texts": madde_texts,
            "relationships": relationships,
        }

    def _collect_madde_texts(self, madde_nos: list, target: dict):
        """madde_no listesinden full_text'leri toplar."""
        if not madde_nos:
            return
        if len(target) >= self.MAX_MADDE_COLLECT:
            return
        missing = [mno for mno in madde_nos if mno and mno not in target]
        if not missing:
            return
        # Global limite uygun kadarini al
        remaining_capacity = self.MAX_MADDE_COLLECT - len(target)
        missing = missing[:remaining_capacity]
        rows = self.neo4j.run_query("""
            MATCH (m:Entity:Madde)
            WHERE m.madde_no IN $mnos
            RETURN m.madde_no AS mno, m.name AS name, m.full_text AS full_text
        """, {"mnos": missing})
        for r in rows:
            if r.get("full_text") and r["mno"] not in target:
                target[r["mno"]] = {
                    "name": r["name"],
                    "full_text": r["full_text"],
                }

    # ------------------------------------------------------------------
    # 4. CONTEXT RERANKING — LLM ile en ilgili maddeleri sec
    # ------------------------------------------------------------------

    def _rerank_madde_texts(self, question: str,
                            madde_texts: dict[int, dict],
                            must_include: list[int]) -> dict[int, dict]:
        """
        Toplanan madde metinleri cok fazlaysa (>15), LLM ile filtrele.
        must_include maddeleri her zaman dahil edilir.
        Az sayida madde varsa filtreleme yapmaz.
        """
        if len(madde_texts) <= 15:
            return madde_texts

        # must_include maddeleri ayir
        kept = {mno: madde_texts[mno] for mno in must_include if mno in madde_texts}

        # Geri kalanlari LLM'e gonder
        candidates = {mno: mt for mno, mt in madde_texts.items() if mno not in kept}
        if not candidates:
            return kept

        # Madde listesini ozetle (token tasarrufu icin ilk 150 karakter)
        madde_list_str = "\n".join(
            f"Madde {mno}: {mt['full_text'][:150]}..."
            for mno, mt in sorted(candidates.items())
        )

        try:
            chain = self.rerank_prompt | self.llm
            response = chain.invoke({
                "question": question,
                "madde_list": madde_list_str,
            })

            # Parse: "141, 142, 143" -> [141, 142, 143]
            selected_mnos = []
            for token in re.findall(r'\d+', response.content):
                mno = int(token)
                if mno in candidates:
                    selected_mnos.append(mno)

            for mno in selected_mnos[:10]:
                kept[mno] = candidates[mno]

        except Exception:
            # LLM basarisiz olursa ilk 10'u al
            for mno in list(candidates.keys())[:10]:
                kept[mno] = candidates[mno]

        return kept

    # ------------------------------------------------------------------
    # 5. CONTEXT BUILDER
    # ------------------------------------------------------------------

    def _build_context(self, entities: list[dict], traversal: dict,
                       analysis: dict, question: str) -> tuple[str, list[int]]:
        """
        Yapilandirilmis context olusturur.
        Reranking sonrasi sadece ilgili madde metinleri dahil edilir.

        Returns:
            (context_string, used_madde_nos) — LLM'e fiilen giden madde numaralari
        """
        # Reranking: cok fazla madde varsa LLM filtrelesin
        madde_texts = self._rerank_madde_texts(
            question,
            traversal["madde_texts"],
            must_include=analysis["madde_nos"],
        )

        sections: list[str] = []
        used_tokens = 0
        used_madde_nos: list[int] = []

        # --- Bolum 1: Madde Metinleri ---
        if madde_texts:
            madde_lines = ["=== TCK MADDE METINLERI ==="]

            # Siralama: once dogrudan sorulan maddeler, sonra entity maddeleri, sonra geri kalan
            priority_mnos = analysis["madde_nos"]
            entity_mnos = [
                e["madde_no"] for e in entities
                if e.get("madde_no") and e["madde_no"] not in priority_mnos
            ]
            ordered = priority_mnos + entity_mnos
            remaining = [mno for mno in sorted(madde_texts.keys()) if mno not in ordered]
            all_mnos = ordered + remaining

            for mno in all_mnos:
                if mno not in madde_texts:
                    continue
                mt = madde_texts[mno]
                text = mt["full_text"]
                line = f"\n[Madde {mno}]\n{text}"
                line_tokens = _count_tokens(line)

                # Token limiti kontrolu — cok genisse kirp
                if used_tokens + line_tokens > self.MAX_CONTEXT_TOKENS * 0.75:
                    line = f"\n[Madde {mno}] (kisaltilmis)\n{text[:600]}..."
                    line_tokens = _count_tokens(line)
                if used_tokens + line_tokens > self.MAX_CONTEXT_TOKENS * 0.90:
                    break

                madde_lines.append(line)
                used_tokens += line_tokens
                used_madde_nos.append(mno)

            if len(madde_lines) > 1:
                sections.append("\n".join(madde_lines))

        # --- Bolum 2: Graph Iliskileri ---
        rels = traversal["relationships"]
        if rels:
            rel_lines = ["\n=== GRAPH ILISKILERI ==="]
            seen_rels: set[str] = set()
            for r in rels:
                key = f"{r['source']}-{r['rel_type']}-{r['target']}"
                if key in seen_rels:
                    continue
                seen_rels.add(key)
                line = f"  {r['source']} --[{r['rel_type']}]--> {r['target']}"
                desc = r.get("description", "")
                if desc:
                    line += f"  ({desc[:80]})"
                rel_lines.append(line)

            if len(rel_lines) > 1:
                sections.append("\n".join(rel_lines[:50]))  # max 50 iliski

        # --- Bolum 3: Entity Ozeti ---
        if entities:
            ent_lines = ["\n=== BULUNAN ENTITYLER ==="]
            for e in entities[:15]:
                desc = e.get("description") or ""
                etype = e.get("type", "")
                line = f"  [{etype}] {e['name']}"
                if desc:
                    line += f" - {desc[:120]}"
                ent_lines.append(line)
            sections.append("\n".join(ent_lines))

        if not sections:
            return "Graph'ta bu soruyla ilgili bilgi bulunamadi.", used_madde_nos

        return "\n".join(sections), used_madde_nos

    # ------------------------------------------------------------------
    # 6. MAIN QUERY
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict:
        """
        Ana sorgu fonksiyonu.

        Pipeline:
        1. LLM + regex/sozluk ile sorgu analizi
        2. Graph'ta entity bul
        3. Sinirlamasiz graph traversal
        4. LLM reranking (gerekirse)
        5. Zengin context olustur
        6. LLM ile yanit uret
        """
        # 1. Query Analysis
        analysis = self._analyze_query(question)

        # 2. Entity Retrieval
        entities = self._find_entities(analysis)

        # 3. Graph Traversal
        traversal = self._traverse_graph(entities)

        # 4+5. Context Build (reranking dahil)
        context, used_madde_nos = self._build_context(entities, traversal, analysis, question)

        # 6. Answer Generation
        chain = self.answer_prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": question,
        })

        # Kaynaklari topla
        all_entity_names = list(set(
            e.get("name", "") for e in entities
        ))[:15]

        madde_sources = sorted(set(used_madde_nos))

        used_keywords = (
            analysis["entity_names"]
            + [str(m) for m in analysis["madde_nos"]]
            + analysis["keywords"]
        )

        return {
            "question": question,
            "answer": response.content,
            "keywords": used_keywords[:10],
            "sources": all_entity_names,
            "madde_sources": madde_sources,
            "context": context,
        }


# Test
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QueryService Test")
    parser.add_argument("--provider", type=str, default=None, help="LLM provider: openai, anthropic")
    parser.add_argument("--model", type=str, default=None, help="Model ismi")
    args = parser.parse_args()
    
    print("=" * 60)
    print("QueryService Test")
    print("=" * 60)

    service = QueryService(provider=args.provider, model=args.model)

    test_questions = [
        "Hirsizlik sucunun cezasi nedir?",
        "Madde 81 ne diyor?",
        "Kasten oldurme ile taksirle oldurme arasindaki fark nedir?",
        "Gece vakti birinin evine girip esya calmanin cezasi ne olur?",
        "Mesru mudafaa nedir, hangi maddede duzenlenmistir?",
    ]

    for q in test_questions:
        print(f"\nSoru: {q}")
        result = service.query(q)
        print(f"Keywords: {result['keywords']}")
        print(f"Madde kaynaklari: {result['madde_sources']}")
        print(f"Yanit:\n{result['answer'][:300]}...")
        print("-" * 60)
