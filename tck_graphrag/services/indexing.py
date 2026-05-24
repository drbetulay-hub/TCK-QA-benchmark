"""
GraphRAG Servisi - V3 (Faz 1 Güncellemesi)

Bu modül, LangChain kullanarak metinden entity ve relationship çıkarır
ve Neo4j veritabanına kaydeder.

V3 Değişiklikleri (Faz 1):
--------------------------
1. Çoklu label sistemi: Entity tipine göre ek Neo4j label'ı
2. Native ilişki tipleri: RELATES_TO yerine CEZA_OLARAK, TANIMLAR vb.
3. Entity normalizasyon: Kaydetmeden önce isim standartlaştırma
4. Prompt'lar ayrı modülde: app/prompts/extraction_prompt.py
"""

import os
import json
import re
from typing import Optional
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from tck_graphrag.core.database import get_neo4j
from tck_graphrag.core.config import get_settings
from tck_graphrag.core.normalization import (
    normalize_entity_name,
    normalize_entity_type,
    normalize_relationship_type,
    get_neo4j_label,
    extract_madde_no,
    VALID_ENTITY_TYPES,
    VALID_RELATIONSHIP_TYPES
)
from tck_graphrag.prompts.extraction_prompt import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE
)

from tck_graphrag._paths import load_project_dotenv
load_project_dotenv()


@dataclass
class Entity:
    """Bir entity (varlık) temsil eder - source_text dahil"""
    name: str
    type: str
    description: str = ""
    source_text: str = ""  # Kaynak metin
    madde_no: Optional[int] = None  # İlgili madde numarası

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "source_text": self.source_text,
            "madde_no": self.madde_no
        }


@dataclass
class Relationship:
    """İki entity arasındaki ilişki - source_text dahil"""
    source: str
    target: str
    type: str
    description: str = ""
    source_text: str = ""  # İlişkinin geçtiği kaynak metin

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "description": self.description,
            "source_text": self.source_text
        }


@dataclass
class ExtractionResult:
    """Bir metin chunk'ından çıkarılan tüm bilgiler"""
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    source_page: int = 0
    source_text: str = ""  # Tüm chunk metni

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "source_page": self.source_page,
            "source_text": self.source_text[:500] if self.source_text else ""
        }


class GraphRAGService:
    """
    GraphRAG Ana Servisi - V3

    Bu servis, metinden Knowledge Graph oluşturur.
    Çoklu label + native ilişki tipleri + normalizasyon desteği.
    """

    # Entity tipleri (referans için — normalization modülünden de gelen VALID_ENTITY_TYPES ile aynı)
    ENTITY_TYPES = list(VALID_ENTITY_TYPES)

    # Relationship tipleri
    RELATIONSHIP_TYPES = list(VALID_RELATIONSHIP_TYPES)

    def __init__(self, model: str = "gpt-4o"):
        settings = get_settings()
        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY bulunamadı!")

        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            api_key=api_key
        )

        self.model = model
        self.neo4j = get_neo4j()

        self._setup_extraction_prompt()
        print(f"✅ GraphRAGService V3 başlatıldı (Model: {model})")

    def _setup_extraction_prompt(self):
        """
        Extraction prompt'unu app/prompts/ modülünden yükler.
        Prompt mühendisliği extraction_prompt.py dosyasında yapılır.
        """
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", EXTRACTION_SYSTEM_PROMPT),
            ("human", EXTRACTION_USER_TEMPLATE)
        ])

        self.json_parser = JsonOutputParser()

    def extract_from_text(self, text: str, page_no: int = 0) -> ExtractionResult:
        """
        Metinden entity ve relationship çıkarır.
        Çıkarılan entity'ler normalizasyon'dan geçirilir.
        """
        try:
            chain = self.extraction_prompt | self.llm
            response = chain.invoke({"text": text})

            content = response.content

            # JSON temizle
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            data = json.loads(content)

            # Entity'leri oluştur — normalizasyon uygula
            entities = []
            for e in data.get("entities", []):
                raw_name = e.get("name", "")
                raw_type = e.get("type", "KAVRAM")

                if not raw_name:
                    continue  # Boş entity'leri atla

                # Normalizasyon
                normalized_name = normalize_entity_name(raw_name)
                normalized_type = normalize_entity_type(raw_type)

                # Madde numarası: ya LLM verdi ya da isimden çıkar
                madde_no = e.get("madde_no")
                if madde_no is None and normalized_type == "MADDE":
                    madde_no = extract_madde_no(normalized_name)

                entities.append(Entity(
                    name=normalized_name,
                    type=normalized_type,
                    description=e.get("description", ""),
                    source_text=e.get("source_text", ""),
                    madde_no=madde_no
                ))

            # Relationship'leri oluştur — normalizasyon uygula
            relationships = []
            for r in data.get("relationships", []):
                raw_source = r.get("source", "")
                raw_target = r.get("target", "")
                raw_type = r.get("type", "ILGILI")

                if not raw_source or not raw_target:
                    continue  # Eksik kaynak/hedef atla

                relationships.append(Relationship(
                    source=normalize_entity_name(raw_source),
                    target=normalize_entity_name(raw_target),
                    type=normalize_relationship_type(raw_type),
                    description=r.get("description", ""),
                    source_text=r.get("source_text", "")
                ))

            return ExtractionResult(
                entities=entities,
                relationships=relationships,
                source_page=page_no,
                source_text=text[:1000]
            )

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse hatası: {e}")
            return ExtractionResult(source_page=page_no, source_text=text[:500])
        except Exception as e:
            print(f"❌ Extraction hatası: {e}")
            return ExtractionResult(source_page=page_no, source_text=text[:500])

    def save_to_neo4j(self, result: ExtractionResult) -> dict:
        """
        Extraction sonuçlarını Neo4j'e kaydeder.

        V3 Değişiklikleri:
        - Çoklu label: Entity tipine göre ek label eklenir
          (:Entity) → (:Entity:Suc), (:Entity:Madde), (:Entity:Ceza) vb.
        - Native ilişki tipleri: RELATES_TO yerine gerçek ilişki tipleri
          [:RELATES_TO {type: 'X'}] → [:CEZA_OLARAK], [:TANIMLAR] vb.
        """
        if not self.neo4j.driver:
            self.neo4j.connect()

        saved_entities = 0
        saved_relationships = 0

        # Entity'leri kaydet — çoklu label ile
        for entity in result.entities:
            # Entity tipine göre Neo4j label belirle
            extra_label = get_neo4j_label(entity.type)

            # ON CREATE: İlk kez oluşturuluyorsa tüm alanları yaz
            # ON MATCH:  Zaten varsa madde_no_list'e ekle, 
            #            source_text daha uzunsa güncelle
            # Bu sayede "Hırsızlık" ilk kez Madde 141'de oluşturulur,
            # Madde 290'da tekrar geçtiğinde madde_no=141 korunur
            query = f"""
            MERGE (e:Entity {{name: $name}})
            ON CREATE SET
                e.type = $type,
                e.description = $description,
                e.source_text = $source_text,
                e.source_page = $source_page,
                e.madde_no = $madde_no,
                e.madde_no_list = CASE WHEN $madde_no IS NOT NULL THEN [$madde_no] ELSE [] END
            ON MATCH SET
                e.madde_no = CASE
                    WHEN e.madde_no IS NULL AND $madde_no IS NOT NULL THEN $madde_no
                    WHEN $type = 'MADDE' AND $madde_no IS NOT NULL THEN $madde_no
                    WHEN $type = 'SUC' AND size(coalesce($source_text, '')) > size(coalesce(e.source_text, '')) AND $madde_no IS NOT NULL THEN $madde_no
                    WHEN $type <> 'MADDE' AND size(coalesce($source_text, '')) > size(coalesce(e.source_text, '')) + 20 AND $madde_no IS NOT NULL THEN $madde_no
                    ELSE e.madde_no
                END,
                e.madde_no_list = CASE
                    WHEN $madde_no IS NULL THEN coalesce(e.madde_no_list, [])
                    WHEN $madde_no IN coalesce(e.madde_no_list, []) THEN e.madde_no_list
                    ELSE coalesce(e.madde_no_list, []) + [$madde_no]
                END,
                e.source_text = CASE
                    WHEN size(coalesce(e.source_text, '')) < size(coalesce($source_text, ''))
                    THEN $source_text
                    ELSE e.source_text
                END,
                e.description = CASE
                    WHEN size(coalesce(e.description, '')) < size(coalesce($description, ''))
                    THEN $description
                    ELSE e.description
                END
            SET e:{extra_label}
            RETURN e
            """
            try:
                self.neo4j.run_query(query, {
                    "name": entity.name,
                    "type": entity.type,
                    "description": entity.description,
                    "source_text": entity.source_text,
                    "source_page": result.source_page,
                    "madde_no": entity.madde_no
                })
                saved_entities += 1
            except Exception as e:
                print(f"⚠️ Entity kayıt hatası ({entity.name}): {e}")

        # Relationship'leri kaydet — native ilişki tipleri ile
        for rel in result.relationships:
            # İlişki tipini doğrula (whitelist)
            rel_type = rel.type if rel.type in VALID_RELATIONSHIP_TYPES else "ILGILI"

            # Native ilişki tipi ile MERGE sorgusu
            # Örn: MERGE (s)-[:CEZA_OLARAK]->(t)
            query = f"""
            MATCH (source:Entity {{name: $source}})
            MATCH (target:Entity {{name: $target}})
            MERGE (source)-[r:{rel_type}]->(target)
            SET r.description = $description,
                r.source_text = $source_text
            RETURN r
            """
            try:
                self.neo4j.run_query(query, {
                    "source": rel.source,
                    "target": rel.target,
                    "description": rel.description,
                    "source_text": rel.source_text
                })
                saved_relationships += 1
            except Exception as e:
                print(f"⚠️ Relationship kayıt hatası: {e}")

        return {
            "saved_entities": saved_entities,
            "saved_relationships": saved_relationships
        }

    def clear_graph(self):
        """Neo4j'deki tüm verileri siler."""
        if not self.neo4j.driver:
            self.neo4j.connect()
        self.neo4j.run_query("MATCH (n) DETACH DELETE n")
        print("🗑️ Graph temizlendi")

    def get_graph_stats(self) -> dict:
        """Graph istatistiklerini döndürür"""
        if not self.neo4j.driver:
            self.neo4j.connect()
        return self.neo4j.get_database_info()


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 GraphRAG Service V3 Test")
    print("=" * 60)

    service = GraphRAGService(model="gpt-4o")

    # Test metni
    test_text = """
    Hırsızlık
    Madde 141- (1) Zilyedinin rızası olmadan başkasına ait taşınır bir malı,
    kendisine veya başkasına bir yarar sağlamak maksadıyla bulunduğu yerden
    alan kimseye bir yıldan üç yıla kadar hapis cezası verilir.

    Nitelikli hırsızlık
    Madde 142- (1) Hırsızlık suçunun;
    a) Kime ait olursa olsun kamu kurum ve kuruluşlarında veya ibadete
    ayrılmış yerlerde bulunan eşya hakkında,
    işlenmesi hâlinde, üç yıldan yedi yıla kadar hapis cezasına hükmolunur.
    (2) Suçun gece vakti işlenmesi halinde, ceza yarı oranında artırılır.
    """

    print("\n📝 Test metni işleniyor...")
    result = service.extract_from_text(test_text, page_no=52)

    print(f"\n📊 Sonuçlar:")
    print(f"   Entity sayısı: {len(result.entities)}")
    print(f"   Relationship sayısı: {len(result.relationships)}")

    print("\n🔖 Entity'ler:")
    for e in result.entities:
        print(f"   [{e.type:8}] {e.name}")
        if e.source_text:
            print(f"            └─ \"{e.source_text[:60]}...\"")

    print("\n🔗 Relationship'ler:")
    for r in result.relationships:
        print(f"   {r.source} --[{r.type}]--> {r.target}")
        if r.source_text:
            print(f"   └─ \"{r.source_text[:60]}...\"")
