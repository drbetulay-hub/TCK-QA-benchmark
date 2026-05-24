"""
Basic RAG Indexing Script

tck_maddeler.jsonl → OpenAI embedding → PostgreSQL (pgvector)

Kullanım:
    python scripts/index_basic_rag.py                # tüm maddeleri indexle
    python scripts/index_basic_rag.py --clear         # tabloyu sıfırla + indexle
    python scripts/index_basic_rag.py --limit 10      # ilk 10 madde
"""

import argparse
import json
import sys
import os

import psycopg2
from psycopg2.extras import execute_values
from langchain_openai import OpenAIEmbeddings

from tck_graphrag._paths import load_project_dotenv
load_project_dotenv()

JSONL_PATH = "data/tck_maddeler.jsonl"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "postgres"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )


def setup_table(conn):
    """pgvector extension + tablo oluştur (index veri yüklendikten sonra)."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS tck_embeddings (
                id SERIAL PRIMARY KEY,
                madde_no INTEGER UNIQUE NOT NULL,
                baslik TEXT,
                icerik TEXT,
                kitap TEXT,
                kisim TEXT,
                bolum TEXT,
                embedding vector({EMBEDDING_DIM})
            )
        """)
    conn.commit()


def create_index(conn):
    """IVFFlat index — veri yüklendikten sonra çağrılmalı."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_tck_embedding
            ON tck_embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 20)
        """)
    conn.commit()
    print("IVFFlat index oluşturuldu.")


def clear_table(conn):
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS tck_embeddings")
    conn.commit()
    print("Tablo silindi.")


def load_maddeler(limit=None):
    maddeler = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            maddeler.append(json.loads(line))
    if limit:
        maddeler = maddeler[:limit]
    return maddeler


def build_text(madde: dict) -> str:
    """Embedding için madde metnini oluştur."""
    parts = [f"Madde {madde['madde_no']} - {madde['baslik']}"]
    if madde.get("bolum"):
        parts.append(f"Bölüm: {madde['bolum']}")
    parts.append(madde["icerik"])
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Basic RAG Indexing")
    parser.add_argument("--clear", action="store_true", help="Tabloyu sıfırla")
    parser.add_argument("--limit", type=int, default=None, help="Kaç madde indexlensin")
    args = parser.parse_args()

    conn = get_conn()

    if args.clear:
        clear_table(conn)

    setup_table(conn)

    # Zaten indexlenmiş maddeleri bul
    with conn.cursor() as cur:
        cur.execute("SELECT madde_no FROM tck_embeddings")
        existing = {row[0] for row in cur.fetchall()}

    maddeler = load_maddeler(args.limit)
    to_index = [m for m in maddeler if m["madde_no"] not in existing]

    if not to_index:
        print(f"Tüm maddeler zaten indexli ({len(existing)} madde).")
        conn.close()
        return

    print(f"{len(to_index)} madde indexlenecek (mevcut: {len(existing)})")

    # Embedding
    embeddings_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    texts = [build_text(m) for m in to_index]

    print("Embedding oluşturuluyor...")
    vectors = embeddings_model.embed_documents(texts)
    print(f"{len(vectors)} embedding oluşturuldu.")

    # Postgres'e yaz
    rows = []
    for m, vec in zip(to_index, vectors):
        rows.append((
            m["madde_no"],
            m.get("baslik", ""),
            m["icerik"],
            m.get("kitap", ""),
            m.get("kisim", ""),
            m.get("bolum", ""),
            vec,
        ))

    with conn.cursor() as cur:
        execute_values(
            cur,
            """INSERT INTO tck_embeddings
               (madde_no, baslik, icerik, kitap, kisim, bolum, embedding)
               VALUES %s
               ON CONFLICT (madde_no) DO UPDATE SET
                   baslik = EXCLUDED.baslik,
                   icerik = EXCLUDED.icerik,
                   embedding = EXCLUDED.embedding""",
            rows,
            template="(%s, %s, %s, %s, %s, %s, %s::vector)",
        )
    conn.commit()

    # IVFFlat index'i veri yüklendikten sonra oluştur
    create_index(conn)

    conn.close()

    print(f"Tamamlandı: {len(rows)} madde indexlendi.")


if __name__ == "__main__":
    main()
