#!/usr/bin/env python3
"""
Neo4j Schema Setup Scripti

Bu script, Neo4j veritabanında index ve constraint'ler oluşturur.
Graph şemasının kurallarını tanımlar.

Kullanım:
    python scripts/setup_schema.py              # Index ve constraint kur
    python scripts/setup_schema.py --clear      # Önce graph'ı temizle, sonra kur
    python scripts/setup_schema.py --info       # Mevcut şema bilgisini göster

Bu scripti çalıştırmadan önce Neo4j'in açık olması gerekir!
"""

import argparse
import sys
from pathlib import Path

# Proje kök dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tck_graphrag.core.database import get_neo4j


# ===========================
# SCHEMA TANIMLARI
# ===========================

# Constraint'ler — veri bütünlüğü kuralları
CONSTRAINTS = [
    {
        "name": "entity_name_unique",
        "query": "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        "description": "Her entity'nin adı benzersiz olmalı"
    }
]

# Index'ler — sorgu performansı
INDEXES = [
    {
        "name": "entity_type_index",
        "query": "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        "description": "Entity tipi üzerinden hızlı arama"
    },
    {
        "name": "entity_madde_no_index",
        "query": "CREATE INDEX entity_madde_no_index IF NOT EXISTS FOR (e:Entity) ON (e.madde_no)",
        "description": "Madde numarası üzerinden hızlı arama"
    },
    {
        "name": "entity_fulltext",
        "query": "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description]",
        "description": "Entity isim ve aciklama uzerinden fulltext arama"
    }
]


def setup_schema(neo4j):
    """Index ve constraint'leri oluşturur."""
    
    print("\n📋 Constraint'ler oluşturuluyor...")
    for constraint in CONSTRAINTS:
        try:
            neo4j.run_query(constraint["query"])
            print(f"   ✅ {constraint['name']}: {constraint['description']}")
        except Exception as e:
            if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                print(f"   ℹ️  {constraint['name']}: Zaten mevcut")
            else:
                print(f"   ❌ {constraint['name']}: {e}")
    
    print("\n📊 Index'ler oluşturuluyor...")
    for index in INDEXES:
        try:
            neo4j.run_query(index["query"])
            print(f"   ✅ {index['name']}: {index['description']}")
        except Exception as e:
            if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
                print(f"   ℹ️  {index['name']}: Zaten mevcut")
            else:
                print(f"   ❌ {index['name']}: {e}")


def clear_graph(neo4j):
    """Graph'taki tüm verileri siler."""
    print("\n🗑️  Graph temizleniyor...")
    neo4j.run_query("MATCH (n) DETACH DELETE n")
    print("   ✅ Tüm node ve relationship'ler silindi")


def show_info(neo4j):
    """Mevcut graph bilgisini gösterir."""
    print("\n📊 Mevcut Graph Bilgisi:")
    
    # Node ve relationship sayısı
    info = neo4j.get_database_info()
    print(f"   Node sayısı: {info['node_count']}")
    print(f"   Relationship sayısı: {info['relationship_count']}")
    
    # Label'lar
    try:
        labels = neo4j.run_query("CALL db.labels()")
        label_names = [l.get("label", str(l)) for l in labels]
        print(f"   Label'lar: {', '.join(label_names) if label_names else 'yok'}")
    except Exception:
        print("   Label'lar: bilgi alınamadı")
    
    # Relationship tipleri
    try:
        rel_types = neo4j.run_query("CALL db.relationshipTypes()")
        rel_names = [r.get("relationshipType", str(r)) for r in rel_types]
        print(f"   İlişki tipleri: {', '.join(rel_names) if rel_names else 'yok'}")
    except Exception:
        print("   İlişki tipleri: bilgi alınamadı")
    
    # Index'ler
    try:
        indexes = neo4j.run_query("SHOW INDEXES")
        print(f"   Index sayısı: {len(indexes)}")
        for idx in indexes:
            name = idx.get("name", "?")
            state = idx.get("state", "?")
            print(f"      - {name} ({state})")
    except Exception:
        print("   Index bilgisi: alınamadı")
    
    # Constraint'ler
    try:
        constraints = neo4j.run_query("SHOW CONSTRAINTS")
        print(f"   Constraint sayısı: {len(constraints)}")
        for con in constraints:
            name = con.get("name", "?")
            print(f"      - {name}")
    except Exception:
        print("   Constraint bilgisi: alınamadı")
    
    # Entity tip dağılımı
    if info['node_count'] > 0:
        try:
            type_counts = neo4j.run_query("""
                MATCH (e:Entity)
                RETURN e.type as type, count(*) as count
                ORDER BY count DESC
            """)
            print(f"\n   Entity tip dağılımı:")
            for tc in type_counts:
                print(f"      {tc['type']}: {tc['count']}")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Neo4j Schema Setup — Index ve Constraint yönetimi"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Graph'ı temizle ve schema'yı kur"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Mevcut graph bilgisini göster"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔧 Neo4j Schema Setup")
    print("=" * 60)
    
    # Neo4j bağlantısı
    neo4j = get_neo4j()
    if not neo4j.connect():
        print("\n❌ Neo4j'e bağlanılamadı! Neo4j Desktop'ta instance'ın çalıştığından emin olun.")
        sys.exit(1)
    
    try:
        if args.info:
            show_info(neo4j)
        else:
            if args.clear:
                clear_graph(neo4j)
            
            setup_schema(neo4j)
            print("\n✅ Schema setup tamamlandı!")
            
            # Bilgileri göster
            show_info(neo4j)
    
    finally:
        neo4j.close()


if __name__ == "__main__":
    main()
