#!/usr/bin/env python3
"""
TCK Indexing Script - V5 (Optimize & Cache)

Pipeline:
  1. PDF → JSONL (export_maddeler.py ile bir kez yapılır)
  2. JSONL → LLM → Cache (LLM yanıtları data/llm_cache/ klasörüne kaydedilir)
  3. Cache → Neo4j (LLM cache'den okunur, Neo4j'e yazılır)

Yenilikler (V5):
  - LLM yanıt cache: Her LLM yanıtı diske kaydedilir, tekrar çağrı gerekmez
  - Madde full-text: JSONL'deki tam madde metni Madde node'una "full_text" olarak yazılır
  - Maliyet tahmini: Input + output token dahil
  - ON CREATE/ON MATCH: Entity madde_no üzerine yazılmaz, array olarak biriktirilir

Kullanım:
    python scripts/index_tck.py                    # Tüm maddeleri indexle
    python scripts/index_tck.py --model gpt-4o-mini # Ucuz model
    python scripts/index_tck.py --limit 10          # Sadece ilk 10 madde
    python scripts/index_tck.py --start 50           # Madde 50'den başla
    python scripts/index_tck.py --clear              # Önce graph'ı temizle
    python scripts/index_tck.py --dry-run            # Neo4j'e kaydetme, sadece test
    python scripts/index_tck.py --from-cache         # LLM çağrısı yapmadan cache'den yaz

Ön koşul:
    python scripts/export_maddeler.py   # JSONL dosyasını oluştur
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Proje kök dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tck_graphrag.services.indexing import GraphRAGService

# Dosya yolları
JSONL_FILE = project_root / "data" / "tck_maddeler.jsonl"
CHECKPOINT_FILE = project_root / "data" / "indexing_checkpoint.json"
LLM_CACHE_DIR = project_root / "data" / "llm_cache"


def load_jsonl() -> list[dict]:
    """JSONL dosyasından maddeleri yükle."""
    if not JSONL_FILE.exists():
        print(f"❌ JSONL dosyası bulunamadı: {JSONL_FILE}")
        print("   Önce export edin: python scripts/export_maddeler.py")
        sys.exit(1)

    maddeler = []
    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                maddeler.append(json.loads(line))

    return maddeler


def load_checkpoint() -> dict:
    """Checkpoint dosyasını yükle."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "processed_madde_numbers": [],
        "last_updated": None,
        "model": None,
        "total_entities": 0,
        "total_relationships": 0,
        "errors": []
    }


def save_checkpoint(checkpoint: dict):
    """Checkpoint dosyasını kaydet."""
    checkpoint["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def clear_checkpoint():
    """Checkpoint dosyasını sil."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("   ✅ Checkpoint dosyası silindi")


# ===========================
# LLM Cache
# ===========================
def get_cache_path(madde_no: int) -> Path:
    """LLM cache dosya yolunu döndür."""
    return LLM_CACHE_DIR / f"madde_{madde_no:03d}.json"


def load_from_cache(madde_no: int) -> Optional[dict]:
    """LLM yanıtını cache'den oku."""
    path = get_cache_path(madde_no)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_to_cache(madde_no: int, result_data: dict):
    """LLM yanıtını cache'e kaydet."""
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = get_cache_path(madde_no)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)


def clear_cache():
    """LLM cache klasörünü temizle."""
    if LLM_CACHE_DIR.exists():
        for f in LLM_CACHE_DIR.glob("*.json"):
            f.unlink()
        print("   ✅ LLM cache temizlendi")


# ===========================
# Madde Full-Text Yazma
# ===========================
def write_madde_full_texts(graphrag_service, maddeler: list[dict]):
    """
    Her Madde node'una JSONL'deki tam madde metnini (full_text) yazar.
    Bu LLM çağrısı YAPMAZ — sadece JSONL'den Neo4j'e yazar.
    """
    print("\n📝 Madde full-text'leri yazılıyor...")
    written = 0
    errors = 0

    for madde in maddeler:
        query = """
        MERGE (m:Entity:Madde {name: $name})
        SET m.full_text = $full_text,
            m.baslik = $baslik,
            m.kitap = $kitap,
            m.kisim = $kisim,
            m.bolum = $bolum,
            m.fikra_sayisi = $fikra
        """
        try:
            graphrag_service.neo4j.run_query(query, {
                "name": f"Madde {madde['madde_no']}",
                "full_text": madde["icerik"],
                "baslik": madde.get("baslik", ""),
                "kitap": madde.get("kitap", ""),
                "kisim": madde.get("kisim", ""),
                "bolum": madde.get("bolum", ""),
                "fikra": madde.get("fikra_count", 0)
            })
            written += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"   ⚠️ Full-text yazma hatası (Madde {madde['madde_no']}): {e}")

    print(f"   ✅ {written} maddeye full-text yazıldı ({errors} hata)")
    return written


# ===========================
# Cross-Reference Yazma
# ===========================
def add_cross_references(graphrag_service, maddeler: list[dict]):
    """
    Cross-reference'ları graph'a REFERANS_VERIR ilişkisi olarak ekler.
    Bu fonksiyon LLM çağrısı YAPMAZ.
    """
    print("\n🔗 Cross-reference'lar ekleniyor...")
    total_refs = 0
    errors = 0

    existing_madde_nos = {m["madde_no"] for m in maddeler}

    for madde in maddeler:
        for ref_no in madde.get("cross_references", []):
            if ref_no not in existing_madde_nos:
                continue

            query = """
            MATCH (source:Entity:Madde {madde_no: $source_no})
            MATCH (target:Entity:Madde {madde_no: $target_no})
            MERGE (source)-[r:REFERANS_VERIR]->(target)
            SET r.description = $description
            RETURN r
            """
            try:
                graphrag_service.neo4j.run_query(query, {
                    "source_no": madde["madde_no"],
                    "target_no": ref_no,
                    "description": f"Madde {madde['madde_no']} → Madde {ref_no} atıf"
                })
                total_refs += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"   ⚠️ Referans hatası (Madde {madde['madde_no']} → {ref_no}): {e}")

    print(f"   ✅ {total_refs} cross-reference eklendi ({errors} hata)")
    return total_refs


def main():
    parser = argparse.ArgumentParser(
        description="TCK maddelerini JSONL'den okuyup Neo4j'e yükle (V5 — cache destekli)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python scripts/index_tck.py                    # Tüm maddeleri indexle
  python scripts/index_tck.py --limit 5          # İlk 5 madde (test)
  python scripts/index_tck.py --clear            # Sıfırdan başla
  python scripts/index_tck.py --from-cache       # Cache'den oku, LLM çağırma
  python scripts/index_tck.py --model gpt-4o-mini # Ucuz model
        """
    )

    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        help="Kullanılacak OpenAI modeli (varsayılan: gpt-4o)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="İşlenecek maksimum madde sayısı"
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help="Başlangıç madde numarası"
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="Bitiş madde numarası"
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Önce graph ve checkpoint'i temizle (cache korunur)"
    )
    parser.add_argument(
        "--clear-cache", action="store_true",
        help="LLM cache'i de temizle (--clear ile birlikte kullanilabilir)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Sadece simülasyon yap, Neo4j'e kaydetme"
    )
    parser.add_argument(
        "--skip-refs", action="store_true",
        help="Cross-reference eklemeyi atla"
    )
    parser.add_argument(
        "--from-cache", action="store_true",
        help="LLM çağrısı yapmadan cache'den oku ve Neo4j'e yaz"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("📚 TCK GraphRAG Indexing - V5 (Cache Destekli)")
    print("=" * 60)

    # ===========================
    # 1. JSONL'den maddeleri yükle
    # ===========================
    print(f"\n📋 JSONL yükleniyor: {JSONL_FILE.name}")
    maddeler = load_jsonl()
    print(f"   ✅ {len(maddeler)} madde yüklendi")

    # ===========================
    # 2. Madde filtreleme
    # ===========================
    if args.start:
        maddeler = [m for m in maddeler if m["madde_no"] >= args.start]
    if args.end:
        maddeler = [m for m in maddeler if m["madde_no"] <= args.end]
    if args.limit:
        maddeler = maddeler[:args.limit]

    print(f"\n   İşlenecek madde: {len(maddeler)}")
    if maddeler:
        print(f"   Aralık: Madde {maddeler[0]['madde_no']} - Madde {maddeler[-1]['madde_no']}")

    # ===========================
    # 3. GraphRAG servisini başlat
    # ===========================
    print(f"\n🤖 Model: {args.model}")
    graphrag_service = GraphRAGService(model=args.model)

    # Graph'i temizle
    if args.clear:
        print("\n🗑️ Graph ve checkpoint temizleniyor...")
        if not args.dry_run:
            graphrag_service.clear_graph()
            clear_checkpoint()

    # Cache'i temizle (sadece --clear-cache ile)
    if args.clear_cache:
        print("\n🗑️ LLM cache temizleniyor...")
        if not args.dry_run:
            clear_cache()

    # ===========================
    # 4. Checkpoint yükle
    # ===========================
    checkpoint = load_checkpoint()

    already_processed = set(checkpoint["processed_madde_numbers"])
    if already_processed:
        print(f"\n📋 Checkpoint: {len(already_processed)} madde zaten işlenmiş")
        maddeler = [m for m in maddeler if m["madde_no"] not in already_processed]
        print(f"   Kalan: {len(maddeler)} madde")

    if not maddeler:
        print("\n✅ Tüm maddeler zaten işlenmiş!")
        sys.exit(0)

    # ===========================
    # 5. Cache kontrolü
    # ===========================
    cached_count = sum(1 for m in maddeler if get_cache_path(m["madde_no"]).exists())
    need_llm = len(maddeler) - cached_count

    print(f"\n📦 Cache durumu:")
    print(f"   Cache'de var: {cached_count}")
    print(f"   LLM gerekli: {need_llm}")

    # ===========================
    # 6. Maliyet tahmini (input + output)
    # ===========================
    if need_llm > 0 and not args.from_cache:
        uncached = [m for m in maddeler if not get_cache_path(m["madde_no"]).exists()]
        total_chars = sum(m["char_count"] for m in uncached)
        estimated_input_tokens = total_chars * 3  # prompt + metin
        estimated_output_tokens = estimated_input_tokens * 0.5  # yanıt ~yarısı

        if args.model == "gpt-4o":
            input_cost = (estimated_input_tokens / 1_000_000) * 2.50
            output_cost = (estimated_output_tokens / 1_000_000) * 10.00
        else:  # gpt-4o-mini
            input_cost = (estimated_input_tokens / 1_000_000) * 0.15
            output_cost = (estimated_output_tokens / 1_000_000) * 0.60

        total_cost = input_cost + output_cost

        print(f"\n💰 Maliyet Tahmini (input + output):")
        print(f"   LLM gereken madde: {need_llm}")
        print(f"   Tahmini input token: {estimated_input_tokens:,.0f}")
        print(f"   Tahmini output token: {estimated_output_tokens:,.0f}")
        print(f"   Tahmini maliyet: ~${total_cost:.2f} USD")
        print(f"   Tahmini süre: ~{need_llm * 10 / 60:.0f} dakika")

        if not args.dry_run:
            print("\n⚠️  LLM çağrısı başlayacak. Devam etmek için Enter'a bas, iptal için Ctrl+C")
            try:
                input()
            except KeyboardInterrupt:
                print("\n❌ İptal edildi")
                sys.exit(0)
    elif args.from_cache and cached_count == 0:
        print("\n❌ Cache boş! Önce --from-cache olmadan çalıştırın.")
        sys.exit(1)

    # ===========================
    # 7. Her maddeyi işle
    # ===========================
    print(f"\n{'=' * 60}")
    print("🚀 İndexleme başlıyor...")
    print("=" * 60)

    total_entities = checkpoint.get("total_entities", 0)
    total_relationships = checkpoint.get("total_relationships", 0)
    total_time = 0
    errors = checkpoint.get("errors", [])
    cache_hits = 0

    for i, madde in enumerate(maddeler, 1):
        start_time = time.time()
        madde_no = madde["madde_no"]
        baslik = madde.get("baslik", f"Madde {madde_no}")
        icerik = madde["icerik"]

        # Çok kısa maddeleri atla
        if madde["char_count"] < 30:
            print(f"   [{i}/{len(maddeler)}] Madde {madde_no}: Atlandı (çok kısa)")
            continue

        print(f"   [{i}/{len(maddeler)}] Madde {madde_no} ({baslik})...",
              end=" ", flush=True)

        try:
            # Cache'den oku veya LLM çağır
            cached = load_from_cache(madde_no)

            if cached and args.from_cache:
                # Cache'den yükle — LLM çağrısı yok
                from tck_graphrag.services.indexing import ExtractionResult, Entity, Relationship
                result = ExtractionResult(
                    entities=[Entity(**e) for e in cached["entities"]],
                    relationships=[Relationship(**r) for r in cached["relationships"]],
                    source_text=cached.get("source_text", ""),
                    source_page=cached.get("source_page", madde_no)
                )
                cache_hits += 1
                elapsed = time.time() - start_time
                total_time += elapsed
                print(f"📦 cache ({len(result.entities)} entity, {len(result.relationships)} rel)", end=" ")

            elif cached and not args.from_cache:
                # Cache var ama LLM modu — cache'den yükle (LLM çağırma)
                from tck_graphrag.services.indexing import ExtractionResult, Entity, Relationship
                result = ExtractionResult(
                    entities=[Entity(**e) for e in cached["entities"]],
                    relationships=[Relationship(**r) for r in cached["relationships"]],
                    source_text=cached.get("source_text", ""),
                    source_page=cached.get("source_page", madde_no)
                )
                cache_hits += 1
                elapsed = time.time() - start_time
                total_time += elapsed
                print(f"📦 cache ({len(result.entities)} entity, {len(result.relationships)} rel)", end=" ")

            else:
                if args.from_cache:
                    print(f"⏭️ cache'de yok, atlanıyor")
                    continue

                # LLM çağrısı
                result = graphrag_service.extract_from_text(
                    icerik,
                    page_no=madde_no
                )
                elapsed = time.time() - start_time
                total_time += elapsed

                # Cache'e kaydet
                cache_data = {
                    "madde_no": madde_no,
                    "baslik": baslik,
                    "source_text": result.source_text,
                    "source_page": result.source_page,
                    "entities": [
                        {
                            "name": e.name, "type": e.type,
                            "description": e.description,
                            "source_text": e.source_text,
                            "madde_no": e.madde_no
                        }
                        for e in result.entities
                    ],
                    "relationships": [
                        {
                            "source": r.source, "target": r.target,
                            "type": r.type, "description": r.description,
                            "source_text": r.source_text
                        }
                        for r in result.relationships
                    ],
                    "cached_at": datetime.now().isoformat()
                }
                save_to_cache(madde_no, cache_data)
                print(f"✅ {len(result.entities)} entity, {len(result.relationships)} rel ({elapsed:.1f}s)", end=" ")

            entity_count = len(result.entities)
            rel_count = len(result.relationships)

            # Neo4j'e kaydet (dry-run değilse)
            if not args.dry_run:
                save_result = graphrag_service.save_to_neo4j(result)
                total_entities += save_result["saved_entities"]
                total_relationships += save_result["saved_relationships"]
            else:
                total_entities += entity_count
                total_relationships += rel_count

            print("→ Neo4j ✅" if not args.dry_run else "")

            # Checkpoint güncelle
            checkpoint["processed_madde_numbers"].append(madde_no)
            checkpoint["total_entities"] = total_entities
            checkpoint["total_relationships"] = total_relationships
            checkpoint["model"] = args.model

            if not args.dry_run:
                save_checkpoint(checkpoint)

        except KeyboardInterrupt:
            print(f"\n\n⚠️ Kullanıcı tarafından durduruldu (Madde {madde_no})")
            if not args.dry_run:
                save_checkpoint(checkpoint)
                print(f"   Checkpoint kaydedildi.")
            break

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Hata ({elapsed:.1f}s): {e}")
            errors.append({"madde_no": madde_no, "error": str(e)})
            checkpoint["errors"] = errors
            if not args.dry_run:
                save_checkpoint(checkpoint)

    # ===========================
    # 8. Madde full-text'lerini yaz
    # ===========================
    if not args.dry_run:
        all_maddeler = load_jsonl()
        write_madde_full_texts(graphrag_service, all_maddeler)

    # ===========================
    # 9. Cross-reference'ları ekle
    # ===========================
    if not args.dry_run and not args.skip_refs:
        all_maddeler = load_jsonl()
        add_cross_references(graphrag_service, all_maddeler)

    # ===========================
    # 10. Özet
    # ===========================
    print(f"\n{'=' * 60}")
    print("📊 İndexleme Özeti")
    print("=" * 60)
    print(f"   İşlenen madde: {len(checkpoint['processed_madde_numbers'])}")
    print(f"   Cache hit: {cache_hits}")
    print(f"   LLM çağrısı: {len(checkpoint['processed_madde_numbers']) - cache_hits}")
    print(f"   Toplam entity: {total_entities}")
    print(f"   Toplam relationship: {total_relationships}")
    print(f"   Toplam süre: {total_time:.1f} saniye ({total_time / 60:.1f} dakika)")
    if maddeler:
        print(f"   Ortalama süre/madde: {total_time / max(len(maddeler), 1):.1f} saniye")

    if errors:
        print(f"\n⚠️ Hatalar ({len(errors)} adet):")
        for err in errors[:5]:
            print(f"   - Madde {err['madde_no']}: {str(err['error'])[:60]}...")

    if not args.dry_run:
        try:
            stats = graphrag_service.get_graph_stats()
            print(f"\n📈 Neo4j Graph İstatistikleri:")
            print(f"   Node sayısı: {stats['node_count']}")
            print(f"   Relationship sayısı: {stats['relationship_count']}")
        except Exception:
            print("\n⚠️ Neo4j istatistikleri alınamadı")
    else:
        print("\n⚠️ Dry-run modu: Neo4j'e kaydedilmedi")

    print("\n✅ İşlem tamamlandı!")


if __name__ == "__main__":
    main()
