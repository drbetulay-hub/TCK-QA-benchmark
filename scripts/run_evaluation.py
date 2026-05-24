"""
Evaluation - Adım 1: Ham Sonuçları Topla

Her soru için GraphRAG ve/veya BasicRAG'ı çalıştırıp sonuçları kaydeder.
Sonuçlar evaluate_ragas.py'de kullanılacak.

Kullanım:
    # GPT-4o ile (varsayılan)
    python scripts/run_evaluation.py --limit 5                    # ilk 5 soru, her iki sistem
    python scripts/run_evaluation.py --limit 5 --system graphrag  # sadece graphrag
    python scripts/run_evaluation.py --system basicrag            # tüm sorular, sadece basicrag
    python scripts/run_evaluation.py                              # tüm sorular, her iki sistem
    python scripts/run_evaluation.py --resume                     # kaldığı yerden devam et
    
    # Claude ile
    python scripts/run_evaluation.py --provider anthropic --model claude-sonnet-4-20250514
    python scripts/run_evaluation.py --provider anthropic --model claude-sonnet-4-20250514 --limit 5
    
    # Farklı output suffix (aynı sistemin farklı modellerle karşılaştırması için)
    python scripts/run_evaluation.py --provider openai --model gpt-4o --suffix gpt4o
    python scripts/run_evaluation.py --provider anthropic --model claude-sonnet-4-20250514 --suffix claude
"""

import argparse
import json
import os
import sys
import time

# Proje root'unu path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tck_graphrag._paths import load_project_dotenv
load_project_dotenv()


DATASET_PATH = "data/evaluation/TCK-QA_dataset.jsonl"
OUTPUT_DIR = "data/evaluation/results"


def load_dataset(limit: int = 0) -> list[dict]:
    """Evaluation dataset'ini yükle."""
    data = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    if limit > 0:
        data = data[:limit]
    print(f"Dataset: {len(data)} soru yüklendi")
    return data


def load_existing_results(path: str) -> dict:
    """Mevcut sonuçları yükle (resume için). Soru -> sonuç dict'i döner."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)
    return {r["question"]: r for r in results}


def query_with_retry(service, question: str, max_retries: int = 8) -> dict:
    """API rate limit / kota hatalarında üstel bekleme ile yeniden dene."""
    for attempt in range(max_retries):
        try:
            return service.query(question)
        except Exception as e:
            err = str(e).lower()
            retryable = any(
                x in err
                for x in ("429", "rate", "quota", "timeout", "overloaded", "503")
            )
            if not retryable or attempt == max_retries - 1:
                raise
            wait = min(120, 5 * (2 ** attempt))
            print(f"    Bekleniyor {wait}s (deneme {attempt + 2}/{max_retries})...")
            time.sleep(wait)


def run_graphrag(
    dataset: list[dict],
    resume: bool = False,
    provider: str = None,
    model: str = None,
    suffix: str = None,
) -> list[dict]:
    """
    GraphRAG ile tüm soruları çalıştır.
    
    Args:
        dataset: Evaluation dataset
        resume: Kaldığı yerden devam et
        provider: LLM provider ("openai" veya "anthropic")
        model: LLM model ismi
        suffix: Output dosya suffix'i (örn: "gpt4o", "claude")
    """
    from tck_graphrag.services.graphrag import QueryService

    # Output dosya adını belirle
    file_suffix = f"_graphrag_{suffix}" if suffix else "_graphrag"
    output_path = os.path.join(OUTPUT_DIR, f"results{file_suffix}.json")
    existing = load_existing_results(output_path) if resume else {}

    if existing:
        print(f"  Resume: {len(existing)} mevcut sonuç bulundu, kalan sorular çalıştırılacak")

    service = QueryService(provider=provider, model=model)
    system_name = f"graphrag_{suffix}" if suffix else "graphrag"
    results = list(existing.values())

    for i, item in enumerate(dataset):
        soru = item["soru"]

        # Resume: zaten varsa atla
        if soru in existing:
            continue

        print(f"  [{i+1}/{len(dataset)}] {soru[:80]}...")

        try:
            start = time.time()
            result = service.query(soru)
            elapsed = round(time.time() - start, 2)

            results.append({
                "question": soru,
                "ground_truth_answer": item["cevap"],
                "ground_truth_context": item["context"],
                "relevant_madde_list": item["relevant_madde_list"],
                "soru_tipi": item["soru_tipi"],
                "system_answer": result["answer"],
                "system_context": result.get("context", ""),
                "system_madde_sources": result.get("madde_sources", []),
                "system_keywords": result.get("keywords", []),
                "system_sources": result.get("sources", []),
                "processing_time": elapsed,
                "system": system_name,
                "llm_provider": service.provider,
                "llm_model": service.model,
            })

        except Exception as e:
            print(f"    HATA: {e}")
            results.append({
                "question": soru,
                "ground_truth_answer": item["cevap"],
                "ground_truth_context": item["context"],
                "relevant_madde_list": item["relevant_madde_list"],
                "soru_tipi": item["soru_tipi"],
                "system_answer": f"HATA: {e}",
                "system_context": "",
                "system_madde_sources": [],
                "system_keywords": [],
                "system_sources": [],
                "processing_time": 0,
                "system": system_name,
                "llm_provider": provider,
                "llm_model": model,
                "error": True,
            })

        # Her 5 soruda bir kaydet (crash'e karşı)
        if (i + 1) % 5 == 0:
            _save_results(results, output_path)

    _save_results(results, output_path)
    return results


def run_basicrag(
    dataset: list[dict],
    resume: bool = False,
    provider: str = None,
    model: str = None,
    suffix: str = None,
) -> list[dict]:
    """
    BasicRAG ile tüm soruları çalıştır.
    
    Args:
        dataset: Evaluation dataset
        resume: Kaldığı yerden devam et
        provider: LLM provider ("openai" veya "anthropic")
        model: LLM model ismi
        suffix: Output dosya suffix'i (örn: "gpt4o", "claude")
    """
    from tck_graphrag.services.basic_rag import BasicRAGService

    # Output dosya adını belirle
    file_suffix = f"_basicrag_{suffix}" if suffix else "_basicrag"
    output_path = os.path.join(OUTPUT_DIR, f"results{file_suffix}.json")
    existing = load_existing_results(output_path) if resume else {}

    if existing:
        print(f"  Resume: {len(existing)} mevcut sonuç bulundu, kalan sorular çalıştırılacak")

    service = BasicRAGService(provider=provider, model=model)
    system_name = f"basicrag_{suffix}" if suffix else "basicrag"
    results = list(existing.values())

    for i, item in enumerate(dataset):
        soru = item["soru"]

        if soru in existing:
            continue

        print(f"  [{i+1}/{len(dataset)}] {soru[:80]}...")

        try:
            start = time.time()
            result = query_with_retry(service, soru)
            elapsed = round(time.time() - start, 2)

            results.append({
                "question": soru,
                "ground_truth_answer": item["cevap"],
                "ground_truth_context": item["context"],
                "relevant_madde_list": item["relevant_madde_list"],
                "soru_tipi": item["soru_tipi"],
                "system_answer": result["answer"],
                "system_context": result.get("context", ""),
                "system_madde_sources": result.get("madde_sources", []),
                "system_keywords": result.get("keywords", []),
                "system_sources": result.get("sources", []),
                "processing_time": elapsed,
                "system": system_name,
                "llm_provider": service.provider,
                "llm_model": service.model,
            })
            existing[soru] = results[-1]
            time.sleep(1)  # embedding burst'ü azalt

        except Exception as e:
            print(f"    HATA (atlandı, resume ile tekrar denenecek): {e}")

        if (i + 1) % 5 == 0:
            _save_results(results, output_path)

    _save_results(results, output_path)
    return results


def _save_results(results: list[dict], path: str):
    """Sonuçları JSON'a kaydet."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  -> {len(results)} sonuç kaydedildi: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation sonuçlarını topla",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # GPT-4o ile (varsayılan)
  python scripts/run_evaluation.py --limit 5
  
  # Claude ile
  python scripts/run_evaluation.py --provider anthropic --model claude-sonnet-4-20250514 --suffix claude
  
  # Farklı modelleri karşılaştır
  python scripts/run_evaluation.py --provider openai --model gpt-4o --suffix gpt4o
  python scripts/run_evaluation.py --provider anthropic --model claude-sonnet-4-20250514 --suffix claude
        """
    )
    parser.add_argument("--limit", type=int, default=0, help="İlk N soru (0=hepsi)")
    parser.add_argument("--system", choices=["graphrag", "basicrag", "both"],
                        default="both", help="Hangi sistem(ler); both = graphrag + basicrag")
    parser.add_argument("--resume", action="store_true",
                        help="Kaldığı yerden devam et")
    parser.add_argument("--provider", type=str, choices=["openai", "anthropic"],
                        default=None, help="LLM provider (varsayılan: config'den)")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model ismi (varsayılan: config'den)")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Output dosya suffix'i (örn: gpt4o, claude)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = load_dataset(args.limit)
    
    # Provider/model bilgisini göster
    provider_info = args.provider or "config'den"
    model_info = args.model or "config'den"
    print(f"\nLLM: {provider_info} / {model_info}")
    if args.suffix:
        print(f"Output suffix: {args.suffix}")

    if args.system in ("graphrag", "both"):
        print("\n=== GraphRAG Evaluation ===")
        results = run_graphrag(
            dataset,
            resume=args.resume,
            provider=args.provider,
            model=args.model,
            suffix=args.suffix,
        )
        errors = sum(1 for r in results if r.get("error"))
        print(f"Tamamlandı: {len(results)} sonuç, {errors} hata")

    if args.system in ("basicrag", "both"):
        print("\n=== BasicRAG Evaluation ===")
        results = run_basicrag(
            dataset,
            resume=args.resume,
            provider=args.provider,
            model=args.model,
            suffix=args.suffix,
        )
        errors = sum(1 for r in results if r.get("error"))
        print(f"Tamamlandı: {len(results)} sonuç, {errors} hata")


    print("\nBitti! Şimdi evaluate_ragas.py ile metrikleri hesaplayın.")


if __name__ == "__main__":
    main()
