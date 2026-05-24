"""
LLM Karşılaştırma Script'i

Farklı LLM'lerle (GPT-4o, Claude, vb.) çalıştırılan evaluation sonuçlarını karşılaştırır.

Kullanım:
    # İki sonuç dosyasını karşılaştır
    python scripts/compare_llms.py \
        --file1 data/evaluation/results/eval_graphrag_gpt4o_detailed.json \
        --file2 data/evaluation/results/eval_graphrag_claude_detailed.json
    
    # Tüm sonuçları karşılaştır (results klasöründeki tüm eval_*_detailed.json dosyaları)
    python scripts/compare_llms.py --all
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = "data/evaluation/results"

# Karşılaştırılacak metrikler
METRIC_COLS = [
    "madde_precision", "madde_recall", "madde_f1",
    "faithfulness", "answer_relevancy", "response_relevancy",
    "factual_correctness", "semantic_similarity", "answer_similarity",
    "llm_context_precision_with_reference", "context_recall", "llm_context_recall",
    "non_llm_context_recall", "processing_time",
]


def load_results(path: str) -> list[dict]:
    """Sonuç dosyasını yükle."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_system_info(results: list[dict]) -> dict:
    """Sonuçlardan sistem bilgisini çıkar."""
    if not results:
        return {"system": "unknown", "provider": "unknown", "model": "unknown"}
    
    first = results[0]
    return {
        "system": first.get("system", "unknown"),
        "provider": first.get("llm_provider", "unknown"),
        "model": first.get("llm_model", "unknown"),
    }


def calc_averages(results: list[dict]) -> dict:
    """Her metrik için ortalama hesapla."""
    avgs = {}
    for col in METRIC_COLS:
        vals = [r[col] for r in results if col in r and r[col] is not None]
        if vals:
            avgs[col] = sum(vals) / len(vals)
    return avgs


def calc_by_soru_tipi(results: list[dict]) -> dict[str, dict]:
    """Soru tipine göre ortalamalar."""
    by_type = defaultdict(list)
    for r in results:
        by_type[r.get("soru_tipi", "unknown")].append(r)
    
    return {stype: calc_averages(items) for stype, items in by_type.items()}


def compare_two_files(path1: str, path2: str):
    """İki sonuç dosyasını karşılaştır."""
    results1 = load_results(path1)
    results2 = load_results(path2)
    
    info1 = get_system_info(results1)
    info2 = get_system_info(results2)
    
    name1 = f"{info1['system']}_{info1['provider']}_{info1['model']}"
    name2 = f"{info2['system']}_{info2['provider']}_{info2['model']}"
    
    print("=" * 90)
    print(f" KARŞILAŞTIRMA: {name1} vs {name2}")
    print("=" * 90)
    
    avgs1 = calc_averages(results1)
    avgs2 = calc_averages(results2)
    
    print(f"\n{'Metrik':45s} | {name1[:15]:>15s} | {name2[:15]:>15s} | {'Fark':>10s}")
    print("-" * 90)
    
    for col in METRIC_COLS:
        if col in avgs1 and col in avgs2:
            v1, v2 = avgs1[col], avgs2[col]
            diff = v2 - v1
            sign = "+" if diff > 0 else ""
            print(f"  {col:43s} | {v1:15.4f} | {v2:15.4f} | {sign}{diff:9.4f}")
    
    # Soru tipi bazlı
    by_type1 = calc_by_soru_tipi(results1)
    by_type2 = calc_by_soru_tipi(results2)
    
    for stype in sorted(set(by_type1.keys()) | set(by_type2.keys())):
        print(f"\n--- {stype} ---")
        t1 = by_type1.get(stype, {})
        t2 = by_type2.get(stype, {})
        
        for col in ["madde_f1", "faithfulness", "semantic_similarity", "processing_time"]:
            if col in t1 and col in t2:
                v1, v2 = t1[col], t2[col]
                diff = v2 - v1
                sign = "+" if diff > 0 else ""
                print(f"  {col:41s} | {v1:15.4f} | {v2:15.4f} | {sign}{diff:9.4f}")


def compare_all():
    """results klasöründeki tüm eval dosyalarını karşılaştır."""
    pattern = os.path.join(RESULTS_DIR, "eval_*_detailed.json")
    files = glob(pattern)
    
    if len(files) < 2:
        print(f"Karşılaştırma için en az 2 dosya gerekli. Bulunan: {len(files)}")
        return
    
    print("=" * 90)
    print(" TÜM SONUÇLARIN KARŞILAŞTIRMASI")
    print("=" * 90)
    
    # Her dosyayı yükle
    all_results = {}
    for f in files:
        results = load_results(f)
        info = get_system_info(results)
        name = f"{info['system']}_{info['provider']}_{info['model']}"
        all_results[name] = {
            "path": f,
            "results": results,
            "info": info,
            "avgs": calc_averages(results),
        }
    
    # Tablo başlığı
    names = list(all_results.keys())
    header = f"{'Metrik':35s} |" + " | ".join(f"{n[:12]:>12s}" for n in names)
    print(f"\n{header}")
    print("-" * len(header))
    
    # Her metrik için satır
    for col in METRIC_COLS:
        row = f"  {col:33s} |"
        for name in names:
            val = all_results[name]["avgs"].get(col, None)
            if val is not None:
                row += f" {val:12.4f} |"
            else:
                row += f" {'N/A':>12s} |"
        print(row)
    
    # En iyi performans özeti
    print("\n" + "=" * 60)
    print(" EN İYİ PERFORMANS")
    print("=" * 60)
    
    for col in ["madde_f1", "faithfulness", "semantic_similarity"]:
        best_name = None
        best_val = -1
        for name, data in all_results.items():
            val = data["avgs"].get(col, -1)
            if val > best_val:
                best_val = val
                best_name = name
        if best_name:
            print(f"  {col:30s}: {best_name} ({best_val:.4f})")
    
    # Processing time için en düşük
    best_name = None
    best_val = float('inf')
    for name, data in all_results.items():
        val = data["avgs"].get("processing_time", float('inf'))
        if val < best_val:
            best_val = val
            best_name = name
    if best_name:
        print(f"  {'processing_time (en düşük)':30s}: {best_name} ({best_val:.4f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="LLM sonuçlarını karşılaştır",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file1", type=str, help="İlk sonuç dosyası")
    parser.add_argument("--file2", type=str, help="İkinci sonuç dosyası")
    parser.add_argument("--all", action="store_true",
                        help="results klasöründeki tüm dosyaları karşılaştır")
    args = parser.parse_args()
    
    if args.all:
        compare_all()
    elif args.file1 and args.file2:
        compare_two_files(args.file1, args.file2)
    else:
        print("Kullanım:")
        print("  python scripts/compare_llms.py --file1 <path1> --file2 <path2>")
        print("  python scripts/compare_llms.py --all")


if __name__ == "__main__":
    main()
