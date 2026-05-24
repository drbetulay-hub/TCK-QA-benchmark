"""
Evaluation - Adım 2: RAGAS Metrikleri + Custom Metrikler + Karşılaştırma

run_evaluation.py çıktılarını okuyup tüm metrikleri hesaplar.

Kullanım:
    # Tek sistem değerlendir
    python scripts/evaluate_ragas.py --input data/evaluation/results_graphrag.json
    python scripts/evaluate_ragas.py --input data/evaluation/results_basicrag.json

    # İlk N sonuçla test et
    python scripts/evaluate_ragas.py --input data/evaluation/results_graphrag.json --limit 5

    # İki sistemi karşılaştır
    python scripts/evaluate_ragas.py --compare

    # RAGAS olmadan sadece custom metrikler
    python scripts/evaluate_ragas.py --input data/evaluation/results_graphrag.json --skip-ragas

    # Eksik RAGAS metriklerini yeniden hesapla
    python scripts/evaluate_ragas.py --input data/evaluation/results/results_graphrag_claude.json --retry-missing
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tck_graphrag._paths import load_project_dotenv
load_project_dotenv()

OUTPUT_DIR = "data/evaluation/results"


# =====================
# Custom Metrikler
# =====================

def calc_madde_metrics(pred_madde: list[int], truth_madde: list[int]) -> dict:
    """Madde bazlı precision, recall, f1 hesapla."""
    import ast
    
    # String olarak kaydedilmiş olabilir, list'e çevir
    if isinstance(pred_madde, str):
        try:
            pred_madde = ast.literal_eval(pred_madde)
        except:
            pred_madde = []
    if isinstance(truth_madde, str):
        try:
            truth_madde = ast.literal_eval(truth_madde)
        except:
            truth_madde = []
    
    pred = set(pred_madde or [])
    truth = set(truth_madde or [])

    if not pred and not truth:
        return {"madde_precision": 1.0, "madde_recall": 1.0, "madde_f1": 1.0}

    tp = len(pred & truth)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "madde_precision": round(precision, 4),
        "madde_recall": round(recall, 4),
        "madde_f1": round(f1, 4),
    }


def compute_custom_metrics(results: list[dict]) -> list[dict]:
    """Her sonuç için custom metrikleri hesapla."""
    enriched = []
    for r in results:
        if r.get("error"):
            metrics = {"madde_precision": 0, "madde_recall": 0, "madde_f1": 0}
        else:
            metrics = calc_madde_metrics(
                r["system_madde_sources"],
                r["relevant_madde_list"]
            )
        enriched.append({**r, **metrics})
    return enriched


# =====================
# RAGAS Metrikleri
# =====================

def _ragas_metric_missing(value, col: str = "faithfulness") -> bool:
    """RAGAS skoru eksik veya geçersiz mi?"""
    if value is None:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    if str(value).lower() in ("nan", "none", ""):
        return True
    return False


def compute_ragas_metrics(
    results: list[dict],
    questions_to_eval: set | None = None,
) -> list[dict]:
    """RAGAS metrikleri hesapla. questions_to_eval verilirse yalnızca o sorular."""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._factual_correctness import FactualCorrectness
    from ragas.metrics._answer_similarity import SemanticSimilarity
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall, NonLLMContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    # RAGAS dataset oluştur
    samples = []
    eval_row_indices = []
    for i, r in enumerate(results):
        if r.get("error"):
            continue
        if questions_to_eval is not None and r["question"] not in questions_to_eval:
            continue

        # retrieved_contexts: sistemin getirdiği context
        retrieved_ctx = r.get("system_context", "")
        if not retrieved_ctx:
            retrieved_ctx = "Bilgi bulunamadı."

        # reference_contexts: ground truth context
        ref_ctx = r.get("ground_truth_context", "")
        if not ref_ctx:
            ref_ctx = "Bilgi bulunamadı."

        samples.append(SingleTurnSample(
            user_input=r["question"],
            response=r["system_answer"],
            reference=r["ground_truth_answer"],
            retrieved_contexts=[retrieved_ctx],
            reference_contexts=[ref_ctx],
        ))
        eval_row_indices.append(i)

    if not samples:
        print("RAGAS: Hesaplanacak örnek yok")
        return results

    dataset = EvaluationDataset(samples=samples)

    # LLM ve embedding wrapper
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    print(f"RAGAS: {len(samples)} örnek değerlendiriliyor...")

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        FactualCorrectness(),
        SemanticSimilarity(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        NonLLMContextRecall(),
    ]

    ragas_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    # Sonuçları birleştir
    scores_df = ragas_result.to_pandas()
    ragas_cols = [c for c in scores_df.columns if c not in
                  ("user_input", "response", "reference", "retrieved_contexts", "reference_contexts")]

    idx_to_scores = {}
    for j, row_i in enumerate(eval_row_indices):
        idx_to_scores[row_i] = {
            col: round(float(scores_df.iloc[j][col]), 4)
            for col in ragas_cols if col in scores_df.columns
        }

    enriched = []
    for i, r in enumerate(results):
        if i in idx_to_scores:
            enriched.append({**r, **idx_to_scores[i]})
        else:
            enriched.append(dict(r))

    return enriched


# =====================
# Raporlama
# =====================

def print_summary(results: list[dict], system_name: str):
    """Genel ve soru_tipi bazlı özet yazdır."""
    # Metrik sütunlarını bul
    metric_cols = [k for k in results[0].keys() if k in (
        "madde_precision", "madde_recall", "madde_f1",
        "faithfulness", "answer_relevancy", "response_relevancy",
        "factual_correctness", "answer_correctness",
        "semantic_similarity", "answer_similarity",
        "context_precision", "llm_context_precision_with_reference",
        "context_recall", "llm_context_recall",
        "non_llm_context_recall", "context_entity_recall",
        "processing_time",
    )]

    if not metric_cols:
        print("Hesaplanmış metrik bulunamadı.")
        return

    print(f"\n{'='*70}")
    print(f" {system_name.upper()} - EVALUATION SONUÇLARI")
    print(f"{'='*70}")

    # Genel ortalama
    print(f"\n--- Genel Ortalama ({len(results)} soru) ---")
    for col in metric_cols:
        vals = [r[col] for r in results if col in r and r[col] is not None]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {col:45s}: {avg:.4f}")

    # Soru tipi bazlı
    by_type = defaultdict(list)
    for r in results:
        by_type[r["soru_tipi"]].append(r)

    for stype, items in sorted(by_type.items()):
        print(f"\n--- {stype} ({len(items)} soru) ---")
        for col in metric_cols:
            vals = [r[col] for r in items if col in r and r[col] is not None]
            if vals:
                avg = sum(vals) / len(vals)
                print(f"  {col:45s}: {avg:.4f}")


def save_detailed_csv(results: list[dict], output_path: str):
    """Her soru için tüm metrikleri CSV'ye yaz."""
    if not results:
        return

    # CSV sütunları: önce temel alanlar, sonra metrikler
    base_cols = ["question", "soru_tipi", "system", "processing_time"]
    answer_cols = ["system_answer", "ground_truth_answer"]
    madde_cols = ["system_madde_sources", "relevant_madde_list"]

    # Metrik sütunlarını otomatik bul
    skip = set(base_cols + answer_cols + madde_cols + [
        "ground_truth_context", "system_context", "system_keywords",
        "system_sources", "error",
    ])
    metric_cols = [k for k in results[0].keys() if k not in skip]

    fieldnames = base_cols + metric_cols + answer_cols + madde_cols

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            # madde listelerini string yap
            row = {**r}
            row["system_madde_sources"] = str(r.get("system_madde_sources", []))
            row["relevant_madde_list"] = str(r.get("relevant_madde_list", []))
            writer.writerow(row)

    print(f"\nDetaylı CSV: {output_path}")


def save_full_json(results: list[dict], output_path: str):
    """Tüm sonuçları (metrikler dahil) JSON'a yaz."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Tam sonuçlar: {output_path}")


def compare_systems():
    """İki sistemin sonuçlarını yan yana karşılaştır."""
    graphrag_path = os.path.join(OUTPUT_DIR, "eval_graphrag_detailed.json")
    basicrag_path = os.path.join(OUTPUT_DIR, "eval_basicrag_detailed.json")

    if not os.path.exists(graphrag_path) or not os.path.exists(basicrag_path):
        print("Karşılaştırma için her iki sistemin de değerlendirilmiş olması gerekiyor.")
        print(f"  Beklenen: {graphrag_path}")
        print(f"  Beklenen: {basicrag_path}")
        return

    with open(graphrag_path) as f:
        graphrag = json.load(f)
    with open(basicrag_path) as f:
        basicrag = json.load(f)

    # Metrik sütunlarını bul
    metric_cols = [k for k in graphrag[0].keys() if k in (
        "madde_precision", "madde_recall", "madde_f1",
        "faithfulness", "answer_relevancy", "response_relevancy",
        "factual_correctness", "answer_correctness",
        "semantic_similarity", "answer_similarity",
        "context_precision", "llm_context_precision_with_reference",
        "context_recall", "llm_context_recall",
        "non_llm_context_recall", "context_entity_recall",
        "processing_time",
    )]

    print(f"\n{'='*70}")
    print(f" GraphRAG vs BasicRAG KARŞILAŞTIRMA")
    print(f"{'='*70}")

    print(f"\n{'Metrik':45s} | {'GraphRAG':>10s} | {'BasicRAG':>10s} | {'Fark':>10s}")
    print("-" * 82)

    for col in metric_cols:
        g_vals = [r[col] for r in graphrag if col in r and r[col] is not None]
        b_vals = [r[col] for r in basicrag if col in r and r[col] is not None]
        if g_vals and b_vals:
            g_avg = sum(g_vals) / len(g_vals)
            b_avg = sum(b_vals) / len(b_vals)
            diff = g_avg - b_avg
            sign = "+" if diff > 0 else ""
            print(f"  {col:43s} | {g_avg:10.4f} | {b_avg:10.4f} | {sign}{diff:9.4f}")

    # Soru tipi bazlı karşılaştırma
    types = set(r["soru_tipi"] for r in graphrag)
    for stype in sorted(types):
        g_items = [r for r in graphrag if r["soru_tipi"] == stype]
        b_items = [r for r in basicrag if r["soru_tipi"] == stype]

        print(f"\n--- {stype} ({len(g_items)} soru) ---")
        print(f"  {'Metrik':41s} | {'GraphRAG':>10s} | {'BasicRAG':>10s} | {'Fark':>10s}")
        print("  " + "-" * 78)

        for col in metric_cols:
            g_vals = [r[col] for r in g_items if col in r and r[col] is not None]
            b_vals = [r[col] for r in b_items if col in r and r[col] is not None]
            if g_vals and b_vals:
                g_avg = sum(g_vals) / len(g_vals)
                b_avg = sum(b_vals) / len(b_vals)
                diff = g_avg - b_avg
                sign = "+" if diff > 0 else ""
                print(f"  {col:41s} | {g_avg:10.4f} | {b_avg:10.4f} | {sign}{diff:9.4f}")

    # Karşılaştırma CSV
    comparison_path = os.path.join(OUTPUT_DIR, "comparison.csv")
    with open(comparison_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "soru_tipi"] +
                        [f"graphrag_{c}" for c in metric_cols] +
                        [f"basicrag_{c}" for c in metric_cols])

        g_by_q = {r["question"]: r for r in graphrag}
        b_by_q = {r["question"]: r for r in basicrag}

        for q in g_by_q:
            if q in b_by_q:
                g = g_by_q[q]
                b = b_by_q[q]
                row = [q, g["soru_tipi"]]
                row += [g.get(c, "") for c in metric_cols]
                row += [b.get(c, "") for c in metric_cols]
                writer.writerow(row)

    print(f"\nKarşılaştırma CSV: {comparison_path}")


# =====================
# Main
# =====================

def main():
    parser = argparse.ArgumentParser(description="RAGAS + Custom evaluation")
    parser.add_argument("--input", type=str, help="run_evaluation.py çıktısı (JSON)")
    parser.add_argument("--limit", type=int, default=0, help="İlk N sonuç (test için)")
    parser.add_argument("--skip-ragas", action="store_true", help="RAGAS'ı atla, sadece custom metrikler")
    parser.add_argument("--retry-missing", action="store_true",
                        help="Mevcut eval dosyasında eksik RAGAS metriklerini yeniden hesapla")
    parser.add_argument("--eval-json", type=str, default=None,
                        help="--retry-missing için mevcut eval JSON yolu")
    parser.add_argument("--compare", action="store_true", help="İki sistemi karşılaştır")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.compare:
        compare_systems()
        return

    if not args.input:
        print("--input veya --compare gerekli. Örnek:")
        print("  python scripts/evaluate_ragas.py --input data/evaluation/results_graphrag.json")
        print("  python scripts/evaluate_ragas.py --compare")
        return

    base = os.path.splitext(os.path.basename(args.input))[0]
    system_suffix = base.replace("results_", "")
    json_path = args.eval_json or os.path.join(OUTPUT_DIR, f"eval_{system_suffix}_detailed.json")

    if args.retry_missing:
        if not os.path.exists(json_path):
            print(f"Mevcut eval dosyası bulunamadı: {json_path}")
            return
        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        missing_questions = {
            r["question"] for r in results
            if _ragas_metric_missing(r.get("faithfulness"))
        }
        print(f"Yüklendi: {len(results)} kayıt ({results[0].get('system', 'unknown')})")
        print(f"Eksik faithfulness: {len(missing_questions)} soru → yeniden hesaplanacak")
        if not missing_questions:
            print("Eksik RAGAS metriği yok.")
            return
        print("\n--- RAGAS Metrikleri (eksik sorular) ---")
        results = compute_ragas_metrics(results, questions_to_eval=missing_questions)
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            results = json.load(f)
        if args.limit > 0:
            results = results[:args.limit]
        system_name = results[0].get("system", "unknown") if results else "unknown"
        print(f"Yüklendi: {len(results)} sonuç ({system_name})")
        print("\n--- Custom Metrikler ---")
        results = compute_custom_metrics(results)
        if not args.skip_ragas:
            print("\n--- RAGAS Metrikleri ---")
            results = compute_ragas_metrics(results)
        else:
            print("\nRAGAS atlandı (--skip-ragas)")

    system_name = results[0].get("system", "unknown") if results else "unknown"

    # 3. Özet yazdır
    print_summary(results, system_name)

    # 4. Kaydet
    csv_path = os.path.join(OUTPUT_DIR, f"eval_{system_suffix}_detailed.csv")

    save_detailed_csv(results, csv_path)
    save_full_json(results, json_path)

    print(f"\nTamamlandı! Diğer sistemi de değerlendirdikten sonra --compare ile karşılaştırın.")


if __name__ == "__main__":
    main()
