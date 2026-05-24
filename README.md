# TCK-QA-benchmark

**TCK-QA-150** — Türk Ceza Kanunu (TCK) üzerinde GraphRAG (Neo4j) ve BasicRAG (pgvector) karşılaştırması.  
Değerlendirme: **GPT-4o** ve **Claude Sonnet 4.6** — 150 soru (`data/evaluation/TCK-QA_dataset.jsonl`).

> **Güvenlik:** API anahtarlarını yalnızca bu klasördeki `.env` dosyanızda tutun (`.env.example` kopyalayın). `.env` Git'e eklenmez; depoda anahtar commit edilmez. Üst dizindeki başka `.env` dosyaları otomatik yüklenmez.

## İçerik

```
TCK-QA-benchmark/
├── tck_graphrag/          # Python paketi
│   ├── core/              # config, LLM factory, Neo4j
│   ├── services/
│   │   ├── graphrag.py    # GraphRAG sorgu (QueryService)
│   │   ├── basic_rag.py   # BasicRAG
│   │   └── indexing.py    # KG inşası (Neo4j)
│   └── prompts/
├── scripts/               # index, evaluation, RAGAS
├── data/
│   ├── tck_maddeler.jsonl # TCK madde metinleri
│   └── evaluation/        # TCK-QA_dataset.jsonl + sonuçlar
└── docs/                  # raporlar, Algorithm 1
```

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# .env içine OPENAI_API_KEY ve (Claude için) ANTHROPIC_API_KEY yazın
```

Neo4j + PostgreSQL (pgvector) gerekir. Örnek: `docker compose up -d`.

## Hızlı deneme

```bash
python scripts/query_demo.py "Madde 141 ne diyor?" --system graphrag
python scripts/query_demo.py "Hırsızlık cezası?" --system basicrag
```

## İndeksleme

```bash
# Graph (Neo4j) — LLM ile entity/ilişki çıkarımı
python scripts/index_tck.py --limit 5

# BasicRAG (PostgreSQL)
python scripts/index_basic_rag.py --clear
```

## Değerlendirme (yeniden koşmak için)

```bash
# GPT-4o
python scripts/run_evaluation.py --system both

# Claude
python scripts/run_evaluation.py --provider anthropic --model claude-sonnet-4-6 --suffix claude --system both

# RAGAS + madde metrikleri
python scripts/evaluate_ragas.py --input data/evaluation/results/results_graphrag.json
python scripts/evaluate_ragas.py --compare
```

## Hazır sonuçlar (`data/evaluation/results/`)

| Dosya | Sistem |
|-------|--------|
| `results_graphrag.json` | GraphRAG + GPT-4o |
| `results_basicrag.json` | BasicRAG + GPT-4o |
| `results_graphrag_claude.json` | GraphRAG + Claude |
| `results_basicrag_claude.json` | BasicRAG + Claude |
| `eval_*_detailed.json` / `.csv` | RAGAS + madde F1 |

Özet: `docs/EVALUATION_REPORT.md`, `docs/EVALUATION_COMPARISON_CLAUDE.md`.

## Özet bulgular (GPT-4o)

| Metrik | GraphRAG | BasicRAG |
|--------|----------|----------|
| Madde F1 | **0.61** | 0.25 |
| Faithfulness | 0.82 | **0.84** |

GraphRAG madde tespitinde belirgin üstün; BasicRAG daha hızlı ve bazı kalite metriklerinde rekabetçi.

## Lisans

Akademik kullanım için yayınlanmıştır. TCK metinleri kamu kaynağıdır; kullanım koşullarınıza uygun şekilde atıf yapın.
