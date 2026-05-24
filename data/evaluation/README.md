# TCK-QA-150 evaluation data

**TCK-QA-150** — 150-question Turkish Penal Code (TCK) QA benchmark.

| File | Description |
|------|-------------|
| `TCK-QA_dataset.jsonl` | Full benchmark (60 single-hop, 55 multi-hop, 35 reasoning) |
| `results/` | Precomputed runs: GraphRAG & BasicRAG × GPT-4o & Claude |

Each JSONL line: `soru`, `cevap`, `context`, `relevant_madde_list`, `soru_tipi`.
