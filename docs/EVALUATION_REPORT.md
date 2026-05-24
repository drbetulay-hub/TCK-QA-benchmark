# GraphRAG vs BasicRAG Evaluation Raporu

**TCK-QA-150** — Türk Ceza Kanunu (TCK) soru-cevap benchmark değerlendirmesi

📅 Rapor Tarihi: 21 Mayıs 2026  
🤖 LLM'ler: GPT-4o (OpenAI), Claude Sonnet 4.6 (Anthropic)  
📊 Dataset: **TCK-QA-150** (`TCK-QA_dataset.jsonl`) — 150 soru (single-hop, multi-hop, reasoning)

---

## 📋 Özet (GPT-4o)

| Metrik | GraphRAG | BasicRAG | Kazanan | Fark |
|--------|----------|----------|---------|------|
| **Madde Precision** | 0.6113 | 0.1587 | 🏆 GraphRAG | +0.4526 |
| **Madde Recall** | 0.7112 | 0.7923 | BasicRAG | +0.0811 |
| **Madde F1** | 0.6090 | 0.2494 | 🏆 GraphRAG | +0.3596 |
| **Faithfulness** | 0.8226 | 0.8355 | BasicRAG | +0.0129 |
| **Semantic Similarity** | 0.8212 | 0.8254 | BasicRAG | +0.0042 |
| **Context Precision** | 0.8267 | 0.9000 | BasicRAG | +0.0733 |
| **Processing Time (s)** | 6.02 | 3.48 | BasicRAG | -2.53 |

### Ana Bulgular

1. **GraphRAG, madde tespitinde çok daha başarılı** (+35.96% F1 artışı)
   - Precision'da %45 fark: GraphRAG doğru maddeleri buluyor
   - Knowledge Graph yapısı sayesinde ilişkili maddeleri keşfediyor

2. **BasicRAG yanıt kalitesinde hafif üstün** (faithfulness, similarity)
   - Fark minimal (%1-2 arası)
   - Vector search basit sorularda etkili

3. **GraphRAG daha yavaş** (~2.5 saniye fazla)
   - Graph traversal + LLM reranking ek zaman alıyor
   - Ama daha zengin context sağlıyor

---

## 📊 Soru Tipi Bazlı Analiz

### Single-Hop Sorular (60 soru)
*Tek bir maddeye dayanan basit sorular*

| Metrik | GraphRAG | BasicRAG | Kazanan |
|--------|----------|----------|---------|
| Madde F1 | 0.5120 | 0.1870 | 🏆 GraphRAG |
| Faithfulness | 0.8228 | **0.9231** | BasicRAG |
| Semantic Similarity | 0.8059 | **0.8269** | BasicRAG |
| Processing Time | 5.48s | **2.74s** | BasicRAG |

**Yorum**: Basit sorularda BasicRAG yeterli ve daha hızlı. Ancak GraphRAG madde tespitinde yine üstün.

---

### Multi-Hop Sorular (55 soru)
*Birden fazla maddeyi ilişkilendiren karmaşık sorular*

| Metrik | GraphRAG | BasicRAG | Kazanan |
|--------|----------|----------|---------|
| Madde F1 | **0.7037** | 0.3111 | 🏆 GraphRAG |
| Faithfulness | 0.8335 | 0.8390 | BasicRAG |
| Semantic Similarity | 0.8409 | 0.8428 | BasicRAG |
| Processing Time | 6.56s | 4.41s | BasicRAG |

**Yorum**: Multi-hop'ta GraphRAG'ın üstünlüğü belirgin (+39.26% F1). Graph traversal ilişkili maddeleri keşfediyor.

---

### Reasoning Sorular (35 soru)
*Muhakeme gerektiren karşılaştırma/analiz soruları*

| Metrik | GraphRAG | BasicRAG | Kazanan |
|--------|----------|----------|---------|
| Madde F1 | **0.6264** | 0.2592 | 🏆 GraphRAG |
| Faithfulness | **0.8051** | 0.6798 | 🏆 GraphRAG |
| Semantic Similarity | **0.8164** | 0.7955 | 🏆 GraphRAG |
| Processing Time | 6.08s | 3.30s | BasicRAG |

**Yorum**: Reasoning sorularında GraphRAG tüm metriklerde üstün! Özellikle faithfulness'ta %12.5 fark.

---

## 🎯 Sonuçlar ve Öneriler

### GraphRAG Tercih Edilmeli
- ✅ Multi-hop sorular (birden fazla madde ilişkisi)
- ✅ Reasoning soruları (karşılaştırma, analiz)
- ✅ Madde numarası doğruluğu önemli olduğunda
- ✅ Nitelikli suç - temel suç ilişkileri

### BasicRAG Yeterli
- ✅ Basit tek-madde soruları
- ✅ Hız kritik olduğunda
- ✅ Genel bilgi soruları

### Hibrit Yaklaşım Önerisi
Soru analizi yaparak:
1. Basit sorular → BasicRAG (hızlı)
2. Karmaşık sorular → GraphRAG (doğru)

---

## 📈 Metrik Açıklamaları

| Metrik | Açıklama |
|--------|----------|
| **Madde Precision** | Sistemin verdiği maddelerin ne kadarı doğru |
| **Madde Recall** | Ground truth maddelerinin ne kadarı bulundu |
| **Madde F1** | Precision ve Recall'ın harmonik ortalaması |
| **Faithfulness** | Yanıtın context'e sadık kalma oranı |
| **Semantic Similarity** | Yanıtın ground truth'a anlamsal benzerliği |
| **Context Precision** | Getirilen context'in relevance'ı |

---

## 🛠️ Teknik Detaylar

### Sistem Mimarisi

```
GraphRAG Pipeline:
Soru → Query Analysis (LLM) → Entity Retrieval (Neo4j) 
     → Graph Traversal → Context Reranking (LLM) → Answer Generation

BasicRAG Pipeline:  
Soru → Embedding → Vector Search (pgvector) → Answer Generation
```

### Kullanılan Teknolojiler
- **LLM**: GPT-4o (OpenAI)
- **Graph DB**: Neo4j
- **Vector DB**: PostgreSQL + pgvector
- **Framework**: LangChain
- **Evaluation**: RAGAS

---

## 📁 Dosya Yapısı

```
data/evaluation/results/
├── eval_graphrag_detailed.json         # GraphRAG + GPT-4o (150, RAGAS)
├── eval_basicrag_detailed.json         # BasicRAG + GPT-4o (150, RAGAS)
├── eval_graphrag_claude_detailed.json  # GraphRAG + Claude (150, RAGAS)
├── eval_basicrag_claude_detailed.json  # BasicRAG + Claude (150, RAGAS)
├── results_graphrag_claude.json        # GraphRAG + Claude ham
├── results_basicrag_claude.json        # BasicRAG + Claude ham
├── comparison.csv                      # GraphRAG vs BasicRAG (GPT-4o)
└── results_*.json                      # Diğer ham sonuçlar
```

---

---

## 🆕 GPT-4o vs Claude Sonnet 4.6 — Tam Karşılaştırma

**22 Mayıs 2026** — Dört sistem de tamamlandı (150/150 + RAGAS).

### Dört Sistem — Genel Metrik Tablosu

| Metrik | GraphRAG GPT-4o | BasicRAG GPT-4o | GraphRAG Claude | BasicRAG Claude | En İyi |
|--------|-----------------|-----------------|-----------------|-----------------|--------|
| **Madde F1** | **0.609** | 0.249 | 0.513 | 0.249 | GraphRAG GPT-4o |
| Madde Precision | **0.611** | 0.159 | 0.403 | 0.159 | GraphRAG GPT-4o |
| Madde Recall | 0.711 | 0.792 | **0.899** | 0.792 | GraphRAG Claude |
| **Faithfulness** | 0.823 | 0.836 | 0.847 | **0.857** | BasicRAG Claude |
| Semantic Similarity | **0.821** | 0.825 | 0.736 | 0.739 | BasicRAG GPT-4o |
| Answer Relevancy | 0.417 | 0.422 | **0.433** | 0.391 | GraphRAG Claude |
| **İşlem Süresi** | **6.0s** | 3.5s | 66.9s | 20.5s | BasicRAG GPT-4o |

### GraphRAG: GPT-4o vs Claude

| Metrik | GPT-4o | Claude | Fark |
|--------|--------|--------|------|
| Madde F1 | **0.609** | 0.513 | -0.096 |
| Madde Precision | **0.611** | 0.403 | -0.209 |
| Madde Recall | 0.711 | **0.899** | +0.188 |
| Faithfulness | 0.823 | **0.847** | +0.024 |
| Semantic Similarity | **0.821** | 0.736 | -0.085 |
| İşlem Süresi | **6.0s** | 66.9s | +60.9s |

#### Soru Tipi Bazlı (Madde F1)

| Soru Tipi | GPT-4o | Claude |
|-----------|--------|--------|
| Multi-hop | **0.704** | 0.595 |
| Reasoning | **0.626** | 0.521 |
| Single-hop | **0.512** | 0.433 |

### GraphRAG Claude vs BasicRAG GPT-4o

| Metrik | BasicRAG GPT-4o | GraphRAG Claude | Fark |
|--------|-----------------|-----------------|------|
| Madde F1 | 0.249 | **0.513** | +0.264 |
| Faithfulness | 0.836 | **0.847** | +0.011 |
| Semantic Similarity | **0.825** | 0.736 | -0.089 |
| İşlem Süresi | **3.5s** | 66.9s | +63.4s |

### BasicRAG: GPT-4o vs Claude

| Metrik | GPT-4o | Claude | Fark |
|--------|--------|--------|------|
| Madde F1 | 0.249 | 0.249 | 0.000 |
| Faithfulness | 0.836 | **0.857** | +0.022 |
| Semantic Similarity | **0.825** | 0.739 | -0.086 |
| İşlem Süresi | **3.5s** | 20.5s | +17.0s |

Aynı embedding/retrieval kullanıldığı için madde metrikleri özdeş; Claude yanıt kalitesinde faithfulness önde.

### Ana Bulgular (Claude Dahil)

1. **GPT-4o GraphRAG en yüksek madde doğruluğu** — F1 0.609.
2. **Claude GraphRAG yüksek recall** (0.899); ancak precision düşük ve çok yavaş (~67 s/soru).
3. **Graph yapısı LLM'den bağımsız üstün** — GraphRAG + Claude, BasicRAG + Claude'dan madde F1'de +0.26 üstün.
4. **BasicRAG + Claude tamamlandı** — Faithfulness en yüksek (0.857); madde metrikleri GPT-4o BasicRAG ile aynı.

Detaylı karşılaştırma: `EVALUATION_COMPARISON_CLAUDE.md`

### Önerilen Sistem Seçimi

| Senaryo | Öneri |
|---------|-------|
| Madde doğruluğu öncelikli | GraphRAG + GPT-4o |
| En yüksek faithfulness | BasicRAG + Claude |
| Düşük gecikme | BasicRAG + GPT-4o |
| Üretim ortamı | GraphRAG + GPT-4o |

---

## Gelecek çalışmalar

1. Human evaluation ile doğrulama
2. Benchmark genişletme (yeni soru tipleri ve alan uzmanı gözden geçirmesi)
3. Ek LLM ve embedding modelleri ile tekrarlanabilirlik çalışmaları

---

*Bu rapor TCK-QA-150 üzerinde GraphRAG vs BasicRAG karşılaştırmasını özetler (GPT-4o + Claude).*
