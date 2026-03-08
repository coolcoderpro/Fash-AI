# FASHAI — AI-Powered Fashion Market Intelligence

## The Problem

Fashion designers and small business owners face a critical challenge: they invest time, money, and resources into creating designs without knowing whether the market will accept them. Validating a design traditionally requires expensive market research, industry connections, or simply launching and hoping for the best. For independent designers and small businesses with limited budgets, a single bad production run can be devastating.

There is no accessible tool that lets a designer upload a design and instantly understand how it compares to what's already selling in the market — what price point works, whether the trend is rising or declining, who the target audience is, and whether it's worth producing at all.

## The Solution

FASHAI is an AI-powered fashion market intelligence platform that lets designers validate their designs before committing to production. A designer simply uploads their design image (or describes it in text), and the system instantly:

1. Finds the most similar competing products already selling in the market
2. Analyzes pricing, trends, demand, and customer reception across those competitors
3. Generates a comprehensive market intelligence report with a clear **GO / CAUTION / NO GO** production verdict

The entire platform runs on AWS and is accessible through a web browser — no downloads, no subscriptions, no fashion industry expertise required.

## How We Built the Market Database

### Dataset Foundation

We started with the [ASOS E-Commerce Dataset](https://www.kaggle.com/code/rajatraj0502/asos-e-commerce-dataset-30-845-products) containing 30,845 real fashion products. From this, we sampled 5,000 products to build our market database.

### The Problem with Raw Data

The raw dataset was noisy and incomplete for our use case:
- Product names had brand prefixes mixed with descriptions ("ASOS DESIGN lacquer corset mini dress...")
- The `brand` column contained full marketing paragraphs instead of brand names
- Critical fashion attributes like pattern, sleeve type, silhouette, and material were buried inside verbose text descriptions
- There was no structured data for niche, occasion, season, or style tags — all essential for meaningful fashion search

### Product Card Generation (fash_img_generator.py)

We built a product card generator that transforms each raw product into a rich, structured visual document:

**Field Enrichment** — Using rule-based keyword detection (no AI, fully offline), we extracted structured attributes from the raw text:
- **Pattern** detection across 16 categories (floral, stripe, check, polka dot, geometric, etc.)
- **Material** extraction from 19 fabric types (silk, cotton, denim, vinyl, chiffon, etc.)
- **Length** classification (mini, midi, maxi, cropped, knee-length)
- **Sleeve type** detection with expanded synonyms (halter/cami/strappy all map to sleeveless)
- **Silhouette** matching across 13 fit categories (bodycon, A-line, wrap, shift, oversized, etc.)
- **Clothing niche**, **occasion**, **season**, and **style tags** — all keyword-scored from product descriptions
- **Auto-generated description** field (e.g. "Sage sleeveless mini dress with bodycon fit")
- **Brand extraction** from product name prefixes, preventing brand names from dominating the search

**Card Layout** — Each product card (900x1080 PNG) contains:
- Top half: Product photo (letterboxed, not cropped — full product always visible)
- Bottom half: Structured metadata overlay — name, price, color, material, niche, occasion, style tags, and market performance stats

**Market Performance Data** — Simulated but realistically balanced: ~33% low / ~33% mid / ~33% high performers with correlated metrics (low performers have high return rates, low ratings, and falling trends — not random noise).

### Why Product Cards Matter for Search

The card design is intentional. When Amazon Titan creates an embedding, it reads both the photo AND the text printed on the card. This means each product embedding captures three information channels:

1. **Visual features from the photo** — color, texture, silhouette, pattern
2. **Text on the card image** — Titan's vision model reads the printed metadata (niche, price, style tags, etc.)
3. **Explicit metadata text** — clean structured text passed alongside the image as a safety net

This triple reinforcement makes embeddings robust. If Titan's OCR misreads something on the card, the explicit text still provides it clearly.

## Creating the Embeddings (embedding_pipeline.py)

Each of the 5,000 products gets a single **1024-dimensional hybrid embedding** from Amazon Titan Embed Image V1. The embedding fuses the card image + structured metadata text into one vector:

```
metadata_to_text() output:
"Sage sleeveless mini dress with bodycon fit. Color: Sage. Material: Vinyl, Polyester.
 Length: Mini. Sleeve: Sleeveless. Silhouette: Bodycon. Niche: Y2K / 2000s Revival.
 Occasion: Night Out / Club. Season: All-season. Style tags: corset, mini, bodycon."
```

This structured format is critical — during search, all query text is generated in the **exact same field format** (Color:, Pattern:, Material:, Length:, etc.). When the query vector and indexed vectors use matching formats, they land close together in the 1024-dimensional embedding space, producing accurate search results.

All 5,000 embeddings are indexed into **Amazon OpenSearch Serverless** using HNSW (Hierarchical Navigable Small World) with cosine similarity for fast approximate nearest-neighbor search.

## How Search Works — Three Modalities

FASHAI supports three ways to search, each producing a 1024-dim query vector aligned with the indexed format.

### Mode A: Image Only

The designer uploads a design photo with no description.

1. **Amazon Rekognition** analyzes the image and extracts garment labels (>90% confidence only) — garment type, pattern, fabric. These are used for context in the final report but NOT included in the search embedding to avoid generic label noise.

2. **Gemma 3 27B IT (Vision)** sees the image and extracts structured attributes in the same format as the indexed products:
   ```
   Description: Black floral print long sleeve dress
   Color: Black (garment color only, NOT pattern colors)
   Pattern: Blue floral print (pattern type + pattern colors)
   Material: Silk | Length: Midi | Sleeve: Long sleeve
   Silhouette: A-line | Niche: Casual Wear
   Occasion: Daily | Season: Summer | Style tags: A-line, fitted
   ```
   The Color/Pattern separation is important — without it, a "blue floral print on a black dress" would confuse the embedding into thinking the dress is blue.

3. **Amazon Titan** creates a hybrid embedding from the uploaded image + Gemma's structured text → 1024-dim query vector.

4. **OpenSearch** finds the 15 nearest products using HNSW kNN search.

5. **Cohere Rerank V3.5** (cross-encoder re-ranker) re-scores all 15 candidates using deep text matching and returns the best 10.

6. **Market Readiness Score** (0–100) is computed from the matched products' real market data — trend momentum, price competitiveness, customer validation, and demand signals.

7. **Gemma 3 27B IT** generates a 7-section market intelligence report from the top 5 matches, ending with a **GO / CAUTION / NO GO** production verdict.

### Mode B: Image + Text

The designer uploads an image AND describes their intent (e.g. "Orange summer skirt with floral sleeveless top").

The pipeline is the same as Image Only, but Gemma Vision now sees both the image AND the designer's description. It fuses visual analysis with the designer's intent — if the image is ambiguous (e.g. multicolored), the designer's text steers the output (Color: Orange, because the designer said "orange").

The designer's raw text is NOT separately added to the embedding — it's baked into Gemma's structured output, which already captures the intent. This prevents noisy free-text from diluting the structured format alignment.

### Mode C: Text Only

The designer types a description with no image (e.g. "red party dress" or "black floral print long sleeve dress").

1. **Gemma 3 27B IT (text-only)** imagines what this garment would look like as a product listing and outputs the same structured fields. It also classifies the query as **Generic** or **Detailed**.

2. **Amazon Titan** creates a text-only embedding from the raw user text + Gemma's structured output combined. The raw text is preserved alongside the structured output so original keywords (like "red") aren't lost.

3. **Search strategy adapts to query type:**
   - **Generic queries** (e.g. "party dresses", "swimwear", "office wear") → Hybrid search: kNN vector similarity + **BM25 keyword matching** on clothing niche, occasion, and style tags. BM25 ensures that "party dresses" finds products tagged with "party" in their niche/occasion fields, even if the vector similarity alone would miss category-level matches.
   - **Detailed queries** (e.g. "black floral print long sleeve dress") → kNN only. BM25 is skipped because keyword matching on detailed descriptions can introduce noise.

4. The rest of the pipeline (Cohere Rerank → Market Readiness → Gemma Insights) is identical to Image mode.

### Compare Mode

Designers can upload two designs side by side to compare their market potential. Each design goes through its own full search pipeline independently (supporting mixed modalities — e.g. image for Design 1, text for Design 2). Gemma then generates a 6-section collection comparison report with individual verdicts for each design and a combined collection verdict.

## Key Technical Decisions

### Format Alignment Across the Entire Pipeline

The single most important design decision: every component that produces or consumes text uses the **exact same structured field format**. The indexing function `metadata_to_text()`, all three Gemma prompts (STYLE_PROMPT, STYLE_PROMPT_WITH_DESC, TEXT_SEARCH_PROMPT), and the Cohere Rerank text function `_product_to_text()` all output fields in the same order: Description, Color, Pattern, Material, Length, Sleeve, Silhouette, Niche, Occasion, Season, Style tags.

This alignment ensures that query vectors land close to relevant indexed vectors in the 1024-dimensional embedding space.

### Rekognition Labels for Insights Only

Early testing showed that including Rekognition's generic labels (e.g. "Clothing", "Dress", "Apparel") in the embedding text reduced search accuracy. These labels are too broad and dilute the specific attributes that matter for fashion matching. Instead, Rekognition labels are only passed to Gemma's insights prompt as additional context — they never touch the embedding.

### Deterministic AI Output

Gemma Vision runs at temperature 0 across all modes, ensuring consistent and reproducible attribute extraction. The same image always produces the same structured output.

### Cross-Encoder Re-ranking

After kNN retrieves the top 15 candidates based on vector similarity, Cohere Rerank V3.5 applies deep cross-attention between the query text and each candidate's product text. This catches semantic matches that pure vector distance might miss, improving the final top 10 results.

## Architecture

### Offline Pipeline (One-Time)
```
ASOS Dataset (5,000 products)
    │
    ▼
┌──────────────────┐     ┌──────────────────┐     ┌───────────────────────┐
│  Image Generator │────▶│   Amazon S3      │────▶│  Amazon Titan         │
│  fash_img_gen.py │     │  cards/ + JSON   │     │  Embed Image V1       │
│  900x1080 cards  │     │                  │     │  image + text → 1024d │
└──────────────────┘     └──────────────────┘     └───────────┬───────────┘
                                                              │
                                                              ▼
                                                  ┌───────────────────────┐
                                                  │  OpenSearch Serverless│
                                                  │  HNSW index (cosine)  │
                                                  │  5,000 vectors        │
                                                  └───────────────────────┘
```

### Online Search Pipeline
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Designer  │────▶│ AWS Amplify │────▶│ API Gateway │────▶│ AWS Lambda  │
│  (Browser)  │     │  Frontend   │     │  REST API   │     │ Orchestrator│
└─────────────┘     └─────────────┘     └─────────────┘     └──────┬──────┘
                                                                   │
                    ┌──────────────────────────────────────────────┐│
                    │              AI / ML Layer                    ││
                    │                                              ▼│
                    │  ┌──────────────┐   ┌────────────────────┐   │
                    │  │  Rekognition │   │  Gemma 3 27B IT    │   │
                    │  │  DetectLabels│   │  Vision: attributes│   │
                    │  │  >90% conf.  │   │  Text: imagination │   │
                    │  │  (insights)  │   │  Temp: 0           │   │
                    │  └──────────────┘   └────────┬───────────┘   │
                    │                              │               │
                    │                              ▼               │
                    │                   ┌────────────────────┐     │
                    │                   │  Amazon Titan       │     │
                    │                   │  Embed Image V1     │     │
                    │                   │  Hybrid: img + text │     │
                    │                   │  → 1024-dim vector  │     │
                    │                   └────────┬───────────┘     │
                    └────────────────────────────│────────────────┘
                                                │
                    ┌───────────────────────────▼──────────────────┐
                    │              Search & Re-rank                 │
                    │                                               │
                    │  OpenSearch ──(15)──▶ Cohere Rerank ──(10)──▶│
                    │  HNSW kNN          V3.5 cross-encoder        │
                    │  + BM25 (text)     re-ranking                │
                    └───────────────────────────┬──────────────────┘
                                                │
                    ┌───────────────────────────▼──────────────────┐
                    │              Intelligence                     │
                    │                                               │
                    │  Market Readiness ──▶ Gemma 3 27B IT ──▶     │
                    │  Score (0-100)       7-section report         │
                    │  Trend│Price│        GO / CAUTION / NO GO    │
                    │  Valid│Demand        production verdict       │
                    └───────────────────────────┬──────────────────┘
                                                │
                                                ▼
                                    ┌───────────────────┐
                                    │  Designer sees:   │
                                    │  • 10 matches     │
                                    │  • Readiness gauge│
                                    │  • AI report      │
                                    │  • Final verdict  │
                                    └───────────────────┘
```

## AWS Services

| Service | Purpose |
|---|---|
| **AWS Amplify** | Frontend hosting (static site) |
| **Amazon API Gateway** | REST endpoint with CORS |
| **AWS Lambda** | Serverless backend — all pipeline orchestration |
| **Amazon Bedrock** | Gemma 3 27B IT (vision + text), Amazon Titan Embed Image V1 (1024-dim), Cohere Rerank V3.5 |
| **Amazon Rekognition** | DetectLabels API (garment type, pattern, fabric — >90% confidence) |
| **Amazon OpenSearch Serverless** | HNSW vector index with cosine similarity (5,000 products) |
| **Amazon S3** | Product card images, enriched metadata, embeddings |

## Tech Stack

- **Backend:** Python, boto3, opensearch-py, Pillow
- **Frontend:** HTML, CSS, JavaScript (single-page app, no framework)
- **AI/ML:** Multimodal hybrid embeddings, HNSW vector search, kNN, BM25, cosine similarity, cross-encoder re-ranking
- **Infrastructure:** Fully serverless on AWS (ap-south-1)

## Project Structure

```
├── frontend/                     # Frontend assets
│   ├── index.html                # Single-page application
│   ├── fash_1.png, fash_2.png, fash_3.png, loading.png, market.png
├── fashion_design_cards/         # Sample product cards (15 of 5,000)
├── lambda_handler.py             # AWS Lambda entry point (routes to 4 modes)
├── query_pipeline.py             # Full search pipeline
├── embedding_pipeline.py         # Offline: Titan hybrid embedding generation
├── fash_img_generator.py         # Offline: product card image generation
├── index_to_opensearch.py        # Offline: OpenSearch bulk indexing
├── enriched_metadata_5000.json   # Structured metadata for all 5,000 products
├── asos_design_dataset.csv       # Source dataset (5,000 sampled ASOS products)
├── architecture.md               # Detailed search flow documentation
├── embeddings_logic.md           # Embedding pipeline walkthrough
├── dataset_summary.md            # Dataset enrichment documentation
├── architecture_technical_diagram.html
├── process_flow_diagram.html
└── Project-Summary.md            # This file
```

## Dataset

- **Source:** [ASOS E-Commerce Dataset](https://www.kaggle.com/code/rajatraj0502/asos-e-commerce-dataset-30-845-products) (30,845 products)
- **Sampled:** 5,000 products for the market database
- **Enrichment:** Rule-based keyword detection — pattern (16 types), material (19 types), length, sleeve type (with synonym expansion), silhouette (13 types), niche, occasion, season, style tags
- **Card Images:** 900x1080 PNG with product photo + structured metadata overlay
- **Embeddings:** 1024-dim multimodal hybrid vectors (Amazon Titan) — image + structured text fused

## Pipeline Numbers

| Metric | Value |
|---|---|
| Products indexed | 5,000 |
| Embedding dimensions | 1,024 |
| Vector search algorithm | HNSW (cosine similarity) |
| Candidates fetched per search | 15 |
| Results after re-ranking | 10 |
| Products sent to AI insights | 5 |
| Market Readiness sub-scores | 4 (Trend, Price, Validation, Demand) |
| Search modes supported | 3 + Compare |

Built for the **AI for Bharat Hackathon 2026**.
