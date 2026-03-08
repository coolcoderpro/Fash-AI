# FASHAI Dataset Pipeline — Summary

## Source Dataset

[ASOS E-Commerce Dataset (30,845 products)](https://www.kaggle.com/code/rajatraj0502/asos-e-commerce-dataset-30-845-products)

Raw CSV fields: `name`, `category`, `color`, `images`, `category_type`, `product_details`, `brand`, `about_me`, `price`

## Pipeline Overview

```
Raw CSV (30,845 rows)
    |
Data Cleaning & Sampling → final_fashion_dataset.csv (5,000 rows)
    |
fash_img_generator.py → Product card images (fash_img_0001.png ... fash_img_5000.png)
                       → enriched_metadata.json (structured metadata per product)
    |
embedding_pipeline.py → Titan hybrid embeddings (image + metadata text → 1024-dim vectors)
                       → Indexed into OpenSearch Serverless
```

## Step 1: Data Cleaning

The raw ASOS dataset had noisy, inconsistent fields:

- **Product names** contained brand prefixes ("ASOS DESIGN", "Miss Selfridge", etc.) mixed with product descriptions
- **`brand` column** contained full marketing paragraphs instead of brand names (e.g. "This is ASOS DESIGN your go-to for all the latest trends...")
- **`product_details`** had template artifacts, product codes, and run-together sentences (e.g. "Dresses by ASOS DESIGNA round of applause for the dressSweetheart neckDrape details")
- **`about_me`** (materials) was verbose with full fabric descriptions (e.g. "Vinyl: glossy fabric with a smooth, rubberised feelCoating: 100% Polyurethane, Main: 94% Polyester, 6% Elastane")
- **`category_type`** was almost always "Dress" — low signal
- Many fields that matter for fashion search (pattern, sleeve type, length, silhouette) were not explicit — buried in product names and descriptions

## Step 2: Product Card Generation (fash_img_generator.py)

### Image Fetching (3-layer strategy)

1. Direct HTTP requests to ASOS CDN (fast but often blocked)
2. Selenium headless Chrome fallback (handles ASOS auth/cookies, JS/CSS/fonts blocked for speed)
3. Placeholder if both fail

Images are letterboxed (not cropped) into a 900×520px photo strip so the full product is always visible.

### Field Enrichment — Rule-Based Keyword Detection

All new fields are derived from existing CSV text using pure keyword matching — no AI, no internet, runs completely offline. The detection scans a `combined` text string built from: product name + product details + about_me + category_type + color.

#### Original fields (from CSV):
| Field | Source | Example |
|---|---|---|
| name | CSV `name` (brand prefix stripped) | "lacquer corset mini dress with drape skirt in sage" |
| brand | Extracted from name prefix | "ASOS DESIGN" |
| price | CSV `price` | 34.0 |
| color | CSV `color` | "Sage" |

#### New fields (keyword-detected):

| Field | Detection Logic | Example |
|---|---|---|
| **pattern** | Matches against 16 pattern categories (floral, stripe, check, polka dot, animal print, abstract, graphic, lace, embroidered, sequin/beaded, tie-dye, camo, paisley, tropical, textured, color block) | "Floral print", "Lace", "None" |
| **material** | Extracts up to 2 key fabric names from 19 material types (leather, denim, silk, satin, velvet, linen, wool, chiffon, mesh, lace, knit, cotton, sequin, vinyl, nylon, polyester, viscose, elastane, crochet) | "Vinyl, Polyester" |
| **length** | Matches 6 length categories (mini, midi, maxi, knee-length, cropped, longline) | "Mini", "Cropped" |
| **sleeve_type** | Matches 6 sleeve categories with expanded synonyms (sleeveless includes halter/cami/strappy/spaghetti/bardot/off-shoulder; short sleeve includes flutter/frill; etc.) | "Sleeveless", "Long sleeve" |
| **silhouette** | Matches 13 fit categories (bodycon, A-line, fitted, relaxed, oversized, wrap, shift, skater, straight, pleated, peplum, tiered, smock/babydoll, draped) | "Bodycon", "Wrap" |
| **description** | Auto-generated natural sentence from color + pattern + sleeve + length + category + silhouette | "Sage sleeveless mini dress with bodycon fit" |

#### Derived intelligence fields (keyword-scored):

| Field | Logic | Example |
|---|---|---|
| clothing_niche | Best-match from 8 niches (Y2K, Cottagecore, Streetwear, Minimalist, Party, Athleisure, Boho, Classic) | "Y2K / 2000s Revival" |
| target_audience | Best-match from 5 audience segments | "Women 18–24 (trend-led)" |
| occasion | Best-match from 6 occasion types | "Night Out / Club" |
| season | Best-match from 5 season categories | "All-season" |
| style_tags | Up to 6 matching tags from 30 seed tags | ["corset", "mini", "bodycon"] |
| price_tier | Rule-based on price thresholds | "Mid-range ($20–$50)" |

### Brand Extraction

Brand names are stripped from product names using a priority-ordered list of 45+ known brand prefixes (longest-first matching so "ASOS DESIGN Curve" matches before "ASOS DESIGN"):

```
"ASOS DESIGN lacquer corset mini dress with drape skirt in sage"
    → brand: "ASOS DESIGN"
    → name:  "lacquer corset mini dress with drape skirt in sage"
```

This prevents "ASOS DESIGN" from dominating every product's embedding vector.

### Market Performance Data — Balanced Distribution

Market fields are seeded by product name (deterministic — same product always gets same numbers) and driven by a performance tier system:

```
Performance roll (seeded RNG) → ~33% Low / ~33% Mid / ~33% High
```

| Metric | Low Performer | Mid Performer | High Performer |
|---|---|---|---|
| Daily sales rate | 5–25% of max | 25–60% of max | 60–100% of max |
| Return rate | High end of category range | Middle of range | Low end of range |
| Avg rating | 2.2–3.4 | 3.3–4.2 | 4.0–5.0 |
| Inventory | 200–600 (excess stock) | 50–300 | 0–80 (running low) |
| Sales trend | 75% falling, 25% stable | 20% falling, 50% stable, 30% rising | 75% rising, 25% stable |
| Conversion rate | Low end of price tier range | Middle | High end |
| Revenue | units × price × (1 - return_rate) | Derived | Derived |

All fields are correlated — a low performer won't have a 5-star rating or rising trend. Max daily sales ceiling is price-tier dependent (budget items sell more units, luxury items fewer).

### Product Card Layout

```
┌─────────────────────────────────┐
│                                 │
│        Product Photo            │
│     (letterboxed, 900×520)      │
│                                 │
├─────────────────────────────────┤
│ ASOS DESIGN              (grey)│
│ lacquer corset mini dress...    │
│ $34.00                [Mid-range│
│ Color: Sage                     │
│ Material: Vinyl • Length: Mini  │
│ • Sleeve: Sleeveless • Fit: ..  │
│ Sage sleeveless mini dress...   │
│─────────────────────────────────│
│ Clothing Niche  Target Audience │
│ Occasion        Season          │
│─────────────────────────────────│
│ Style Tags: [pill] [pill] ...   │
│═════════════════════════════════│
│ Units  Revenue  Return  Conv.   │
│ Days   Inventory Rating  Trend  │
└─────────────────────────────────┘
```

## Step 3: Embedding Pipeline (embedding_pipeline.py)

Each product gets a single hybrid embedding:

```python
# metadata_to_text() builds structured text from enriched_metadata.json
text = "Product Name. Brand: X. Category: Y. Color: Red. Price: $45. Niche: Casual. ..."

# Single Titan API call with both image + text
body = {
    "inputText":  text,
    "inputImage": image_b64,
    "embeddingConfig": {"outputEmbeddingLength": 1024}
}
```

This produces one 1024-dim vector per product that fuses visual features with metadata semantics. All three search modes (image, image+text, text-only) generate query text aligned with this indexed format for optimal vector matching.

## Iteration History

### V1 — Initial Version
- Basic card with all raw CSV fields displayed
- Brand field was the full marketing paragraph
- Product details shown with template artifacts and product codes
- Materials & Care shown as full verbose text
- Category always "Dress" — no useful signal
- No pattern, length, sleeve, silhouette, or material extraction
- Market data skewed towards high performers (most products showed high units, rising trends)

### V2 — Field Enrichment
- Added keyword-based detection for: pattern, length, sleeve_type, silhouette
- Added clean material extraction (up to 2 key fabrics from 19 types)
- Added auto-generated description field
- Expanded keyword lists for better coverage (e.g. "halter", "cami", "strappy" all map to Sleeveless)

### V3 — Cleanup & Balancing (Current)
- Stripped brand prefix from product name → clean brand field + clean product name
- Removed noisy fields from card and metadata: category (always "Dress"), product_details (template noise), materials_full (verbose)
- Kept raw CSV text in `combined` variable for keyword detection — nothing lost
- Balanced market performance data: ~33% low / ~33% mid / ~33% high performers
- All market metrics correlated (low performer = low sales + high returns + low rating + falling trend)
- Reduced card height (INFO_H 760→560) since Product Details and Materials & Care sections removed

## Final Metadata Schema

```json
{
  "name": "lacquer corset mini dress with drape skirt in sage",
  "brand": "ASOS DESIGN",
  "price": 34.0,
  "color": "Sage",
  "pattern": "None",
  "material": "Vinyl, Polyester",
  "length": "Mini",
  "sleeve_type": "Sleeveless",
  "silhouette": "Bodycon",
  "clothing_niche": "Y2K / 2000s Revival",
  "target_audience": "Women 18-24 (trend-led)",
  "occasion": "Night Out / Club",
  "season": "All-season",
  "price_tier": "Mid-range ($20-$50)",
  "style_tags": ["corset", "mini", "bodycon", "zip-back", "sweetheart neck"],
  "description": "Sage sleeveless mini dress with bodycon fit",
  "units_sold": 181,
  "revenue_generated": 4979,
  "return_rate_pct": 19.1,
  "conversion_rate_pct": 5.2,
  "days_since_launch": 39,
  "sales_trend": "rising",
  "inventory_remaining": 7,
  "avg_rating": 4.1,
  "product_id": "0001",
  "image_path": "product_outputs/images/fash_img_0001.png"
}
```

## Key Design Decisions

1. **Keyword detection over AI** — All field extraction runs offline with zero API calls. Fast, deterministic, free. Works because fashion product names are descriptively rich.

2. **Brand stripping** — Prevents "ASOS DESIGN" from dominating every embedding vector. The clean product name carries more differentiating signal.

3. **Pattern/Color separation** — Pattern field captures "Floral print" separately from Color "Sage". Prevents pattern colors (e.g. "blue floral") from being confused with garment color in embeddings.

4. **Material simplification** — Extracts "Vinyl, Polyester" from verbose fabric descriptions. Cleaner embeddings, matches how users search ("vinyl dress" not "glossy fabric with a smooth rubberised feel").

5. **Balanced market data** — Equal distribution of low/mid/high performers with correlated metrics. Enables meaningful market analysis (trend detection, price positioning) rather than everything looking successful.

6. **Card = Metadata** — What's on the product card image matches exactly what's in enriched_metadata.json. No hidden fields, no verbose blobs — clean and aligned for both visual inspection and embedding generation.
