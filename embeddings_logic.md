# Embedding Pipeline — How It Works

## Overview

The embedding pipeline (`embedding_pipeline.py`) is an offline indexing script that converts 5000 fashion products into searchable vectors. Each product gets a single 1024-dimensional embedding that captures both visual and semantic information.

## End-to-End Example: Product 0001

### The Card Image

Each product has a **card image** (`fash_img_XXXX.png`) that is NOT just a photo — it's a rich visual document containing:
- **Top half**: Product photo (model wearing the garment)
- **Bottom half**: Text overlay with product name, price, color, material, niche, occasion, season, style tags, and market performance stats

Example card for product 0001:
- Photo: Model wearing a sage lacquer corset mini dress
- Text on card: "lacquer corset mini dress", "$34.00", "Color: Sage", "Material: Vinyl, Polyester", "Y2K / 2000s Revival", "Night Out / Club", style tags, market stats

### Step 1: Load Metadata from S3

Downloads `enriched_metadata_5000.json` from S3. Each product has structured fields:

```json
{
  "product_id": "0001",
  "name": "lacquer corset mini dress with drape skirt in sage",
  "color": "Sage",
  "pattern": "None",
  "material": "Vinyl, Polyester",
  "length": "Mini",
  "sleeve_type": "Sleeveless",
  "silhouette": "Bodycon",
  "description": "Sage sleeveless mini dress with bodycon fit",
  "clothing_niche": "Y2K / 2000s Revival",
  "occasion": "Night Out / Club",
  "season": "All-season",
  "style_tags": ["corset", "mini", "bodycon", "zip-back", "sweetheart neck"],
  "units_sold": 821,
  "revenue_generated": 21605,
  "avg_rating": 3.8,
  "sales_trend": "stable"
}
```

### Step 2: Load Card Image from S3

Fetches `cards/fash_img_0001.png` — the **entire card** (photo + text overlay). Pipeline resizes it to 512x512 pixels and converts to base64-encoded JPEG.

### Step 3: Convert Metadata to Text (`metadata_to_text`)

Titan's `inputText` expects a single string, not JSON. So `metadata_to_text()` flattens the structured fields:

```
Sage sleeveless mini dress with bodycon fit. Color: Sage. Material: Vinyl, Polyester.
Length: Mini. Sleeve: Sleeveless. Silhouette: Bodycon. Niche: Y2K / 2000s Revival.
Occasion: Night Out / Club. Season: All-season. Style tags: corset, mini, bodycon,
zip-back, sweetheart neck.
```

**Why not just use the image?** The photo captures visual features (color, shape, texture) but can't convey semantic meaning — occasion, season, niche, style tags. The text fills that gap.

**Why this format matters:** During search, Gemma Vision produces structured text in the **same field format** (Color:, Pattern:, Niche:, etc.). Matching formats between indexing and querying ensures vectors land close together in the embedding space.

### Step 4: Titan Creates One Hybrid Embedding

Both inputs go into a **single API call** to `amazon.titan-embed-image-v1`:

```python
{
    "inputImage": "<base64 of the full 512x512 card image>",
    "inputText":  "Sage sleeveless mini dress with bodycon fit. Color: Sage. Material: Vinyl, Polyester...",
    "embeddingConfig": {"outputEmbeddingLength": 1024}
}
```

Titan internally fuses **three sources of information**:

1. **Visual features from the photo** — the sage/green color, shiny vinyl texture, bodycon silhouette, mini length
2. **Text burned into the card image** — Titan's vision model can read text in images, so it picks up "Y2K / 2000s Revival", "$34.00", "Vinyl", style tags, etc. printed on the card
3. **The explicit inputText string** — clean, structured metadata ensuring nothing is missed

The intentional **double reinforcement** (same info via image text + inputText) makes embeddings more robust — if Titan's OCR misreads something on the card, the inputText still provides it clearly.

Output: a **1024-dimensional vector** (array of 1024 floating-point numbers). Each dimension represents some learned feature (not human-interpretable, but mathematically meaningful for similarity search).

### Step 5: Save to S3

All embeddings + metadata are saved as `embeddings/embeddings.json` on S3. Each entry contains the 1024-dim vector plus all product metadata fields. These are later indexed into OpenSearch for kNN search.

## How Search Uses These Embeddings

When a user uploads a design, the query pipeline creates a **query vector the same way**:

| Search Mode | Query Image | Query Text |
|---|---|---|
| Image Only | User's uploaded image | Gemma Vision structured output (same field format as `metadata_to_text`) |
| Image + Text | User's uploaded image | Gemma Vision output fused with user's description |
| Text Only | None (text embedding only) | Raw user text + Gemma structured output |

OpenSearch finds the products whose stored vectors are **geometrically closest** to the query vector using kNN (k-nearest neighbors). Products with similar visual appearance AND similar semantic attributes score highest.

## Pipeline Configuration

| Setting | Value |
|---|---|
| S3 bucket | `fashion-assistant-images` |
| Metadata key | `metadata/enriched_metadata_5000.json` |
| Card images prefix | `cards/` |
| Output key | `embeddings/embeddings.json` |
| Titan model | `amazon.titan-embed-image-v1` |
| Image resize | 512x512 |
| Vector dimensions | 1024 |
| Sleep between calls | 0.5s |
| Total products | 5000 |

## Running the Pipeline

```bash
python embedding_pipeline.py --test          # first 5 products (verification)
python embedding_pipeline.py --all           # all 5000 products (~42 min)
python embedding_pipeline.py --start 0 --end 500   # specific range
```
