# FASHAI Search Architecture

## Three Search Modes

### 1. Image Only (upload image, no description)

```
Upload Image
    |
Rekognition DetectLabels (>90% confidence, Apparel and Accessories only)
    -> Garment Type: Dress (95%)
    -> Pattern: Floral (92%)
    -> Materials: Silk (91%)
    |
Gemma Vision (Converse API, temperature=0, STYLE_PROMPT)
    -> Description: Black floral print long sleeve dress
    -> Color: Black (garment color only, NOT pattern colors)
    -> Pattern: Blue floral print (pattern type + pattern colors)
    -> Material: Silk
    -> Length: Midi
    -> Sleeve: Long sleeve
    -> Silhouette: A-line
    -> Niche: Casual Wear
    -> Occasion: Daily
    -> Season: Summer
    -> Style tags: A-line, fitted
    |
Search text = Gemma Vision output only (Rekognition NOT in embeddings)
    "Description: Black floral print long sleeve dress. Color: Black.
     Pattern: Blue floral print. Material: Silk. Length: Midi..."
    |
Single Titan hybrid embedding (inputImage + inputText) -> 1024-dim query vector
    |
kNN search on OpenSearch (top 15)
    |
Cohere Rerank (cohere.rerank-v3-5:0) -> top 10 re-ordered
    |
Market Readiness Score (trend + price + validation + demand)
    |
Gemma text insights (top 5, 7-section report with GO/CAUTION/NO GO verdict)
    (Rekognition labels passed to insights prompt for context)
```

### 2. Image + Text (upload image + describe design)

```
Upload Image + User Description (e.g. "Orange summer skirt with floral sleeveless top")
    |
Rekognition DetectLabels (>90% confidence)
    -> same label extraction as Image Only
    |
Gemma Vision (Converse API, temperature=0, STYLE_PROMPT_WITH_DESC)
    -> Sees image AND user's description
    -> Fuses visual analysis with user intent into single structured output:
       Description: Orange floral sleeveless co-ord set
       Color: Orange (steered by user description)
       Pattern: Floral print
       Material: Cotton
       Length: Midi
       Sleeve: Sleeveless
       Silhouette: A-line
       Niche: Casual Wear
       Occasion: Daily
       Season: Summer
       Style tags: floral, co-ord
    |
Search text = Gemma Vision fused output only (Rekognition NOT in embeddings)
    |
Single Titan hybrid embedding (inputImage + inputText) -> 1024-dim query vector
    |
kNN search on OpenSearch (top 15)
    |
Cohere Rerank (cohere.rerank-v3-5:0) -> top 10 re-ordered
    |
Market Readiness Score
    |
Gemma text insights (top 5)
    (Rekognition labels passed to insights prompt for context)
```

Key difference: `STYLE_PROMPT_WITH_DESC` bakes the user's description into the Gemma Vision prompt. Gemma combines what it sees with what the user said — user intent wins for ambiguous fields.

### 3. Text Only (no image, just type description)

```
User Description (e.g. "red party dress" or "Black floral print long sleeve dress")
    |
Gemma text-only (TEXT_SEARCH_PROMPT, InvokeModel)
    -> Imagines what this garment would look like as a product listing
    -> Classifies query as Generic or Detailed
    -> Outputs structured fields with Color/Pattern separation:
       Query type: Generic (or Detailed)
       Color: Red
       Pattern: None
       Material: Satin, Sequin
       Length: Mini
       Sleeve: Sleeveless
       Silhouette: Bodycon
       Niche: Party Wear
       Occasion: Party, Night Out
       Season: All-Season
       Style tags: bodycon, glamorous
    |
Search text = raw user description + Gemma structured output
    "red party dress Color: Red. Pattern: None. Material: Satin. Length: Mini..."
    (raw text preserves original keywords like "red" as safety net)
    |
Titan Text Embedding -> 1024-dim query vector
    |
    +-- Generic query (e.g. "party dresses", "swimwear")
    |       -> Hybrid search: kNN + BM25 on clothing_niche, occasion, style_tags
    |       -> BM25 boosts products matching "party" in occasion/niche fields
    |
    +-- Detailed query (e.g. "Black floral print long sleeve dress")
            -> kNN only (BM25 skipped to avoid keyword noise)
    |
Top 15 candidates from OpenSearch
    |
Cohere Rerank (cohere.rerank-v3-5:0) -> top 10 re-ordered
    |
Market Readiness Score
    |
Gemma text insights (top 5)
```

Gemma classifies and structures in one call — zero extra latency. Generic queries get BM25 keyword boosting for better category-level matching. Detailed queries stay pure kNN for precision.

### 4. Compare Mode

```
Design 01 (image/image+text/text) + Design 02 (image/image+text/text)
    |
Two parallel match_only API calls (one per design)
    -> Image: Rekognition + Gemma Vision + hybrid embedding + kNN + Cohere Rerank
    -> Text-only: Gemma structured + text embedding + kNN/BM25 + Cohere Rerank
    -> Mixed modalities supported (e.g. image for Design 01, text for Design 02)
    |
Single compare_insights API call with both product sets (top 5 per design)
    -> Gemma generates 6-section collection comparison report
    -> COLLECTION VERDICT: GO / CAUTION / NO GO for each design + pair
```

Compare mode uses the same pipeline as single-image/text search (no legacy 70/30 blending).

## Key Differences

| Aspect | Image Only | Image + Text | Text Only |
|---|---|---|---|
| Rekognition | Yes (insights only) | Yes (insights only) | No |
| Gemma Vision | STYLE_PROMPT (image only) | STYLE_PROMPT_WITH_DESC (image + user text) | TEXT_SEARCH_PROMPT (text-only, imagines product) |
| Structured fields | Description, Color, Pattern, Material, Length, Sleeve, Silhouette, Niche, Occasion, Season, Style tags | Same fields | Same fields + Query type |
| Color/Pattern separation | Yes | Yes | Yes |
| Description field | Yes (natural sentence) | Yes (natural sentence) | No (raw text preserved) |
| Query classification | No | No | Yes (Generic/Detailed) |
| Image Embedding | Hybrid (image + text) | Hybrid (image + text) | Text only |
| Search method | kNN | kNN | Generic: kNN + BM25 / Detailed: kNN |
| BM25 fields | N/A | N/A | clothing_niche, occasion, style_tags |
| Search text | Gemma output (no Rekognition) | Gemma fused output (no Rekognition) | Raw description + Gemma structured |
| Cohere Rerank | Yes | Yes | Yes |
| Results fetched | 15 | 15 | 15 |
| Results displayed | 10 | 10 | 10 |
| Results to insights | 5 | 5 | 5 |
| Market Readiness | Yes | Yes | Yes |
| Gemma Insights | Yes | Yes | Yes |

## Indexing (embedding_pipeline.py)

Each product is indexed using a **hybrid embedding** that fuses three information sources into one 1024-dim vector.

### What goes into each embedding

**Card image** (`cards/fash_img_XXXX.png`): A rich product card containing the product photo (top half) plus text overlay (bottom half) with name, price, color, material, niche, occasion, style tags, and market stats. Titan's vision model reads both the visual features AND the text printed on the card.

**Metadata text** (`metadata_to_text()`): Structured string built from the JSON metadata:
```
Sage sleeveless mini dress with bodycon fit. Color: Sage. Material: Vinyl, Polyester.
Length: Mini. Sleeve: Sleeveless. Silhouette: Bodycon. Niche: Y2K / 2000s Revival.
Occasion: Night Out / Club. Season: All-season. Style tags: corset, mini, bodycon,
zip-back, sweetheart neck.
```

### Titan API call

Both inputs go into a single call to `amazon.titan-embed-image-v1`:

```python
body = {
    "inputImage": "<base64 of the full 512x512 card image>",
    "inputText":  "Sage sleeveless mini dress with bodycon fit. Color: Sage...",
    "embeddingConfig": {"outputEmbeddingLength": 1024}
}
```

Titan fuses three information channels:
1. **Visual features from the photo** — color, texture, silhouette, pattern
2. **Text on the card image** — Titan's vision reads printed text (name, price, niche, tags, stats)
3. **Explicit inputText** — clean structured metadata as a safety net

The double reinforcement (same info via card text + inputText) makes embeddings robust — if OCR misreads something, inputText still provides it clearly.

### Why format alignment matters

All three search modes generate query text in the **same field format** as `metadata_to_text()` (Color:, Pattern:, Niche:, etc.). Matching formats between indexing and querying ensures vectors land close together in the 1024-dim embedding space.

See `embeddings_logic.md` for a detailed end-to-end walkthrough with a real product example.
