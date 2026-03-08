import boto3
import json
import base64
from io import BytesIO
from PIL import Image
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

import math

REGION              = "ap-south-1"
S3_BUCKET           = "fashion-assistant-images"
INDEX_NAME          = "fashion-products"
TITAN_MODEL_ID      = "amazon.titan-embed-image-v1"
GEMMA_MODEL_ID      = "google.gemma-3-27b-it"
RERANK_MODEL_ID     = "cohere.rerank-v3-5:0"
OPENSEARCH_ENDPOINT = "vmryh04ckerilvnvd5se.ap-south-1.aoss.amazonaws.com"
FETCH_K             = 15
RETURN_K            = 10
INSIGHTS_K          = 5

s3          = boto3.client("s3",              region_name=REGION)
bedrock     = boto3.client("bedrock-runtime", region_name=REGION)
rekognition = boto3.client("rekognition",     region_name=REGION)

# Fashion-relevant label categories for Rekognition filtering (grouped)
GARMENT_TYPES = {
    "dress", "shirt", "pants", "skirt", "jacket", "coat", "blouse", "sweater",
    "hoodie", "jeans", "shorts", "suit", "blazer", "vest", "cardigan", "leggings",
    "jumpsuit", "romper", "gown", "tunic", "cape", "top", "trouser", "kurta",
    "saree", "sari", "lehenga", "salwar", "kameez", "dupatta", "sherwani",
    "t-shirt", "tank top", "crop top", "maxi dress", "mini dress", "midi dress",
    "wrap dress", "a-line", "bodycon", "sundress", "cocktail dress",
}
PATTERN_KEYWORDS = {
    "floral", "striped", "plaid", "polka dot", "checkered", "geometric", "abstract",
    "solid", "printed", "embroidered", "sequin", "tie-dye", "camouflage", "paisley",
    "animal print", "leopard", "zebra", "houndstooth", "tartan", "ikat", "batik",
    "tribal", "graphic", "motif", "block print",
}
FABRIC_KEYWORDS = {
    "denim", "silk", "cotton", "linen", "leather", "wool", "polyester", "satin",
    "velvet", "chiffon", "lace", "knit", "tweed", "corduroy", "suede", "nylon",
    "organza", "tulle", "georgette", "crepe", "jersey", "mesh", "fleece",
}
DETAIL_KEYWORDS = {
    "sleeve", "collar", "pocket", "zipper", "button", "hem", "ruffle", "pleated",
    "v-neck", "round neck", "off-shoulder", "halter", "backless", "slit",
}
# Combined set for general matching
FASHION_LABELS = (GARMENT_TYPES | PATTERN_KEYWORDS | FABRIC_KEYWORDS |
                  DETAIL_KEYWORDS |
                  {"clothing", "apparel", "person", "woman", "man", "fashion"})

def get_os_client():
    credentials = boto3.Session(region_name=REGION).get_credentials()
    auth = AWSV4SignerAuth(credentials, REGION, "aoss")
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30
    )

def image_to_base64(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((512, 512), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def get_query_embedding(image_b64):
    response = bedrock.invoke_model(
        modelId=TITAN_MODEL_ID,
        body=json.dumps({
            "inputImage": image_b64,
            "embeddingConfig": {"outputEmbeddingLength": 1024}
        }),
        contentType="application/json",
        accept="application/json"
    )
    return json.loads(response["body"].read())["embedding"]

def _categorize_label(name_lower):
    """Return the fashion category for a label, or None if not fashion-relevant."""
    if any(k in name_lower for k in GARMENT_TYPES) or any(name_lower in k for k in GARMENT_TYPES):
        return "garment_type"
    if any(k in name_lower for k in PATTERN_KEYWORDS) or any(name_lower in k for k in PATTERN_KEYWORDS):
        return "pattern"
    if any(k in name_lower for k in FABRIC_KEYWORDS) or any(name_lower in k for k in FABRIC_KEYWORDS):
        return "fabric"
    if any(k in name_lower for k in DETAIL_KEYWORDS) or any(name_lower in k for k in DETAIL_KEYWORDS):
        return "detail"
    if name_lower in {"clothing", "apparel", "fashion"}:
        return "garment_type"
    if name_lower in {"person", "woman", "man"}:
        return None  # skip generic person labels
    return None

SKIN_COLORS = {
    "bisque", "peachpuff", "navajowhite", "moccasin", "wheat", "tan",
    "burlywood", "sandybrown", "peru", "sienna", "saddlebrown", "chocolate",
    "rosybrown", "indianred", "mistyrose", "linen", "oldlace", "seashell",
    "antiquewhite", "blanchedalmond", "papayawhip", "cornsilk", "beige",
}

def _is_skin_tone(css_name, r, g, b):
    """Heuristic: skip common skin-tone colors."""
    if css_name.lower() in SKIN_COLORS:
        return True
    # Warm mid-tones with high red, moderate green, lower blue = likely skin
    if r > 150 and g > 100 and b > 60 and (r - b) > 40 and abs(r - g) < 80:
        return True
    return False

def _extract_garment_colors(labels):
    """Extract dominant colors from garment label instances (bounding boxes), not the full image."""
    colors = []
    for label in labels:
        name_lower = label["Name"].lower()
        # Only extract colors from garment/clothing instances
        if not any(k in name_lower for k in GARMENT_TYPES) and name_lower not in {"clothing", "apparel"}:
            continue
        for instance in label.get("Instances", []):
            for c in instance.get("DominantColors", []):
                css_name = c.get("CSSColor", "")
                hex_code = c.get("HexCode", "")
                pct = round(c.get("PixelPercent", 0), 1)
                r, g, b = c.get("Red", 0), c.get("Green", 0), c.get("Blue", 0)
                if not css_name or pct < 4:
                    continue
                if _is_skin_tone(css_name, r, g, b):
                    continue
                colors.append({
                    "label": css_name,
                    "hex": hex_code,
                    "rgb": [r, g, b],
                    "pixel_percent": pct,
                    "category": "color",
                    "confidence": pct
                })
    # Deduplicate and sort by coverage
    seen = set()
    unique = []
    for c in sorted(colors, key=lambda x: x["pixel_percent"], reverse=True):
        if c["label"].lower() not in seen:
            seen.add(c["label"].lower())
            unique.append(c)
    return unique[:4]

def _extract_fallback_colors(image_properties):
    """Fallback: use foreground dominant colors if no garment instances found, filtering skin tones."""
    colors = []
    fg = image_properties.get("Foreground", {})
    dom_colors = fg.get("DominantColors", [])
    if not dom_colors:
        dom_colors = image_properties.get("DominantColors", [])
    for c in dom_colors[:6]:
        css_name = c.get("CSSColor", "")
        hex_code = c.get("HexCode", "")
        pct = round(c.get("PixelPercent", 0), 1)
        r, g, b = c.get("Red", 0), c.get("Green", 0), c.get("Blue", 0)
        if not css_name or pct < 5:
            continue
        if _is_skin_tone(css_name, r, g, b):
            continue
        colors.append({
            "label": css_name,
            "hex": hex_code,
            "rgb": [r, g, b],
            "pixel_percent": pct,
            "category": "color",
            "confidence": pct
        })
    return colors[:3]

def detect_visual_attributes(image_bytes):
    """Call Rekognition to detect garment type, pattern, fabric, details — only >90% confidence."""
    try:
        response = rekognition.detect_labels(
            Image={"Bytes": image_bytes},
            MaxLabels=40,
            MinConfidence=90.0,
            Features=["GENERAL_LABELS"],
            Settings={
                "GeneralLabels": {
                    "LabelCategoryInclusionFilters": [
                        "Apparel and Accessories",
                    ]
                }
            }
        )

        attrs = []
        for label in response.get("Labels", []):
            name_lower = label["Name"].lower()
            category = _categorize_label(name_lower)
            if category:
                attrs.append({
                    "label": label["Name"],
                    "confidence": round(label["Confidence"], 1),
                    "category": category
                })

        # Sort by confidence descending, deduplicate
        attrs.sort(key=lambda x: x["confidence"], reverse=True)
        seen = set()
        unique = []
        for a in attrs:
            key = a["label"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(a)

        return unique[:10]
    except Exception as e:
        print(f"Rekognition error (non-fatal): {e}")
        return []

def _attrs_to_search_text(visual_attributes):
    """Build search text from high-confidence Rekognition labels (>90%)."""
    garments = []
    patterns = []
    fabrics = []
    details = []

    for a in visual_attributes:
        cat = a.get("category", "")
        label = a["label"]
        if cat == "garment_type":
            garments.append(label)
        elif cat == "pattern":
            patterns.append(label)
        elif cat == "fabric":
            fabrics.append(label)
        elif cat == "detail":
            details.append(label)

    if not any([garments, patterns, fabrics]):
        return ""

    parts = []
    if garments:
        parts.append(f"Category: {', '.join(garments)}.")
    if patterns:
        parts.append(f"Style tags: {', '.join(patterns)}.")
    if fabrics:
        parts.append(f"Materials: {', '.join(fabrics)}.")

    return " ".join(parts)

def get_hybrid_embedding(image_b64, text):
    """Get 1024-dim embedding from Titan using both image and text — same as indexing."""
    body = {"inputImage": image_b64, "embeddingConfig": {"outputEmbeddingLength": 1024}}
    if text:
        body["inputText"] = text
    response = bedrock.invoke_model(
        modelId=TITAN_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )
    return json.loads(response["body"].read())["embedding"]

def get_text_embedding(text):
    """Get 1024-dim embedding from Titan using text input."""
    response = bedrock.invoke_model(
        modelId=TITAN_MODEL_ID,
        body=json.dumps({
            "inputText": text,
            "embeddingConfig": {"outputEmbeddingLength": 1024}
        }),
        contentType="application/json",
        accept="application/json"
    )
    return json.loads(response["body"].read())["embedding"]

def combine_vectors(image_vec, text_vec, image_weight=0.7):
    """Weighted average of image and text embeddings, L2-normalized."""
    tw = 1 - image_weight
    combined = [image_weight * a + tw * b for a, b in zip(image_vec, text_vec)]
    norm = math.sqrt(sum(x * x for x in combined))
    if norm > 0:
        combined = [x / norm for x in combined]
    return combined

def rerank_by_attributes(products, visual_attributes, return_k=RETURN_K):
    """Rerank products using 70% OpenSearch similarity + 30% attribute overlap."""
    if not visual_attributes:
        return products[:return_k]

    attr_labels = set(a["label"].lower() for a in visual_attributes)

    for p in products:
        # Collect all text fields to match against
        match_fields = []
        match_fields.append((p.get("clothing_niche") or "").lower())
        match_fields.append((p.get("occasion") or "").lower())
        match_fields.append((p.get("target_audience") or "").lower())
        for tag in (p.get("style_tags") or []):
            match_fields.append(tag.lower())

        product_text = " ".join(match_fields)

        # Count how many Rekognition labels appear in the product's metadata
        matches = sum(1 for al in attr_labels if al in product_text)
        overlap_score = matches / len(attr_labels) if attr_labels else 0

        os_score = p.get("score", 0)
        p["rerank_score"] = 0.7 * os_score + 0.3 * overlap_score

    products.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return products[:return_k]

def compute_market_readiness(products):
    """Compute a 0-100 market readiness score from product data."""
    if not products:
        return {"score": 0, "breakdown": {"trend": 0, "price": 0, "validation": 0, "demand": 0}}

    # Trend signal (0-25): % of matches with "rising" trend
    rising = sum(1 for p in products if (p.get("sales_trend") or "").lower() == "rising")
    trend_score = round((rising / len(products)) * 25)

    # Price competitiveness (0-25): favor mid-range pricing, penalize extremes
    tier_scores = {"budget": 15, "low": 20, "mid": 25, "high": 18, "premium": 12, "luxury": 8}
    tiers = [tier_scores.get((p.get("price_tier") or "").lower(), 15) for p in products]
    price_score = round(sum(tiers) / len(tiers)) if tiers else 12

    # Market validation (0-25): avg rating scaled (0-5 → 0-25)
    ratings = [float(p.get("avg_rating") or 0) for p in products]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    validation_score = round((avg_rating / 5.0) * 25)

    # Demand signal (0-25): avg sales velocity (units/day), capped at 5+/day = 25
    velocities = []
    for p in products:
        units = float(p.get("units_sold") or 0)
        days = max(float(p.get('days_listed') or 1), 1)
        velocities.append(units / days)
    avg_vel = sum(velocities) / len(velocities) if velocities else 0
    demand_score = round(min(avg_vel / 5.0, 1.0) * 25)

    total = trend_score + price_score + validation_score + demand_score

    return {
        "score": total,
        "breakdown": {
            "trend": trend_score,
            "price": price_score,
            "validation": validation_score,
            "demand": demand_score
        }
    }

def search_similar(query_vector, top_k=FETCH_K):
    client = get_os_client()
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": top_k
                    }
                }
            },
            "_source": {"excludes": ["embedding"]}
        }
    )
    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        source["score"] = hit["_score"]
        results.append(source)
    return results

def search_hybrid(query_vector, text_query, top_k=FETCH_K):
    """Hybrid search: kNN vector similarity + BM25 keyword matching. Used for text-only search."""
    client = get_os_client()
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": top_k
                                }
                            }
                        },
                        {
                            "multi_match": {
                                "query": text_query,
                                "fields": ["clothing_niche", "occasion", "style_tags"]
                            }
                        }
                    ]
                }
            },
            "_source": {"excludes": ["embedding"]}
        }
    )
    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        source["score"] = hit["_score"]
        results.append(source)
    return results

def _product_to_text(p):
    """Convert a product result to text for reranking. Mirrors metadata_to_text() format."""
    tags = ", ".join(p.get("style_tags") or [])
    parts = [f"{p.get('name', '')}."]
    parts.append(f"Color: {p.get('color', '')}.")
    pattern = p.get('pattern', 'None')
    if pattern and pattern != 'None':
        parts.append(f"Pattern: {pattern}.")
    parts.append(f"Material: {p.get('material', '')}.")
    length = p.get('length', 'None')
    if length and length != 'None':
        parts.append(f"Length: {length}.")
    sleeve = p.get('sleeve_type', 'None')
    if sleeve and sleeve != 'None':
        parts.append(f"Sleeve: {sleeve}.")
    silhouette = p.get('silhouette', 'None')
    if silhouette and silhouette != 'None':
        parts.append(f"Silhouette: {silhouette}.")
    parts.append(f"Niche: {p.get('clothing_niche', '')}.")
    parts.append(f"Occasion: {p.get('occasion', '')}.")
    parts.append(f"Season: {p.get('season', '')}.")
    parts.append(f"Style tags: {tags}.")
    return " ".join(parts)

def rerank_results(query_text, products, top_n=RETURN_K):
    """Rerank search results using Cohere Rerank on Bedrock."""
    if not products or not query_text:
        return products[:top_n]
    try:
        documents = [_product_to_text(p) for p in products]
        response = bedrock.invoke_model(
            modelId=RERANK_MODEL_ID,
            body=json.dumps({
                "query": query_text,
                "documents": documents,
                "top_n": top_n
            }),
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response["body"].read())
        reranked = []
        for r in result["results"]:
            idx = r["index"]
            products[idx]["rerank_score"] = r["relevance_score"]
            reranked.append(products[idx])
        print(f"Reranked {len(products)} → top {len(reranked)}")
        return reranked
    except Exception as e:
        print(f"Rerank error (non-fatal, falling back to kNN order): {e}")
        return products[:top_n]

def get_image_url(image_s3_key):
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": image_s3_key},
            ExpiresIn=3600
        )
    except:
        return None

def call_gemma(prompt, max_tokens=500):
    response = bedrock.invoke_model(
        modelId=GEMMA_MODEL_ID,
        body=json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }),
        contentType="application/json",
        accept="application/json"
    )
    return json.loads(response["body"].read())["choices"][0]["message"]["content"]

def call_gemma_vision(image_bytes, prompt, max_tokens=100, temperature=0.0):
    """Call Gemma with an image using the Bedrock Converse API."""
    response = bedrock.converse(
        modelId=GEMMA_MODEL_ID,
        messages=[{
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "jpeg",
                        "source": {"bytes": image_bytes}
                    }
                },
                {
                    "text": prompt
                }
            ]
        }],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature}
    )
    return response["output"]["message"]["content"][0]["text"]

STYLE_PROMPT = """Look at the outfit and extract these fields. Use short, factual values only. No sentences.

Description: (one short natural sentence describing the garment focusing on color, pattern, and style, e.g. Black floral print long sleeve dress)
Color: (primary garment color only, NOT pattern colors, e.g. Orange, Teal, Dusty Rose)
Pattern: (pattern type + pattern colors if any, e.g. Blue floral print, Geometric gold and black, None if solid)
Material: (fabric appearance, e.g. Chiffon, Cotton, Silk, Knit, Denim)
Length: (e.g. Mini, Midi, Maxi, Cropped, Knee-length, None if not applicable)
Sleeve: (e.g. Sleeveless, Short sleeve, Long sleeve, Puff sleeve, None if not applicable)
Silhouette: (e.g. Bodycon, A-line, Fitted, Relaxed, Oversized, Wrap, None if not applicable)
Niche: (e.g. Casual Wear, Ethnic Wear, Formal Wear, Streetwear)
Occasion: (e.g. Daily, Party, Wedding, Office, Beach)
Season: (e.g. Summer, Winter, All-Season, Monsoon)
Style tags: (2-4 tags, e.g. A-line, fitted, bohemian, minimal)"""

STYLE_PROMPT_WITH_DESC = """The designer describes this as: "{description}"

Look at the outfit and the designer's description above. Extract these fields by combining what you see in the image with the designer's description. Use short, factual values only. No sentences.

Description: (one short natural sentence describing the garment focusing on color, pattern, and style)
Color: (primary garment color only, NOT pattern colors, e.g. Orange, Teal, Dusty Rose)
Pattern: (pattern type + pattern colors if any, e.g. Blue floral print, Geometric gold and black, None if solid)
Material: (fabric appearance, e.g. Chiffon, Cotton, Silk, Knit, Denim)
Length: (e.g. Mini, Midi, Maxi, Cropped, Knee-length, None if not applicable)
Sleeve: (e.g. Sleeveless, Short sleeve, Long sleeve, Puff sleeve, None if not applicable)
Silhouette: (e.g. Bodycon, A-line, Fitted, Relaxed, Oversized, Wrap, None if not applicable)
Niche: (e.g. Casual Wear, Ethnic Wear, Formal Wear, Streetwear)
Occasion: (e.g. Daily, Party, Wedding, Office, Beach)
Season: (e.g. Summer, Winter, All-Season, Monsoon)
Style tags: (2-4 tags, e.g. A-line, fitted, bohemian, minimal)"""

TEXT_SEARCH_PROMPT = """A user is searching for fashion products using this description: "{description}"

Imagine what this garment would realistically look like as a product listing. If the description is brief, fill in all fields with realistic values. If detailed, extract what is given and fill the rest. Always separate garment color from pattern colors.

Use short, factual values only. No sentences.

Query type: (Generic or Detailed — Generic if the description is vague/broad like "party dresses" or "swimwear" or "traditional wear", Detailed if it specifies color/pattern/style like "red floral long sleeve dress")
Color: (primary garment color only, NOT pattern colors, e.g. Orange, Teal, Dusty Rose)
Pattern: (pattern type + pattern colors if any, e.g. Blue floral print, Geometric gold and black, None if solid)
Material: (fabric appearance, e.g. Chiffon, Cotton, Silk, Knit, Denim)
Length: (e.g. Mini, Midi, Maxi, Cropped, Knee-length, None if not applicable)
Sleeve: (e.g. Sleeveless, Short sleeve, Long sleeve, Puff sleeve, None if not applicable)
Silhouette: (e.g. Bodycon, A-line, Fitted, Relaxed, Oversized, Wrap, None if not applicable)
Niche: (e.g. Casual Wear, Ethnic Wear, Formal Wear, Streetwear)
Occasion: (e.g. Daily, Party, Wedding, Office, Beach)
Season: (e.g. Summer, Winter, All-Season, Monsoon)
Style tags: (2-4 tags, e.g. A-line, fitted, bohemian, minimal)"""

def generate_text_search_description(description):
    """Use Gemma text-only to convert user description into structured product-like format."""
    try:
        prompt = TEXT_SEARCH_PROMPT.format(description=description)
        result = call_gemma(prompt, max_tokens=150)
        print(f"Text search description: {result}")
        return result.strip()
    except Exception as e:
        print(f"Gemma text reformat error (non-fatal): {e}")
        return description

def generate_style_description(image_bytes_compressed, design_description=""):
    """Use Gemma vision to extract structured fashion attributes from an image, optionally guided by user description."""
    try:
        if design_description:
            prompt = STYLE_PROMPT_WITH_DESC.format(description=design_description)
        else:
            prompt = STYLE_PROMPT
        desc = call_gemma_vision(image_bytes_compressed, prompt, max_tokens=150, temperature=0.0)
        print(f"Style description: {desc}")
        return desc.strip()
    except Exception as e:
        print(f"Gemma vision error (non-fatal): {e}")
        return ""

def generate_insights(similar_products, design_description="", visual_attributes=None):
    products_text = ""
    for i, p in enumerate(similar_products, 1):
        units = float(p.get('units_sold') or 0)
        days  = float(p.get('days_listed') or 1)
        velocity = round(units / days, 1) if days else 0
        conv_raw = p.get('conv. rate') or p.get('conv_rate') or p.get('conversion_rate')
        if conv_raw is not None:
            cv = float(conv_raw)
            conv_pct = f"{cv*100:.1f}" if 0 < cv < 1 else f"{cv:.1f}"
        else:
            conv_pct = 'N/A'
        products_text += f"""
Product {i}: {p.get('name')}
  Price: ${p.get('price')} ({p.get('price_tier')}) | Revenue: ${p.get('revenue', 'N/A')}
  Units sold: {int(units)} | Days listed: {int(days)} | Velocity: {velocity} units/day
  Inventory: {p.get('inventory', 'N/A')} units | Conv. rate: {conv_pct}%
  Niche: {p.get('clothing_niche')} | Audience: {p.get('target_audience')}
  Occasion: {p.get('occasion')} | Season: {p.get('season')}
  Style tags: {', '.join(p.get('style_tags', []))}
  Sales trend: {p.get('sales_trend')} | Avg rating: {p.get('avg_rating')}
  Similarity score: {p.get('score', 0):.3f}
"""
    va_text = ""
    # Style description from Gemma vision (colors, garment type, style)
    if design_description:
        va_text += f"AI-generated style description of the uploaded design: {design_description}\n"
    # High-confidence Rekognition labels (>90%)
    if visual_attributes:
        labels = ", ".join(f"{a['label']} ({a['confidence']}%)" for a in visual_attributes)
        va_text += f"Detected garment attributes: {labels}\n"

    prompt = f"""You are a fashion market intelligence expert advising a first-time small business owner with limited budget.

A designer has uploaded a fashion design image for market analysis.
{f'Design description: {design_description}' if design_description else ''}
{va_text}
Here are the {len(similar_products)} most similar products currently in the market:
{products_text}

Give practical, specific, data-driven advice a beginner can act on immediately. Keep total response under 420 words.
Use EXACTLY these 7 section headers. No markdown, no asterisks, no hashtags. Plain text only.

TREND STATUS: Analyze all {len(similar_products)} products' sales trends together. State [Rising/Stable/Declining] as a final verdict and explain what it means for a new small business entering this space.

RECOMMENDED PRICE: Give ONE final price point (e.g. $45) — not a range. Justify it in one sentence using competitor pricing and conv. rate data only.

TARGET AUDIENCE: Who to market to and why, based on the niche and audience data above.

LAUNCH TIMING: Best season and occasion to launch. Give specific timing advice based on when similar products perform best.

MARKET ENTRY BARRIER: [Easy/Medium/Hard] — based on competitor units sold, avg ratings, and days listed. One sentence explanation of what this means for a small business owner.

PRODUCTION BATCH SIZE: Based on competitor inventory levels and sales velocity data above, recommend a specific starting batch size (e.g. "start with 30 units"). Explain how long that stock will last at the average sell rate.

VERDICT: Write exactly in this format —
GO (or PROCEED WITH CAUTION or NO GO): [One sentence on trend + price position]. [One sentence on who buys it and the strongest competitor to watch]. [One sentence on the single differentiator to lead with and the immediate next action for the owner]."""

    return call_gemma(prompt, max_tokens=800)

def generate_compare_insights(products1, products2, desc1="", desc2=""):
    def fmt(products):
        lines = []
        for p in products:
            units = float(p.get('units_sold') or 0)
            days  = float(p.get('days_listed') or 1)
            vel   = round(units / days, 1) if days else 0
            lines.append(
                f"  - {p.get('name')} | ${p.get('price')} ({p.get('price_tier')}) | "
                f"Revenue: ${p.get('revenue','N/A')} | Velocity: {vel}/day | "
                f"Niche: {p.get('clothing_niche')} | Audience: {p.get('target_audience')} | "
                f"Occasion: {p.get('occasion')} | Season: {p.get('season')} | "
                f"Trend: {p.get('sales_trend')} | Rating: {p.get('avg_rating')}"
            )
        return "\n".join(lines)

    d1_label = f'Design 01{f" ({desc1})" if desc1 else ""}'
    d2_label = f'Design 02{f" ({desc2})" if desc2 else ""}'

    prompt = (
        "You are a fashion market analyst advising a small business owner. "
        "Do NOT use markdown, asterisks, hashtags, or ** formatting. Plain text only.\n\n"
        f"{d1_label} — similar market products:\n{fmt(products1)}\n\n"
        f"{d2_label} — similar market products:\n{fmt(products2)}\n\n"
        "Respond using EXACTLY these 6 section headers in this order. "
        "For every dimension, state what Design 01 does, what Design 02 does, then where they converge or diverge.\n\n"
        "DESIGN 01 PROFILE:\n"
        f"Describe {d1_label} in 2 sentences: its niche, who buys it, when they wear it, and its trend direction.\n\n"
        "DESIGN 02 PROFILE:\n"
        f"Describe {d2_label} in 2 sentences: its niche, who buys it, when they wear it, and its trend direction.\n\n"
        "AUDIENCE & OCCASION COMPARISON:\n"
        f"Design 01 targets [audience/occasion]. Design 02 targets [audience/occasion]. "
        "State whether they overlap or diverge and what that means for marketing both together.\n\n"
        "PRICE & REVENUE COMPARISON:\n"
        f"Design 01 market prices at $X with revenue $Y. Design 02 at $X with $Y. "
        "Which has stronger revenue potential and why. Suggest a pricing strategy for selling both.\n\n"
        "COLLECTION SYNERGY:\n"
        "Do these two designs work as a collection? Evaluate style coherence, occasion alignment, "
        "and seasonal fit. State clearly whether they strengthen or weaken each other.\n\n"
        "COLLECTION VERDICT:\n"
        f"{d1_label}: [GO / PROCEED WITH CAUTION / NO GO] — one sentence reason.\n"
        f"{d2_label}: [GO / PROCEED WITH CAUTION / NO GO] — one sentence reason.\n"
        "Collection as pair: [GO / PROCEED WITH CAUTION / NO GO] — one sentence on whether to launch them together."
    )
    return call_gemma(prompt, max_tokens=900)

# ── PUBLIC FUNCTIONS called by lambda_handler ─────────────────────────────────

def run_query(image_bytes, design_description=""):
    print("Step 1: Detecting garment attributes via Rekognition (>90% only)...")
    visual_attributes = detect_visual_attributes(image_bytes)
    print(f"Detected {len(visual_attributes)} high-confidence attributes")

    image_b64 = image_to_base64(image_bytes)
    compressed_bytes = base64.b64decode(image_b64)

    print("Step 2: Generating style description via Gemma Vision...")
    style_description = generate_style_description(compressed_bytes, design_description)
    search_text = style_description.strip()

    print(f"Step 3: Generating hybrid Titan embedding (image + text)...")
    print(f"  Text: {search_text}")
    query_vector = get_hybrid_embedding(image_b64, search_text)

    print("Step 4: Searching OpenSearch for similar products...")
    similar = search_similar(query_vector)

    print("Step 5: Reranking results via Cohere Rerank...")
    similar = rerank_results(search_text, similar)

    print("Step 6: Generating presigned S3 image URLs...")
    for p in similar:
        p["image_url"] = get_image_url(p.get("image_s3_key", ""))

    print("Step 7: Computing market readiness score...")
    market_readiness = compute_market_readiness(similar)

    print("Step 8: Generating AI insights via Gemma...")
    insights = generate_insights(similar[:INSIGHTS_K], style_description, visual_attributes)

    return {
        "similar_products": similar,
        "insights": insights,
        "visual_attributes": visual_attributes,
        "market_readiness": market_readiness,
        "style_description": style_description
    }

def run_match_only(image_bytes, design_description=""):
    print("Step 1: Detecting garment attributes via Rekognition (>90% only)...")
    visual_attributes = detect_visual_attributes(image_bytes)
    print(f"Detected {len(visual_attributes)} high-confidence attributes")

    image_b64 = image_to_base64(image_bytes)
    compressed_bytes = base64.b64decode(image_b64)

    print("Step 2: Generating style description via Gemma Vision...")
    style_description = generate_style_description(compressed_bytes, design_description)
    search_text = style_description.strip()

    print(f"Step 3: Generating hybrid Titan embedding (image + text)...")
    print(f"  Text: {search_text}")
    query_vector = get_hybrid_embedding(image_b64, search_text)

    print("Step 4: Searching OpenSearch for similar products...")
    similar = search_similar(query_vector)

    print("Step 5: Reranking results via Cohere Rerank...")
    similar = rerank_results(search_text, similar)

    print("Step 6: Generating presigned S3 image URLs...")
    for p in similar:
        p["image_url"] = get_image_url(p.get("image_s3_key", ""))

    print("Step 7: Computing market readiness score...")
    market_readiness = compute_market_readiness(similar)

    return {
        "similar_products": similar,
        "visual_attributes": visual_attributes,
        "market_readiness": market_readiness
    }

def run_text_search(description):
    """Text-only search: Gemma imagines structured product description, then Titan text embedding + kNN search."""
    print(f"Step 1: Converting description to structured format via Gemma...")
    structured_description = generate_text_search_description(description)

    print(f"Step 2: Generating text embedding...")
    search_text = f"{description} {structured_description}".strip()
    text_vector = get_text_embedding(search_text)

    is_generic = "generic" in structured_description.lower().split("query type:")[-1].split("\n")[0].lower() if "query type:" in structured_description.lower() else True
    if is_generic:
        print("Step 3: Searching OpenSearch (hybrid: kNN + BM25 — generic query)...")
        similar = search_hybrid(text_vector, description)
    else:
        print("Step 3: Searching OpenSearch (kNN only — detailed query)...")
        similar = search_similar(text_vector)
    print(f"Fetched {len(similar)} candidates")

    print("Step 4: Reranking results via Cohere Rerank...")
    similar = rerank_results(search_text, similar)

    print("Step 5: Generating image URLs...")
    for p in similar:
        p["image_url"] = get_image_url(p.get("image_s3_key", ""))

    print("Step 6: Computing market readiness score...")
    market_readiness = compute_market_readiness(similar)

    print("Step 7: Generating AI insights via Gemma...")
    insights = generate_insights(similar[:INSIGHTS_K], description)

    return {
        "similar_products": similar,
        "insights": insights,
        "visual_attributes": [],
        "market_readiness": market_readiness
    }

def run_text_match_only(description):
    """Text-only match for compare mode: same as run_text_search but without insights."""
    print(f"Step 1: Converting description to structured format via Gemma...")
    structured_description = generate_text_search_description(description)

    print(f"Step 2: Generating text embedding...")
    search_text = f"{description} {structured_description}".strip()
    text_vector = get_text_embedding(search_text)

    is_generic = "generic" in structured_description.lower().split("query type:")[-1].split("\n")[0].lower() if "query type:" in structured_description.lower() else True
    if is_generic:
        print("Step 3: Searching OpenSearch (hybrid: kNN + BM25 — generic query)...")
        similar = search_hybrid(text_vector, description)
    else:
        print("Step 3: Searching OpenSearch (kNN only — detailed query)...")
        similar = search_similar(text_vector)
    print(f"Fetched {len(similar)} candidates")

    print("Step 4: Reranking results via Cohere Rerank...")
    similar = rerank_results(search_text, similar)

    print("Step 5: Generating image URLs...")
    for p in similar:
        p["image_url"] = get_image_url(p.get("image_s3_key", ""))

    print("Step 6: Computing market readiness score...")
    market_readiness = compute_market_readiness(similar)

    return {
        "similar_products": similar,
        "visual_attributes": [],
        "market_readiness": market_readiness
    }

def run_compare_insights(products1, products2, desc1="", desc2=""):
    print("Generating collection comparison insights...")
    insights = generate_compare_insights(products1[:INSIGHTS_K], products2[:INSIGHTS_K], desc1, desc2)
    return {"insights": insights}
