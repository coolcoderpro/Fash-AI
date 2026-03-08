"""
Embedding Pipeline
==================
Reads enriched_metadata.json + card PNGs from S3
→ Calls Amazon Titan Multimodal Embeddings G1
→ Saves vectors + metadata to S3 as embeddings.json

Usage:
    python3 embedding_pipeline.py --test          # test on first 5 products
    python3 embedding_pipeline.py --all           # full 5000 products
    python3 embedding_pipeline.py --start 100 --end 200  # specific range
"""

import boto3
import json
import base64
import argparse
import time
from io import BytesIO
from PIL import Image
from tqdm import tqdm

REGION          = "ap-south-1"
S3_BUCKET       = "fashion-assistant-images"
S3_CARDS_PREFIX = "cards/"
S3_METADATA_KEY = "metadata/enriched_metadata_5000.json"
S3_OUTPUT_KEY   = "embeddings/embeddings.json"
TITAN_MODEL_ID  = "amazon.titan-embed-image-v1"
IMAGE_SIZE      = (512, 512)
SLEEP_BETWEEN   = 0.5

s3      = boto3.client("s3",              region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

def load_metadata():
    print("Loading metadata from S3...")
    resp = s3.get_object(Bucket=S3_BUCKET, Key=S3_METADATA_KEY)
    data = json.load(resp["Body"])
    print(f"Loaded {len(data)} products from S3")
    return data

def image_from_s3(product_id):
    key = f"{S3_CARDS_PREFIX}fash_img_{product_id}.png"
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        img  = Image.open(BytesIO(resp["Body"].read())).convert("RGB")
        img  = img.resize(IMAGE_SIZE, Image.LANCZOS)
        buf  = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"\n  WARNING: Could not load image {key}: {e}")
        return None

def metadata_to_text(m):
    tags = ", ".join(m.get("style_tags", []))
    desc = m.get("description", "")
    pattern = m.get("pattern", "None")
    length = m.get("length", "None")
    sleeve = m.get("sleeve_type", "None")
    silhouette = m.get("silhouette", "None")

    parts = [f"{desc}." if desc else f"{m['name']}."]
    parts.append(f"Color: {m.get('color', '')}.")
    if pattern != "None":
        parts.append(f"Pattern: {pattern}.")
    parts.append(f"Material: {m.get('material', '')}.")
    if length != "None":
        parts.append(f"Length: {length}.")
    if sleeve != "None":
        parts.append(f"Sleeve: {sleeve}.")
    if silhouette != "None":
        parts.append(f"Silhouette: {silhouette}.")
    parts.append(f"Niche: {m.get('clothing_niche', '')}.")
    parts.append(f"Occasion: {m.get('occasion', '')}.")
    parts.append(f"Season: {m.get('season', '')}.")
    parts.append(f"Style tags: {tags}.")

    return " ".join(parts)

def get_titan_embedding(image_b64, text):
    body = {
        "inputText":  text,
        "inputImage": image_b64,
        "embeddingConfig": {"outputEmbeddingLength": 1024}
    }
    response = bedrock.invoke_model(
        modelId     = TITAN_MODEL_ID,
        body        = json.dumps(body),
        contentType = "application/json",
        accept      = "application/json"
    )
    result = json.loads(response["body"].read())
    return result["embedding"]

def save_to_s3(embeddings):
    body = json.dumps(embeddings, indent=2)
    s3.put_object(
        Bucket      = S3_BUCKET,
        Key         = S3_OUTPUT_KEY,
        Body        = body,
        ContentType = "application/json"
    )
    print(f"\nSaved {len(embeddings)} embeddings to s3://{S3_BUCKET}/{S3_OUTPUT_KEY}")

def run(metadata, start, end):
    results = []
    ok = fail = 0
    print(f"\nEmbedding Pipeline — products {start} to {end} ({end-start} total)\n")
    t_start = time.time()

    for i, meta in enumerate(tqdm(metadata[start:end], desc="Embedding")):
        pid = meta.get("product_id", f"{i+1:04d}")
        try:
            image_b64 = image_from_s3(pid)
            if not image_b64:
                fail += 1
                continue
            text   = metadata_to_text(meta)
            vector = get_titan_embedding(image_b64, text)
            results.append({
                "product_id":          pid,
                "embedding":           vector,
                "name":                meta.get("name"),
                "brand":               meta.get("brand"),
                "price":               meta.get("price"),
                "color":               meta.get("color"),
                "pattern":             meta.get("pattern"),
                "material":            meta.get("material"),
                "length":              meta.get("length"),
                "sleeve_type":         meta.get("sleeve_type"),
                "silhouette":          meta.get("silhouette"),
                "description":         meta.get("description"),
                "clothing_niche":      meta.get("clothing_niche"),
                "target_audience":     meta.get("target_audience"),
                "occasion":            meta.get("occasion"),
                "season":              meta.get("season"),
                "price_tier":          meta.get("price_tier"),
                "style_tags":          meta.get("style_tags"),
                "units_sold":          meta.get("units_sold"),
                "revenue_generated":   meta.get("revenue_generated"),
                "return_rate_pct":     meta.get("return_rate_pct"),
                "conversion_rate_pct": meta.get("conversion_rate_pct"),
                "days_since_launch":   meta.get("days_since_launch"),
                "sales_trend":         meta.get("sales_trend"),
                "inventory_remaining": meta.get("inventory_remaining"),
                "avg_rating":          meta.get("avg_rating"),
                "image_s3_key":        f"{S3_CARDS_PREFIX}fash_img_{pid}.png",
            })
            ok += 1
            time.sleep(SLEEP_BETWEEN)
        except Exception as e:
            print(f"\n  FAILED on product {pid}: {e}")
            fail += 1
            continue

    if results:
        save_to_s3(results)

    elapsed = time.time() - t_start
    print(f"\nDone! OK={ok} Failed={fail} Time={int(elapsed//60)}m {int(elapsed%60)}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  action="store_true", help="Run on first 5 products")
    parser.add_argument("--all",   action="store_true", help="Run all 1000 products")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end",   type=int, default=10)
    args = parser.parse_args()

    metadata = load_metadata()

    if args.test:
        run(metadata, 0, 5)
    elif args.all:
        run(metadata, 0, len(metadata))
    else:
        run(metadata, args.start, args.end)

if __name__ == "__main__":
    main()
