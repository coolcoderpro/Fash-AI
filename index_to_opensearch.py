"""
OpenSearch Indexing Script
==========================
Reads embeddings.json from S3 and bulk-indexes into OpenSearch Serverless.

Usage:
    python index_to_opensearch.py --test          # index first 5 products
    python index_to_opensearch.py --all           # index all products
    python index_to_opensearch.py --create-index  # create/recreate the index first, then index all
"""

import boto3
import json
import argparse
import time
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, helpers

REGION              = "ap-south-1"
S3_BUCKET           = "fashion-assistant-images"
S3_EMBEDDINGS_KEY   = "embeddings/embeddings.json"
INDEX_NAME          = "fashion-products"
OPENSEARCH_ENDPOINT = "vmryh04ckerilvnvd5se.ap-south-1.aoss.amazonaws.com"
BATCH_SIZE          = 100

s3 = boto3.client("s3", region_name=REGION)

def get_os_client():
    credentials = boto3.Session(region_name=REGION).get_credentials()
    auth = AWSV4SignerAuth(credentials, REGION, "aoss")
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60
    )

def create_index(client):
    """Create the fashion-products index with knn_vector mapping."""
    index_body = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 1024,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 512,
                            "m": 16
                        }
                    }
                },
                "name":                {"type": "text"},
                "brand":               {"type": "keyword"},
                "price":               {"type": "float"},
                "color":               {"type": "keyword"},
                "pattern":             {"type": "keyword"},
                "material":            {"type": "text"},
                "length":              {"type": "keyword"},
                "sleeve_type":         {"type": "keyword"},
                "silhouette":          {"type": "keyword"},
                "description":         {"type": "text"},
                "clothing_niche":      {"type": "text"},
                "target_audience":     {"type": "text"},
                "occasion":            {"type": "text"},
                "season":              {"type": "text"},
                "price_tier":          {"type": "keyword"},
                "style_tags":          {"type": "text"},
                "units_sold":          {"type": "integer"},
                "revenue":             {"type": "float"},
                "return_rate_pct":     {"type": "float"},
                "conv_rate":           {"type": "float"},
                "days_listed":         {"type": "integer"},
                "sales_trend":         {"type": "keyword"},
                "inventory":           {"type": "integer"},
                "avg_rating":          {"type": "float"},
                "image_s3_key":        {"type": "keyword"},
            }
        }
    }

    if client.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists. Deleting...")
        client.indices.delete(index=INDEX_NAME)
        print("Deleted. Waiting 5s for propagation...")
        time.sleep(5)

    print(f"Creating index '{INDEX_NAME}'...")
    client.indices.create(index=INDEX_NAME, body=index_body)
    print(f"Index '{INDEX_NAME}' created successfully.")
    time.sleep(2)

def load_embeddings():
    print(f"Loading embeddings from s3://{S3_BUCKET}/{S3_EMBEDDINGS_KEY}...")
    resp = s3.get_object(Bucket=S3_BUCKET, Key=S3_EMBEDDINGS_KEY)
    data = json.loads(resp["Body"].read())
    print(f"Loaded {len(data)} embeddings.")
    return data

def build_doc(item):
    """Build an OpenSearch document from an embedding item."""
    return {
        "embedding":           item["embedding"],
        "name":                item.get("name"),
        "brand":               item.get("brand"),
        "price":               item.get("price"),
        "color":               item.get("color"),
        "pattern":             item.get("pattern"),
        "material":            item.get("material"),
        "length":              item.get("length"),
        "sleeve_type":         item.get("sleeve_type"),
        "silhouette":          item.get("silhouette"),
        "description":         item.get("description"),
        "clothing_niche":      item.get("clothing_niche"),
        "target_audience":     item.get("target_audience"),
        "occasion":            item.get("occasion"),
        "season":              item.get("season"),
        "price_tier":          item.get("price_tier"),
        "style_tags":          item.get("style_tags"),
        "units_sold":          item.get("units_sold"),
        "revenue":             item.get("revenue_generated"),
        "return_rate_pct":     item.get("return_rate_pct"),
        "conv_rate":           item.get("conversion_rate_pct"),
        "days_listed":         item.get("days_since_launch"),
        "sales_trend":         item.get("sales_trend"),
        "inventory":           item.get("inventory_remaining"),
        "avg_rating":          item.get("avg_rating"),
        "image_s3_key":        item.get("image_s3_key"),
    }

def index_products(client, embeddings, limit=None):
    """Index products in batches."""
    if limit:
        embeddings = embeddings[:limit]

    total = len(embeddings)
    ok = fail = 0
    t_start = time.time()

    print(f"\nIndexing {total} products in batches of {BATCH_SIZE}...\n")

    for i in range(0, total, BATCH_SIZE):
        batch = embeddings[i:i + BATCH_SIZE]
        actions = []
        for item in batch:
            actions.append({
                "_index": INDEX_NAME,
                "_source": build_doc(item)
            })

        try:
            success, errors = helpers.bulk(client, actions, raise_on_error=False)
            ok += success
            if errors:
                fail += len(errors)
                for e in errors[:3]:
                    print(f"  Error: {e}")
        except Exception as e:
            print(f"  Batch {i//BATCH_SIZE + 1} failed: {e}")
            fail += len(batch)

        elapsed = time.time() - t_start
        print(f"  Batch {i//BATCH_SIZE + 1}/{(total + BATCH_SIZE - 1)//BATCH_SIZE} — "
              f"indexed {min(i + BATCH_SIZE, total)}/{total} "
              f"({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t_start
    print(f"\nDone! OK={ok} Failed={fail} Time={int(elapsed)}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Index first 5 products")
    parser.add_argument("--all", action="store_true", help="Index all products")
    parser.add_argument("--create-index", action="store_true", help="Create/recreate the index before indexing")
    args = parser.parse_args()

    client = get_os_client()

    if args.create_index:
        create_index(client)

    embeddings = load_embeddings()

    if args.test:
        index_products(client, embeddings, limit=5)
    elif args.all or args.create_index:
        index_products(client, embeddings)
    else:
        print("Usage: --test (5 products), --all (all products), --create-index (recreate index + index all)")

if __name__ == "__main__":
    main()
