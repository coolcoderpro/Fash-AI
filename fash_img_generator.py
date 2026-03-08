"""
Fashion Product Image Generator — Final
========================================
Converts CSV fashion dataset rows into rich product card images.

Changes in this version:
  - Output files named fash_img_0001.png, fash_img_0002.png ...
  - Full product image shown (no cropping, letterboxed if needed)
  - Text section starts lower, giving the image more breathing room
  - Market/performance fields populated with realistic random values
  - All fields written to enriched_metadata.json for RAG pipeline

Setup:
    pip install pillow requests pandas selenium webdriver-manager

Usage:
    python fash_img_generator.py              # first 10 rows
    python fash_img_generator.py --rows 100
    python fash_img_generator.py --all
    python fash_img_generator.py --all --skip-existing   # resume after crash
    python fash_img_generator.py --no-selenium           # skip Chrome fallback
"""

import ast
import json
import re
import argparse
import random
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

OUTPUT_DIR   = Path("product_outputs")
IMAGE_DIR    = OUTPUT_DIR / "images"
METADATA_OUT = OUTPUT_DIR / "enriched_metadata.json"

CARD_W   = 900
PHOTO_H  = 520   # taller photo area so full image fits better
INFO_H   = 560   # text section height (trimmed — no product details/materials sections)
CARD_H   = PHOTO_H + INFO_H
MARGIN   = 44

# Colours
C_BG        = (255, 255, 255)
C_PHOTO_BG  = (240, 240, 240)
C_TITLE     = (22,  22,  22)
C_HEADING   = (44,  62,  80)
C_TEXT      = (75,  75,  75)
C_PRICE     = (192, 57,  43)
C_TAG_BG    = (234, 237, 240)
C_TAG_TEXT  = (44,  62,  80)
C_DIVIDER   = (215, 215, 215)
C_BADGE_BG  = (44,  62,  80)
C_BADGE_FG  = (255, 255, 255)
C_PERF_BG   = (247, 249, 250)
C_PERF_VAL  = (30,  30,  30)
C_PERF_LBL  = (130, 130, 130)
C_TREND_UP  = (39,  174, 96)
C_TREND_DN  = (192, 57,  43)
C_TREND_FL  = (120, 120, 120)

REQUEST_TIMEOUT = 3  # fail fast — ASOS blocks direct requests, Selenium is the real fetcher
IMAGE_WIDTH     = 700   # request this width from CDN

_session = requests.Session()
_session.headers.update({
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer":         "https://www.asos.com/",
    "Accept":          "image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
})
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ---------------------------------------------------------------------------
# FONT LOADER
# ---------------------------------------------------------------------------

def _try_font(name: str, size: int):
    for path in [
        name,
        f"C:/Windows/Fonts/{name}",
        "/usr/share/fonts/truetype/msttcorefonts/" + name,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


class Fonts:
    def __init__(self):
        self.title   = _try_font("arialbd.ttf", 21)
        self.heading = _try_font("arialbd.ttf", 14)
        self.body    = _try_font("arial.ttf",   13)
        self.small   = _try_font("arial.ttf",   11)
        self.price   = _try_font("arialbd.ttf", 24)
        self.tag     = _try_font("arial.ttf",   11)
        self.badge   = _try_font("arialbd.ttf", 11)
        self.perf_v  = _try_font("arialbd.ttf", 16)
        self.perf_l  = _try_font("arial.ttf",   10)

# ---------------------------------------------------------------------------
# ENRICHMENT — rule-based field derivation
# ---------------------------------------------------------------------------

NICHE_KW = {
    "Y2K / 2000s Revival":    ["y2k", "bodycon", "corset", "lacquer", "vinyl", "halter"],
    "Cottagecore / Romantic": ["ditsy", "floral", "ruffle", "prairie", "lace", "shirred"],
    "Streetwear":             ["oversized", "hoodie", "jogger", "cargo", "graphic", "crop"],
    "Minimalist / Clean":     ["minimal", "tailored", "straight", "neutral", "beige"],
    "Party / Nightlife":      ["sequin", "glitter", "cut-out", "mini", "satin", "slip"],
    "Athleisure":             ["sports", "gym", "legging", "biker", "active"],
    "Boho":                   ["bohemian", "fringe", "crochet", "kimono", "ethnic"],
    "Classic / Workwear":     ["blazer", "trouser", "shirt", "formal", "office"],
}
OCCASION_KW = {
    "Night Out / Club":  ["bodycon", "mini", "lacquer", "sequin", "satin", "halter", "vinyl"],
    "Casual / Everyday": ["oversized", "hoodie", "jogger", "t-shirt", "jeans", "relaxed"],
    "Formal / Work":     ["blazer", "trouser", "shirt", "formal", "tailored"],
    "Festival":          ["crochet", "fringe", "festival", "boho", "maxi"],
    "Date Night":        ["slip", "lace", "cut-out", "satin", "romantic"],
    "Beach / Holiday":   ["bikini", "swimsuit", "beach", "resort", "linen"],
}
AUDIENCE_KW = {
    "Women 18–24 (trend-led)":      ["y2k", "corset", "mini", "halter", "crop", "bodycon"],
    "Women 25–35 (versatile)":      ["maxi", "midi", "tailored", "blazer", "shirt"],
    "Women 18–35 (inclusive)":      ["curve", "tall", "petite", "maternity", "plus"],
    "Teens / Gen-Z":                ["graphic", "streetwear", "oversized", "cargo"],
    "Active / Fitness Enthusiasts": ["gym", "sports", "biker", "performance"],
}
SEASON_KW = {
    "Summer":        ["linen", "bikini", "sundress", "shorts", "lightweight", "beach"],
    "Winter":        ["knit", "wool", "coat", "puffer", "chunky", "thermal"],
    "Spring/Summer": ["floral", "ditsy", "pastel", "midi", "ruffle"],
    "Autumn/Winter": ["satin", "velvet", "leather", "corset", "blazer"],
    "All-season":    ["polyester", "elastane", "jersey", "stretch"],
}
# --- Brand prefixes to strip from product names (longest first for greedy match) ---
BRAND_PREFIXES = sorted([
    "ASOS DESIGN Maternity", "ASOS DESIGN Curve", "ASOS DESIGN Petite", "ASOS DESIGN Tall",
    "ASOS DESIGN", "ASOS EDITION", "ASOS LUXE", "ASOS 4505", "ASOS",
    "COLLUSION Unisex", "COLLUSION",
    "River Island Plus", "River Island",
    "Miss Selfridge", "New Look", "Vero Moda", "Noisy May", "Daisy Street",
    "& Other Stories", "The North Face", "adidas Originals", "adidas",
    "Urban Revivo", "In The Style", "Reclaimed Vintage", "Wednesday's Girl",
    "The Frolic", "Forever New", "Closet London", "Pretty Lavish",
    "Nobody's Child", "Extro & Vert", "Hope & Ivy",
    "Nike Running", "Nike", "Topshop", "Monki", "Stradivarius",
    "Whistles", "Weekday", "Bershka", "Pull&Bear", "Mango",
    "French Connection", "Ted Baker", "Warehouse", "Oasis",
    "Puma", "Reebok", "Under Armour", "Converse",
], key=len, reverse=True)  # longest first so "ASOS DESIGN Curve" matches before "ASOS DESIGN"

def _strip_brand(name: str) -> tuple:
    """Return (brand, clean_name) with brand prefix removed."""
    for prefix in BRAND_PREFIXES:
        if name.startswith(prefix):
            clean = name[len(prefix):].strip()
            return prefix, clean if clean else name
    return "", name

TAG_SEEDS = [
    "corset","halter","crop","mini","midi","maxi","slip","bodycon",
    "oversized","tailored","ruffle","lace","satin","velvet","knit",
    "floral","stripe","print","graphic","cut-out","sequin","sheer",
    "zip-back","sweetheart neck","high neck","off-shoulder","wrap",
    "elasticated","sleeveless","long sleeve",
]

# --- Pattern detection ---
PATTERN_KW = {
    "Floral print":     ["floral", "ditsy", "flower", "botanical", "rose", "daisy", "poppy", "blossom", "garden"],
    "Stripe":           ["stripe", "striped", "pinstripe", "breton", "nautical stripe"],
    "Check / Plaid":    ["check", "plaid", "gingham", "tartan", "houndstooth", "windowpane", "buffalo check", "argyle"],
    "Polka dot":        ["polka", "dot", "spotted", "spot print"],
    "Animal print":     ["leopard", "snake", "zebra", "animal", "croc", "snakeskin", "python", "cheetah", "tiger", "cow print"],
    "Abstract print":   ["abstract", "geometric", "marble", "swirl", "wave print", "retro print", "mosaic"],
    "Graphic print":    ["graphic", "logo", "slogan", "text print", "letter", "motif", "printed"],
    "Lace":             ["lace", "lace overlay", "eyelash lace", "crochet"],
    "Embroidered":      ["embroidered", "embroidery", "broderie", "applique"],
    "Sequin / Beaded":  ["sequin", "beaded", "embellished", "glitter", "sparkle", "rhinestone", "crystal"],
    "Tie-dye":          ["tie-dye", "tie dye", "dip-dye", "ombre", "gradient"],
    "Camo":             ["camo", "camouflage"],
    "Paisley":          ["paisley"],
    "Tropical":         ["tropical", "palm", "hawaiian", "paradise", "jungle"],
    "Textured":         ["textured", "ribbed", "quilted", "waffle", "puckered", "crinkle", "pleated texture"],
    "Color block":      ["color block", "colour block", "colorblock", "colourblock", "contrast panel"],
}

# --- Material detection (priority order — first match wins) ---
MATERIAL_KW = {
    "Leather":     ["leather", "faux leather", "pu leather", "vegan leather"],
    "Denim":       ["denim", "chambray"],
    "Silk":        ["silk", "silky"],
    "Satin":       ["satin", "sateen"],
    "Velvet":      ["velvet", "velour"],
    "Linen":       ["linen", "flax"],
    "Wool":        ["wool", "woolen", "merino", "cashmere"],
    "Chiffon":     ["chiffon", "georgette"],
    "Mesh":        ["mesh", "net", "tulle"],
    "Lace":        ["lace", "eyelash lace", "guipure"],
    "Knit":        ["knit", "knitted", "jersey", "rib knit", "ribbed knit"],
    "Cotton":      ["cotton", "organic cotton"],
    "Sequin":      ["sequin", "sequined"],
    "Vinyl":       ["vinyl", "lacquer", "pvc", "patent"],
    "Nylon":       ["nylon", "nylon blend"],
    "Polyester":   ["polyester", "poly"],
    "Viscose":     ["viscose", "rayon"],
    "Elastane":    ["elastane", "spandex", "lycra", "stretch"],
    "Crochet":     ["crochet", "macrame"],
}

# --- Length detection ---
LENGTH_KW = {
    "Mini":        ["mini", "micro", "short dress", "short skirt"],
    "Midi":        ["midi", "mid-length", "mid length", "below knee", "below-knee", "calf-length"],
    "Maxi":        ["maxi", "floor length", "floor-length", "full-length", "full length", "ankle length", "ankle-length"],
    "Knee-length": ["knee-length", "knee length", "to the knee"],
    "Cropped":     ["cropped", "crop", "bralette", "bralet"],
    "Longline":    ["longline", "long line", "thigh-length", "thigh length", "tunic"],
}

# --- Sleeve type detection ---
SLEEVE_KW = {
    "Sleeveless":    ["sleeveless", "strapless", "bandeau", "tube", "strappy", "spaghetti",
                      "one shoulder", "one-shoulder", "halter", "cami", "vest", "tank",
                      "sweetheart neck", "bardot", "off-shoulder", "off shoulder", "backless"],
    "Short sleeve":  ["short sleeve", "short-sleeve", "cap sleeve", "cap-sleeve", "t-shirt",
                      "tee", "flutter sleeve", "flutter-sleeve", "frill sleeve"],
    "Long sleeve":   ["long sleeve", "long-sleeve", "bishop sleeve", "blouson sleeve"],
    "3/4 sleeve":    ["3/4 sleeve", "three-quarter", "bracelet sleeve"],
    "Puff sleeve":   ["puff sleeve", "puff-sleeve", "balloon sleeve", "balloon-sleeve",
                      "volume sleeve", "mutton sleeve", "leg-of-mutton"],
    "Bell sleeve":   ["bell sleeve", "bell-sleeve", "flared sleeve", "trumpet sleeve",
                      "flare sleeve", "angel sleeve"],
}

# --- Silhouette / Fit detection ---
SILHOUETTE_KW = {
    "Bodycon":    ["bodycon", "body-con", "figure-hugging", "body-hugging", "figure hugging",
                   "skin-tight", "skin tight", "form-fitting", "form fitting"],
    "A-line":     ["a-line", "a line", "flared skirt", "swing"],
    "Fitted":     ["fitted", "slim fit", "slim-fit", "tailored", "slim", "sculpted",
                   "contour", "structured"],
    "Relaxed":    ["relaxed", "loose", "loose-fit", "easy fit", "comfort fit", "casual fit"],
    "Oversized":  ["oversized", "oversize", "boxy", "baggy", "dropped shoulder", "drop shoulder"],
    "Wrap":       ["wrap dress", "wrap front", "wrap style", "wrap top", "wrap skirt", "surplice"],
    "Shift":      ["shift dress", "shift style", "column"],
    "Skater":     ["skater", "fit and flare", "fit-and-flare", "circle skirt"],
    "Straight":   ["straight", "straight-cut", "straight cut", "regular fit", "regular-fit"],
    "Pleated":    ["pleated", "pleat", "accordion", "knife pleat"],
    "Peplum":     ["peplum"],
    "Tiered":     ["tiered", "tier", "layered skirt"],
    "Smock":      ["smock", "smocked", "babydoll", "baby doll", "trapeze"],
    "Draped":     ["draped", "drape", "cowl", "gathered"],
}

def _detect_materials(text: str, max_items: int = 2) -> str:
    """Return up to max_items matched materials, or 'Unknown' if none found."""
    t = text.lower()
    found = []
    for label, kws in MATERIAL_KW.items():
        if any(k in t for k in kws):
            found.append(label)
            if len(found) >= max_items:
                break
    return ", ".join(found) if found else "Unknown"

def _detect_field(text: str, kw_map: dict) -> str:
    """Return the first matching label from kw_map, or 'None' if nothing matches."""
    t = text.lower()
    for label, kws in kw_map.items():
        if any(k in t for k in kws):
            return label
    return "None"

def _generate_description(meta: dict) -> str:
    """Generate a natural one-sentence product description from metadata fields."""
    parts = []
    color = meta.get("color", "").strip()
    if color:
        parts.append(color)
    pattern = meta.get("pattern", "None")
    if pattern != "None":
        parts.append(pattern.lower())
    sleeve = meta.get("sleeve_type", "None")
    if sleeve != "None":
        parts.append(sleeve.lower())
    length = meta.get("length", "None")
    if length != "None":
        parts.append(length.lower())
    # Category not in meta — infer from name keywords
    name_lower = meta.get("name", "").lower()
    for cat in ["dress", "top", "skirt", "trouser", "blazer", "coat", "jumpsuit", "shorts"]:
        if cat in name_lower:
            parts.append(cat)
            break
    silhouette = meta.get("silhouette", "None")
    if silhouette != "None":
        parts.append(f"with {silhouette.lower()} fit")
    return " ".join(parts).strip().capitalize() if parts else meta.get("name", "")

def _score(text: str, kw_map: dict) -> str:
    t = text.lower()
    best, best_n = list(kw_map.keys())[0], 0
    for label, kws in kw_map.items():
        n = sum(1 for k in kws if k in t)
        if n > best_n:
            best_n, best = n, label
    return best

def _price_tier(p: float) -> str:
    if p < 20:  return "Budget (<$20)"
    if p < 50:  return "Mid-range ($20–$50)"
    if p < 100: return "Premium ($50–$100)"
    return "Luxury ($100+)"

def _style_tags(text: str) -> List[str]:
    found = [t for t in TAG_SEEDS if t in text.lower()]
    return found[:6] or ["fashion", "womenswear"]

# ---------------------------------------------------------------------------
# MARKET FIELD PATTERNS
# Real fashion retail benchmarks used as base ranges.
# Each field is derived from others so the numbers tell a coherent story.
# Seeded by product name so the SAME product always gets the SAME numbers.
# ---------------------------------------------------------------------------

# Return rate benchmarks by category (fashion industry averages)
CATEGORY_RETURN_RATES = {
    "dress":      (18.0, 32.0),   # dresses return heavily — fit issues
    "top":        (12.0, 22.0),
    "jeans":      (20.0, 35.0),   # sizing is the #1 fashion return reason
    "coat":       (10.0, 20.0),
    "swimwear":   (5.0,  12.0),   # low — hygiene policy limits returns
    "activewear": (8.0,  16.0),
    "default":    (14.0, 26.0),
}

# Conversion rate benchmarks by price tier
# Cheaper items convert better — less purchase hesitation
PRICE_CONV_RATES = {
    "budget":    (5.0,  9.0),
    "mid":       (2.5,  6.0),
    "premium":   (1.2,  3.5),
    "luxury":    (0.5,  2.0),
}

def _market_fields(name: str, price: float, category: str) -> dict:
    """
    Generate seeded, internally consistent market fields with balanced distribution.

    Performance tier (~33% each) drives all fields:
      - Low performer:  few sales, high returns, low rating, falling trend, high inventory
      - Mid performer:  moderate sales, average returns, decent rating, stable trend
      - High performer: strong sales, low returns, high rating, rising trend, low inventory

    All fields stay correlated — a low performer won't have 5-star rating or rising trend.
    Seeded by product name so the SAME product always gets the SAME numbers.
    """
    rng = random.Random(hash(name))   # seeded — deterministic per product

    # 1. Performance tier — evenly distributed (~33% each)
    perf_roll = rng.random()
    if perf_roll < 0.33:
        perf = "low"
    elif perf_roll < 0.66:
        perf = "mid"
    else:
        perf = "high"

    # 2. How long has it been listed?
    days = rng.randint(14, 600)

    # 3. Price tier → affects daily sales ceiling
    if price < 20:
        tier = "budget"
    elif price < 50:
        tier = "mid"
    elif price < 100:
        tier = "premium"
    else:
        tier = "luxury"

    # Max daily sales by price tier
    max_daily_by_tier = {"budget": 12.0, "mid": 6.0, "premium": 2.5, "luxury": 0.8}
    max_daily = max_daily_by_tier[tier]

    # 4. Daily rate — driven by performance tier
    if perf == "low":
        daily_rate = rng.uniform(0.05, max_daily * 0.25)
    elif perf == "mid":
        daily_rate = rng.uniform(max_daily * 0.25, max_daily * 0.6)
    else:
        daily_rate = rng.uniform(max_daily * 0.6, max_daily)

    # 5. Units sold = days × daily_rate
    units = max(2, int(days * daily_rate))

    # 6. Return rate — category + performance driven
    cat_lower = category.lower()
    rate_range = next(
        (v for k, v in CATEGORY_RETURN_RATES.items() if k in cat_lower),
        CATEGORY_RETURN_RATES["default"]
    )
    if perf == "low":
        # Low performers: high end of return range + penalty
        return_rate = round(rng.uniform(rate_range[1] * 0.8, rate_range[1] * 1.2), 1)
    elif perf == "mid":
        # Mid performers: middle of range
        mid = (rate_range[0] + rate_range[1]) / 2
        return_rate = round(rng.uniform(rate_range[0], mid + 3), 1)
    else:
        # High performers: low end of return range
        return_rate = round(rng.uniform(rate_range[0] * 0.5, rate_range[0] + 3), 1)
    return_rate = max(2.0, min(return_rate, 48.0))

    # 7. Avg rating — correlated with performance
    if perf == "low":
        avg_rating = round(rng.uniform(2.2, 3.4), 1)
    elif perf == "mid":
        avg_rating = round(rng.uniform(3.3, 4.2), 1)
    else:
        avg_rating = round(rng.uniform(4.0, 5.0), 1)

    # 8. Inventory — low performers have excess stock, high performers running low
    if perf == "low":
        inventory = rng.randint(200, 600)
    elif perf == "mid":
        inventory = rng.randint(50, 300)
    else:
        inventory = rng.randint(0, 80)

    # 9. Sales trend — directly from performance tier (with slight randomness)
    trend_roll = rng.random()
    if perf == "low":
        trend = "falling" if trend_roll < 0.75 else "stable"
    elif perf == "mid":
        if trend_roll < 0.2:
            trend = "falling"
        elif trend_roll < 0.7:
            trend = "stable"
        else:
            trend = "rising"
    else:
        trend = "rising" if trend_roll < 0.75 else "stable"

    # 10. Conversion rate — price tier + performance driven
    conv_range = PRICE_CONV_RATES[tier]
    if perf == "low":
        conv_rate = round(rng.uniform(conv_range[0] * 0.3, conv_range[0]), 1)
    elif perf == "mid":
        conv_rate = round(rng.uniform(conv_range[0], (conv_range[0] + conv_range[1]) / 2), 1)
    else:
        conv_rate = round(rng.uniform((conv_range[0] + conv_range[1]) / 2, conv_range[1]), 1)
    conv_rate = max(0.2, conv_rate)

    # 11. Revenue — units sold minus returned items
    effective_units = units * (1 - return_rate / 100)
    revenue = round(effective_units * price, 2)

    return {
        "units_sold":          units,
        "revenue_generated":   revenue,
        "return_rate_pct":     return_rate,
        "conversion_rate_pct": conv_rate,
        "days_since_launch":   days,
        "sales_trend":         trend,
        "inventory_remaining": inventory,
        "avg_rating":          avg_rating,
    }


def _clean_details(text: str) -> str:
    """
    Clean up product details text:
    - Remove unfilled template placeholders like {if applicable}
    - Remove duplicate sentences
    - Cap at 400 chars so it never overflows the card
    """
    # Remove {if applicable} and similar unfilled template tokens
    text = re.sub(r'\{[^}]*\}', '', text)
    # Add spaces between run-together sentences (camelCase boundaries)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    # Remove extra whitespace
    text = re.sub(r' {2,}', ' ', text).strip()
    # Cap length
    if len(text) > 400:
        text = text[:400].rsplit(' ', 1)[0] + '...'
    return text

def enrich(row: Any) -> dict:
    raw_name = str(row.get("name", ""))
    combined = " ".join([
        raw_name,
        str(row.get("product_details", "")),
        str(row.get("about_me", "")),
        str(row.get("category_type", "")),
        str(row.get("color", "")),
    ])
    brand, clean_name = _strip_brand(raw_name)
    price = float(row.get("price", 0) or 0)
    category = str(row.get("category_type", ""))
    meta = {
        "name":            clean_name,
        "brand":           brand,
        "price":           price,
        "color":           str(row.get("color", "")),
        "pattern":         _detect_field(combined, PATTERN_KW),
        "material":        _detect_materials(combined),
        "length":          _detect_field(combined, LENGTH_KW),
        "sleeve_type":     _detect_field(combined, SLEEVE_KW),
        "silhouette":      _detect_field(combined, SILHOUETTE_KW),
        "clothing_niche":  _score(combined, NICHE_KW),
        "target_audience": _score(combined, AUDIENCE_KW),
        "occasion":        _score(combined, OCCASION_KW),
        "season":          _score(combined, SEASON_KW),
        "price_tier":      _price_tier(price),
        "style_tags":      _style_tags(combined),
    }
    # Generate natural description from structured fields
    meta["description"] = _generate_description(meta)
    meta.update(_market_fields(
        name     = meta["name"],
        price    = price,
        category = category,
    ))
    return meta

# ---------------------------------------------------------------------------
# IMAGE DOWNLOADER — 3-layer strategy
# ---------------------------------------------------------------------------

def parse_image_urls(raw) -> List[str]:
    if not raw or str(raw).strip() in ("", "nan"):
        return []
    try:
        urls = ast.literal_eval(str(raw))
        return [u for u in urls if isinstance(u, str) and u.startswith("http")]
    except Exception:
        return re.findall(r'https?://[^\s\'"]+', str(raw))

def _fetch_via_requests(urls: List[str]) -> Optional[Image.Image]:
    for url in urls:
        try:
            fast_url = re.sub(r'wid=\d+', f'wid={IMAGE_WIDTH}', url)
            resp = _session.get(fast_url, timeout=REQUEST_TIMEOUT, stream=True)
            print(f"    [HTTP] status={resp.status_code} | url={fast_url[:70]}...")
            if resp.status_code == 200:
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                print(f"    [HTTP] image size: {img.size}")
                return img
            else:
                print(f"    [HTTP] non-200 response: {resp.status_code} — trying original URL")
            # Fallback to original URL if resized fails
            if fast_url != url:
                resp = _session.get(url, timeout=REQUEST_TIMEOUT, stream=True)
                print(f"    [HTTP] fallback status={resp.status_code}")
                if resp.status_code == 200:
                    return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            print(f"    [HTTP] EXCEPTION: {e.__class__.__name__}: {str(e)[:80]}")
            continue
    return None

_selenium_driver = None

def _get_selenium_driver():
    global _selenium_driver
    if _selenium_driver is not None:
        return _selenium_driver
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager

        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=800,1000")
        opts.add_argument(f"user-agent={REQUEST_HEADERS['User-Agent']}")

        # Block all non-image resources — ads, JS, CSS, fonts
        # This makes the page load MUCH faster since we only need the image
        opts.add_argument("--blink-settings=imagesEnabled=true")
        prefs = {
            "profile.managed_default_content_settings.javascript":    2,  # block JS
            "profile.managed_default_content_settings.stylesheets":   2,  # block CSS
            "profile.managed_default_content_settings.fonts":         2,  # block fonts
            "profile.managed_default_content_settings.notifications":  2,  # block popups
        }
        opts.add_experimental_option("prefs", prefs)

        _selenium_driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=opts)
        print("  [Selenium] Headless Chrome started (JS/CSS/fonts blocked for speed)")
        return _selenium_driver
    except Exception as e:
        print(f"  [Selenium] Could not start: {e}")
        return None

def _fetch_via_selenium(urls: List[str]) -> Optional[Image.Image]:
    """
    Open the image URL directly in Chrome (not the product page).
    Chrome handles auth/cookies that ASOS CDN requires.
    JS/CSS blocked so only the raw image loads — much faster.
    """
    driver = _get_selenium_driver()
    if not driver:
        return None

    for url in urls:
        try:
            # Request smaller image directly
            fast_url = re.sub(r'wid=\d+', f'wid={IMAGE_WIDTH}', url)

            # Navigate directly to the image URL
            driver.get(fast_url)

            # Wait briefly for image to load
            import time as _t
            _t.sleep(1.5)

            # Grab the <img> element and screenshot just it
            from selenium.webdriver.common.by import By
            imgs = driver.find_elements(By.TAG_NAME, "img")
            if imgs:
                png = imgs[0].screenshot_as_png
            else:
                # Direct image URL renders as the image itself in Chrome
                png = driver.get_screenshot_as_png()

            img = Image.open(BytesIO(png)).convert("RGB")
            if img.width > 50 and img.height > 50:
                print(f"    [Selenium] fetched image: {img.size}")
                return img
        except Exception as e:
            print(f"    [Selenium] failed: {e.__class__.__name__}: {str(e)[:60]}")
            continue
    return None

def download_image(urls: List[str], use_selenium: bool = True) -> Optional[Image.Image]:
    """Go straight to Selenium — skips requests entirely since ASOS blocks direct calls."""
    if urls:
        return _fetch_via_selenium(urls)
    return None


def shutdown_selenium():
    global _selenium_driver
    if _selenium_driver:
        try: _selenium_driver.quit()
        except Exception: pass
        _selenium_driver = None

# ---------------------------------------------------------------------------
# CARD RENDERER
# ---------------------------------------------------------------------------

def _wrap(text: str, font, max_w: int, draw: ImageDraw.ImageDraw) -> List[str]:
    words, lines, cur = text.split(), [], []
    for word in words:
        test = " ".join(cur + [word])
        if draw.textbbox((0, 0), test, font=font)[2] <= max_w:
            cur.append(word)
        else:
            if cur: lines.append(" ".join(cur))
            cur = [word]
    if cur: lines.append(" ".join(cur))
    return lines or [""]

def _tag_pills(draw, tags, x, y, font, max_x, gap=6):
    pad_x, pad_y, r = 10, 5, 8
    cx = x
    for tag in tags:
        tw   = draw.textbbox((0, 0), tag, font=font)[2]
        pw   = tw + pad_x * 2
        ph   = font.size + pad_y * 2
        if cx + pw > max_x:
            cx  = x
            y  += ph + gap
        draw.rounded_rectangle([cx, y, cx + pw, y + ph], radius=r, fill=C_TAG_BG)
        draw.text((cx + pad_x, y + pad_y), tag, font=font, fill=C_TAG_TEXT)
        cx += pw + gap
    return y + (font.size + pad_y * 2)

def make_photo_strip(photo: Optional[Image.Image]) -> Image.Image:
    """
    Fit the entire product image inside the strip using letterboxing.
    No cropping — the full image is always visible.
    """
    strip = Image.new("RGB", (CARD_W, PHOTO_H), C_PHOTO_BG)
    if photo is None:
        d = ImageDraw.Draw(strip)
        d.text((CARD_W // 2, PHOTO_H // 2), "[ No image available ]",
               font=_try_font("arial.ttf", 16), fill=(180, 180, 180), anchor="mm")
        return strip

    # Scale so the whole image fits inside CARD_W × PHOTO_H (no crop)
    img_w, img_h = photo.size
    scale   = min(CARD_W / img_w, PHOTO_H / img_h)
    new_w   = int(img_w * scale)
    new_h   = int(img_h * scale)
    resized = photo.resize((new_w, new_h), Image.LANCZOS)

    # Centre it
    x_off = (CARD_W - new_w) // 2
    y_off = (PHOTO_H - new_h) // 2
    strip.paste(resized, (x_off, y_off))
    return strip

def render_card(meta: dict, strip: Image.Image, fonts: Fonts) -> Image.Image:
    card = Image.new("RGB", (CARD_W, CARD_H), C_BG)
    card.paste(strip, (0, 0))
    draw = ImageDraw.Draw(card)
    cw   = CARD_W - MARGIN * 2

    # Start text 20px below the photo
    y = PHOTO_H + 20

    # ── Brand ────────────────────────────────────────────────────────────
    brand = meta.get("brand", "")
    if brand:
        draw.text((MARGIN, y), brand, font=fonts.small, fill=C_PERF_LBL)
        y += 16

    # ── Product name ──────────────────────────────────────────────────────
    for line in _wrap(meta["name"], fonts.title, cw, draw):
        draw.text((MARGIN, y), line, font=fonts.title, fill=C_TITLE)
        y += 27
    y += 4

    # ── Price + badge ─────────────────────────────────────────────────────
    draw.text((MARGIN, y), f"${meta['price']:.2f}", font=fonts.price, fill=C_PRICE)
    badge = meta["price_tier"].split("(")[0].strip()
    bw    = draw.textbbox((0, 0), badge, font=fonts.badge)[2] + 16
    bx    = CARD_W - MARGIN - bw
    draw.rounded_rectangle([bx, y + 3, bx + bw, y + 25], radius=5, fill=C_BADGE_BG)
    draw.text((bx + 8, y + 5), badge, font=fonts.badge, fill=C_BADGE_FG)
    y += 36

    # ── Color / Pattern ──────────────────────────────────────────────────
    pattern = meta.get("pattern", "None")
    if pattern != "None":
        color_line = f"Color: {meta['color']}   •   Pattern: {pattern}"
    else:
        color_line = f"Color: {meta['color']}"
    draw.text((MARGIN, y), color_line, font=fonts.body, fill=C_TEXT)
    y += 20

    # ── Material / Length / Sleeve / Silhouette ──────────────────────────
    detail_parts = []
    material = meta.get("material", "Unknown")
    if material != "Unknown":
        detail_parts.append(f"Material: {material}")
    length = meta.get("length", "None")
    if length != "None":
        detail_parts.append(f"Length: {length}")
    sleeve = meta.get("sleeve_type", "None")
    if sleeve != "None":
        detail_parts.append(f"Sleeve: {sleeve}")
    silhouette = meta.get("silhouette", "None")
    if silhouette != "None":
        detail_parts.append(f"Fit: {silhouette}")
    if detail_parts:
        draw.text((MARGIN, y), "   •   ".join(detail_parts),
                  font=fonts.body, fill=C_TEXT)
        y += 20

    # ── Description ──────────────────────────────────────────────────────
    desc = meta.get("description", "")
    if desc:
        draw.text((MARGIN, y), desc, font=fonts.small, fill=C_TEXT)
        y += 18

    # ── Divider ───────────────────────────────────────────────────────────
    y += 6
    draw.line([(MARGIN, y), (CARD_W - MARGIN, y)], fill=C_DIVIDER, width=1)
    y += 12

    # ── 2×2 intelligence grid ─────────────────────────────────────────────
    intel = [
        ("Clothing Niche",  meta["clothing_niche"]),
        ("Target Audience", meta["target_audience"]),
        ("Occasion",        meta["occasion"]),
        ("Season",          meta["season"]),
    ]
    col2_x = MARGIN + (cw // 2) + 10
    for i, (lbl, val) in enumerate(intel):
        cx = MARGIN if i % 2 == 0 else col2_x
        cy = y + (i // 2) * 44
        draw.text((cx, cy),      lbl, font=fonts.heading, fill=C_HEADING)
        draw.text((cx, cy + 18), val, font=fonts.small,   fill=C_TEXT)
    y += 2 * 44 + 6

    # ── Style tags ────────────────────────────────────────────────────────
    draw.text((MARGIN, y), "Style Tags", font=fonts.heading, fill=C_HEADING)
    y += 18
    y = _tag_pills(draw, meta["style_tags"], MARGIN, y,
                   fonts.tag, CARD_W - MARGIN) + 10

    # ── Market performance panel ──────────────────────────────────────────
    panel_y = CARD_H - 145
    draw.rectangle([(0, panel_y), (CARD_W, CARD_H)], fill=C_PERF_BG)
    draw.line([(0, panel_y), (CARD_W, panel_y)], fill=C_DIVIDER, width=1)

    trend_colour = {
        "rising": C_TREND_UP, "falling": C_TREND_DN, "stable": C_TREND_FL
    }.get(str(meta.get("sales_trend", "stable")).lower(), C_TREND_FL)
    trend_icon = {
        "rising": "↑ Rising", "falling": "↓ Falling", "stable": "→ Stable"
    }.get(str(meta.get("sales_trend", "stable")).lower(), "→ Stable")

    perf_items = [
        ("Units Sold",    str(meta.get("units_sold", "—")),          C_PERF_VAL),
        ("Revenue",       f"${meta.get('revenue_generated', 0):,.0f}", C_PERF_VAL),
        ("Return Rate",   f"{meta.get('return_rate_pct', '—')}%",    C_PERF_VAL),
        ("Conv. Rate",    f"{meta.get('conversion_rate_pct', '—')}%", C_PERF_VAL),
        ("Days Listed",   str(meta.get("days_since_launch", "—")),   C_PERF_VAL),
        ("Inventory",     str(meta.get("inventory_remaining", "—")), C_PERF_VAL),
        ("Avg Rating",    f"★ {meta.get('avg_rating', '—')}",         C_PERF_VAL),
        ("Sales Trend",   trend_icon,                                 trend_colour),
    ]

    cols    = 4
    col_w   = (CARD_W - MARGIN * 2) // cols
    p_start = panel_y + 18

    for i, (lbl, val, col) in enumerate(perf_items):
        row_i = i // cols
        col_i = i % cols
        px    = MARGIN + col_i * col_w
        py    = p_start + row_i * 52
        draw.text((px, py),      val, font=fonts.perf_v, fill=col)
        draw.text((px, py + 22), lbl, font=fonts.perf_l, fill=C_PERF_LBL)

    return card

# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def process_dataset(csv_path: str, max_rows: Optional[int],
                    use_selenium: bool = True,
                    skip_existing: bool = False) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)
    if max_rows:
        df = df.head(max_rows)

    print(f"\n{'='*65}")
    print(f"  Fashion Image Generator — Final")
    print(f"  Dataset       : {csv_path}  ({len(df)} rows)")
    print(f"  Output        : {IMAGE_DIR.absolute()}")
    print(f"  Selenium      : {'enabled' if use_selenium else 'disabled'}")
    print(f"  Skip existing : {'yes' if skip_existing else 'no'}")
    print(f"{'='*65}\n")

    fonts      = Fonts()
    all_meta   = []
    ok = no_img = fail = 0
    total      = len(df)
    times      = []          # per-product elapsed times for ETA

    import time as _time
    run_start = _time.time()

    def _fmt(secs):
        """Format seconds as mm:ss or hh:mm:ss."""
        secs = int(secs)
        h, rem = divmod(secs, 3600)
        m, s   = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"

    try:
        for idx, (_, row) in enumerate(df.iterrows()):
            pid        = f"{idx + 1:04d}"
            out_path   = IMAGE_DIR / f"fash_img_{pid}.png"
            name_short = str(row.get("name", ""))[:50]
            t_start    = _time.time()

            print(f"[{idx+1}/{total}] fash_img_{pid} — {name_short}...")

            # Skip if already done
            if skip_existing and out_path.exists():
                print(f"  ↷ already exists, skipping")
                all_meta.append({**enrich(row),
                                 "product_id": pid,
                                 "image_path": str(out_path)})
                ok += 1
                continue

            try:
                # Enrich
                meta = enrich(row)

                # Download image
                urls  = parse_image_urls(row.get("images"))
                photo = download_image(urls, use_selenium=use_selenium)
                if photo:
                    print(f"  ✓ image fetched")
                else:
                    print(f"  ✗ no image — placeholder used")
                    no_img += 1

                # Build card
                strip = make_photo_strip(photo)
                card  = render_card(meta, strip, fonts)
                card.save(str(out_path), "PNG", optimize=True)

                meta["product_id"] = pid
                meta["image_path"] = str(out_path)
                all_meta.append(meta)
                ok += 1

            except Exception as e:
                print(f"  ✗ ERROR on row {idx+1}: {e} — skipping, will retry with --skip-existing")
                fail = locals().get("fail", 0) + 1
                continue

            # ── Timing ────────────────────────────────────────────────
            elapsed   = _time.time() - t_start
            times.append(elapsed)
            avg_time  = sum(times) / len(times)
            remaining = total - (idx + 1)
            eta_secs  = avg_time * remaining
            total_run = _time.time() - run_start

            print(f"  → saved: {out_path.name}  "
                  f"[{elapsed:.1f}s this item | "
                  f"avg {avg_time:.1f}s | "
                  f"elapsed {_fmt(total_run)} | "
                  f"ETA {_fmt(eta_secs)}]")

    finally:
        shutdown_selenium()

    total_time = _time.time() - run_start

    with open(METADATA_OUT, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*65}")
    print(f"  Done!")
    print(f"  Cards created  : {ok}  |  Placeholders: {no_img}")
    print(f"  Total time     : {_fmt(total_time)}")
    print(f"  Avg per item   : {(total_time/max(ok,1)):.1f}s")
    print(f"  Metadata       : {METADATA_OUT.absolute()}")
    print(f"{'='*65}\n")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Fashion CSV → product card images")
    p.add_argument("--csv",            default="final_fashion_dataset.csv")
    p.add_argument("--rows",           type=int, default=10)
    p.add_argument("--all",            action="store_true")
    p.add_argument("--no-selenium",    action="store_true")
    p.add_argument("--skip-existing",  action="store_true")
    args = p.parse_args()

    process_dataset(
        csv_path      = args.csv,
        max_rows      = None if args.all else args.rows,
        use_selenium  = not args.no_selenium,
        skip_existing = args.skip_existing,
    )

if __name__ == "__main__":
    main()
