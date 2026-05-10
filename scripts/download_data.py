"""
scripts/download_data.py
-------------------------
Downloads and prepares datasets for Multi-Modal Search Engine.

TEXT  → MS MARCO (HuggingFace version - stable)
IMAGE → MS-COCO 2017 (val split)

Usage:
    python scripts/download_data.py --all --sample 5000
"""

import json
import logging
import argparse
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_IMAGES = ROOT / "data" / "images"
DATA_PROCESSED = ROOT / "data" / "processed"

for d in [DATA_RAW, DATA_IMAGES, DATA_PROCESSED]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Download helper
# ─────────────────────────────────────────────

def _download_file(url: str, dest: Path, desc: str = ""):
    if dest.exists():
        logger.info(f"Already exists: {dest.name}")
        return dest

    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc or dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return dest


# ─────────────────────────────────────────────
# MS MARCO (FIXED - HuggingFace)
# ─────────────────────────────────────────────

def download_msmarco(sample: int = 5000):
    logger.info("=== Downloading MS MARCO (HuggingFace) ===")

    from datasets import load_dataset

    ds = load_dataset("ms_marco", "v1.1", split="train")

    docs = []
    queries = {}

    for i, item in enumerate(ds):
        if i >= sample:
            break

        queries[str(i)] = item["query"]

        for p in item["passages"]["passage_text"]:
            docs.append({
                "doc_id": str(len(docs)),
                "text": p
            })

        if len(docs) >= sample:
            break

    out_docs = DATA_PROCESSED / "msmarco_passages.json"
    out_queries = DATA_PROCESSED / "msmarco_queries.json"

    with open(out_docs, "w", encoding="utf-8") as f:
        json.dump(docs[:sample], f, indent=2)

    with open(out_queries, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)

    logger.info(f"Saved {len(docs[:sample])} passages → {out_docs}")
    logger.info(f"Saved {len(queries)} queries → {out_queries}")


# ─────────────────────────────────────────────
# MS-COCO (FIXED)
# ─────────────────────────────────────────────

def download_coco_val(sample: int = 5000):
    logger.info("=== Downloading MS-COCO 2017 ===")

    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    img_url = "http://images.cocodataset.org/zips/val2017.zip"

    ann_zip = DATA_RAW / "coco_annotations.zip"
    img_zip = DATA_RAW / "coco_val2017.zip"

    _download_file(ann_url, ann_zip, "COCO annotations")
    _download_file(img_url, img_zip, "COCO images")

    logger.info("Extracting COCO data...")

    with zipfile.ZipFile(ann_zip) as z:
        z.extractall(DATA_RAW)

    with zipfile.ZipFile(img_zip) as z:
        z.extractall(DATA_IMAGES)

    ann_file = DATA_RAW / "annotations" / "captions_val2017.json"

    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    records = []
    seen = set()

    for ann in coco["annotations"]:
        img_id = ann["image_id"]

        if img_id in seen:
            continue

        seen.add(img_id)

        records.append({
            "image_id": str(img_id),
            "image_path": str(DATA_IMAGES / "val2017" / id_to_file[img_id]),
            "caption": ann["caption"]
        })

        if len(records) >= sample:
            break

    out = DATA_PROCESSED / "coco_captions.json"

    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    logger.info(f"Saved {len(records)} COCO samples → {out}")


# ─────────────────────────────────────────────
# CLI (FIXED)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--text", action="store_true")
    parser.add_argument("--coco", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--sample", type=int, default=5000)

    args = parser.parse_args()

    if args.all:
        download_msmarco(args.sample)
        download_coco_val(min(args.sample, 5000))

    else:
        if args.text:
            download_msmarco(args.sample)

        if args.coco:
            download_coco_val(min(args.sample, 5000))

    if not any([args.text, args.coco, args.all]):
        print("Use --text / --coco / --all")