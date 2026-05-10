import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────
# SAFE JSON LOADER
# ─────────────────────────────
def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────
# COCO CAPTIONS (JSON)
# ─────────────────────────────
def load_coco():
    path = ROOT / "data/raw/annotations/captions_val2017.json"

    data = load_json(path)

    return [
        {
            "image_id": ann["image_id"],
            "text": ann["caption"],
            "doc_id": f"coco_{ann['id']}",
            "source": "coco"
        }
        for ann in data.get("annotations", [])
    ]


# ─────────────────────────────
# FLICKR CAPTIONS (TXT - FIXED)
# ─────────────────────────────
def load_flickr():
    path = ROOT / "data/raw/annotations/flickr_captions.txt"

    if not path.exists():
        print(f"[WARN] Flickr file missing: {path}")
        return []

    records = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Expected format: image_id \t caption
            parts = line.split("\t")

            if len(parts) == 2:
                image_id, caption = parts
            else:
                image_id = f"flickr_{i}"
                caption = line

            records.append({
                "image_id": image_id,
                "text": caption,
                "doc_id": f"flickr_{i}",
                "source": "flickr"
            })

    return records


# ─────────────────────────────
# TEXT DATA = COCO + FLICKR
# ─────────────────────────────
def load_all_text():
    return load_coco() + load_flickr()


# ─────────────────────────────
# IMAGE META = COCO + FLICKR
# ─────────────────────────────
def load_all_image_meta():
    return load_coco() + load_flickr()