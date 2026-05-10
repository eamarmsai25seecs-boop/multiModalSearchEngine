import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COCO_ANN_DIR = ROOT / "data/raw/annotations"
FLICKR_FILE = ROOT / "data/raw/annotations/flickr_captions.txt"  # adjust if needed

OUT_DIR = ROOT / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUERIES_OUT = OUT_DIR / "queries.json"
QRELS_OUT = OUT_DIR / "qrels.json"


def load_coco(coco_path):
    data = json.load(open(coco_path, "r"))

    queries = {}
    qrels = {}

    for i, ann in enumerate(data["annotations"]):
        qid = f"coco_q{i}"
        queries[qid] = ann["caption"].lower()
        qrels[qid] = [str(ann["image_id"])]

    return queries, qrels


def load_flickr(flickr_path):
    queries = {}
    qrels = {}

    with open(flickr_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split("\t")

        if len(parts) < 2:
            continue

        img_caption = parts[1]

        img_id = parts[0].split("#")[0]

        qid = f"flickr_q{i}"

        queries[qid] = img_caption.lower()
        qrels[qid] = [img_id]

    return queries, qrels


def build():

    all_queries = {}
    all_qrels = {}

    print("Loading COCO...")

    coco_files = [
        COCO_ANN_DIR / "captions_train2017.json",
        COCO_ANN_DIR / "captions_val2017.json"
    ]

    for f in coco_files:
        if f.exists():
            q, r = load_coco(f)
            all_queries.update(q)
            all_qrels.update(r)

    print("Loading Flickr...")

    if FLICKR_FILE.exists():
        q, r = load_flickr(FLICKR_FILE)
        all_queries.update(q)
        all_qrels.update(r)

    print("Saving files...")

    json.dump(all_queries, open(QUERIES_OUT, "w"), indent=2)
    json.dump(all_qrels, open(QRELS_OUT, "w"), indent=2)

    print("DONE ✔")
    print("queries:", len(all_queries))
    print("qrels:", len(all_qrels))


if __name__ == "__main__":
    build()