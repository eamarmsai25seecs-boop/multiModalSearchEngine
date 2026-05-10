import sys
import logging
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.text_encoder import TextEncoder
from core.image_encoder import ImageEncoder
from core.index_builder import IndexBuilder
from core.dataset_loader import load_all_text, load_all_image_meta

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_IMAGES = ROOT / "data" / "images"
INDEX_DIR = ROOT / "indexes"


# ✅ BATCH TEXT ENCODING (FIXED)
def encode_texts_in_batches(encoder, texts, batch_size):
    all_embeds = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vec = encoder.encode(batch)
        all_embeds.append(np.array(vec, dtype=np.float32))

        print(f"Encoded {i + len(batch)}/{len(texts)}")

    return np.vstack(all_embeds)


# ─────────────────────────────
def build_text_index(builder, batch_size=32, limit=None):

    logger.info("TEXT INDEX (COCO + FLICKR)")

    records = load_all_text()

    if limit:
        records = records[:limit]

    texts = [r["text"] for r in records]

    encoder = TextEncoder()

    # ✅ FIXED (BATCHED)
    embeddings = encode_texts_in_batches(encoder, texts, batch_size)

    builder.build_text_index(
        embeddings=embeddings,
        documents=texts,
        doc_ids=[r["doc_id"] for r in records],
        extra_meta=[
            {"text": r["text"], "source": r["source"], "image_id": r["image_id"]}
            for r in records
        ],
    )

    logger.info(f"TEXT DONE: {len(texts)}")


# ─────────────────────────────
def build_image_index(builder, batch_size=32, limit=None):

    logger.info("IMAGE INDEX (COCO + FLICKR)")

    records = load_all_image_meta()

    if limit:
        records = records[:limit]

    COCO_DIR = DATA_IMAGES / "val2017"
    FLICKR_DIR = DATA_IMAGES / "flickr"

    valid = []

    for r in records:
        img_id = str(r["image_id"])

        coco_path = COCO_DIR / (img_id.zfill(12) + ".jpg")
        flickr_path = FLICKR_DIR / img_id

        if coco_path.exists():
            valid.append({**r, "image_path": str(coco_path)})
        elif flickr_path.exists():
            valid.append({**r, "image_path": str(flickr_path)})

    logger.info(f"VALID IMAGES: {len(valid)}")

    encoder = ImageEncoder()
    embeddings = []

    from tqdm import tqdm

    for i in tqdm(range(0, len(valid), batch_size)):
        batch = valid[i:i + batch_size]
        paths = [r["image_path"] for r in batch]

        vec = encoder.encode_images(paths)
        embeddings.append(np.array(vec, dtype=np.float32))

    image_embeddings = np.vstack(embeddings)

    builder.build_image_index(
        embeddings=image_embeddings,
        image_paths=[r["image_path"] for r in valid],
        captions=[r["text"] for r in valid],
        extra_meta=[
            {"text": r["text"], "source": r["source"], "image_id": r["image_id"]}
            for r in valid
        ],
    )

    logger.info(f"IMAGE DONE: {len(valid)}")


# ─────────────────────────────
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--all", action="store_true")
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    builder = IndexBuilder(index_dir=str(INDEX_DIR))

    if args.all or args.text:
        build_text_index(builder, args.batch_size, args.limit)

    if args.all or args.images:
        build_image_index(builder, args.batch_size, args.limit)

    builder.save()
    logger.info("DONE ✔")


if __name__ == "__main__":
    main()