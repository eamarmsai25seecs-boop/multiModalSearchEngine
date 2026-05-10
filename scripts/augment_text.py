""""
scripts/augment_text.py
------------------------
Synthetic text augmentation for MS MARCO passages.
Generates 3 augmented versions per original passage using:
  1. Synonym replacement (WordNet)
  2. Random word deletion + swap
  3. Back-translation (English → German → English)

Usage:
    python scripts/augment_text.py --input data/processed/msmarco_passages.json
                                    --output data/processed/msmarco_augmented.json
                                    --methods synonym deletion
"""

import json
import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict

import nltk
import nlpaug.augmenter.word as naw
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Download required NLTK data
for pkg in ["wordnet", "averaged_perceptron_tagger", "stopwords"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"


def augment_synonym(texts: List[str], aug_max: int = 3) -> List[str]:
    """Replace up to aug_max words per sentence with WordNet synonyms."""
    logger.info("Running synonym replacement augmentation...")
    aug = naw.SynonymAug(aug_src="wordnet", aug_max=aug_max)
    results = []
    for text in tqdm(texts, desc="Synonym aug"):
        try:
            augmented = aug.augment(text)
            results.append(augmented[0] if isinstance(augmented, list) else augmented)
        except Exception:
            results.append(text)  # fallback to original on error
    return results


def augment_deletion(texts: List[str], aug_p: float = 0.15) -> List[str]:
    """Randomly delete ~15% of words."""
    logger.info("Running random deletion augmentation...")
    aug = naw.RandomWordAug(action="delete", aug_p=aug_p)
    results = []
    for text in tqdm(texts, desc="Deletion aug"):
        try:
            augmented = aug.augment(text)
            results.append(augmented[0] if isinstance(augmented, list) else augmented)
        except Exception:
            results.append(text)
    return results


def augment_swap(texts: List[str], aug_p: float = 0.1) -> List[str]:
    """Randomly swap ~10% of adjacent words."""
    logger.info("Running word swap augmentation...")
    aug = naw.RandomWordAug(action="swap", aug_p=aug_p)
    results = []
    for text in tqdm(texts, desc="Swap aug"):
        try:
            augmented = aug.augment(text)
            results.append(augmented[0] if isinstance(augmented, list) else augmented)
        except Exception:
            results.append(text)
    return results


def augment_back_translation(texts: List[str], batch_size: int = 32) -> List[str]:
    """
    Back-translation: English → German → English using googletrans.
    NOTE: This requires internet access and is slow (~1 sec/sentence).
          Use only on a small subset or skip if offline.
    """
    logger.info("Running back-translation augmentation (slow, requires internet)...")
    try:
        from googletrans import Translator
        translator = Translator()
    except ImportError:
        logger.warning("googletrans not installed. Skipping back-translation.")
        return texts

    results = []
    for text in tqdm(texts, desc="Back-translation"):
        try:
            de = translator.translate(text, src="en", dest="de").text
            en = translator.translate(de, src="de", dest="en").text
            results.append(en)
        except Exception:
            results.append(text)  # fallback
    return results


def run_augmentation(
    input_path: str,
    output_path: str,
    methods: List[str],
    max_docs: int = None,
):
    """
    Load passages, apply each augmentation method, save combined dataset.

    Each original passage gets one augmented copy per method.
    Final dataset = original + (len(methods) × original_count) documents.
    """
    logger.info(f"Loading passages from {input_path}...")
    with open(input_path) as f:
        original_docs = json.load(f)

    if max_docs:
        original_docs = original_docs[:max_docs]

    texts = [d["text"] for d in original_docs]
    logger.info(f"Loaded {len(texts)} original passages.")

    all_docs = list(original_docs)  # start with originals

    method_funcs = {
        "synonym":           augment_synonym,
        "deletion":          augment_deletion,
        "swap":              augment_swap,
        "back_translation":  augment_back_translation,
    }

    for method in methods:
        if method not in method_funcs:
            logger.warning(f"Unknown method '{method}'. Skipping.")
            continue

        aug_texts = method_funcs[method](texts)

        aug_docs = []
        for i, (orig, aug_text) in enumerate(zip(original_docs, aug_texts)):
            aug_docs.append({
                "doc_id":  f"{orig['doc_id']}_aug_{method}",
                "text":    aug_text,
                "source":  "augmented",
                "method":  method,
                "orig_id": orig["doc_id"],
            })
        all_docs.extend(aug_docs)
        logger.info(f"Added {len(aug_docs)} docs via '{method}'. Total so far: {len(all_docs)}")

    logger.info(f"Saving {len(all_docs)} documents to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(all_docs, f)
    logger.info("Done.")
    logger.info(f"Augmentation ratio: {len(all_docs)/len(original_docs):.1f}×")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment MS MARCO text passages")
    parser.add_argument("--input",   default=str(DATA_PROCESSED / "msmarco_passages.json"))
    parser.add_argument("--output",  default=str(DATA_PROCESSED / "msmarco_augmented.json"))
    parser.add_argument("--methods", nargs="+",
                        default=["synonym", "deletion"],
                        choices=["synonym", "deletion", "swap", "back_translation"],
                        help="Augmentation methods to apply")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Limit number of source docs (for testing)")
    args = parser.parse_args()

    run_augmentation(
        input_path=args.input,
        output_path=args.output,
        methods=args.methods,
        max_docs=args.max_docs,
    )