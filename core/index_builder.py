import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import faiss

logger = logging.getLogger(__name__)

INDEX_DIR = Path(__file__).resolve().parent.parent / "indexes"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def _build_faiss_index(dim: int, n_vectors: int):
    return faiss.IndexFlatIP(dim)

class IndexBuilder:
    def __init__(self, index_dir: str = None):
        self.index_dir = Path(index_dir) if index_dir else INDEX_DIR
        self.text_index: Optional[faiss.Index] = None
        self.image_index: Optional[faiss.Index] = None
        self.metadata = {"text": [], "image": []}

    def build_text_index(self, embeddings, documents, doc_ids=None, extra_meta=None):
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        n, dim = embeddings.shape
        
        self.text_index = _build_faiss_index(dim, n)
        self.text_index.add(embeddings)
        
        # Reset metadata to prevent appending to old data
        self.metadata["text"] = [] 
        for i, doc in enumerate(documents):
            meta = {"text": doc, "doc_id": str(doc_ids[i] if doc_ids else i)}
            if extra_meta and i < len(extra_meta):
                meta.update(extra_meta[i])
            self.metadata["text"].append(meta)
        logger.info(f"Text index ready: {self.text_index.ntotal} items")

    def build_image_index(self, embeddings, image_paths, captions=None, extra_meta=None):
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        n, dim = embeddings.shape
        
        self.image_index = _build_faiss_index(dim, n)
        self.image_index.add(embeddings)
        
        # Reset metadata to prevent appending to old data
        self.metadata["image"] = []
        for i, path in enumerate(image_paths):
            meta = {
                "image_path": str(Path(path).resolve()),
                "caption": captions[i] if captions else ""
            }
            if extra_meta and i < len(extra_meta):
                meta.update(extra_meta[i])
            self.metadata["image"].append(meta)
        logger.info(f"Image index ready: {self.image_index.ntotal} items")

    def save(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.text_index:
            faiss.write_index(self.text_index, str(self.index_dir / "text.faiss"))
        if self.image_index:
            faiss.write_index(self.image_index, str(self.index_dir / "image.faiss"))
        with open(self.index_dir / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info("Indexes saved ✔")

    def load(self):
        text_path = self.index_dir / "text.faiss"
        image_path = self.index_dir / "image.faiss"
        meta_path = self.index_dir / "metadata.pkl"

        if text_path.exists(): self.text_index = faiss.read_index(str(text_path))
        if image_path.exists(): self.image_index = faiss.read_index(str(image_path))
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        logger.info(f"Loaded: text={len(self.metadata['text'])}, image={len(self.metadata['image'])}")

    @property
    def text_meta(self): return self.metadata["text"]

    @property
    def image_meta(self): return self.metadata["image"]

    def index_stats(self):
        return {
            "text_vectors": self.text_index.ntotal if self.text_index else 0,
            "image_vectors": self.image_index.ntotal if self.image_index else 0,
        }