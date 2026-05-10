import logging
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np

from core.text_encoder import TextEncoder
from core.image_encoder import ImageEncoder
from core.index_builder import IndexBuilder
from core.query_expansion import QueryExpander
from core.models import SearchResult
from core.reranker import CrossEncoderReranker, CLIPReranker, HybridReranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModalSearcher:
    def __init__(
        self,
        index_dir: str = "indexes",
        text_model: str = "ViT-B/32",
        clip_model: str = "ViT-B/32",
        expand_queries: bool = True,
        expansion_strategy: str = "wordnet",
        device: str = None,
    ):
        self.index_dir = Path(index_dir)
        self.text_encoder = TextEncoder(model_name=text_model, device=device)
        self.image_encoder = ImageEncoder(model_name=clip_model, device=device)
        self.store = IndexBuilder(index_dir=str(self.index_dir))
        self.expand_queries = expand_queries
        self.expander = QueryExpander(strategy=expansion_strategy) if expand_queries else None

        self.reranker_mode = "none"
        self.ce_reranker = CrossEncoderReranker()
        self.clip_reranker = CLIPReranker()
        self.hybrid_reranker = HybridReranker()

    def load(self):
        self.store.load()
        logger.info(f"[Searcher] Loaded: {self.store.index_stats()}")

    def set_reranker(self, mode: str):
        self.reranker_mode = mode

    def _expand(self, query: str) -> str:
        if self.expand_queries and self.expander:
            return self.expander.expand(query)
        return query

    def _faiss_search(self, index, query_vec, top_k):
        vec = np.array(query_vec, dtype=np.float32)
        if vec.ndim == 1: vec = vec.reshape(1, -1)
        if vec.shape[1] != index.d:
            raise ValueError(f"❌ Dimension mismatch! Query={vec.shape[1]} Index={index.d}")
        vec = vec / np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-8, None)
        scores, ids = index.search(vec, top_k)
        return scores[0], ids[0]

    def _get_meta_list(self, meta_obj) -> list:
        if isinstance(meta_obj, dict): return list(meta_obj.values())
        return meta_obj

    def _build_results(self, scores, ids, meta_data, modality):
        results = []
        meta_list = self._get_meta_list(meta_data)
        seen_paths = set()
        for score, idx in zip(scores, ids):
            if idx == -1 or idx >= len(meta_list): continue
            meta = meta_list[idx]
            content_id = meta.get('image_path') or meta.get('path') or meta.get('text') or str(meta)
            if content_id not in seen_paths:
                results.append(SearchResult(
                    rank=len(results) + 1,
                    score=float(score),
                    modality=modality,
                    metadata=meta,
                ))
                seen_paths.add(content_id)
        return results

    def _apply_reranker(self, query, results, mode):
        if not results or mode == "none":
            return results
        
        # Cross-modal handling: Ensure images have 'text' for the reranker
        if mode in ["cross_encoder", "hybrid"]:
            for r in results:
                if "text" not in r.metadata and "caption" in r.metadata:
                    r.metadata["text"] = r.metadata["caption"]

        if mode == "cross_encoder": return self.ce_reranker.rerank(query, results)
        if mode == "clip": return self.clip_reranker.rerank(query, results)
        if mode == "hybrid": return self.hybrid_reranker.rerank(query, results)
        return results

    def search_text_to_text(self, query: str, top_k: int = 10):
        query_expanded = self._expand(query)
        vec = self.text_encoder.encode(query_expanded)
        scores, ids = self._faiss_search(self.store.text_index, vec, top_k * 2)
        results = self._build_results(scores, ids, self.store.text_meta, "text")
        return self._apply_reranker(query, results[:top_k], self.reranker_mode)

    def search_text_to_image(self, query: str, top_k: int = 10):
        vec = self.text_encoder.encode(f"a photo of {query}")
        scores, ids = self._faiss_search(self.store.image_index, vec, top_k * 2)
        results = self._build_results(scores, ids, self.store.image_meta, "image")
        return self._apply_reranker(query, results[:top_k], self.reranker_mode)

    def search_image_to_text(self, image_source, top_k: int = 10):
        vec = self.image_encoder.encode_images(image_source)
        scores, ids = self._faiss_search(self.store.text_index, vec, top_k * 2)
        results = self._build_results(scores, ids, self.store.text_meta, "text")
        return self._apply_reranker("", results[:top_k], self.reranker_mode)

    def search_image_to_image(self, image_source, top_k: int = 10):
        vec = self.image_encoder.encode_images(image_source)
        scores, ids = self._faiss_search(self.store.image_index, vec, top_k * 2)
        results = self._build_results(scores, ids, self.store.image_meta, "image")
        return self._apply_reranker("", results[:top_k], self.reranker_mode)

    def search_hybrid(self, query: str, top_k: int = 10):
        vec = self.text_encoder.encode(f"a photo of {query}")
        t_scores, t_ids = self._faiss_search(self.store.text_index, vec, top_k * 2)
        i_scores, i_ids = self._faiss_search(self.store.image_index, vec, top_k * 2)

        combined = []
        seen_paths = set()
        
        for m_data, scores, ids, mod in [(self.store.text_meta, t_scores, t_ids, "text"), 
                                         (self.store.image_meta, i_scores, i_ids, "image")]:
            meta_list = self._get_meta_list(m_data)
            for s, i in zip(scores, ids):
                if i != -1 and i < len(meta_list):
                    meta = meta_list[i]
                    content_id = meta.get('image_path') or meta.get('path') or meta.get('text') or str(meta)
                    if content_id not in seen_paths:
                        combined.append((mod, float(s), meta))
                        seen_paths.add(content_id)

        combined.sort(key=lambda x: x[1], reverse=True)
        final = [SearchResult(rank=i+1, score=s, modality=m, metadata=meta) 
                 for i, (m, s, meta) in enumerate(combined[:top_k * 2])] # Fetch more for reranker

        return self._apply_reranker(query, final[:top_k], self.reranker_mode)

    def search(self, query, mode="text_to_text", top_k=10, **kwargs):
        search_funcs = {
            "text_to_text": self.search_text_to_text,
            "text_to_image": self.search_text_to_image,
            "image_to_text": self.search_image_to_text,
            "image_to_image": self.search_image_to_image,
            "hybrid": self.search_hybrid,
        }
        return search_funcs[mode](query, top_k=top_k, **kwargs)