import logging
from typing import List, Optional, Union
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from PIL import Image
from core.models import SearchResult
from core.image_encoder import ImageEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = None,
        max_length: int = 512,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[CrossEncoderReranker] Loading '{model_name}' on {self.device}...")
        self.model = CrossEncoder(model_name, max_length=max_length, device=self.device)
        logger.info("[CrossEncoderReranker] Ready.")

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
        text_field: str = "text",
    ) -> List[SearchResult]:
        if not results:
            return results

        # MODIFIED: Fallback logic to find ANY text content (text or caption)
        pairs = []
        for r in results:
            content = r.metadata.get(text_field) or r.metadata.get("caption") or r.metadata.get("text") or ""
            pairs.append((query, str(content)))

        scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)

        for r, s in zip(results, scores):
            r.score = float(s)

        reranked = sorted(results, key=lambda r: r.score, reverse=True)
        for i, r in enumerate(reranked):
            r.rank = i + 1

        return reranked[:top_k] if top_k else reranked

class CLIPReranker:
    def __init__(self, clip_model: str = "ViT-B/32", device: str = None):
        self.encoder = ImageEncoder(model_name=clip_model, device=device)
        logger.info("[CLIPReranker] Ready.")

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
        image_field: str = "image_path",
    ) -> List[SearchResult]:
        if not results:
            return results

        query_vec = self.encoder.encode_text(query)

        for r in results:
            img_path = r.metadata.get(image_field)
            if not img_path:
                r.score = 0.0
                continue
            try:
                img_vec = self.encoder.encode_images(img_path)
                r.score = float(np.dot(query_vec, img_vec))
            except Exception as e:
                logger.warning(f"[CLIPReranker] Error loading '{img_path}': {e}")
                r.score = 0.0

        reranked = sorted(results, key=lambda r: r.score, reverse=True)
        for i, r in enumerate(reranked):
            r.rank = i + 1
        return reranked[:top_k] if top_k else reranked

class HybridReranker:
    def __init__(
        self,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        alpha: float = 0.3,
        device: str = None,
    ):
        self.ce_ranker = CrossEncoderReranker(cross_encoder_model, device=device)
        self.alpha = alpha

    @staticmethod
    def _min_max_norm(scores: List[float]) -> List[float]:
        arr = np.array(scores, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return [0.5] * len(scores)
        return ((arr - mn) / (mx - mn)).tolist()

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
        text_field: str = "text",
    ) -> List[SearchResult]:
        if not results:
            return results

        orig_scores = [r.score for r in results]
        self.ce_ranker.rerank(query, results, text_field=text_field)
        ce_scores = [r.score for r in results]

        orig_norm = self._min_max_norm(orig_scores)
        ce_norm = self._min_max_norm(ce_scores)

        for r, o, c in zip(results, orig_norm, ce_norm):
            r.score = self.alpha * o + (1 - self.alpha) * c

        reranked = sorted(results, key=lambda r: r.score, reverse=True)
        for i, r in enumerate(reranked):
            r.rank = i + 1
        return reranked[:top_k] if top_k else reranked