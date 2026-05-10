import numpy as np
import torch
import clip
from typing import List, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEncoder:

    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"[CLIP TextEncoder] Loading CLIP on {self.device}...")
        self.model, _ = clip.load(model_name, device=self.device)
        self.model.eval()

    def encode(self, texts, normalize=True):

        single = isinstance(texts, str)
        if single:
            texts = [texts]

        # ✅ ONLY ONE PROMPT HERE
        texts = [f"a photo of {t}" for t in texts]

        tokens = clip.tokenize(texts, truncate=True).to(self.device)

        with torch.no_grad():
            feats = self.model.encode_text(tokens).float()

        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)

        out = feats.cpu().numpy()
        return out[0] if single else out