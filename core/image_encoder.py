import numpy as np
import torch
import clip
from PIL import Image
from typing import Union, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageEncoder:

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"[CLIP ImageEncoder] Loading CLIP on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        self.embedding_dim = self.model.visual.output_dim
        logger.info(f"[CLIP ImageEncoder] Ready. Dim = {self.embedding_dim}")

    def _load(self, img):
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        return Image.open(img).convert("RGB")

    def encode_images(
        self,
        images: Union[str, List[str], Image.Image],
        normalize: bool = True,
    ) -> np.ndarray:

        single = not isinstance(images, list)
        if single:
            images = [images]

        batch = torch.stack([
            self.preprocess(self._load(i)) for i in images
        ]).to(self.device)

        with torch.no_grad():
            feats = self.model.encode_image(batch).float()

        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)

        out = feats.cpu().numpy()
        return out[0] if single else out