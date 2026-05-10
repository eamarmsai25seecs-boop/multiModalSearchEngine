from typing import Dict, Any

class SearchResult:
    def __init__(self, rank: int, score: float, modality: str, metadata: Dict[str, Any]):
        self.rank = rank
        self.score = score
        self.modality = modality
        self.metadata = metadata

    def __repr__(self):
        return f"SearchResult(rank={self.rank}, score={self.score:.4f}, modality={self.modality})"