import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


def load_corpus(path: str = "data/corpus.jsonl") -> List[Dict[str, str]]:
    items = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return items


class DenseRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.items: List[Dict[str, str]] = []
        self.embs: np.ndarray | None = None

    def index(self, items: List[Dict[str, str]]) -> None:
        self.items = items
        texts = [it["text"] for it in items]
        embs = self.model.encode(texts, normalize_embeddings=True)
        self.embs = np.array(embs)

    def topk(self, query: str, k: int = 3) -> List[Tuple[Dict[str, str], float]]:
        assert self.embs is not None
        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.embs @ q
        idx = np.argsort(-sims)[:k]
        return [(self.items[int(i)], float(sims[int(i)])) for i in idx]