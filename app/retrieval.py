"""Hybrid retrieval: dense embeddings (BAAI/bge-small-en-v1.5 via fastembed) + BM25,
fused with Reciprocal Rank Fusion.

Why hybrid: dense vectors catch paraphrases ("lost my laptop" -> "Loss or theft"),
BM25 catches exact tokens that embeddings blur (amounts like "NGN 50,000", "probation",
"anchor days"). RRF merges both rankings without having to calibrate their score scales.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .ingest import Chunk

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge models expect this prefix on queries (not on passages) for retrieval tasks.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an and are as at be by can do does for from how i if in is it its me my of on or "
    "so that the their there this to was what when where which who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Hit:
    chunk: Chunk
    score: float        # fused RRF score
    dense: float        # cosine similarity, for logging/debugging


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], cache_dir: str | None = None):
        from fastembed import TextEmbedding  # imported lazily: heavy (onnxruntime)

        self.chunks = chunks
        self.model = TextEmbedding(EMBED_MODEL, cache_dir=cache_dir)
        self.matrix = self._embed([c.search_text for c in chunks])
        self.bm25 = BM25Okapi([tokenize(c.search_text) for c in chunks])

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self.model.embed(texts)), dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    def search(self, query: str, k: int = 6) -> list[Hit]:
        q = self._embed([QUERY_PREFIX + query])[0]
        dense = self.matrix @ q
        sparse = self.bm25.get_scores(tokenize(query))

        dense_rank = np.argsort(-dense)
        sparse_rank = np.argsort(-sparse)
        fused = np.zeros(len(self.chunks))
        for rank, idx in enumerate(dense_rank):
            fused[idx] += 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(sparse_rank):
            if sparse[idx] > 0:  # a chunk with no keyword overlap earns no BM25 credit
                fused[idx] += 1.0 / (RRF_K + rank + 1)

        top = np.argsort(-fused)[:k]
        return [Hit(self.chunks[i], float(fused[i]), float(dense[i])) for i in top]
