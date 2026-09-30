"""Text embeddings: sentence-transformers locally on CPU (no keys), or a deterministic fake for
tests.

Model choice (measured 2026-09-30 on 5k products of the sample, P@10 = share of the top 10 in the
expected category, 5 queries): all-MiniLM-L6-v2 0.66 on English queries vs 0.56 for
paraphrase-multilingual-MiniLM-L12-v2, and it is faster (177 vs 141 texts/s). Neither handles
Ukrainian queries (0.04 / 0.24), so the query parser (KAN-60) hands the embedder English text.
Product texts embed well in English even for French products: OFF category tags are English.
"""

import hashlib
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.config import settings
from app.ingest.source import Product

EMBEDDING_DIM = 384  # the MiniLM family; the pgvector column and the Qdrant collection use it too

_WORD_RE = re.compile(r"\w+")


def embedding_text(product: Product) -> str:
    """product_name + categories + ingredients. Tags lose the language prefix and dashes
    ("en:hazelnut-spreads" -> "hazelnut spreads"). The model truncates at 256 tokens, so the most
    telling parts come first."""
    categories = ", ".join(t.split(":", 1)[-1].replace("-", " ") for t in product.categories_tags)
    parts = (product.product_name, categories, product.ingredients_text)
    return ". ".join(p for p in parts if p)


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """float32 array (len(texts), dim), L2-normalized: cosine similarity == dot product."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        # heavy: torch loads here, not when the module is imported
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = self.model.get_sentence_embedding_dimension()
        if self.dim != EMBEDDING_DIM:
            raise ValueError(
                f"{model_name} gives {self.dim}-d vectors, the schema is {EMBEDDING_DIM}-d"
            )
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


class FakeEmbedder:
    """Hashed bag of words: offline, instant, and texts sharing words land close together, which
    is enough to test search plumbing without a model."""

    dim = EMBEDDING_DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD_RE.findall(text.lower()):
                digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
                out[row, int.from_bytes(digest[:4], "little") % self.dim] += 1.0
            norm = np.linalg.norm(out[row])
            out[row] = out[row] / norm if norm else 1.0 / np.sqrt(self.dim)
        return out


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """One model per process: RQ runs jobs in-process (SimpleWorker), so it loads once per
    worker."""
    if settings.embedding_provider == "fake":
        return FakeEmbedder()
    return SentenceTransformerEmbedder(settings.embedding_model)
