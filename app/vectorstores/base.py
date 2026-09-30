"""One interface over Qdrant and pgvector: search and the comparison (KAN-61) treat them alike."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.ingest.source import Product
from app.schemas import SearchFilters

# Filter fields copied next to each vector (Qdrant payload; in Postgres they are the row's columns).
PAYLOAD_FIELDS = (
    "code",
    "allergens_tags",
    "traces_tags",
    "vegan",
    "nutriscore_grade",
    "sugars_100g",
    "categories_tags",
)


# HNSW candidate list at query time, the same in both stores (pgvector hnsw.ef_search, Qdrant
# hnsw_ef): with filters the graph walk needs headroom beyond pgvector's default 40.
SEARCH_EF = 100


@dataclass(frozen=True)
class Hit:
    code: str
    score: float  # cosine similarity, higher is closer


def payload(product: Product) -> dict:
    return product.model_dump(include=set(PAYLOAD_FIELDS))


class VectorStore(Protocol):
    name: str

    def upsert(self, products: list[Product], vectors: np.ndarray) -> None:
        """Idempotent: the same product overwrites its previous vector."""
        ...

    def count(self) -> int:
        """Products that have a vector."""
        ...

    def get_vector(self, code: str) -> np.ndarray | None: ...

    def search(
        self,
        vector: np.ndarray,
        limit: int = 10,
        exclude_code: str | None = None,
        filters: SearchFilters | None = None,
    ) -> list[Hit]:
        """Top `limit` by cosine similarity among products passing `filters`, filtered inside the
        store while it walks the index (no over-fetch and post-filter in Python)."""
        ...
