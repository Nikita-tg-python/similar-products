"""One interface over Qdrant and pgvector: search and the comparison (KAN-61) treat them alike."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.ingest.source import Product

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
        self, vector: np.ndarray, limit: int = 10, exclude_code: str | None = None
    ) -> list[Hit]: ...
