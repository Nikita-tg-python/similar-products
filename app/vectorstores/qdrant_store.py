"""Qdrant: the collection, deterministic point ids and payload indexes for the filters."""

import uuid

import numpy as np
from qdrant_client import QdrantClient, models

from app.embeddings import EMBEDDING_DIM
from app.ingest.source import Product
from app.vectorstores.base import Hit, payload

# Qdrant ids are unsigned ints or UUIDs. Barcodes are neither (leading zeros, the odd letter), so
# the id is a UUIDv5 of the code: stable across runs, which keeps upserts idempotent.
_ID_NAMESPACE = uuid.UUID("6f1d3c1e-0b8a-4d0e-9a57-5a3b2f6c1d20")

# Same graph parameters as the pgvector index (migrations/002_pgvector.sql): a fair comparison.
HNSW = models.HnswConfigDiff(m=16, ef_construct=64)

KEYWORD_FIELDS = (
    "code",
    "allergens_tags",
    "traces_tags",
    "categories_tags",
    "vegan",
    "nutriscore_grade",
)


def point_id(code: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, code))


class QdrantStore:
    name = "qdrant"

    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection

    def ensure_collection(self) -> bool:
        """Create the collection and its payload indexes if missing; True if it was created."""
        if self.client.collection_exists(self.collection):
            return False
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
            hnsw_config=HNSW,
        )
        for field in KEYWORD_FIELDS:
            self.client.create_payload_index(
                self.collection, field, models.PayloadSchemaType.KEYWORD
            )
        self.client.create_payload_index(
            self.collection, "sugars_100g", models.PayloadSchemaType.FLOAT
        )
        return True

    def upsert(self, products: list[Product], vectors: np.ndarray) -> None:
        points = [
            models.PointStruct(id=point_id(p.code), vector=v.tolist(), payload=payload(p))
            for p, v in zip(products, vectors, strict=True)
        ]
        if points:
            self.client.upsert(self.collection, points=points, wait=True)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def get_vector(self, code: str) -> np.ndarray | None:
        points = self.client.retrieve(self.collection, [point_id(code)], with_vectors=True)
        return np.asarray(points[0].vector, dtype=np.float32) if points else None

    def search(
        self, vector: np.ndarray, limit: int = 10, exclude_code: str | None = None
    ) -> list[Hit]:
        query_filter = None
        if exclude_code is not None:
            query_filter = models.Filter(
                must_not=[
                    models.FieldCondition(key="code", match=models.MatchValue(value=exclude_code))
                ]
            )
        response = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            query_filter=query_filter,
            limit=limit,
            with_payload=["code"],
        )
        return [Hit(code=p.payload["code"], score=p.score) for p in response.points]
