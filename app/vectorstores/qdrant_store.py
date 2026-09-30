"""Qdrant: the collection, deterministic point ids and payload indexes for the filters."""

import uuid

import numpy as np
from qdrant_client import QdrantClient, models

from app.embeddings import EMBEDDING_DIM
from app.ingest.source import Product
from app.schemas import SearchFilters
from app.vectorstores.base import SEARCH_EF, Hit, payload

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
        self,
        vector: np.ndarray,
        limit: int = 10,
        exclude_code: str | None = None,
        filters: SearchFilters | None = None,
    ) -> list[Hit]:
        response = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            query_filter=qdrant_filter(filters or SearchFilters(), exclude_code),
            search_params=models.SearchParams(hnsw_ef=SEARCH_EF),
            limit=limit,
            with_payload=["code"],
        )
        return [Hit(code=p.payload["code"], score=p.score) for p in response.points]


def qdrant_filter(filters: SearchFilters, exclude_code: str | None) -> models.Filter | None:
    must: list[models.Condition] = []
    must_not: list[models.Condition] = []
    if exclude_code is not None:
        must_not.append(
            models.FieldCondition(key="code", match=models.MatchValue(value=exclude_code))
        )
    if filters.exclude_allergens:
        # MatchAny on an array payload: true if any element is in the list
        banned = models.MatchAny(any=filters.exclude_allergens)
        must_not += [
            models.FieldCondition(key="allergens_tags", match=banned),
            models.FieldCondition(key="traces_tags", match=banned),
        ]
    if filters.vegan_only:
        must.append(models.FieldCondition(key="vegan", match=models.MatchValue(value="yes")))
    if (grades := filters.nutriscore_grades) is not None:
        must.append(
            models.FieldCondition(key="nutriscore_grade", match=models.MatchAny(any=grades))
        )
    if filters.sugars_max is not None:
        # a null / missing sugars_100g never satisfies a range: unknown sugar is excluded
        must.append(
            models.FieldCondition(key="sugars_100g", range=models.Range(lte=filters.sugars_max))
        )
    if not (must or must_not):
        return None
    return models.Filter(must=must or None, must_not=must_not or None)
