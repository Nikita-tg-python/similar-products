from functools import lru_cache

from qdrant_client import QdrantClient

from app.config import settings
from app.vectorstores.qdrant_store import QdrantStore


@lru_cache(maxsize=1)
def get_qdrant_store() -> QdrantStore:
    return QdrantStore(
        QdrantClient(url=settings.qdrant_url, timeout=30), settings.qdrant_collection
    )
