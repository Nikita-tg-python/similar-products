from functools import lru_cache

from qdrant_client import QdrantClient

from app.config import settings
from app.vectorstores.qdrant_store import QdrantStore


@lru_cache(maxsize=1)
def get_qdrant_store() -> QdrantStore:
    client = QdrantClient(
        url=settings.qdrant_url,
        prefer_grpc=settings.qdrant_prefer_grpc,
        grpc_port=settings.qdrant_grpc_port,
        timeout=30,
    )
    return QdrantStore(client, settings.qdrant_collection)
