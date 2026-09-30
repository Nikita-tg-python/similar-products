from functools import lru_cache

from qdrant_client import QdrantClient

from app.config import settings
from app.vectorstores.qdrant_store import QdrantStore

# Ping an idle channel every 60 s: after ~35 idle minutes the API's gRPC connection had gone
# half-open and the next search hung until its 30 s deadline (KAN-64 rehearsal: a 500).
GRPC_KEEPALIVE = {
    "grpc.keepalive_time_ms": 60_000,
    "grpc.keepalive_timeout_ms": 10_000,
    "grpc.keepalive_permit_without_calls": 1,
}


@lru_cache(maxsize=1)
def get_qdrant_store() -> QdrantStore:
    client = QdrantClient(
        url=settings.qdrant_url,
        prefer_grpc=settings.qdrant_prefer_grpc,
        grpc_port=settings.qdrant_grpc_port,
        grpc_options=GRPC_KEEPALIVE,
        timeout=30,  # collection setup and batch upserts; searches pass their own, shorter one
    )
    return QdrantStore(client, settings.qdrant_collection)
