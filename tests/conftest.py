import os

# Before any app import: tests never load a real embedding model (offline, instant).
os.environ["EMBEDDING_PROVIDER"] = "fake"

import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import psycopg  # noqa: E402
import pytest  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from app import db  # noqa: E402
from app.config import settings  # noqa: E402
from app.embeddings import FakeEmbedder, embedding_text  # noqa: E402
from app.ingest.source import iter_local_rows, normalize  # noqa: E402
from app.vectorstores.pgvector_store import PgvectorStore  # noqa: E402
from app.vectorstores.qdrant_store import QdrantStore  # noqa: E402
from scripts.build_fixture import FIXTURE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def pg_schema():
    """A throwaway schema in the compose Postgres; skipped when it isn't running. `public` stays on
    the search_path: the `vector` type lives there (migrations pin the extension to it)."""
    try:
        conn = psycopg.connect(settings.database_url, connect_timeout=2, autocommit=True)
    except psycopg.OperationalError:
        pytest.skip("Postgres is not running (make up)")
    schema = f"test_{uuid.uuid4().hex[:8]}"
    conn.execute(f"CREATE SCHEMA {schema}")
    conn.execute(f"SET search_path TO {schema}, public")
    try:
        yield conn
    finally:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")
        conn.close()


@pytest.fixture
def qdrant_store():
    """Qdrant's local in-memory mode: the real client and filter semantics, no server."""
    store = QdrantStore(QdrantClient(":memory:"), "products_test")
    store.ensure_collection()
    return store


@pytest.fixture(scope="module")
def products():
    return [normalize(r) for r in iter_local_rows(ROOT / FIXTURE)]


@pytest.fixture(scope="module")
def vectors(products):
    return FakeEmbedder().encode([embedding_text(p) for p in products])


@pytest.fixture
def pgvector_store(pg_schema, products):
    db.apply_migrations(pg_schema)
    db.upsert_products(pg_schema, products)
    return PgvectorStore(pg_schema)
