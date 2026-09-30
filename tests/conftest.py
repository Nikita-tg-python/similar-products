import uuid

import psycopg
import pytest

from app.config import settings


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
