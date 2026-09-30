"""Postgres access shared by the API and the ingest workers (psycopg 3: sync for RQ jobs)."""

from collections.abc import Iterable
from pathlib import Path

import psycopg

from app.ingest.source import Product

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
# Arbitrary constant: serialises migrations when the API and the producer start at once.
_MIGRATION_LOCK_ID = 5_302_2026

PRODUCT_COLUMNS = list(Product.model_fields)

UPSERT_PRODUCT_SQL = (
    f"INSERT INTO products ({', '.join(PRODUCT_COLUMNS)}) "
    f"VALUES ({', '.join(f'%({c})s' for c in PRODUCT_COLUMNS)}) "
    "ON CONFLICT (code) DO UPDATE SET "
    + ", ".join(f"{c} = EXCLUDED.{c}" for c in PRODUCT_COLUMNS if c != "code")
    + ", updated_at = now()"
)


def connect(database_url: str) -> psycopg.Connection:
    # lock_timeout: a job blocked by someone else's row locks fails (and RQ retries it) instead of
    # hanging; seen when a killed worker left an open transaction behind (KAN-58).
    return psycopg.connect(database_url, connect_timeout=5, options="-c lock_timeout=60s")


def apply_migrations(conn: psycopg.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Run every migrations/*.sql in name order. Each file is idempotent (IF NOT EXISTS etc.), so
    re-running is a no-op; the advisory lock keeps concurrent starters from racing on DDL."""
    applied = []
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_ID,))
        for path in sorted(migrations_dir.glob("*.sql")):
            conn.execute(path.read_text(encoding="utf-8"))
            applied.append(path.name)
    return applied


def upsert_products(conn: psycopg.Connection, products: Iterable[Product]) -> int:
    """Insert or overwrite by `code`: re-ingesting the same rows leaves the table unchanged."""
    params = [p.model_dump() for p in products]
    if params:
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(UPSERT_PRODUCT_SQL, params)
    return len(params)
