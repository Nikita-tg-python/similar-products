import math
from pathlib import Path

import pytest
from fakeredis import FakeStrictRedis
from rq import Queue

from app import db
from app.ingest import producer, worker
from app.ingest.source import iter_local_rows, normalize
from scripts.build_fixture import FIXTURE
from tests.test_source import raw_row

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = ROOT / FIXTURE


@pytest.fixture
def captured(monkeypatch):
    """Replace Postgres with a dict keyed by code — the same semantics as the upsert."""
    store: dict[str, object] = {}

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def upsert(_conn, products):
        products = list(products)
        store.update({p.code: p for p in products})
        return len(products)

    monkeypatch.setattr(db, "connect", lambda _url: FakeConn())
    monkeypatch.setattr(db, "upsert_products", upsert)
    return store


def test_batches_split_without_losing_rows():
    rows = [{"code": str(i)} for i in range(1234)]
    out = list(producer.batches(rows, 500))
    assert [len(b) for b in out] == [500, 500, 234]
    assert [r for b in out for r in b] == rows


def test_bad_rows_do_not_fail_the_batch(captured):
    rows = [raw_row(), raw_row(code=""), raw_row(code="2", nova_group="three")]
    stats = worker.process_products(rows)
    assert stats == {"received": 3, "upserted": 1, "skipped": 1, "invalid": 1}
    assert list(captured) == ["0000101209159"]


def test_queue_round_trip_is_idempotent(captured):
    """Producer → RQ (synchronous, fake Redis) → worker, twice: same products, no duplicates."""
    redis = FakeStrictRedis()
    queue = Queue(worker.QUEUE_NAME, connection=redis, is_async=False)
    rows = list(iter_local_rows(FIXTURE_PATH))

    job_ids = producer.enqueue_file(queue, FIXTURE_PATH, batch_size=100)
    assert len(job_ids) == math.ceil(len(rows) / 100)
    statuses, totals = producer.job_summary(redis, job_ids)
    assert statuses == {"finished": len(job_ids)}
    # the fixture holds useful rows only
    assert totals["received"] == totals["upserted"] == len(rows)
    assert set(captured) == {r["code"] for r in rows}

    producer.enqueue_file(queue, FIXTURE_PATH, batch_size=100)
    assert len(captured) == len(rows)


def test_upsert_sql_updates_every_column_on_conflict():
    sql = db.UPSERT_PRODUCT_SQL
    assert "ON CONFLICT (code) DO UPDATE" in sql
    for column in db.PRODUCT_COLUMNS:
        assert f"%({column})s" in sql
        if column != "code":
            assert f"{column} = EXCLUDED.{column}" in sql


def test_upsert_into_postgres_is_idempotent(pg_schema):
    conn = pg_schema
    db.apply_migrations(conn)
    db.apply_migrations(conn)  # re-running migrations is a no-op
    products = [normalize(r) for r in iter_local_rows(FIXTURE_PATH)]

    assert db.upsert_products(conn, products) == len(products)
    assert db.upsert_products(conn, products) == len(products)
    count = conn.execute("SELECT count(*), count(DISTINCT code) FROM products").fetchone()
    assert count == (len(products), len(products))

    changed = products[0].model_copy(update={"product_name": "Renamed"})
    db.upsert_products(conn, [changed])
    name = conn.execute("SELECT product_name FROM products WHERE code = %s", (changed.code,))
    assert name.fetchone() == ("Renamed",)
