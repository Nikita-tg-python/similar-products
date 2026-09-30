"""RQ job: normalize a batch of raw Open Food Facts rows, upsert them into Postgres, embed them once
and write the same vectors to pgvector and Qdrant in parallel.

Run by `rq worker ingest --worker-class=rq.worker.SimpleWorker` (docker compose service `worker`):
jobs run in the worker process itself, so the embedding model loads once per worker, not once per
forked job (and torch is never used across a fork). Jobs carry raw rows, not `Product`s, so the
normalization rules live in one place (`app/ingest/source.py`) and change without re-enqueuing.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import ValidationError

from app import db, telemetry
from app.config import settings
from app.embeddings import embedding_text, get_embedder
from app.ingest.source import Product, normalize
from app.telemetry import in_current_context, span
from app.vectorstores import get_qdrant_store
from app.vectorstores.pgvector_store import PgvectorStore

# A child of rq's logger: `rq worker` configures only that one, so batch stats show up in its log.
logger = logging.getLogger("rq.worker.ingest")

QUEUE_NAME = "ingest"


def normalize_batch(rows: list[dict[str, Any]]) -> tuple[list[Product], int, int]:
    """(products, skipped, invalid): skipped rows are useless for search by design (see
    `normalize`), invalid ones have values of an unexpected type. Neither fails the batch."""
    products: list[Product] = []
    skipped = invalid = 0
    for row in rows:
        try:
            product = normalize(row)
        except (ValidationError, TypeError, AttributeError, ValueError):
            logger.warning("invalid row, code=%r", row.get("code"), exc_info=True)
            invalid += 1
            continue
        if product is None:
            skipped += 1
        else:
            products.append(product)
    return products, skipped, invalid


def _write(store, products: list[Product], vectors) -> float:
    with span(f"write_{store.name}", rows=len(products)):
        started = time.monotonic()
        store.upsert(products, vectors)
        return time.monotonic() - started


def process_products(rows: list[dict[str, Any]]) -> dict[str, int]:
    """The RQ job. Any store error raises, so RQ retries the whole batch: every write is an
    idempotent upsert, and a retry rewrites both stores, so they cannot drift apart for good."""
    telemetry.configure("similar-products-worker")  # once per worker process
    with span("ingest_batch", rows=len(rows)) as batch_span:
        stats = _process(rows)
        for key, value in stats.items():
            batch_span.set_attribute(key, value)
    return stats


def _process(rows: list[dict[str, Any]]) -> dict[str, int]:
    with span("normalize"):
        products, skipped, invalid = normalize_batch(rows)
    with span("load_embedder"):  # the model loads on the worker's first job, then ~0 ms
        embedder = get_embedder()
    started = time.monotonic()
    with span("embed", texts=len(products)):
        vectors = embedder.encode([embedding_text(p) for p in products])
    seconds = {"encode_s": time.monotonic() - started}
    with db.connect(settings.database_url) as conn:
        with span("upsert_products"):
            upserted = db.upsert_products(conn, products)
        stores = (PgvectorStore(conn), get_qdrant_store())
        # Same vectors into both stores at once; .result() re-raises either store's error.
        with ThreadPoolExecutor(max_workers=len(stores)) as pool:
            futures = {
                s.name: pool.submit(in_current_context(_write), s, products, vectors)
                for s in stores
            }
            seconds |= {f"{name}_s": future.result() for name, future in futures.items()}
    stats = {
        "received": len(rows),
        "upserted": upserted,
        "embedded": len(vectors),
        "skipped": skipped,
        "invalid": invalid,
        **{k: round(v, 2) for k, v in seconds.items()},
    }
    logger.info("batch done: %s", stats)
    return stats
