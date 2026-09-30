"""RQ job: normalize a batch of raw Open Food Facts rows and upsert them into Postgres.

Run by `rq worker ingest` (docker compose service `worker`). Jobs carry raw rows, not `Product`s, so
the normalization rules live in one place (`app/ingest/source.py`) and change without re-enqueuing.
"""

import logging
from typing import Any

from pydantic import ValidationError

from app import db
from app.config import settings
from app.ingest.source import Product, normalize

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


def process_products(rows: list[dict[str, Any]]) -> dict[str, int]:
    """The RQ job. A DB error raises, so RQ retries the whole batch (the upsert is idempotent)."""
    products, skipped, invalid = normalize_batch(rows)
    with db.connect(settings.database_url) as conn:
        upserted = db.upsert_products(conn, products)
    stats = {"received": len(rows), "upserted": upserted, "skipped": skipped, "invalid": invalid}
    logger.info("batch done: %s", stats)
    return stats
