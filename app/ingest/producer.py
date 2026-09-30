"""Enqueue the local sample into the RQ `ingest` queue in batches, then report progress until done.

    python -m app.ingest.producer [--source data/sample_50k.parquet] [--batch-size 500] [--no-wait]

Batches, not single rows: RQ forks a work horse per job, so 50k one-row jobs would spend most of
their time forking. Exit code 1 if any batch failed after its retries.
"""

import argparse
import itertools
import logging
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from redis import Redis
from redis.exceptions import RedisError
from rq import Queue, Retry
from rq.job import Job, JobStatus

from app import db
from app.config import settings
from app.ingest.source import iter_local_rows
from app.ingest.worker import QUEUE_NAME, process_products
from app.vectorstores import get_qdrant_store

logger = logging.getLogger(__name__)

DEFAULT_SOURCE = Path("data/sample_50k.parquet")
RETRY = Retry(max=3, interval=[5, 30, 60])


def batches(rows: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    it = iter(rows)
    while batch := list(itertools.islice(it, size)):
        yield batch


def enqueue_file(queue: Queue, source: Path, batch_size: int) -> list[str]:
    job_ids = []
    start = 0
    for batch in batches(iter_local_rows(source), batch_size):
        job = queue.enqueue(
            process_products,
            batch,
            retry=RETRY,
            job_timeout=600,
            result_ttl=24 * 3600,
            failure_ttl=7 * 24 * 3600,
            description=f"{source.name} rows {start}..{start + len(batch) - 1}",
        )
        job_ids.append(job.id)
        start += len(batch)
    logger.info("enqueued %d rows in %d jobs to queue %r", start, len(job_ids), queue.name)
    return job_ids


def job_summary(redis: Redis, job_ids: list[str]) -> tuple[Counter, Counter]:
    """(jobs by status, summed job results) for the given jobs."""
    statuses: Counter = Counter()
    totals: Counter = Counter()
    for job in Job.fetch_many(job_ids, connection=redis):
        if job is None:  # expired: only after result_ttl, long after a run
            statuses["expired"] += 1
            continue
        status = job.get_status()
        statuses[status.value if status else "unknown"] += 1
        if status == JobStatus.FINISHED and isinstance(job.return_value(), dict):
            totals.update(job.return_value())
    return statuses, totals


# Progress polling tolerates Redis hiccups (seen in the KAN-64 rehearsal on a loaded machine: one
# socket timeout killed `make ingest` while the workers carried on). Give up after ~1 minute.
MAX_POLL_FAILURES = 30


def wait_for(redis: Redis, queue: Queue, job_ids: list[str], poll_s: float = 2.0) -> Counter:
    terminal = {JobStatus.FINISHED.value, JobStatus.FAILED.value, "expired"}
    failures = 0
    while True:
        try:
            statuses, totals = job_summary(redis, job_ids)
            in_queue = queue.count
        except RedisError as exc:
            failures += 1
            if failures >= MAX_POLL_FAILURES:
                raise
            logger.warning("redis did not answer (%s), polling again", type(exc).__name__)
            time.sleep(poll_s)
            continue
        failures = 0
        done = sum(statuses[s] for s in terminal)
        logger.info(
            "jobs %d/%d done (failed %d), in queue %d | rows upserted %d, skipped %d, invalid %d",
            done,
            len(job_ids),
            statuses[JobStatus.FAILED.value],
            in_queue,
            totals["upserted"],
            totals["skipped"],
            totals["invalid"],
        )
        if done == len(job_ids):
            return statuses
        time.sleep(poll_s)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--no-wait", action="store_true", help="enqueue and exit")
    args = parser.parse_args()
    if not args.source.exists():
        sys.exit(f"{args.source} not found: run `make sample` first")

    # Schema and collection must exist before the first job lands, whichever of API/producer starts
    # first.
    with db.connect(settings.database_url) as conn:
        logger.info("migrations applied: %s", db.apply_migrations(conn))
    if get_qdrant_store().ensure_collection():
        logger.info("qdrant collection %r created", settings.qdrant_collection)

    redis = Redis.from_url(settings.redis_url, socket_timeout=30, retry_on_timeout=True)
    queue = Queue(QUEUE_NAME, connection=redis)
    started = time.monotonic()
    job_ids = enqueue_file(queue, args.source, args.batch_size)
    if args.no_wait:
        return
    statuses = wait_for(redis, queue, job_ids)
    with db.connect(settings.database_url) as conn:
        count = conn.execute("SELECT count(*) FROM products").fetchone()[0]
    logger.info("done in %.0fs; products in Postgres: %d", time.monotonic() - started, count)
    if failed := statuses[JobStatus.FAILED.value]:
        sys.exit(f"{failed} batches failed: see `rq info` / the failed job registry")


if __name__ == "__main__":
    main()
