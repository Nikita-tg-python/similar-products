import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from psycopg_pool import ConnectionPool

from app import db, health, telemetry
from app.api import search
from app.config import settings
from app.embeddings import get_embedder
from app.errors import register_error_handlers
from app.llm import create_llm_client
from app.vectorstores import get_qdrant_store

logger = logging.getLogger(__name__)


def _migrate() -> list[str]:
    with db.connect(settings.database_url) as conn:
        applied = db.apply_migrations(conn)
    if get_qdrant_store().ensure_collection():
        logger.info("qdrant collection %r created", settings.qdrant_collection)
    return applied


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO)
    logger.info("migrations applied: %s", await asyncio.to_thread(_migrate))
    app.state.llm = create_llm_client(settings)
    # Load the embedding model now, not on the first POST /search (several seconds).
    await asyncio.to_thread(get_embedder)
    # Sync pool: DB work runs in FastAPI's thread pool, never on the event loop.
    app.state.pool = ConnectionPool(
        settings.database_url, kwargs=db.CONNECT_KWARGS, min_size=1, max_size=10, open=False
    )
    await asyncio.to_thread(app.state.pool.open, wait=True)
    yield
    await asyncio.to_thread(app.state.pool.close)


telemetry.configure("similar-products-api")
app = FastAPI(
    title="Similar Products",
    version="0.1.0",
    lifespan=lifespan,
    # FastAPI's own OpenTelemetry support is off: tracing goes through app/telemetry.py and the
    # instrumentor below. Left on, it adds an OTLP exporter whenever OTEL_EXPORTER_OTLP_ENDPOINT
    # is set, even with TRACING_EXPORTER=tree (seen in KAN-62: warnings about an absent Jaeger).
    telemetry={"tracing": False, "metrics": False, "logs": False, "auto_configure": False},
)
# One server span per request; no per-chunk receive/send spans, no /health noise.
FastAPIInstrumentor.instrument_app(app, excluded_urls="health", exclude_spans=["receive", "send"])
register_error_handlers(app)
app.include_router(search.router)


@app.get("/health")
async def get_health() -> JSONResponse:
    services = await health.check_all(settings)
    all_ok = all(services.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={
            "status": "ok" if all_ok else "degraded",
            **{name: "ok" if ok else "error" for name, ok in services.items()},
            "llm_provider": settings.llm_provider,
        },
    )
