import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from psycopg_pool import ConnectionPool

from app import db, health
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


app = FastAPI(title="Similar Products", version="0.1.0", lifespan=lifespan)
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
