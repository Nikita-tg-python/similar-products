import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app import db, health
from app.config import settings
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
    yield


app = FastAPI(title="Similar Products", version="0.1.0", lifespan=lifespan)


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
