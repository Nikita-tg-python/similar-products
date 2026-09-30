"""Liveness probes of the backing services, each bounded by a short timeout and never raising."""

import asyncio

import httpx
import psycopg
from redis.asyncio import Redis

from app.config import Settings

TIMEOUT_S = 2.0


async def check_postgres(settings: Settings) -> bool:
    try:
        async with await psycopg.AsyncConnection.connect(
            settings.database_url, connect_timeout=TIMEOUT_S
        ) as conn:
            row = await (await conn.execute("SELECT 1")).fetchone()
            return row == (1,)
    except Exception:
        return False


async def check_qdrant(settings: Settings) -> bool:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            response = await client.get(f"{settings.qdrant_url}/readyz")
            return response.status_code == 200
    except Exception:
        return False


async def check_redis(settings: Settings) -> bool:
    client = Redis.from_url(settings.redis_url, socket_timeout=TIMEOUT_S)
    try:
        return bool(await client.ping())
    except Exception:
        return False
    finally:
        await client.aclose()


async def check_all(settings: Settings) -> dict[str, bool]:
    postgres, qdrant, redis = await asyncio.gather(
        check_postgres(settings), check_qdrant(settings), check_redis(settings)
    )
    return {"postgres": postgres, "qdrant": qdrant, "redis": redis}
