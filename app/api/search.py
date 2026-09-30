"""GET /products/{code}/similar: neighbours of a product's own vector, filtered inside the store."""

import time
from collections.abc import Iterator
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, Query, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app import db
from app.errors import ApiError
from app.schemas import Backend, NutriScore, SearchFilters, SimilarProduct, SimilarResponse
from app.vectorstores import get_qdrant_store
from app.vectorstores.base import VectorStore
from app.vectorstores.pgvector_store import PgvectorStore
from app.vectorstores.qdrant_store import QdrantStore

router = APIRouter(tags=["search"])


def get_conn(request: Request) -> Iterator[psycopg.Connection]:
    with request.app.state.pool.connection() as conn:
        yield conn


def get_qdrant() -> QdrantStore:
    return get_qdrant_store()


def parse_filters(
    exclude_allergens: Annotated[
        list[str],
        Query(description="Repeat or comma-separate: `nuts`, `milk,gluten`. Also drops traces."),
    ] = [],  # noqa: B006 (FastAPI copies query defaults)
    vegan_only: Annotated[bool, Query(description="Only vegan status 'yes'")] = False,
    nutriscore_max: Annotated[
        NutriScore | None, Query(description="Worst grade allowed: `c` keeps a, b, c")
    ] = None,
) -> SearchFilters:
    try:
        return SearchFilters(
            exclude_allergens=exclude_allergens,
            vegan_only=vegan_only,
            nutriscore_max=nutriscore_max,
        )
    except ValidationError as exc:
        # FastAPI does not turn a ValidationError raised in a dependency into a 422 by itself
        raise RequestValidationError(exc.errors()) from exc


@router.get("/products/{code}/similar", response_model=SimilarResponse)
def similar_products(
    code: str,
    conn: Annotated[psycopg.Connection, Depends(get_conn)],
    qdrant: Annotated[QdrantStore, Depends(get_qdrant)],
    filters: Annotated[SearchFilters, Depends(parse_filters)],
    backend: Backend = "qdrant",
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> SimilarResponse:
    product = db.fetch_products(conn, [code]).get(code)
    if product is None:
        raise ApiError(404, "product_not_found", f"no product with code {code!r}")
    store: VectorStore = PgvectorStore(conn) if backend == "pgvector" else qdrant
    vector = store.get_vector(code)
    if vector is None:
        raise ApiError(409, "not_indexed", f"product {code!r} has no vector in {backend} yet")

    started = time.perf_counter()
    hits = store.search(vector, limit=limit, exclude_code=code, filters=filters)
    took_ms = (time.perf_counter() - started) * 1000

    # Product details always come from Postgres, whichever store ranked them.
    details = db.fetch_products(conn, [h.code for h in hits])
    return SimilarResponse(
        product=product,
        backend=backend,
        filters_applied=filters,
        took_ms=round(took_ms, 2),
        results=[
            SimilarProduct(**details[h.code].model_dump(), score=round(h.score, 4))
            for h in hits
            if h.code in details
        ],
    )
