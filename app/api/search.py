"""Search endpoints. The ranking is always a vector search and the filtering always happens inside
the store; the LLM (POST /search only) just turns the text into filters and an English phrase.

- GET  /products/{code}/similar: neighbours of a product's own vector
- POST /search: free text -> ParsedQuery (one LLM call) -> embedding + filters -> neighbours
"""

import time
from collections.abc import Iterator
from typing import Annotated

import numpy as np
import psycopg
from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app import db
from app.embeddings import get_embedder
from app.errors import ApiError
from app.llm import LLMClient
from app.llm.parse_query import parse_query
from app.schemas import (
    Backend,
    NutriScore,
    SearchFilters,
    SearchRequest,
    SearchResponse,
    SimilarProduct,
    SimilarResponse,
)
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


def get_llm(request: Request) -> LLMClient:
    return request.app.state.llm


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

    results, took_ms = run_search(conn, store, vector, filters, limit, exclude_code=code)
    return SimilarResponse(
        product=product,
        backend=backend,
        filters_applied=filters,
        took_ms=took_ms,
        results=results,
    )


@router.post("/search", response_model=SearchResponse)
async def search_text(
    body: SearchRequest,
    conn: Annotated[psycopg.Connection, Depends(get_conn)],
    qdrant: Annotated[QdrantStore, Depends(get_qdrant)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> SearchResponse:
    parsed = await parse_query(llm, body.query)  # the only LLM call
    filters, notes = parsed.to_filters()
    vector = (await run_in_threadpool(get_embedder().encode, [parsed.semantic_text]))[0]
    store: VectorStore = PgvectorStore(conn) if body.backend == "pgvector" else qdrant
    results, took_ms = await run_in_threadpool(run_search, conn, store, vector, filters, body.limit)
    return SearchResponse(
        query=body.query,
        parsed=parsed,
        filters_applied=filters,
        notes=notes,
        backend=body.backend,
        took_ms=took_ms,
        results=results,
    )


def run_search(
    conn: psycopg.Connection,
    store: VectorStore,
    vector: np.ndarray,
    filters: SearchFilters,
    limit: int,
    exclude_code: str | None = None,
) -> tuple[list[SimilarProduct], float]:
    """Vector search with in-store filters; product details always come from Postgres, whichever
    store ranked them. Returns (results, milliseconds spent in the vector search)."""
    started = time.perf_counter()
    hits = store.search(vector, limit=limit, exclude_code=exclude_code, filters=filters)
    took_ms = round((time.perf_counter() - started) * 1000, 2)
    details = db.fetch_products(conn, [h.code for h in hits])
    results = [
        SimilarProduct(**details[h.code].model_dump(), score=round(h.score, 4))
        for h in hits
        if h.code in details
    ]
    return results, took_ms
