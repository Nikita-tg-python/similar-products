"""pgvector: the `embedding` column of `products` (migrations/002_pgvector.sql), cosine distance."""

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from app.ingest.source import Product
from app.schemas import SearchFilters
from app.vectorstores.base import SEARCH_EF, Hit


class PgvectorStore:
    name = "pgvector"

    def __init__(self, conn: psycopg.Connection) -> None:
        # The filter fields are the row's own columns, so only the vector is written here.
        register_vector(conn)
        self.conn = conn

    def upsert(self, products: list[Product], vectors: np.ndarray) -> None:
        params = [(v, p.code) for p, v in zip(products, vectors, strict=True)]
        if params:
            with self.conn.transaction(), self.conn.cursor() as cur:
                cur.executemany("UPDATE products SET embedding = %s WHERE code = %s", params)

    def count(self) -> int:
        return self.conn.execute(
            "SELECT count(*) FROM products WHERE embedding IS NOT NULL"
        ).fetchone()[0]

    def get_vector(self, code: str) -> np.ndarray | None:
        row = self.conn.execute(
            "SELECT embedding FROM products WHERE code = %s AND embedding IS NOT NULL", (code,)
        ).fetchone()
        return row[0].to_numpy().astype(np.float32) if row else None

    def search(
        self,
        vector: np.ndarray,
        limit: int = 10,
        exclude_code: str | None = None,
        filters: SearchFilters | None = None,
    ) -> list[Hit]:
        where, params = sql_filter(filters or SearchFilters(), exclude_code)
        params |= {"v": vector, "limit": limit}
        with self.conn.transaction():
            # Iterative scan: with a selective WHERE, keep walking the HNSW graph until `limit`
            # rows pass, instead of filtering only the first ef_search candidates (and returning
            # fewer). relaxed_order may shuffle near-ties, so the outer query sorts again.
            self.conn.execute(f"SET LOCAL hnsw.ef_search = {SEARCH_EF}")
            self.conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            rows = self.conn.execute(
                "WITH hits AS MATERIALIZED ("
                "  SELECT code, embedding <=> %(v)s AS distance FROM products"
                f"  WHERE {where} ORDER BY distance LIMIT %(limit)s"
                ") SELECT code, 1 - distance FROM hits ORDER BY distance",
                params,
            ).fetchall()
        return [Hit(code=code, score=float(score)) for code, score in rows]


def sql_filter(filters: SearchFilters, exclude_code: str | None) -> tuple[str, dict]:
    """WHERE clause over the row's own columns; values always go in as parameters."""
    clauses = ["embedding IS NOT NULL"]
    params: dict = {}
    if exclude_code is not None:
        clauses.append("code <> %(exclude_code)s")
        params["exclude_code"] = exclude_code
    if filters.exclude_allergens:
        clauses.append("NOT (allergens_tags && %(banned)s OR traces_tags && %(banned)s)")
        params["banned"] = filters.exclude_allergens
    if filters.vegan_only:
        clauses.append("vegan = 'yes'")
    if (grades := filters.nutriscore_grades) is not None:
        clauses.append("nutriscore_grade = ANY(%(grades)s)")
        params["grades"] = grades
    return " AND ".join(clauses), params
