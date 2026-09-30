"""pgvector: the `embedding` column of `products` (migrations/002_pgvector.sql), cosine distance."""

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from app.ingest.source import Product
from app.vectorstores.base import Hit


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
        self, vector: np.ndarray, limit: int = 10, exclude_code: str | None = None
    ) -> list[Hit]:
        rows = self.conn.execute(
            "SELECT code, 1 - (embedding <=> %(v)s) FROM products "
            "WHERE embedding IS NOT NULL AND code IS DISTINCT FROM %(exclude)s "
            "ORDER BY embedding <=> %(v)s LIMIT %(limit)s",
            {"v": vector, "exclude": exclude_code, "limit": limit},
        ).fetchall()
        return [Hit(code=code, score=float(score)) for code, score in rows]
