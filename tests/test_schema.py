import json

import pytest

from app import db

N_ROWS = 300


def plan(conn, sql: str) -> str:
    """The EXPLAIN plan as JSON text: index names show up in it wherever an index is used."""
    return json.dumps(conn.execute(f"EXPLAIN (FORMAT JSON) {sql}").fetchone()[0])


@pytest.fixture
def filled(pg_schema):
    conn = pg_schema
    db.apply_migrations(conn)
    conn.execute(
        """
        INSERT INTO products (code, product_name, categories_tags, allergens_tags, vegan, embedding)
        SELECT i::text, 'product ' || i,
               CASE WHEN i %% 50 = 0 THEN '{en:biscuits}'::text[] ELSE '{en:snacks}' END,
               CASE WHEN i %% 7 = 0 THEN '{en:nuts}'::text[] ELSE '{}' END,
               CASE WHEN i %% 3 = 0 THEN 'yes' ELSE 'no' END,
               (SELECT array_agg(random() - 0.5) FROM generate_series(1, 384) WHERE i > 0)::vector
        FROM generate_series(1, %s) AS i
        """,
        (N_ROWS,),
    )
    conn.execute("ANALYZE products")
    # On a few hundred rows a seq scan is rightly cheaper; what must hold is that the indexes are
    # *usable* for these queries (operator class matches `<=>` / `&&`). That the planner picks HNSW
    # on its own at 50k rows is shown with EXPLAIN ANALYZE in the README.
    conn.execute("SET enable_seqscan = off")
    return conn


def test_migrations_are_idempotent_and_create_the_vector_schema(pg_schema):
    conn = pg_schema
    assert db.apply_migrations(conn) == db.apply_migrations(conn)
    column = conn.execute(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'products'::regclass AND attname = 'embedding'"
    ).fetchone()
    assert column == ("vector(384)",)
    rows = conn.execute(
        "SELECT indexname FROM pg_indexes "
        "WHERE tablename = 'products' AND schemaname = current_schema()"
    )
    indexes = {r[0] for r in rows}
    assert {
        "products_embedding_hnsw",
        "products_allergens_gin",
        "products_traces_gin",
        "products_categories_gin",
    } <= indexes


def test_knn_uses_the_hnsw_index(filled):
    query = "ORDER BY embedding <=> (SELECT embedding FROM products WHERE code = '1') LIMIT 10"
    assert "products_embedding_hnsw" in plan(filled, f"SELECT code FROM products {query}")


def test_filtered_knn_still_uses_the_hnsw_index(filled):
    filled.execute("SET hnsw.iterative_scan = relaxed_order")
    sql = (
        "SELECT code FROM products WHERE vegan = 'yes' AND NOT allergens_tags && '{en:nuts}' "
        "ORDER BY embedding <=> (SELECT embedding FROM products WHERE code = '1') LIMIT 10"
    )
    assert "products_embedding_hnsw" in plan(filled, sql)
    assert len(filled.execute(sql).fetchall()) == 10


def test_category_filter_uses_gin(filled):
    sql = "SELECT code FROM products WHERE categories_tags && '{en:biscuits}'"
    assert "products_categories_gin" in plan(filled, sql)
