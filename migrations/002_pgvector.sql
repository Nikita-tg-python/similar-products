-- pgvector column and search indexes. Idempotent: runs on every start, also over a filled table.
-- SCHEMA public: the extension must never land in a throwaway test schema (DROP ... CASCADE there
-- would take the column with it).
CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;

-- 384 = sentence-transformers MiniLM family (KAN-58). Filled by the ingest workers; NULL until then.
ALTER TABLE products ADD COLUMN IF NOT EXISTS embedding vector(384);

-- Cosine distance (`<=>`): the embeddings are compared by angle, as in Qdrant (KAN-58/KAN-61).
-- m / ef_construction are pgvector's defaults, written out so the comparison states them.
CREATE INDEX IF NOT EXISTS products_embedding_hnsw
    ON products USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Tag filters: `&&` (overlaps) / `@>` (contains). An exclusion (NOT &&) cannot use GIN; it is
-- applied to the vector index's candidates instead.
CREATE INDEX IF NOT EXISTS products_allergens_gin ON products USING gin (allergens_tags);
CREATE INDEX IF NOT EXISTS products_traces_gin ON products USING gin (traces_tags);
CREATE INDEX IF NOT EXISTS products_categories_gin ON products USING gin (categories_tags);
