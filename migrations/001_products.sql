-- Normalized Open Food Facts products (app/ingest/source.py: Product). Idempotent: runs on every start.
-- Vector columns and search indexes are added in KAN-57.
CREATE TABLE IF NOT EXISTS products (
    code                      text PRIMARY KEY,
    product_name              text NOT NULL,
    brands                    text,
    lang                      text,
    categories_tags           text[] NOT NULL DEFAULT '{}',
    ingredients_text          text,
    allergens_tags            text[] NOT NULL DEFAULT '{}',
    traces_tags               text[] NOT NULL DEFAULT '{}',
    vegan                     text NOT NULL DEFAULT 'unknown'
                              CHECK (vegan IN ('yes', 'no', 'maybe', 'unknown')),
    nutriscore_grade          text CHECK (nutriscore_grade IN ('a', 'b', 'c', 'd', 'e')),
    sugars_100g               real CHECK (sugars_100g BETWEEN 0 AND 100),
    nova_group                smallint,
    environmental_score_grade text,
    completeness              real,
    countries_tags            text[] NOT NULL DEFAULT '{}',
    updated_at                timestamptz NOT NULL DEFAULT now()
);
