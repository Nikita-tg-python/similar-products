import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import search
from app.ingest.source import Product
from app.main import app
from app.schemas import SearchFilters, allergen_tag

NUTS = "en:nuts"


# --- filters ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "tag"),
    [
        ("nuts", NUTS),
        ("en:nuts", NUTS),
        ("Tree nuts", NUTS),
        ("soy", "en:soybeans"),
        ("sesame", "en:sesame-seeds"),
        ("DAIRY", "en:milk"),
    ],
)
def test_allergen_names_and_aliases(name, tag):
    assert allergen_tag(name) == tag


def test_unknown_allergen_is_rejected_not_ignored():
    with pytest.raises(ValidationError, match="unknown allergen 'nutz'"):
        SearchFilters(exclude_allergens=["nutz"])


def test_filters_accept_repeated_and_comma_separated_values():
    f = SearchFilters(exclude_allergens=["nuts,milk", "peanuts", "nuts"], nutriscore_max="b")
    assert f.exclude_allergens == ["en:milk", "en:nuts", "en:peanuts"]
    assert f.nutriscore_grades == ["a", "b"]
    assert SearchFilters().nutriscore_grades is None


def passes(p: Product, f: SearchFilters) -> bool:
    banned = set(f.exclude_allergens)
    return (
        not banned & (set(p.allergens_tags) | set(p.traces_tags))
        and (not f.vegan_only or p.vegan == "yes")
        and (f.nutriscore_grades is None or p.nutriscore_grade in f.nutriscore_grades)
        and (f.sugars_max is None or (p.sugars_100g is not None and p.sugars_100g <= f.sugars_max))
    )


FILTERS = [
    SearchFilters(exclude_allergens=["nuts"]),
    SearchFilters(vegan_only=True),
    SearchFilters(nutriscore_max="b"),
    SearchFilters(sugars_max=5),
    SearchFilters(exclude_allergens=["nuts"], vegan_only=True, sugars_max=5),
    SearchFilters(exclude_allergens=["nuts", "milk"], vegan_only=True, nutriscore_max="c"),
]


@pytest.fixture(params=["qdrant", "pgvector"])
def filled_store(request, products, vectors):
    store = request.getfixturevalue(f"{request.param}_store")
    store.upsert(products, vectors)
    return store


@pytest.mark.parametrize("filters", FILTERS, ids=lambda f: f.model_dump_json(exclude_defaults=True))
def test_every_hit_passes_the_filters(filled_store, products, vectors, filters):
    by_code = {p.code: p for p in products}
    eligible = sum(passes(p, filters) for p in products)
    for i in range(0, len(products), 25):
        hits = filled_store.search(
            vectors[i], limit=10, exclude_code=products[i].code, filters=filters
        )
        assert all(passes(by_code[h.code], filters) for h in hits)
        assert products[i].code not in {h.code for h in hits}
        # filtered inside the store: a selective filter still fills the page when it can
        assert len(hits) == min(10, eligible - passes(products[i], filters))


def test_stores_agree_under_filters(qdrant_store, pgvector_store, products, vectors):
    for s in (qdrant_store, pgvector_store):
        s.upsert(products, vectors)
    f = FILTERS[-1]
    for i in range(0, len(products), 50):
        q, p = (
            {h.code for h in s.search(vectors[i], 10, products[i].code, f)}
            for s in (qdrant_store, pgvector_store)
        )
        assert len(q & p) >= 8


# --- API -------------------------------------------------------------------------------------


@pytest.fixture
def client(pg_schema, pgvector_store, qdrant_store, products, vectors):
    pgvector_store.upsert(products, vectors)
    qdrant_store.upsert(products, vectors)

    def conn():
        yield pg_schema

    app.dependency_overrides = {search.get_conn: conn, search.get_qdrant: lambda: qdrant_store}
    try:
        yield TestClient(app)  # no `with`: lifespan (real DB pool, migrations) stays off
    finally:
        app.dependency_overrides = {}


def nut_product(products) -> Product:
    return next(p for p in products if NUTS in p.allergens_tags)


@pytest.mark.parametrize("backend", ["qdrant", "pgvector"])
def test_similar_returns_products_and_the_filters_applied(client, products, backend):
    code = products[0].code
    r = client.get(f"/products/{code}/similar", params={"backend": backend, "limit": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["product"]["code"] == code
    assert body["backend"] == backend
    assert body["filters_applied"] == {
        "exclude_allergens": [],
        "vegan_only": False,
        "nutriscore_max": None,
        "sugars_max": None,
    }
    assert len(body["results"]) == 5
    assert code not in {r["code"] for r in body["results"]}
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert {"product_name", "vegan", "allergens_tags", "traces_tags"} <= set(body["results"][0])


@pytest.mark.parametrize("backend", ["qdrant", "pgvector"])
def test_exclude_nuts_drops_nuts_in_allergens_and_traces(client, products, backend):
    code = nut_product(products).code
    unfiltered = client.get(f"/products/{code}/similar", params={"backend": backend, "limit": 30})
    with_nuts = [
        r
        for r in unfiltered.json()["results"]
        if NUTS in r["allergens_tags"] or NUTS in r["traces_tags"]
    ]
    assert with_nuts, "the neighbours of a nut product should include nut products"

    r = client.get(
        f"/products/{code}/similar",
        params={"backend": backend, "limit": 30, "exclude_allergens": "nuts"},
    )
    body = r.json()
    assert body["filters_applied"]["exclude_allergens"] == [NUTS]
    assert len(body["results"]) == 30
    for item in body["results"]:
        assert NUTS not in item["allergens_tags"] and NUTS not in item["traces_tags"]


def test_all_filters_through_the_api(client, products):
    code = products[0].code
    r = client.get(
        f"/products/{code}/similar",
        params=[
            ("exclude_allergens", "nuts"),
            ("exclude_allergens", "milk,gluten"),
            ("vegan_only", "true"),
            ("nutriscore_max", "c"),
        ],
    )
    body = r.json()
    assert body["filters_applied"] == {
        "exclude_allergens": ["en:gluten", "en:milk", "en:nuts"],
        "vegan_only": True,
        "nutriscore_max": "c",
        "sugars_max": None,
    }
    for item in body["results"]:
        assert item["vegan"] == "yes" and item["nutriscore_grade"] in {"a", "b", "c"}
        tags = set(item["allergens_tags"]) | set(item["traces_tags"])
        assert not tags & {"en:gluten", "en:milk", "en:nuts"}


def test_unknown_code_is_404(client):
    r = client.get("/products/0000000000000/similar")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "product_not_found"


@pytest.mark.parametrize(
    "params",
    [{"exclude_allergens": "nutz"}, {"backend": "faiss"}, {"nutriscore_max": "f"}, {"limit": 0}],
)
def test_bad_parameters_are_structured_422(client, products, params):
    r = client.get(f"/products/{products[0].code}/similar", params=params)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"
    assert r.json()["error"]["details"]


def test_product_without_vector_is_409(client, pg_schema):
    pg_schema.execute("INSERT INTO products (code, product_name) VALUES ('new-1', 'Brand new')")
    r = client.get("/products/new-1/similar", params={"backend": "pgvector"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "not_indexed"
