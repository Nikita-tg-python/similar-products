"""Query parsing with FakeLLM only: no network, no key."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.api import search
from app.llm import create_llm_client
from app.llm.base import LLMError
from app.llm.fake import FakeLLM
from app.llm.parse_query import SYSTEM_PROMPT, parse_query
from app.main import app
from app.schemas import LOW_SUGAR_MAX_G, ParsedQuery, ProductOut, SearchFilters

EXAMPLE = "веганське печиво без горіхів з низьким вмістом цукру"
NUTS = "en:nuts"


def parse(llm, query=EXAMPLE) -> ParsedQuery:
    return asyncio.run(parse_query(llm, query))


def test_the_task_example_parses_into_filters():
    parsed = parse(FakeLLM())
    assert parsed == ParsedQuery(
        semantic_text="cookies biscuits",
        exclude_allergens=["nuts"],
        vegan_required=True,
        sugar_level="low",
    )
    filters, notes = parsed.to_filters()
    assert filters == SearchFilters(
        exclude_allergens=[NUTS], vegan_only=True, sugars_max=LOW_SUGAR_MAX_G
    )
    assert any("5 g/100 g" in n for n in notes)


def test_one_llm_call_and_it_never_sees_products():
    llm = FakeLLM()
    parse(llm)
    assert len(llm.calls) == 1
    system, user = llm.calls[0]
    assert (system.role, system.content) == ("system", SYSTEM_PROMPT)
    assert (user.role, user.content) == ("user", EXAMPLE)


def test_invalid_json_is_retried_once():
    good = json.dumps({"semantic_text": "yogurt", "exclude_allergens": ["milk"]})
    llm = FakeLLM(["not json at all", good])
    assert parse(llm, "йогурт без лактози").exclude_allergens == ["milk"]
    assert len(llm.calls) == 2
    assert "not valid for the schema" in llm.calls[1][-1].content


def test_invalid_twice_is_a_502_not_a_guess():
    llm = FakeLLM(['{"semantic_text": ""}', '{"sugar_level": "very low"}'])
    with pytest.raises(LLMError) as exc:
        parse(llm)
    assert (exc.value.status, exc.value.code) == (502, "llm_bad_response")
    assert len(llm.calls) == 2


def test_code_fences_are_tolerated():
    llm = FakeLLM(['```json\n{"semantic_text": "chocolate spread"}\n```'])
    assert parse(llm).semantic_text == "chocolate spread"


def test_unknown_allergen_is_reported_not_silently_dropped():
    filters, notes = ParsedQuery(
        semantic_text="sausages", exclude_allergens=["pork", "soy"]
    ).to_filters()
    assert filters.exclude_allergens == ["en:soybeans"]
    assert any("'pork'" in n and "not filtered" in n for n in notes)


def test_constraint_without_a_filter_is_reported():
    parsed = parse(FakeLLM(), "sausages without pork")
    filters, notes = parsed.to_filters()
    assert filters == SearchFilters()
    assert notes == ["not filtered (no such filter): 'without pork'"]


def test_no_constraints_means_no_filters():
    parsed = parse(FakeLLM(), "sparkling water")
    assert parsed.semantic_text == "sparkling water"
    assert parsed.to_filters() == (SearchFilters(), [])


def test_vegan_false_is_no_filter():
    filters, _ = ParsedQuery(semantic_text="cheese", vegan_required=False).to_filters()
    assert filters.vegan_only is False


def test_factory_gives_fakellm_for_the_fake_provider():
    from app.config import settings

    assert settings.llm_provider == "fake"  # forced in conftest: tests never reach a real LLM
    assert isinstance(create_llm_client(settings), FakeLLM)


# --- POST /search end to end (FakeLLM, fake embeddings, real filters in both stores) ---------


def test_search_endpoint_fully_offline(monkeypatch, qdrant_store, products, vectors):
    """No Postgres, no network: Qdrant in memory, product details served from the fixture."""
    from app import db

    qdrant_store.upsert(products, vectors)
    by_code = {p.code: p for p in products}
    monkeypatch.setattr(
        db,
        "fetch_products",
        lambda _conn, codes: {
            c: ProductOut(**by_code[c].model_dump(include=set(ProductOut.model_fields)))
            for c in codes
        },
    )

    def no_db():
        yield None

    app.dependency_overrides = {
        search.get_conn: no_db,
        search.get_qdrant: lambda: qdrant_store,
        search.get_llm: lambda: FakeLLM(),
    }
    try:
        r = TestClient(app).post("/search", json={"query": EXAMPLE, "limit": 10})
    finally:
        app.dependency_overrides = {}
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert results
    for item in results:
        assert item["vegan"] == "yes"
        assert NUTS not in item["allergens_tags"] + item["traces_tags"]
        assert item["sugars_100g"] is not None and item["sugars_100g"] <= LOW_SUGAR_MAX_G


@pytest.fixture
def client(pg_schema, pgvector_store, qdrant_store, products, vectors):
    pgvector_store.upsert(products, vectors)
    qdrant_store.upsert(products, vectors)
    llm = FakeLLM()

    def conn():
        yield pg_schema

    app.dependency_overrides = {
        search.get_conn: conn,
        search.get_qdrant: lambda: qdrant_store,
        search.get_llm: lambda: llm,
    }
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides = {}


@pytest.mark.parametrize("backend", ["qdrant", "pgvector"])
def test_example_query_returns_only_vegan_nut_free_low_sugar(client, products, backend):
    r = client.post("/search", json={"query": EXAMPLE, "backend": backend, "limit": 20})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["parsed"]["semantic_text"] == "cookies biscuits"
    assert body["filters_applied"] == {
        "exclude_allergens": [NUTS],
        "vegan_only": True,
        "nutriscore_max": None,
        "sugars_max": LOW_SUGAR_MAX_G,
    }
    eligible = [
        p
        for p in products
        if p.vegan == "yes"
        and NUTS not in p.allergens_tags + p.traces_tags
        and p.ingredients_text is not None
        and p.sugars_100g is not None
        and p.sugars_100g <= LOW_SUGAR_MAX_G
    ]
    assert len(body["results"]) == min(20, len(eligible)) > 0
    for item in body["results"]:
        assert item["vegan"] == "yes"
        assert NUTS not in item["allergens_tags"] and NUTS not in item["traces_tags"]
        assert item["sugars_100g"] is not None and item["sugars_100g"] <= LOW_SUGAR_MAX_G


def test_llm_failure_is_a_structured_error(client):
    app.dependency_overrides[search.get_llm] = lambda: FakeLLM(["{", "{"])
    r = client.post("/search", json={"query": EXAMPLE})
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "llm_bad_response"


@pytest.mark.parametrize("body", [{}, {"query": ""}, {"query": "x", "backend": "faiss"}])
def test_bad_search_request_is_422(client, body):
    r = client.post("/search", json=body)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"
