import io

import pytest
from fastapi.testclient import TestClient

from app.api import search
from app.ingest import worker
from app.ingest.source import iter_local_rows
from app.llm.fake import FakeLLM
from app.main import app
from app.telemetry import TreeConsoleExporter, memory_exporter, span
from tests.test_ingest import FIXTURE_PATH
from tests.test_parse_query import EXAMPLE


@pytest.fixture(autouse=True)
def fresh_spans():
    memory_exporter.clear()
    yield
    memory_exporter.clear()


def by_name():
    return {s.name: s for s in memory_exporter.get_finished_spans()}


@pytest.fixture
def client(pg_schema, pgvector_store, qdrant_store, products, vectors):
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


def test_search_request_is_one_trace_with_named_steps(client):
    assert client.post("/search", json={"query": EXAMPLE, "limit": 5}).status_code == 200
    spans = by_name()
    root = spans["POST /search"]
    steps = ["llm_parse", "embed_query", "vector_search", "fetch_products"]
    assert set(steps) <= set(spans)
    for name in steps:
        assert spans[name].context.trace_id == root.context.trace_id
        assert spans[name].parent.span_id == root.context.span_id, name
    # in this order: parse, then embed, then search
    starts = [spans[n].start_time for n in steps]
    assert starts == sorted(starts)
    assert spans["llm_parse"].attributes["llm.provider"] == "fake"
    assert spans["llm_parse"].attributes["llm.attempts"] == 1
    assert spans["vector_search"].attributes["backend"] == "qdrant"
    assert spans["vector_search"].attributes["hits"] == 5


def test_similar_request_traces_the_vector_lookup_and_search(client, products):
    r = client.get(f"/products/{products[0].code}/similar", params={"exclude_allergens": "nuts"})
    assert r.status_code == 200
    spans = by_name()
    assert {"GET /products/{code}/similar", "get_vector", "vector_search"} <= set(spans)
    assert "en:nuts" in spans["vector_search"].attributes["filters"]


def test_health_is_not_traced(client):
    client.get("/health")
    assert not [s for s in memory_exporter.get_finished_spans() if "health" in s.name]


def test_ingest_batch_trace_includes_both_store_writes(captured):
    rows = list(iter_local_rows(FIXTURE_PATH))[:50]
    worker.process_products(rows)
    spans = by_name()
    batch = spans["ingest_batch"]
    assert batch.attributes["upserted"] == 50
    for name in ("normalize", "load_embedder", "embed", "upsert_products", "write_qdrant"):
        assert spans[name].parent.span_id == batch.context.span_id, name


def test_tree_exporter_prints_the_tree_when_the_root_ends():
    out = io.StringIO()
    exporter = TreeConsoleExporter(out)
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("POST /search"):
        with tracer.start_as_current_span("llm_parse", attributes={"llm.provider": "fake"}):
            assert out.getvalue() == ""  # nothing until the root ends
        with tracer.start_as_current_span("vector_search"):
            pass
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("── trace ")
    assert lines[1].startswith("POST /search")
    assert lines[2].startswith("  llm_parse") and "llm.provider=fake" in lines[2]
    assert lines[3].startswith("  vector_search")


def test_span_helper_skips_none_attributes():
    with span("x", a=1, b=None) as s:
        assert "b" not in s.attributes and s.attributes["a"] == 1


def test_fastapi_native_telemetry_cannot_add_exporters():
    config = app._telemetry
    assert config["auto_configure"] is False and config["tracing"] is False
