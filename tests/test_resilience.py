"""Store outages: bounded retries on transient Qdrant errors, a structured 503 instead of a 500."""

import grpc
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import search
from app.llm.fake import FakeLLM
from app.main import app
from app.vectorstores import GRPC_KEEPALIVE
from app.vectorstores.qdrant_store import read_with_retry


class FakeRpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode) -> None:
        self._code = code

    def code(self) -> grpc.StatusCode:
        return self._code


def flaky(*outcomes):
    """A call that raises / returns the given outcomes in order, counting its calls."""
    calls = []

    def call():
        outcome = outcomes[len(calls)]
        calls.append(outcome)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return call, calls


@pytest.mark.parametrize("code", [grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.UNAVAILABLE])
def test_transient_error_is_retried_once(code):
    call, calls = flaky(FakeRpcError(code), "hits")
    assert read_with_retry(call) == "hits"
    assert len(calls) == 2


def test_second_transient_error_is_raised_not_retried_forever():
    error = FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)
    call, calls = flaky(error, error, "never")
    with pytest.raises(FakeRpcError):
        read_with_retry(call)
    assert len(calls) == 2


def test_non_transient_error_is_not_retried():
    call, calls = flaky(FakeRpcError(grpc.StatusCode.INVALID_ARGUMENT), "never")
    with pytest.raises(FakeRpcError):
        read_with_retry(call)
    assert len(calls) == 1


def test_grpc_channel_keeps_alive_when_idle():
    assert GRPC_KEEPALIVE["grpc.keepalive_permit_without_calls"] == 1
    assert GRPC_KEEPALIVE["grpc.keepalive_time_ms"] <= 60_000


class DownStore:
    name = "qdrant"

    def get_vector(self, code):
        raise FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)

    def search(self, *args, **kwargs):
        raise FakeRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)


@pytest.fixture
def client_with_down_qdrant(monkeypatch):
    from app import db
    from app.schemas import ProductOut

    product = ProductOut(
        code="1",
        product_name="x",
        brands=None,
        vegan="yes",
        nutriscore_grade=None,
        sugars_100g=None,
        allergens_tags=[],
        traces_tags=[],
        categories_tags=[],
    )
    monkeypatch.setattr(db, "fetch_products", lambda _c, codes: {c: product for c in codes})
    monkeypatch.setattr(
        search, "get_embedder", lambda: type("E", (), {"encode": lambda s, t: np.zeros((1, 384))})()
    )

    def no_db():
        yield None

    app.dependency_overrides = {
        search.get_conn: no_db,
        search.get_qdrant: lambda: DownStore(),
        search.get_llm: lambda: FakeLLM(),
    }
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides = {}


def test_store_outage_on_search_is_a_503(client_with_down_qdrant):
    r = client_with_down_qdrant.post("/search", json={"query": "sparkling water"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "backend_unavailable"
    assert "qdrant" in r.json()["error"]["message"]


def test_store_outage_on_similar_is_a_503(client_with_down_qdrant):
    r = client_with_down_qdrant.get("/products/1/similar")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "backend_unavailable"


def test_database_outage_is_a_503(monkeypatch):
    import psycopg

    from app import db

    def broken(_conn, _codes):
        raise psycopg.OperationalError("server closed the connection unexpectedly")

    monkeypatch.setattr(db, "fetch_products", broken)

    def no_db():
        yield None

    app.dependency_overrides = {search.get_conn: no_db}
    try:
        r = TestClient(app, raise_server_exceptions=False).get("/products/1/similar")
    finally:
        app.dependency_overrides = {}
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "database_unavailable"


def test_producer_polling_survives_redis_hiccups(monkeypatch):
    from collections import Counter

    from redis.exceptions import TimeoutError as RedisTimeout

    from app.ingest import producer

    answers = [RedisTimeout("Timeout reading from socket"), (Counter(finished=2), Counter())]

    def summary(_redis, _ids):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(producer, "job_summary", summary)
    queue = type("Q", (), {"count": 0})()
    statuses = producer.wait_for(None, queue, ["a", "b"], poll_s=0)
    assert statuses["finished"] == 2
