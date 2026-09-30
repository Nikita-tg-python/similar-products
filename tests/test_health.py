import pytest
from fastapi.testclient import TestClient

from app import health
from app.main import app


def _fixed_check(ok: bool):
    async def check(_settings) -> bool:
        return ok

    return check


def _stub_checks(monkeypatch, **results: bool) -> None:
    for name in ("postgres", "qdrant", "redis"):
        monkeypatch.setattr(health, f"check_{name}", _fixed_check(results.get(name, True)))


def test_health_ok_when_all_services_up(monkeypatch):
    _stub_checks(monkeypatch)
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert (body["postgres"], body["qdrant"], body["redis"]) == ("ok", "ok", "ok")


@pytest.mark.parametrize("down", ["postgres", "qdrant", "redis"])
def test_health_503_names_the_service_that_is_down(monkeypatch, down):
    _stub_checks(monkeypatch, **{down: False})
    response = TestClient(app).get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()[down] == "error"
