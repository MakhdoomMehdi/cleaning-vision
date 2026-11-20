from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_health_ok() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "Cleaning Vision AI"
    assert isinstance(body["openai_configured"], bool)


def test_health_never_exposes_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-value")
    from app.core import config

    config.get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            response = client.get("/health")
        assert response.json()["openai_configured"] is True
        assert "sk-test-secret-value" not in response.text
    finally:
        config.get_settings.cache_clear()


def test_cors_origins_parsed_from_comma_separated_string() -> None:
    settings = Settings(cors_origins="http://a.test, http://b.test")  # type: ignore[arg-type]
    assert settings.cors_origins == ["http://a.test", "http://b.test"]


def test_cors_allows_local_frontend_only() -> None:
    with TestClient(create_app()) as client:
        allowed = client.get("/health", headers={"Origin": "http://localhost:5173"})
        blocked = client.get("/health", headers={"Origin": "http://evil.example"})

    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in blocked.headers
