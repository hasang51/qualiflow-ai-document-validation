from __future__ import annotations

from fastapi.testclient import TestClient


def test_unhandled_exception_does_not_leak_internal_details(client: TestClient):
    app = client.app

    @app.get("/__boom")
    def boom() -> None:
        raise RuntimeError("secret-db-password")

    response = client.get("/__boom")
    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Internal server error."
    assert "secret-db-password" not in response.text
    assert "request_id" in body
