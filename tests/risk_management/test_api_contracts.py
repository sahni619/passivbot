import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from risk_management.configuration import AccountConfig, AuthConfig, RealtimeConfig
from risk_management.web import AuthManager, create_app


class _SimpleAuth(AuthManager):
    def __init__(self) -> None:
        super().__init__(secret_key="secret", users={"admin": "pw"}, https_only=False)

    def authenticate(self, username: str, password: str) -> bool:  # type: ignore[override]
        return self.users.get(username) == password


class _StubController:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def fetch_presentable_snapshot(self):
        self.calls.append("snapshot")
        return {"status": "ok"}

    async def list_order_types(self, account_name: str):
        self.calls.append(f"list:{account_name}")
        return ["limit", "market"]


@pytest.fixture
def app(tmp_path: Path) -> TestClient:
    config = RealtimeConfig(
        accounts=[AccountConfig(name="demo", exchange="binance")],
        auth=AuthConfig(secret_key="secret", users={"admin": "pw"}, https_only=False),
        config_root=tmp_path,
    )
    app = create_app(config, auth_manager=_SimpleAuth())
    app.state.controller = _StubController()
    client = TestClient(app)
    login = client.post("/login", data={"username": "admin", "password": "pw"})
    assert login.status_code in {200, 303}
    return client


def test_snapshot_endpoint_uses_controller(app: TestClient):
    response = app.get("/api/snapshot")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "snapshot" in app.app.state.controller.calls


def test_list_order_types_uses_controller(app: TestClient):
    response = app.get("/api/trading/accounts/demo/order-types")
    assert response.status_code == 200
    body = response.json()
    assert body["order_types"] == ["limit", "market"]
    assert "list:demo" in app.app.state.controller.calls
