"""Tests for risk management web server helpers."""

from __future__ import annotations

import importlib
import subprocess
import sys
import types

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "uvicorn" not in sys.modules:
    uvicorn_stub = types.ModuleType("uvicorn")

    def _noop_run(*_args, **_kwargs) -> None:  # pragma: no cover - helper for import
        return None

    uvicorn_stub.run = _noop_run
    sys.modules["uvicorn"] = uvicorn_stub

from risk_management.configuration import AccountConfig, AuthConfig, RealtimeConfig  # noqa: E402
from risk_management.web_server import (  # noqa: E402
    _INVALID_HTTP_REQUEST_FILTER_NAME,
    _determine_uvicorn_logging,
    _apply_https_only_policy,
)


def _make_config(global_debug: bool = False, account_debug: bool = False) -> RealtimeConfig:
    account = AccountConfig(
        name="Example",
        exchange="binance",
        credentials={},
        debug_api_payloads=account_debug,
    )
    return RealtimeConfig(accounts=[account], debug_api_payloads=global_debug)


def test_determine_uvicorn_logging_defaults_when_disabled() -> None:
    config = _make_config()

    log_config, log_level = _determine_uvicorn_logging(config)

    assert log_config is None
    assert log_level == "info"


def test_determine_uvicorn_logging_uses_uvicorn_config(monkeypatch) -> None:
    dummy_logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {},
        "handlers": {"default": {"class": "logging.StreamHandler"}},
        "loggers": {"": {"handlers": ["default"], "level": "INFO"}},
    }
    uvicorn_module = types.ModuleType("uvicorn")
    uvicorn_config_module = types.ModuleType("uvicorn.config")
    uvicorn_config_module.LOGGING_CONFIG = dummy_logging_config

    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_module)
    monkeypatch.setitem(sys.modules, "uvicorn.config", uvicorn_config_module)

    config = _make_config(global_debug=True)

    log_config, log_level = _determine_uvicorn_logging(config)

    assert log_level == "debug"
    assert log_config["loggers"][""]["level"] == "DEBUG"
    assert log_config["loggers"]["risk_management"]["level"] == "DEBUG"
    assert _INVALID_HTTP_REQUEST_FILTER_NAME in log_config["filters"]
    assert _INVALID_HTTP_REQUEST_FILTER_NAME in log_config["loggers"]["uvicorn.error"]["filters"]


def test_determine_uvicorn_logging_handles_missing_uvicorn(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "uvicorn", raising=False)
    monkeypatch.delitem(sys.modules, "uvicorn.config", raising=False)

    original_import = importlib.import_module

    def fake_import(name, package=None):
        if name == "uvicorn.config":
            raise ModuleNotFoundError("uvicorn unavailable")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    config = _make_config(account_debug=True)

    log_config, log_level = _determine_uvicorn_logging(config)

    assert log_config is None
    assert log_level == "debug"


def test_apply_https_only_policy_disabled_when_tls_missing(caplog) -> None:
    config = _make_config()
    config.auth = AuthConfig(secret_key="secret", users={"user": "hash"}, https_only=True)

    enforced = _apply_https_only_policy(config, ssl_enabled=False)

    assert not enforced
    assert config.auth.https_only is False
    assert any("Disabling HTTPS enforcement" in message for message in caplog.messages)


def test_apply_https_only_policy_preserves_https_when_tls_available() -> None:
    config = _make_config()
    config.auth = AuthConfig(secret_key="secret", users={"user": "hash"}, https_only=True)

    enforced = _apply_https_only_policy(config, ssl_enabled=True)

    assert enforced is True
    assert config.auth.https_only is True


def test_apply_https_only_policy_ignored_when_not_requested(caplog) -> None:
    config = _make_config()
    config.auth = AuthConfig(secret_key="secret", users={"user": "hash"}, https_only=False)

    enforced = _apply_https_only_policy(config, ssl_enabled=False)

    assert not enforced
    assert config.auth.https_only is False
    assert not caplog.messages


def test_web_server_script_can_display_help() -> None:
    """Ensure the CLI entrypoint works when executed as a script."""

    # Use the repo root so ``risk_management`` is importable without installation.
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo_root / "risk_management" / "web_server.py"), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Launch the risk dashboard web UI" in result.stdout


def test_main_reports_missing_web_dependencies(monkeypatch, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "risk_management" / "realtime_config.json"

    import risk_management.web_server as web_server

    original_import = importlib.import_module

    def fake_import_module(name):
        if name == "risk_management.web":
            raise ModuleNotFoundError("fastapi")
        return original_import(name)

    monkeypatch.setattr(web_server.importlib, "import_module", fake_import_module)

    with pytest.raises(SystemExit) as excinfo:
        web_server.main(
            [
                "--config",
                str(config_path),
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ]
        )

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "fastapi" in captured.err
    assert "requirements.txt" in captured.err
