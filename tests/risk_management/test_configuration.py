"""Tests for realtime risk management configuration helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_management.configuration import _merge_credentials, _normalise_credentials, load_realtime_config


def _write_config(tmp_path: Path, payload: dict) -> Path:
    config_path = tmp_path / "configs" / "risk" / "realtime.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def _base_payload() -> dict:
    return {
        "accounts": [
            {
                "name": "Example",
                "exchange": "binanceusdm",
                "credentials": {"key": "abc", "secret": "def"},
            }
        ],
    }


def test_load_realtime_config_requires_object_top_level(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(TypeError, match="Realtime configuration must be a JSON object"):
        load_realtime_config(config_path)


def test_custom_endpoint_path_resolves_relative(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["custom_endpoints"] = {"path": "../custom_endpoints.json", "autodiscover": False}
    config_path = _write_config(tmp_path, payload)

    config = load_realtime_config(config_path)

    expected_path = (config_path.parent / Path("../custom_endpoints.json")).resolve()
    assert config.custom_endpoints is not None
    assert config.custom_endpoints.path == str(expected_path)


def test_custom_endpoint_path_keeps_absolute(tmp_path: Path) -> None:
    payload = _base_payload()
    absolute_path = tmp_path / "custom" / "endpoints.json"
    payload["custom_endpoints"] = {"path": str(absolute_path), "autodiscover": True}
    config_path = _write_config(tmp_path, payload)

    config = load_realtime_config(config_path)

    assert config.custom_endpoints is not None
    assert config.custom_endpoints.path == str(absolute_path.resolve())
    assert config.custom_endpoints.autodiscover is True


def test_load_realtime_config_requires_mapping_api_keys(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["api_keys_file"] = "../api-keys.json"
    config_path = _write_config(tmp_path, payload)

    api_keys_path = config_path.parent.parent / "api-keys.json"
    api_keys_path.write_text(json.dumps([{"key": "value"}]), encoding="utf-8")

    with pytest.raises(TypeError, match="API key configuration must be a JSON object"):
        load_realtime_config(config_path)


def test_normalise_credentials_supports_aliases() -> None:
    payload = {
        "key": " key-value ",
        "api_secret": " secret-value ",
        "passPhrase": " pass ",
        "uid": " 123 ",
        "exchange": "binance",
        "headers": {"X-Test": "1"},
        "options": {"defaultType": "swap"},
        "ccxt_config": {"login": "demo"},
        "wallet_address": " wallet ",
        "private_key": " private ",
    }

    normalised = _normalise_credentials(payload)

    assert normalised == {
        "apiKey": "key-value",
        "secret": "secret-value",
        "password": "pass",
        "uid": "123",
        "headers": {"X-Test": "1"},
        "options": {"defaultType": "swap"},
        "ccxt": {"login": "demo"},
        "walletAddress": "wallet",
        "privateKey": "private",
    }


def test_merge_credentials_prioritises_primary_values() -> None:
    primary = {"apiKey": "primary", "headers": {"X-Primary": "1"}}
    secondary = {"key": "secondary", "headers": {"X-Secondary": "2"}, "exchange": "binance"}

    merged = _merge_credentials(primary, secondary)

    assert merged["apiKey"] == "primary"
    assert merged["headers"] == {"X-Secondary": "2", "X-Primary": "1"}
    assert "exchange" not in merged


def test_load_realtime_config_supports_nested_user_entries(tmp_path: Path) -> None:
    api_keys_path = tmp_path / "api-keys.json"
    api_keys_path.write_text(
        json.dumps(
            {
                "referrals": {"binance": "https://example.com"},
                "binance_01": {"exchange": "binance", "key": "a", "secret": "b"},
                "users": {
                    "okx_01": {
                        "exchange": "okx",
                        "key": "c",
                        "secret": "d",
                        "passphrase": "p",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "realtime.json"
    config_path.write_text(
        json.dumps(
            {
                "api_keys_file": "../api-keys.json",
                "accounts": [
                    {
                        "name": "Binance",
                        "api_key_id": "binance_01",
                        "exchange": "binance",
                    },
                    {
                        "name": "OKX",
                        "api_key_id": "okx_01",
                        "exchange": "okx",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    config = load_realtime_config(config_path)

    assert len(config.accounts) == 2
    binance = config.accounts[0]
    okx = config.accounts[1]

    assert binance.credentials["apiKey"] == "a"
    assert binance.credentials["secret"] == "b"

    assert okx.credentials["apiKey"] == "c"
    assert okx.credentials["secret"] == "d"
    assert okx.credentials["password"] == "p"
    assert config.config_root == config_path.parent.resolve()



def test_load_realtime_config_expands_user_path(tmp_path: Path, monkeypatch) -> None:
    home_api_keys = tmp_path / "api-keys.json"
    home_api_keys.write_text(
        json.dumps({"binance": {"exchange": "binance", "key": "x", "secret": "y"}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("HOME", str(tmp_path))

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "api_keys_file": "~/api-keys.json",
                "accounts": [
                    {
                        "name": "Binance",
                        "api_key_id": "binance",
                        "exchange": "binance",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = load_realtime_config(config_path)
    assert config.accounts[0].credentials["apiKey"] == "x"
    assert config.config_root == config_path.parent.resolve()


def test_load_realtime_config_propagates_account_messages(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["account_messages"] = {"Example": "Healthy", "Other": None}
    config_path = _write_config(tmp_path, payload)

    config = load_realtime_config(config_path)

    assert config.account_messages == {"Example": "Healthy"}


def test_auth_https_only_flag_respected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["auth"] = {
        "secret_key": "abc",
        "users": {"demo": "hashed"},
        "https_only": False,
    }

    config_path = _write_config(tmp_path, payload)

    config = load_realtime_config(config_path)

    assert config.auth is not None
    assert config.auth.https_only is False


def test_auth_missing_secret_rejected(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["auth"] = {"users": {"demo": "hashed"}}
    config_path = _write_config(tmp_path, payload)

    with pytest.raises(ValueError, match="secret_key"):
        load_realtime_config(config_path)


def test_alert_threshold_defaults_applied(tmp_path: Path) -> None:
    payload = _base_payload()
    config_path = _write_config(tmp_path, payload)

    config = load_realtime_config(config_path)

    assert config.alert_thresholds.wallet_exposure_pct == 0.6
    assert config.alert_thresholds.loss_threshold_pct == -0.12


def test_notification_channels_validate_sequence(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["notification_channels"] = "not-a-list"
    config_path = _write_config(tmp_path, payload)

    with pytest.raises(TypeError, match="notification_channels"):
        load_realtime_config(config_path)


def test_load_realtime_config_parses_email_settings(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["email"] = {
        "host": "smtp.example.com",
        "port": 2525,
        "username": "alerts@example.com",
        "password": "secret",
        "sender": "alerts@example.com",
        "use_tls": False,
        "use_ssl": True,
    }
    config_path = _write_config(tmp_path, payload)

    config = load_realtime_config(config_path)

    assert config.email is not None
    assert config.email.host == "smtp.example.com"
    assert config.email.port == 2525
    assert config.email.username == "alerts@example.com"
    assert config.email.password == "secret"
    assert config.email.sender == "alerts@example.com"
    assert config.email.use_tls is False
    assert config.email.use_ssl is True


def test_load_realtime_config_discovers_api_keys_file(tmp_path: Path) -> None:
    repo_root = tmp_path / "passivbot"
    repo_root.mkdir()

    api_keys_path = repo_root / "api-keys.json"
    api_keys_path.write_text(
        json.dumps({"binance": {"exchange": "binance", "key": "auto", "secret": "secret"}}),
        encoding="utf-8",
    )

    config_dir = repo_root / "risk_management"
    config_dir.mkdir()
    config_path = config_dir / "realtime.json"
    config_path.write_text(
        json.dumps(
            {
                "accounts": [
                    {
                        "name": "Binance",
                        "exchange": "binance",
                        "api_key_id": "binance",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    config = load_realtime_config(config_path)

    assert config.accounts[0].credentials["apiKey"] == "auto"

