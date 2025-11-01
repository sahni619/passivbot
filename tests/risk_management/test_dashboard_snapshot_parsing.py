import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from risk_management.dashboard import parse_snapshot
from risk_management.domain.models import AlertThresholds


def test_parse_snapshot_skips_accounts_missing_required_fields(caplog):
    generated_at = "2024-07-01T12:00:00Z"
    snapshot = {
        "generated_at": generated_at,
        "accounts": [
            {},
            {"name": "incomplete"},
            {
                "name": "valid",
                "balance": 1000,
                "positions": [],
                "open_orders": [],
            },
        ],
        "alert_thresholds": {},
        "notification_channels": ["email"],
    }

    with caplog.at_level(logging.WARNING):
        parsed_generated_at, accounts, thresholds, notifications = parse_snapshot(snapshot)

    assert parsed_generated_at == datetime(2024, 7, 1, 12, 0, tzinfo=timezone.utc)
    assert len(accounts) == 1
    assert accounts[0].name == "valid"
    defaults = AlertThresholds()
    assert thresholds.wallet_exposure_pct == defaults.wallet_exposure_pct
    assert notifications == ["email"]


    assert any("Skipping account" in message for message in caplog.messages)

    messages = "".join(record.message for record in caplog.records)
    assert "Skipping account" in messages



def test_parse_snapshot_ignores_non_mapping_accounts(caplog):
    snapshot = {
        "generated_at": "2024-07-01T12:00:00Z",
        "accounts": ["not-a-dict", []],
    }

    with caplog.at_level(logging.WARNING):
        _, accounts, _, _ = parse_snapshot(snapshot)

    assert accounts == []

    assert any("not a mapping" in message for message in caplog.messages)


def test_parse_snapshot_logs_when_accounts_payload_not_iterable(caplog):
    snapshot = {
        "generated_at": "2024-07-01T12:00:00Z",
        "accounts": "invalid",
    }

    with caplog.at_level(logging.WARNING):
        _, accounts, _, _ = parse_snapshot(snapshot)

    assert accounts == []
    assert any("not an iterable of mappings" in message for message in caplog.messages)

    messages = "".join(record.message for record in caplog.records)
    assert "not a mapping" in messages

