import json
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_management.audit import (
    AuditSettings,
    get_audit_logger,
    reset_audit_registry,
)


def test_audit_log_hash_chain(tmp_path):
    reset_audit_registry()
    log_path = tmp_path / "audit.log"
    settings = AuditSettings(log_path=log_path)

    writer = get_audit_logger(settings)
    assert writer is not None

    first_hash = writer.log("event.one", "alice", {"foo": "bar"})
    second_hash = writer.log("event.two", "bob", {"foo": "baz"})

    payloads = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert payloads[0]["hash"] == first_hash
    assert payloads[1]["hash"] == second_hash
    assert payloads[1]["prev_hash"] == payloads[0]["hash"]

    canonical = json.dumps(
        {
            "timestamp": payloads[0]["timestamp"],
            "action": payloads[0]["action"],
            "actor": payloads[0]["actor"],
            "details": payloads[0]["details"],
            "prev_hash": payloads[0]["prev_hash"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assert payloads[0]["hash"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_audit_log_redacts_sensitive_fields(tmp_path):
    reset_audit_registry()
    log_path = tmp_path / "audit.log"
    settings = AuditSettings(log_path=log_path, redact_fields=("token", "password"))

    writer = get_audit_logger(settings)
    assert writer is not None

    writer.log(
        "event.redact",
        "system",
        {
            "token": "should-hide",
            "nested": {"password": "super-secret", "other": "visible"},
            "list": [{"apiToken": "secret"}],
        },
    )

    payloads = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert payloads
    details = payloads[0]["details"]
    assert details["token"] == "<redacted>"
    assert details["nested"]["password"] == "<redacted>"
    assert details["nested"]["other"] == "visible"
    assert details["list"][0]["apiToken"] == "<redacted>"
