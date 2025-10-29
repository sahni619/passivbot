"""Append-only audit trail utilities for the risk management dashboard."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_REDACT_FIELDS: tuple[str, ...] = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "private_key",
)


@dataclass(frozen=True)
class AuditS3Settings:
    """Configuration for offloading audit records to S3."""

    bucket: str
    prefix: str = ""
    region_name: Optional[str] = None
    profile_name: Optional[str] = None


@dataclass(frozen=True)
class AuditSyslogSettings:
    """Configuration for forwarding audit records to syslog."""

    address: str = "localhost"
    port: int = 514
    facility: str = "user"


@dataclass(frozen=True)
class AuditSettings:
    """Audit logging preferences loaded from the realtime configuration."""

    log_path: Path
    enabled: bool = True
    redact_fields: Sequence[str] = DEFAULT_REDACT_FIELDS
    s3: Optional[AuditS3Settings] = None
    syslog: Optional[AuditSyslogSettings] = None


class AuditSink:
    """Protocol-like base class for audit sinks."""

    def write(self, payload: str) -> None:  # pragma: no cover - interface method
        raise NotImplementedError


class FileAuditSink(AuditSink):
    """Persist audit records to a JSONL file on disk."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def bootstrap_hash(self) -> str:
        """Return the last stored hash from the log file, if any."""

        try:
            with self._path.open("r", encoding="utf-8") as handle:
                last_line = ""
                for line in handle:
                    line = line.strip()
                    if line:
                        last_line = line
        except FileNotFoundError:
            return "0" * 64
        except OSError as exc:  # pragma: no cover - unexpected filesystem failure
            logger.warning("Unable to read audit log %s: %s", self._path, exc)
            return "0" * 64
        if not last_line:
            return "0" * 64
        try:
            payload = json.loads(last_line)
        except json.JSONDecodeError:  # pragma: no cover - corrupted payload
            logger.error("Encountered invalid JSON in audit log %s", self._path)
            return "0" * 64
        return str(payload.get("hash") or "0" * 64)

    def write(self, payload: str) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(payload)


class S3AuditSink(AuditSink):
    """Send audit records to an S3 bucket as individual objects."""

    def __init__(self, settings: AuditS3Settings) -> None:
        try:
            import boto3  # type: ignore[import]
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "S3 audit offloading requires the 'boto3' package to be installed."
            ) from exc

        session_kwargs: Dict[str, Any] = {}
        if settings.profile_name:
            session_kwargs["profile_name"] = settings.profile_name
        session = boto3.session.Session(**session_kwargs)
        client_kwargs: Dict[str, Any] = {"service_name": "s3"}
        if settings.region_name:
            client_kwargs["region_name"] = settings.region_name
        self._client = session.client(**client_kwargs)
        self._settings = settings

    def write(self, payload: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        key_prefix = self._settings.prefix.rstrip("/")
        if key_prefix:
            key = f"{key_prefix}/{timestamp}.json"
        else:
            key = f"{timestamp}.json"
        try:
            self._client.put_object(
                Bucket=self._settings.bucket,
                Key=key,
                Body=payload.encode("utf-8"),
                ContentType="application/json",
            )
        except Exception as exc:  # pragma: no cover - network errors hard to reproduce
            logger.warning("Failed to offload audit record to S3: %s", exc)


class SyslogAuditSink(AuditSink):
    """Dispatch audit records to a syslog endpoint."""

    def __init__(self, settings: AuditSyslogSettings) -> None:
        from logging.handlers import SysLogHandler

        self._handler = SysLogHandler(
            address=(settings.address, int(settings.port)),
            facility=settings.facility,
        )

    def write(self, payload: str) -> None:
        record = logging.LogRecord(
            name="risk_management.audit",
            level=logging.INFO,
            pathname="audit",
            lineno=0,
            msg=payload,
            args=(),
            exc_info=None,
        )
        self._handler.handle(record)


class AuditLogWriter:
    """Append-only audit writer that maintains a hash chain."""

    def __init__(
        self,
        *,
        file_sink: FileAuditSink,
        redact_fields: Sequence[str],
        extra_sinks: Sequence[AuditSink] = (),
    ) -> None:
        self._file_sink = file_sink
        self._sinks = [file_sink, *extra_sinks]
        self._lock = threading.Lock()
        self._redact_keys = {self._normalise_key(field) for field in redact_fields}
        self._last_hash = file_sink.bootstrap_hash()

    @staticmethod
    def _normalise_key(key: str) -> str:
        return key.replace(" ", "").replace("-", "_").lower()

    def _redact(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            redacted: Dict[str, Any] = {}
            for key, item in value.items():
                norm_key = self._normalise_key(str(key))
                if any(field in norm_key for field in self._redact_keys):
                    redacted[key] = "<redacted>"
                else:
                    redacted[key] = self._redact(item)
            return redacted
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._redact(item) for item in value)
        if isinstance(value, set):
            return [self._redact(item) for item in value]
        return value

    def _canonical_payload(self, record: Mapping[str, Any]) -> str:
        return json.dumps(record, sort_keys=True, separators=(",", ":"))

    def log(self, action: str, actor: str, details: Mapping[str, Any] | None = None) -> str:
        """Append an audit record and return its hash."""

        if details is None:
            details = {}
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            base_record: Dict[str, Any] = {
                "timestamp": timestamp,
                "action": str(action),
                "actor": str(actor),
                "details": self._redact(dict(details)),
                "prev_hash": self._last_hash,
            }
            canonical = self._canonical_payload(base_record)
            record_hash = sha256(canonical.encode("utf-8")).hexdigest()
            base_record["hash"] = record_hash
            payload = json.dumps(base_record, sort_keys=True) + "\n"
            for sink in self._sinks:
                try:
                    sink.write(payload)
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.warning("Failed to write audit record via %s: %s", type(sink).__name__, exc)
            self._last_hash = record_hash
        return record_hash

    @property
    def log_path(self) -> Path:
        return self._file_sink.path


def iter_audit_entries(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield parsed audit records from ``path``."""

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:  # pragma: no cover - corrupted entry
                    logger.warning("Skipping invalid audit record: %s", line)
    except FileNotFoundError:
        return


def read_audit_entries(
    path: Path,
    *,
    limit: Optional[int] = None,
    action: Optional[str] = None,
    actor: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """Return filtered audit entries from ``path`` respecting ``limit``."""

    action_norm = action.lower() if action else None
    actor_norm = actor.lower() if actor else None
    results: list[Dict[str, Any]] = []
    for entry in iter_audit_entries(path):
        if action_norm and str(entry.get("action", "")).lower() != action_norm:
            continue
        if actor_norm and str(entry.get("actor", "")).lower() != actor_norm:
            continue
        results.append(entry)
    if limit is not None:
        return results[-limit:]
    return results


_audit_registry: Dict[Path, AuditLogWriter] = {}
_registry_lock = threading.Lock()


def reset_audit_registry() -> None:
    """Reset the cached audit writers. Intended for tests only."""

    with _registry_lock:
        _audit_registry.clear()


def get_audit_logger(settings: Optional[AuditSettings]) -> Optional[AuditLogWriter]:
    """Return a cached :class:`AuditLogWriter` for ``settings``."""

    if settings is None or not settings.enabled:
        return None
    log_path = settings.log_path
    with _registry_lock:
        writer = _audit_registry.get(log_path)
        if writer is not None:
            return writer
        extra_sinks: list[AuditSink] = []
        if settings.s3:
            try:
                extra_sinks.append(S3AuditSink(settings.s3))
            except Exception as exc:
                logger.warning("Unable to initialise S3 audit sink: %s", exc)
        if settings.syslog:
            try:
                extra_sinks.append(SyslogAuditSink(settings.syslog))
            except Exception as exc:
                logger.warning("Unable to initialise syslog audit sink: %s", exc)
        file_sink = FileAuditSink(log_path)
        writer = AuditLogWriter(
            file_sink=file_sink,
            redact_fields=settings.redact_fields,
            extra_sinks=tuple(extra_sinks),
        )
        _audit_registry[log_path] = writer
        return writer
