"""Notification helpers for realtime risk monitoring."""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone
from typing import Any, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from .audit import AuditLogWriter
from .configuration import RealtimeConfig
from .dashboard import evaluate_alerts, parse_snapshot
from .email_notifications import EmailAlertSender
from .telegram_notifications import TelegramNotifier
from .policies import PolicyActionState, PolicyEvaluationResult

__all__ = [
    "NotificationCoordinator",
]

logger = logging.getLogger(__name__)


LIMIT_LABELS = {
    "venue_concentration_pct": "venue concentration",
    "asset_concentration_pct": "asset concentration",
}


class NotificationCoordinator:
    """Coordinate email and telegram notifications for realtime snapshots."""

    def __init__(
        self,
        config: RealtimeConfig,
        *,
        audit_logger: Optional[AuditLogWriter] = None,
    ) -> None:
        self._email_sender = EmailAlertSender(config.email) if config.email else None
        self._email_recipients = self._extract_email_recipients(config.notification_channels)
        self._telegram_targets = self._extract_telegram_targets(config.notification_channels)
        self._telegram_notifier = TelegramNotifier() if self._telegram_targets else None
        self._active_alerts: set[str] = set()
        self._daily_snapshot_tz = ZoneInfo("America/New_York")
        self._daily_snapshot_sent_date: Optional[date] = None
        self._audit = audit_logger

    def _emit_audit(self, action: str, details: Mapping[str, Any]) -> None:
        if not self._audit:
            return
        try:
            self._audit.log(action=action, actor="system", details=dict(details))
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Failed to emit notification audit entry: %s", exc)

    @staticmethod
    def _extract_email_recipients(channels: Sequence[Any]) -> list[str]:
        recipients: list[str] = []
        for channel in channels:
            if not isinstance(channel, str):
                continue
            if channel.lower().startswith("email:"):
                address = channel.split(":", 1)[1].strip()
                if address:
                    recipients.append(address)
        return recipients

    @staticmethod
    def _extract_telegram_targets(channels: Sequence[Any]) -> list[tuple[str, str]]:
        targets: list[tuple[str, str]] = []
        for channel in channels:
            if not isinstance(channel, str):
                continue
            if not channel.lower().startswith("telegram:"):
                continue
            payload = channel.split(":", 1)[1]
            token = ""
            chat_id = ""
            if "@" in payload:
                token, _, chat_id = payload.partition("@")
            elif "/" in payload:
                token, _, chat_id = payload.partition("/")
            else:
                parts = payload.split(":", 1)
                if len(parts) == 2:
                    token, chat_id = parts
            token = token.strip()
            chat_id = chat_id.strip()
            if token and chat_id:
                targets.append((token, chat_id))
        return targets

    def send_daily_snapshot(self, snapshot: Mapping[str, Any], portfolio_balance: float) -> None:
        if not self._email_sender or not self._email_recipients:
            return
        now_ny = datetime.now(self._daily_snapshot_tz)
        current_date = now_ny.date()
        if self._daily_snapshot_sent_date and current_date > self._daily_snapshot_sent_date:
            self._daily_snapshot_sent_date = None
        if now_ny.time() < time(16, 0):
            return
        if self._daily_snapshot_sent_date == current_date:
            return
        accounts = snapshot.get("accounts", [])
        lines = [
            f"Daily portfolio snapshot ({now_ny.strftime('%Y-%m-%d')} 16:00 ET)",
            f"Total balance: ${portfolio_balance:,.2f}",
            "",
            "Accounts:",
        ]
        for account in accounts or []:
            if not isinstance(account, Mapping):
                continue
            name = str(account.get("name", "unknown"))
            balance = float(account.get("balance", 0.0))
            realised = float(account.get("daily_realized_pnl", 0.0))
            lines.append(
                f"- {name}: balance ${balance:,.2f}, daily realised PnL ${realised:,.2f}"
            )
        body = "\n".join(lines)
        subject = "Daily portfolio balance snapshot"
        self._email_sender.send(subject, body, self._email_recipients)
        self._daily_snapshot_sent_date = current_date
        self._emit_audit(
            "notification.email.daily_snapshot",
            {
                "subject": subject,
                "recipient_count": len(self._email_recipients),
            },
        )

    def dispatch_alerts(self, snapshot: Mapping[str, Any]) -> None:
        if not (self._email_sender or self._telegram_notifier):
            return
        try:
            _, accounts, thresholds, _ = parse_snapshot(dict(snapshot))
            alerts = evaluate_alerts(accounts, thresholds)
        except Exception as exc:  # pragma: no cover - snapshot parsing errors are logged for diagnostics
            logger.debug("Skipping email alert dispatch due to parsing error: %s", exc, exc_info=True)
            return
        limit_alerts: list[str] = []
        accounts_payload = snapshot.get("accounts", [])
        for account_entry in accounts_payload or []:
            if not isinstance(account_entry, Mapping):
                continue
            name = str(account_entry.get("name", "unknown"))
            metadata = account_entry.get("metadata") if isinstance(account_entry.get("metadata"), Mapping) else {}
            breaches = metadata.get("limit_breaches") if isinstance(metadata, Mapping) else None
            if not isinstance(breaches, Mapping):
                breaches = account_entry.get("limit_breaches") if isinstance(account_entry.get("limit_breaches"), Mapping) else None
            if not isinstance(breaches, Mapping):
                continue
            for metric, breach in breaches.items():
                if not isinstance(breach, Mapping) or not breach.get("breached"):
                    continue
                value = breach.get("value")
                limit = breach.get("limit")
                try:
                    value_float = float(value)
                except (TypeError, ValueError):
                    continue
                limit_float: Optional[float] = None
                if limit is not None:
                    try:
                        limit_float = float(limit)
                    except (TypeError, ValueError):
                        limit_float = None
                label = LIMIT_LABELS.get(metric, metric.replace("_", " "))
                if limit_float is not None:
                    message = f"{name}: {label} {value_float:.2%} exceeds limit {limit_float:.2%}"
                else:
                    message = f"{name}: {label} {value_float:.2%} exceeds limit"
                limit_alerts.append(message)
        alerts.extend(limit_alerts)
        alerts_set = set(alerts)
        new_alerts = [alert for alert in alerts if alert not in self._active_alerts]
        self._active_alerts = alerts_set
        if not new_alerts:
            return
        generated_at = snapshot.get("generated_at")
        timestamp = (
            generated_at
            if isinstance(generated_at, str)
            else datetime.now(timezone.utc).isoformat()
        )
        lines = [f"Exposure thresholds were exceeded at {timestamp}.", "", "Alerts:"]
        lines.extend(f"- {alert}" for alert in new_alerts)
        body = "\n".join(lines)
        subject = "Risk alert: exposure threshold breached"
        if self._email_sender and self._email_recipients:
            self._email_sender.send(subject, body, self._email_recipients)
            self._emit_audit(
                "notification.email.alert",
                {
                    "subject": subject,
                    "alert_count": len(new_alerts),
                    "recipient_count": len(self._email_recipients),
                },
            )
        if self._telegram_notifier and self._telegram_targets:
            message = f"Exposure alert at {timestamp}\n" + "\n".join(new_alerts)
            for token, chat_id in self._telegram_targets:
                self._telegram_notifier.send(token, chat_id, message)
            self._emit_audit(
                "notification.telegram.alert",
                {
                    "alert_count": len(new_alerts),
                    "destination_count": len(self._telegram_targets),
                },
            )

    def handle_policy_evaluations(self, result: PolicyEvaluationResult) -> None:
        """Log policy actions and forward notifications to downstream channels."""

        if not result.evaluations:
            return

        for action in result.executed_actions:
            message = action.rendered_message or action.message_template
            if not message:
                value = action.trigger_value
                threshold = action.threshold
                message = (
                    f"Action {action.config.type} executed"
                    f" (value={value!r}, threshold={threshold!r})"
                )
            self._log_policy_action(action, message)
            if action.config.type in {"notify", "require_confirmation"}:
                payload = message
                if action.config.type == "require_confirmation" and action.confirmation_key:
                    payload = f"{message}\nConfirmation key: {action.confirmation_key}"
                self._dispatch_policy_notification(action, payload)

        for pending in result.pending_confirmations:
            logger.warning(
                "Policy %s awaiting confirmation (key=%s)",
                pending.policy_name,
                pending.confirmation_key or "unspecified",
            )

    def _log_policy_action(self, action: PolicyActionState, message: str) -> None:
        log_message = f"Policy {action.policy_name}: {message}"
        severity = (action.severity or "info").lower()
        if severity in {"warning", "warn"}:
            logger.warning(log_message)
        elif severity in {"error", "critical", "fatal"}:
            logger.error(log_message)
        elif severity == "debug":
            logger.debug(log_message)
        else:
            logger.info(log_message)

    def _dispatch_policy_notification(
        self, action: PolicyActionState, message: str
    ) -> None:
        channels = [
            str(channel).strip().lower()
            for channel in action.channels
            if str(channel).strip()
        ]
        if not channels:
            channels = ["email", "telegram"]
        subject = action.config.subject or f"Policy triggered: {action.policy_name}"

        if "email" in channels and self._email_sender and self._email_recipients:
            self._email_sender.send(subject, message, self._email_recipients)
        if "telegram" in channels and self._telegram_notifier and self._telegram_targets:
            for token, chat_id in self._telegram_targets:
                self._telegram_notifier.send(token, chat_id, message)

        for channel in channels:
            if channel in {"email", "telegram"}:
                continue
            logger.info(
                "Policy %s requested unsupported notification channel '%s'",
                action.policy_name,
                channel,
            )

    @property
    def email_sender(self) -> Optional[EmailAlertSender]:
        return self._email_sender

    @property
    def email_recipients(self) -> Sequence[str]:
        return tuple(self._email_recipients)

    @property
    def telegram_targets(self) -> Sequence[tuple[str, str]]:
        return tuple(self._telegram_targets)

