"""Policy evaluation helpers for realtime risk monitoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .configuration import (
    ManualOverrideConfig,
    PolicyActionConfig,
    PolicyConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class PolicyActionState:
    """Represent the state of a policy action for rendering and notifications."""

    policy_name: str
    config: PolicyActionConfig
    status: str
    trigger_value: Optional[float]
    threshold: Optional[float]
    rendered_message: Optional[str]
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def requires_confirmation(self) -> bool:
        return bool(self.config.requires_confirmation)

    @property
    def channels(self) -> Sequence[str]:
        return tuple(self.config.channels)

    @property
    def severity(self) -> str:
        return self.config.severity

    @property
    def confirmation_key(self) -> Optional[str]:
        return self.config.confirmation_key

    @property
    def message_template(self) -> Optional[str]:
        return self.config.message


@dataclass
class PolicyEvaluation:
    """The outcome of evaluating a single policy against a snapshot."""

    config: PolicyConfig
    triggered: bool
    trigger_value: Optional[float]
    actions: List[PolicyActionState] = field(default_factory=list)
    cooldown_active: bool = False
    override_active: bool = False
    override_payload: Optional[Dict[str, Any]] = None

    @property
    def threshold(self) -> Optional[float]:
        return self.config.trigger.value

    @property
    def operator(self) -> str:
        return self.config.trigger.operator


@dataclass
class PolicyEvaluationResult:
    """Aggregated evaluation output for all configured policies."""

    evaluations: List[PolicyEvaluation]

    @property
    def executed_actions(self) -> List[PolicyActionState]:
        return [
            action
            for evaluation in self.evaluations
            for action in evaluation.actions
            if action.status == "triggered"
        ]

    @property
    def pending_confirmations(self) -> List[PolicyActionState]:
        return [
            action
            for action in self.executed_actions
            if action.requires_confirmation
        ]

    def to_payload(self) -> Dict[str, Any]:
        evaluations_payload: List[Dict[str, Any]] = []
        active: List[Dict[str, Any]] = []
        pending: List[Dict[str, Any]] = []
        overrides: List[Dict[str, Any]] = []

        for evaluation in self.evaluations:
            actions_payload: List[Dict[str, Any]] = []
            for action in evaluation.actions:
                message = action.rendered_message or action.message_template
                actions_payload.append(
                    {
                        "type": action.config.type,
                        "status": action.status,
                        "message": message,
                        "requires_confirmation": action.requires_confirmation,
                        "severity": action.severity,
                        "channels": list(action.channels),
                        "confirmation_key": action.confirmation_key,
                    }
                )
                if action.requires_confirmation and action.status == "triggered":
                    pending.append(
                        {
                            "policy": evaluation.config.name,
                            "action": action.config.type,
                            "message": message,
                            "confirmation_key": action.confirmation_key,
                        }
                    )

            override_payload = evaluation.override_payload
            if override_payload:
                overrides.append({"policy": evaluation.config.name, **override_payload})

            evaluations_payload.append(
                {
                    "name": evaluation.config.name,
                    "description": evaluation.config.description,
                    "triggered": evaluation.triggered,
                    "metric": evaluation.config.trigger.metric,
                    "operator": evaluation.operator,
                    "threshold": evaluation.threshold,
                    "value": evaluation.trigger_value,
                    "cooldown_active": evaluation.cooldown_active,
                    "override_active": evaluation.override_active,
                    "actions": actions_payload,
                    "manual_override": override_payload,
                }
            )

            if evaluation.triggered:
                active.append(
                    {
                        "name": evaluation.config.name,
                        "value": evaluation.trigger_value,
                        "threshold": evaluation.threshold,
                        "operator": evaluation.operator,
                    }
                )

        payload: Dict[str, Any] = {"evaluations": evaluations_payload}
        if active:
            payload["active"] = active
        if pending:
            payload["pending_actions"] = pending
        if overrides:
            payload["manual_overrides"] = overrides
        return payload


class PolicyEvaluator:
    """Evaluate configured policies against realtime snapshots."""

    def __init__(self, policies: Sequence[PolicyConfig]):
        self._policies = list(policies)
        self._policies_by_name = {policy.name: policy for policy in self._policies}
        self._last_triggered: Dict[str, datetime] = {}
        self._overrides: Dict[str, Dict[str, Any]] = {}

    @property
    def policies(self) -> Sequence[PolicyConfig]:
        return tuple(self._policies)

    def evaluate(self, snapshot: Mapping[str, Any]) -> PolicyEvaluationResult:
        results: List[PolicyEvaluation] = []
        now = datetime.now(timezone.utc)

        for policy in self._policies:
            value = self._resolve_metric(snapshot, policy.trigger.metric)
            triggered = False
            if value is not None:
                triggered = self._compare(value, policy.trigger.operator, policy.trigger.value)

            override_state = self._resolve_override(policy.name, now)
            override_active = override_state is not None

            should_execute = triggered and not override_active
            cooldown_active = False
            if should_execute and policy.trigger.cooldown_seconds:
                last = self._last_triggered.get(policy.name)
                if last is not None:
                    elapsed = (now - last).total_seconds()
                    if elapsed < policy.trigger.cooldown_seconds:
                        should_execute = False
                        cooldown_active = True

            if should_execute:
                self._last_triggered[policy.name] = now

            actions = self._build_actions(
                policy,
                value,
                snapshot,
                should_execute,
                triggered,
                cooldown_active,
                override_active,
            )

            override_payload = self._build_override_payload(policy.manual_override, override_state)

            evaluation = PolicyEvaluation(
                config=policy,
                triggered=triggered,
                trigger_value=value,
                actions=actions,
                cooldown_active=cooldown_active,
                override_active=override_active,
                override_payload=override_payload,
            )
            results.append(evaluation)

        return PolicyEvaluationResult(results)

    def set_manual_override(
        self,
        policy_name: str,
        *,
        reason: Optional[str] = None,
        expires_after_seconds: Optional[int] = None,
    ) -> None:
        policy = self._policies_by_name.get(policy_name)
        if policy is None:
            raise KeyError(f"Unknown policy '{policy_name}'")

        expires_after = expires_after_seconds
        if expires_after is None and policy.manual_override:
            expires_after = policy.manual_override.expires_after_seconds

        expires_at: Optional[datetime] = None
        if expires_after:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_after)

        self._overrides[policy_name] = {
            "reason": reason,
            "expires_at": expires_at,
        }

    def clear_manual_override(self, policy_name: str) -> None:
        self._overrides.pop(policy_name, None)

    def _resolve_override(
        self, policy_name: str, now: datetime
    ) -> Optional[Dict[str, Any]]:
        state = self._overrides.get(policy_name)
        if not state:
            return None
        expires_at = state.get("expires_at")
        if isinstance(expires_at, datetime) and expires_at <= now:
            self._overrides.pop(policy_name, None)
            return None
        return state

    def _build_override_payload(
        self,
        config: Optional[ManualOverrideConfig],
        state: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not config and not state:
            return None
        payload = {
            "allowed": bool(config.allowed if config else False),
            "instructions": config.instructions if config else None,
            "expires_after_seconds": config.expires_after_seconds if config else None,
            "active": bool(state),
        }
        if state:
            if state.get("reason"):
                payload["reason"] = state["reason"]
            expires_at = state.get("expires_at")
            if isinstance(expires_at, datetime):
                payload["expires_at"] = expires_at.isoformat()
        return payload

    def _build_actions(
        self,
        policy: PolicyConfig,
        value: Optional[float],
        snapshot: Mapping[str, Any],
        should_execute: bool,
        triggered: bool,
        cooldown_active: bool,
        override_active: bool,
    ) -> List[PolicyActionState]:
        actions: List[PolicyActionState] = []
        if not policy.actions:
            return actions

        for action in policy.actions:
            status = "idle"
            if should_execute:
                status = "triggered"
            elif triggered and override_active:
                status = "overridden"
            elif triggered and cooldown_active:
                status = "cooldown"

            message = self._render_message(
                action.message,
                policy=policy,
                value=value,
                threshold=policy.trigger.value,
                snapshot=snapshot,
                status=status,
            )

            actions.append(
                PolicyActionState(
                    policy_name=policy.name,
                    config=action,
                    status=status,
                    trigger_value=value,
                    threshold=policy.trigger.value,
                    rendered_message=message,
                    context={"status": status, "metric": policy.trigger.metric},
                )
            )

        return actions

    @staticmethod
    def _compare(value: float, operator: str, threshold: float) -> bool:
        if operator in {">", "gt"}:
            return value > threshold
        if operator in {">=", "gte"}:
            return value >= threshold
        if operator in {"<", "lt"}:
            return value < threshold
        if operator in {"<=", "lte"}:
            return value <= threshold
        if operator in {"==", "="}:
            return value == threshold
        if operator in {"!=", "<>"}:
            return value != threshold
        # Default to >= when operator is unknown to avoid silent failure
        logger.debug("Unknown operator '%s', defaulting to '>=' comparison", operator)
        return value >= threshold

    def _resolve_metric(self, snapshot: Mapping[str, Any], metric: str) -> Optional[float]:
        metric = metric.strip()
        if not metric:
            return None

        try:
            scope, *rest = metric.split(".")
        except ValueError:
            return None

        if scope == "portfolio":
            return self._resolve_portfolio_metric(snapshot, rest)
        if scope == "account":
            return self._resolve_account_metric(snapshot, rest)
        if scope == "position":
            return self._resolve_position_metric(snapshot, rest)
        return None

    def _resolve_portfolio_metric(
        self, snapshot: Mapping[str, Any], parts: Sequence[str]
    ) -> Optional[float]:
        if not parts:
            return None
        key = parts[0]
        accounts = self._extract_accounts(snapshot)
        if key == "balance":
            return sum(account.get("balance", 0.0) for account in accounts)
        if key == "drawdown_pct":
            stop_loss = snapshot.get("portfolio_stop_loss")
            if isinstance(stop_loss, Mapping):
                value = stop_loss.get("current_drawdown_pct")
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
            return None
        if key == "daily_realized_pnl":
            return sum(account.get("daily_realized_pnl", 0.0) for account in accounts)
        return None

    def _resolve_account_metric(
        self, snapshot: Mapping[str, Any], parts: Sequence[str]
    ) -> Optional[float]:
        if len(parts) < 2:
            return None
        account_name = parts[0]
        metric = parts[1]
        accounts = self._extract_accounts(snapshot)
        account = next(
            (entry for entry in accounts if str(entry.get("name")) == account_name),
            None,
        )
        if account is None:
            return None
        if metric == "balance":
            return _safe_float(account.get("balance"))
        if metric == "daily_realized_pnl":
            return _safe_float(account.get("daily_realized_pnl"))
        if metric == "drawdown_pct":
            stop_losses = snapshot.get("account_stop_losses")
            if isinstance(stop_losses, Mapping):
                state = stop_losses.get(account_name)
                if isinstance(state, Mapping):
                    value = state.get("current_drawdown_pct")
                    if value is not None:
                        return _safe_float(value)
        return None

    def _resolve_position_metric(
        self, snapshot: Mapping[str, Any], parts: Sequence[str]
    ) -> Optional[float]:
        if len(parts) < 3:
            return None
        account_name = parts[0]
        symbol = parts[1]
        metric = parts[2]
        accounts = self._extract_accounts(snapshot)
        account = next(
            (entry for entry in accounts if str(entry.get("name")) == account_name),
            None,
        )
        if account is None:
            return None
        positions = account.get("positions")
        if not isinstance(positions, Iterable):
            return None
        position = next(
            (
                pos
                for pos in positions
                if isinstance(pos, Mapping)
                and str(pos.get("symbol")) == symbol
            ),
            None,
        )
        if position is None:
            return None
        if metric == "wallet_exposure_pct":
            return _safe_float(position.get("wallet_exposure_pct"))
        if metric == "unrealized_pnl":
            return _safe_float(position.get("unrealized_pnl"))
        return None

    @staticmethod
    def _extract_accounts(snapshot: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        accounts_raw = snapshot.get("accounts")
        if not isinstance(accounts_raw, Iterable):
            return []
        accounts: List[Mapping[str, Any]] = []
        for entry in accounts_raw:
            if isinstance(entry, Mapping):
                accounts.append(entry)
        return accounts

    def _render_message(
        self,
        template: Optional[str],
        *,
        policy: PolicyConfig,
        value: Optional[float],
        threshold: Optional[float],
        snapshot: Mapping[str, Any],
        status: str,
    ) -> Optional[str]:
        if template is None:
            return None
        context = {
            "policy": policy.name,
            "metric": policy.trigger.metric,
            "value": value,
            "threshold": threshold,
            "status": status,
        }
        try:
            return template.format(**context)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug(
                "Failed to format policy action message for %s: %s",
                policy.name,
                exc,
                exc_info=True,
            )
            return template


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
