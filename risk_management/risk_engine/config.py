"""Configuration schema for the refactored risk engine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from risk_management.configuration import RealtimeConfig


@dataclass
class ThresholdConfig:
    """Alert and enforcement thresholds expressed as drawdown ratios."""

    alert_drawdown: float = -0.03
    close_positions_drawdown: float = -0.08
    kill_bots_drawdown: float = -0.12


@dataclass
class ActionConfig:
    """Action execution safety rails."""

    dry_run: bool = False
    close_cooldown_seconds: int = 900
    kill_cooldown_seconds: int = 1800
    max_close_notional: Optional[float] = None
    backoff_seconds: float = 5.0
    max_backoff_seconds: float = 60.0
    close_retry_attempts: int = 3


@dataclass
class RiskEngineConfig:
    """Unified configuration for risk rules and execution layers."""

    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    actions: ActionConfig = field(default_factory=ActionConfig)
    state_path: Optional[Path] = None

    @classmethod
    def from_realtime_config(cls, config: RealtimeConfig) -> "RiskEngineConfig":
        """Populate defaults from the existing realtime configuration."""

        thresholds = ThresholdConfig()
        # ``loss_threshold_pct`` is historically represented as a decimal (e.g., -0.08).
        loss_threshold = config.alert_thresholds.get("loss_threshold_pct") if isinstance(
            getattr(config, "alert_thresholds", None), dict
        ) else None
        if isinstance(loss_threshold, (int, float)):
            thresholds.close_positions_drawdown = float(loss_threshold)
            # Use a less aggressive alert threshold when only one value is supplied.
            thresholds.alert_drawdown = float(loss_threshold) / 2
        max_drawdown = config.alert_thresholds.get("max_drawdown_pct") if isinstance(
            getattr(config, "alert_thresholds", None), dict
        ) else None
        if isinstance(max_drawdown, (int, float)) and max_drawdown > 0:
            # Convert positive percentage into negative drawdown ratio.
            thresholds.kill_bots_drawdown = -abs(float(max_drawdown))

        reports_dir = config.reports_dir
        state_path: Optional[Path] = None
        if reports_dir:
            state_path = Path(reports_dir) / "risk_state.json"

        actions = ActionConfig()
        return cls(thresholds=thresholds, actions=actions, state_path=state_path)


@dataclass
class Settings:
    """Single entry point for risk configuration with environment overrides."""

    risk: RiskEngineConfig
    exchange_modes: Dict[str, bool] = field(default_factory=dict)
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    @classmethod
    def from_environment(
        cls, *, realtime: Optional[RealtimeConfig] = None, env: Optional[Dict[str, str]] = None
    ) -> "Settings":
        env = env or os.environ
        realtime = realtime or RealtimeConfig()
        base = RiskEngineConfig.from_realtime_config(realtime)

        alert_dd = _env_float(env.get("RISK_ALERT_DRAWDOWN"))
        close_dd = _env_float(env.get("RISK_CLOSE_DRAWDOWN"))
        kill_dd = _env_float(env.get("RISK_KILL_DRAWDOWN"))
        if alert_dd is not None:
            base.thresholds.alert_drawdown = alert_dd
        if close_dd is not None:
            base.thresholds.close_positions_drawdown = close_dd
        if kill_dd is not None:
            base.thresholds.kill_bots_drawdown = kill_dd

        dry_run = _env_bool(env.get("RISK_DRY_RUN"))
        if dry_run is not None:
            base.actions.dry_run = dry_run
        close_cooldown = _env_int(env.get("RISK_CLOSE_COOLDOWN_SECONDS"))
        kill_cooldown = _env_int(env.get("RISK_KILL_COOLDOWN_SECONDS"))
        if close_cooldown is not None:
            base.actions.close_cooldown_seconds = close_cooldown
        if kill_cooldown is not None:
            base.actions.kill_cooldown_seconds = kill_cooldown
        max_notional = _env_float(env.get("RISK_MAX_CLOSE_NOTIONAL"))
        if max_notional is not None:
            base.actions.max_close_notional = max_notional

        backoff = _env_float(env.get("RISK_BACKOFF_SECONDS"))
        max_backoff = _env_float(env.get("RISK_MAX_BACKOFF_SECONDS"))
        retry_attempts = _env_int(env.get("RISK_CLOSE_RETRY_ATTEMPTS"))
        if backoff is not None:
            base.actions.backoff_seconds = backoff
        if max_backoff is not None:
            base.actions.max_backoff_seconds = max_backoff
        if retry_attempts is not None and retry_attempts > 0:
            base.actions.close_retry_attempts = retry_attempts

        telegram_token = env.get("RISK_TELEGRAM_TOKEN") or getattr(realtime, "tg_bot_token", None)
        telegram_chat_id = env.get("RISK_TELEGRAM_CHAT_ID") or getattr(realtime, "tg_bot_chat_id", None)

        exchange_modes: Dict[str, bool] = {}
        for name, cfg in realtime.accounts.items():
            exchange_modes[name] = bool(getattr(cfg, "one_way_mode", True))
        for key, value in env.items():
            if key.startswith("RISK_EXCHANGE_ONE_WAY_"):
                exchange_name = key.replace("RISK_EXCHANGE_ONE_WAY_", "").lower()
                exchange_modes[exchange_name] = _env_bool(value) if value is not None else True

        return cls(
            risk=base,
            exchange_modes=exchange_modes,
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id,
        )


def _env_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _env_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _env_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
