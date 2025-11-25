"""Pure risk decision logic based on aggregated portfolio data."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict

from .config import RiskEngineConfig
from .portfolio_aggregator import PortfolioView
from .state_store import StateStore

logger = logging.getLogger(__name__)


class RiskDecisionLevel(str, Enum):
    NONE = "none"
    ALERT = "alert"
    CLOSE_POSITIONS = "close_positions"
    KILL_BOTS = "kill_bots"


@dataclass
class RiskDecision:
    level: RiskDecisionLevel
    drawdown: float
    adjusted_equity: float
    high_water: float
    rationale: str
    breach_id: str

    def to_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["level"] = self.level.value
        return payload


class RiskRulesEngine:
    """Evaluate drawdown rules without side effects."""

    def __init__(self, config: RiskEngineConfig) -> None:
        self._config = config

    def evaluate(self, view: PortfolioView, state_store: StateStore) -> RiskDecision:
        state = state_store.load()
        adjusted_equity = view.total_equity - view.net_cashflow
        if view.net_cashflow:
            logger.info(
                "Adjusting equity for net cashflow",
                extra={"net_cashflow": view.net_cashflow, "raw_equity": view.total_equity, "adjusted_equity": adjusted_equity},
            )
        high_water = state.high_water or adjusted_equity
        high_water = max(high_water, adjusted_equity)
        drawdown = 0.0
        if high_water > 0:
            drawdown = (high_water - adjusted_equity) / high_water

        state_store.save(high_water=high_water, baseline_equity=state.baseline_equity or adjusted_equity)

        decision_level = RiskDecisionLevel.NONE
        rationale = ""
        thresholds = self._config.thresholds
        if drawdown >= abs(thresholds.kill_bots_drawdown):
            decision_level = RiskDecisionLevel.KILL_BOTS
            rationale = f"Drawdown {drawdown:.2%} exceeds kill threshold {abs(thresholds.kill_bots_drawdown):.2%}"
        elif drawdown >= abs(thresholds.close_positions_drawdown):
            decision_level = RiskDecisionLevel.CLOSE_POSITIONS
            rationale = (
                f"Drawdown {drawdown:.2%} exceeds close threshold {abs(thresholds.close_positions_drawdown):.2%}"
            )
        elif drawdown >= abs(thresholds.alert_drawdown):
            decision_level = RiskDecisionLevel.ALERT
            rationale = f"Drawdown {drawdown:.2%} exceeds alert threshold {abs(thresholds.alert_drawdown):.2%}"
        else:
            rationale = "Drawdown within tolerance"

        breach_id = state_store.breach_id(drawdown, high_water)
        log_level = logging.INFO
        if decision_level is RiskDecisionLevel.ALERT:
            log_level = logging.WARNING
        elif decision_level in (RiskDecisionLevel.CLOSE_POSITIONS, RiskDecisionLevel.KILL_BOTS):
            log_level = logging.ERROR
        logger.log(
            log_level,
            "Evaluated risk decision",
            extra={
                "decision": decision_level.value,
                "drawdown": drawdown,
                "rationale": rationale,
                "adjusted_equity": adjusted_equity,
                "high_water": high_water,
            },
        )
        return RiskDecision(
            level=decision_level,
            drawdown=drawdown,
            adjusted_equity=adjusted_equity,
            high_water=high_water,
            rationale=rationale,
            breach_id=breach_id,
        )
