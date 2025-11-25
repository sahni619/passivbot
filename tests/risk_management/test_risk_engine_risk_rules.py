from dataclasses import dataclass
from typing import Dict, Optional

import pytest

from risk_management.risk_engine.config import RiskEngineConfig, ThresholdConfig
from risk_management.risk_engine.portfolio_aggregator import PortfolioView
from risk_management.risk_engine.risk_rules import RiskDecisionLevel, RiskRulesEngine
from risk_management.risk_engine.state_store import StateStore, RiskState


@dataclass
class InMemoryState(StateStore):
    baseline_equity: Optional[float] = None
    high_water: Optional[float] = None
    actions: Dict[str, float] = None

    def load(self) -> RiskState:
        return RiskState(self.baseline_equity, self.high_water, self.actions or {})

    def save(self, *, baseline_equity: Optional[float], high_water: Optional[float]) -> None:
        if baseline_equity is not None:
            self.baseline_equity = baseline_equity
        if high_water is not None:
            self.high_water = high_water

    def record_action(self, action_key: str) -> None:  # pragma: no cover - not needed
        (self.actions or {}).update({action_key: 0})

    def should_execute(self, action_key: str, cooldown_seconds: int) -> bool:  # pragma: no cover - not needed
        return True

    def breach_id(self, drawdown: float, high_water: float) -> str:
        return f"{drawdown:.4f}:{high_water:.2f}"


def make_view(total_equity: float, net_cashflow: float = 0.0) -> PortfolioView:
    return PortfolioView(
        total_balance=total_equity,
        total_equity=total_equity,
        total_unrealized_pnl=0.0,
        total_realized_pnl=0.0,
        net_cashflow=net_cashflow,
    )


def test_no_breach_returns_none():
    engine = RiskRulesEngine(RiskEngineConfig())
    store = InMemoryState()
    view = make_view(1000)

    decision = engine.evaluate(view, store)

    assert decision.level is RiskDecisionLevel.NONE
    assert decision.drawdown == pytest.approx(0)
    assert store.high_water == pytest.approx(1000)


def test_alert_on_drawdown_threshold():
    engine = RiskRulesEngine(RiskEngineConfig(thresholds=ThresholdConfig(alert_drawdown=-0.03)))
    store = InMemoryState(high_water=1000)
    view = make_view(970)

    decision = engine.evaluate(view, store)

    assert decision.level is RiskDecisionLevel.ALERT
    assert pytest.approx(decision.drawdown, rel=1e-3) == 0.03


def test_close_positions_on_severe_drawdown():
    engine = RiskRulesEngine(RiskEngineConfig(thresholds=ThresholdConfig(close_positions_drawdown=-0.08)))
    store = InMemoryState(high_water=1000)
    view = make_view(900)

    decision = engine.evaluate(view, store)

    assert decision.level is RiskDecisionLevel.CLOSE_POSITIONS
    assert decision.drawdown == pytest.approx(0.10)


def test_kill_bots_when_configured():
    engine = RiskRulesEngine(
        RiskEngineConfig(thresholds=ThresholdConfig(close_positions_drawdown=-0.05, kill_bots_drawdown=-0.1))
    )
    store = InMemoryState(high_water=1000)
    view = make_view(800)

    decision = engine.evaluate(view, store)

    assert decision.level is RiskDecisionLevel.KILL_BOTS
    assert decision.drawdown == pytest.approx(0.20)


def test_deposits_do_not_reduce_drawdown():
    engine = RiskRulesEngine(RiskEngineConfig())
    store = InMemoryState(high_water=1000)
    view = make_view(total_equity=1100, net_cashflow=100)

    decision = engine.evaluate(view, store)

    assert decision.level is RiskDecisionLevel.NONE
    assert decision.adjusted_equity == pytest.approx(1000)


def test_withdrawals_do_not_trigger_false_loss():
    engine = RiskRulesEngine(RiskEngineConfig())
    store = InMemoryState(high_water=1000)
    view = make_view(total_equity=900, net_cashflow=-200)

    decision = engine.evaluate(view, store)

    assert decision.level is RiskDecisionLevel.NONE
    assert decision.adjusted_equity == pytest.approx(1100)
