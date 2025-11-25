"""Risk engine components for realtime risk management.

The package provides a minimal, testable surface for risk evaluation composed of
exchange adapters, portfolio aggregation, rules evaluation, side-effectful
action execution, and persistence of stateful thresholds.
"""

from .config import RiskEngineConfig, Settings
from .exchange_client import ExchangeClientAdapter
from .metrics import MetricRegistry
from .portfolio_aggregator import PortfolioAggregator, PortfolioView
from .risk_rules import RiskDecision, RiskDecisionLevel, RiskRulesEngine
from .state_store import FileStateStore, StateStore
from .action_executor import ActionExecutor
from .risk_loop import risk_loop

__all__ = [
    "RiskEngineConfig",
    "Settings",
    "ExchangeClientAdapter",
    "MetricRegistry",
    "PortfolioAggregator",
    "PortfolioView",
    "RiskDecision",
    "RiskDecisionLevel",
    "RiskRulesEngine",
    "FileStateStore",
    "StateStore",
    "ActionExecutor",
    "risk_loop",
]
