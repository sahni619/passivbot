"""Persistence for drawdown baselines and executed actions."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RiskState:
    baseline_equity: Optional[float]
    high_water: Optional[float]
    actions: Dict[str, float]


class StateStore:
    def load(self) -> RiskState:  # pragma: no cover - interface
        raise NotImplementedError

    def save(self, *, baseline_equity: Optional[float], high_water: Optional[float]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def record_action(self, action_key: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def should_execute(self, action_key: str, cooldown_seconds: int) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def breach_id(self, drawdown: float, high_water: float) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class FileStateStore(StateStore):
    """Simple JSON-backed persistence for risk decisions."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> RiskState:
        if not self._path.exists():
            return RiskState(baseline_equity=None, high_water=None, actions={})
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read risk state file %s: %s", self._path, exc)
            return RiskState(baseline_equity=None, high_water=None, actions={})
        return RiskState(
            baseline_equity=payload.get("baseline_equity"),
            high_water=payload.get("high_water"),
            actions={k: float(v) for k, v in (payload.get("actions") or {}).items()},
        )

    def save(self, *, baseline_equity: Optional[float], high_water: Optional[float]) -> None:
        state = self.load()
        if baseline_equity is not None:
            state.baseline_equity = baseline_equity
        if high_water is not None:
            state.high_water = high_water
        payload = {
            "baseline_equity": state.baseline_equity,
            "high_water": state.high_water,
            "actions": state.actions,
        }
        try:
            _atomic_write(self._path, json.dumps(payload, indent=2))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to persist risk state %s: %s", self._path, exc)

    def record_action(self, action_key: str) -> None:
        state = self.load()
        state.actions[action_key] = time()
        payload = {
            "baseline_equity": state.baseline_equity,
            "high_water": state.high_water,
            "actions": state.actions,
        }
        try:
            _atomic_write(self._path, json.dumps(payload, indent=2))
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to persist action state %s: %s", self._path, exc)

    def should_execute(self, action_key: str, cooldown_seconds: int) -> bool:
        state = self.load()
        last = state.actions.get(action_key)
        if last is None:
            return True
        return (time() - last) >= cooldown_seconds

    def breach_id(self, drawdown: float, high_water: float) -> str:
        drawdown_bucket = round(drawdown, 3)
        high_water_bucket = round(high_water or 0.0, 1)
        return f"dd:{drawdown_bucket:.3f}|hw:{high_water_bucket:.1f}"


def _atomic_write(path: Path, content: str) -> None:
    tmp = tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8")
    try:
        with tmp as f:
            f.write(content)
            f.flush()
        Path(tmp.name).replace(path)
    except Exception:
        Path(tmp.name).unlink(missing_ok=True)  # type: ignore[arg-type]
        raise
