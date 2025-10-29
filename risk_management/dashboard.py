"""Compatibility wrapper around the refactored dashboard modules."""

from __future__ import annotations

import sys

from .presentation.dashboard_cli import load_snapshot, main, run_dashboard
from .presentation.dashboard_rendering import build_dashboard, render_dashboard

__all__ = ["build_dashboard", "load_snapshot", "main", "render_dashboard", "run_dashboard"]


if __name__ == "__main__":  # pragma: no cover - manual invocation hook
    sys.exit(main())
