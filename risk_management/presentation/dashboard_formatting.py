"""Formatting helpers used by the dashboard renderer."""

from __future__ import annotations

import math
from typing import Optional

__all__ = [
    "format_currency",
    "format_pct",
    "format_price",
    "format_simple_number",
]


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_pct(value: float) -> str:
    return f"{value * 100:6.2f}%"


def format_price(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:,.2f}"


def format_simple_number(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:,.4f}"
