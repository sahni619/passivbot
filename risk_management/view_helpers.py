"""Helpers for assembling template contexts."""

from __future__ import annotations

from typing import Any, Mapping


def dashboard_context(request, user: str, snapshot: Mapping[str, Any], grafana_context: Mapping[str, Any]) -> dict:
    return {
        "request": request,
        "user": user,
        "snapshot": snapshot,
        "grafana_dashboards": grafana_context.get("dashboards", []),
        "grafana_account_dashboards": grafana_context.get("account_dashboards", []),
        "grafana_theme": grafana_context.get("theme"),
    }


def api_keys_context(
    request,
    user: str,
    api_keys: Mapping[str, Any],
    accounts: list[Mapping[str, Any]],
    config_path: str | None,
    api_keys_path: str | None,
    grafana_context: Mapping[str, Any],
) -> dict:
    return {
        "request": request,
        "user": user,
        "api_keys": api_keys,
        "accounts": accounts,
        "config_path": config_path,
        "api_keys_path": api_keys_path,
        "grafana_dashboards": grafana_context.get("dashboards", []),
        "grafana_theme": grafana_context.get("theme"),
    }

