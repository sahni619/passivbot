"""Utilities for loading and persisting API key configuration files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping


def _ensure_mapping(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("API key configuration must be a JSON object")
    return payload


def load_api_keys(path: Path) -> Dict[str, Any]:
    """Load and validate an api-keys.json payload."""

    data = json.loads(path.read_text(encoding="utf-8"))
    mapping = _ensure_mapping(data)
    return {str(key): value for key, value in mapping.items()}


def save_api_keys(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist API keys to ``path`` with a stable formatting."""

    _ensure_mapping(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def validate_api_key_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a normalised API key entry ensuring required fields exist."""

    if not isinstance(entry, Mapping):
        raise TypeError("API key entry must be an object")
    exchange = entry.get("exchange")
    if not exchange:
        raise ValueError("API key entries require an 'exchange' field")

    normalized = {str(key): value for key, value in entry.items()}
    normalized["exchange"] = str(exchange).strip()
    return normalized
