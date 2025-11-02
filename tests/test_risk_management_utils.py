"""Tests for the :mod:`risk_management._utils` helpers."""

from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

utils = importlib.import_module("risk_management._utils")
account_clients = importlib.import_module("risk_management.account_clients")


def test_stringify_payload_serialises_sets_and_bytes() -> None:
    payload = {"assets": {"BTC", "ETH"}, "note": b"risk"}

    result = utils.stringify_payload(payload)

    parsed = json.loads(result)
    assert sorted(parsed["assets"]) == ["BTC", "ETH"]
    assert parsed["note"] == "risk"


@pytest.mark.parametrize(
    "values, expected",
    [
        ((None, ""), None),
        (
            (("1.5",),),
            1.5,
        ),
        ((None, " 42 "), 42.0),
        (
            (("abc", 1.0), "2"),
            2.0,
        ),
    ],
)
def test_first_float(values: tuple[object, ...], expected: float | None) -> None:
    assert utils.first_float(*values) == expected


def test_extract_position_details_prefers_explicit_fields() -> None:
    position = {
        "positionSide": "long",
        "info": {"positionIdx": "2", "positionSide": "short"},
    }

    side, idx, explicit = utils.extract_position_details(position)

    assert side == "LONG"
    assert idx == 2
    assert explicit is True


def test_extract_position_details_infers_side_from_index() -> None:
    position = {"info": {"positionIdx": 1}}

    side, idx, explicit = utils.extract_position_details(position)

    assert side == "LONG"
    assert idx == 1
    assert explicit is False


def test_normalise_order_book_depth_handles_aliases() -> None:
    normalise = account_clients._normalise_order_book_depth

    assert normalise("binanceusdm", 7) == 10
    assert normalise("Binance-USDm", 600) == 1000
    assert normalise("binance_coinm", 1002) == 1000


def test_normalise_order_book_depth_defaults_for_unknown_exchange() -> None:
    normalise = account_clients._normalise_order_book_depth

    assert normalise("unknown", 12) == 12
    assert normalise("", -5) == 1
