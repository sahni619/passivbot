import importlib


def test_import_risk_management_realtime() -> None:
    module = importlib.import_module("risk_management.realtime")
    assert hasattr(module, "RealtimeDataFetcher")
