from types import SimpleNamespace

from risk_management import view_helpers


def test_dashboard_context_includes_grafana():
    request = SimpleNamespace()
    context = view_helpers.dashboard_context(
        request,
        "user",
        {"accounts": []},
        {"dashboards": [1], "account_dashboards": [2], "theme": "dark"},
    )
    assert context["request"] is request
    assert context["user"] == "user"
    assert context["grafana_dashboards"] == [1]
    assert context["grafana_account_dashboards"] == [2]
    assert context["grafana_theme"] == "dark"


def test_api_keys_context_includes_paths():
    request = SimpleNamespace()
    context = view_helpers.api_keys_context(
        request,
        "user",
        {"api": "keys"},
        [{"name": "acc"}],
        "/path/config",
        "/path/api_keys",
        {"dashboards": [3], "theme": "light"},
    )
    assert context["api_keys"] == {"api": "keys"}
    assert context["accounts"] == [{"name": "acc"}]
    assert context["config_path"] == "/path/config"
    assert context["api_keys_path"] == "/path/api_keys"
    assert context["grafana_dashboards"] == [3]
    assert context["grafana_theme"] == "light"
