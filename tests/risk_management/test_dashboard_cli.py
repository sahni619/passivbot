import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from risk_management import dashboard


def test_cli_smoke_with_static_snapshot(tmp_path: Path, capsys):
    snapshot = {
        "generated_at": "2024-01-01T00:00:00Z",
        "accounts": [
            {
                "name": "Demo Account",
                "balance": 1000,
                "positions": [],
            }
        ],
    }
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")

    exit_code = asyncio.run(
        dashboard._run_cli(  # type: ignore[attr-defined]
            dashboard.argparse.Namespace(
                config=path,
                realtime_config=None,
                interval=0,
                iterations=1,
                custom_endpoints=None,
            )
        )
    )
    captured = capsys.readouterr()
    assert "Demo Account" in captured.out
    assert exit_code == 0
