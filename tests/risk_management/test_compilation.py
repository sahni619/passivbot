from __future__ import annotations

import py_compile
from pathlib import Path


def test_risk_management_sources_parse() -> None:
    """Ensure all risk management modules remain free of syntax errors/typos."""

    package_root = Path(__file__).resolve().parents[2] / "risk_management"
    failures: list[tuple[Path, str]] = []

    for path in package_root.rglob("*.py"):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:  # pragma: no cover - defensive check
            failures.append((path, exc.msg))

    assert not failures, "\n".join(f"{path}: {message}" for path, message in failures)
