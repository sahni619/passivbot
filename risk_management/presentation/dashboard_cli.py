"""CLI entry point for the risk management dashboard."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..configuration import CustomEndpointSettings, load_realtime_config
from ..core.domain import Scenario, ScenarioResult, ScenarioShock
from ..services import RiskService
from ..stress import simulate_scenarios
from .dashboard_rendering import build_dashboard

logger = logging.getLogger(__name__)

__all__ = [
    "load_snapshot",
    "main",
    "run_dashboard",
]


def load_snapshot(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_configured_scenarios(
    configured: Sequence[Scenario],
    requested: Optional[Sequence[str]],
) -> List[Scenario]:
    if not configured:
        if requested:
            names = ", ".join(str(name) for name in requested)
            raise ValueError(
                f"No configured scenarios available to match: {names}",
            )
        return []

    if not requested:
        return list(configured)

    index: Dict[str, Scenario] = {}
    for scenario in configured:
        keys = {scenario.name.lower()}
        if scenario.id:
            keys.add(scenario.id.lower())
        for key in keys:
            index.setdefault(key, scenario)

    selected: List[Scenario] = []
    missing: List[str] = []

    for entry in requested:
        key = str(entry).strip().lower()
        if not key:
            continue
        match = index.get(key)
        if match is None:
            missing.append(str(entry))
            continue
        if match not in selected:
            selected.append(match)

    if missing:
        raise ValueError(f"Unknown scenario(s): {', '.join(missing)}")

    return selected


def _parse_ad_hoc_scenario(values: Optional[Sequence[str]]) -> Optional[Scenario]:
    if not values:
        return None

    shocks: List[ScenarioShock] = []
    for raw in values:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        if ":" not in text:
            raise ValueError(
                "Ad-hoc shocks must use the format SYMBOL:PCT (for example BTCUSDT:-0.10).",
            )
        symbol_part, pct_part = text.split(":", 1)
        symbol = symbol_part.strip().upper()
        if not symbol:
            raise ValueError("Shock definitions must include a symbol name before ':'.")
        try:
            pct = float(pct_part)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Shock '{text}' must include a numeric percentage change.",
            ) from exc
        shocks.append(ScenarioShock(symbol=symbol, price_pct=pct))

    if not shocks:
        return None

    description = ", ".join(
        f"{shock.symbol} {shock.price_pct:+.2%}" for shock in shocks
    )
    return Scenario(
        id="adhoc",
        name="Ad-hoc shock",
        description=description,
        shocks=tuple(shocks),
    )


def run_dashboard(config_path: Path) -> str:
    snapshot = load_snapshot(config_path)
    return build_dashboard(snapshot)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render the risk management dashboard")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("dashboard_config.json"),
        help="Path to the JSON snapshot used for the dashboard.",
    )
    parser.add_argument(
        "--realtime-config",
        type=Path,
        default=None,
        help="Path to a realtime configuration file. Overrides --config when provided.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Refresh interval in seconds. Set to >0 to continuously monitor the file.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of times to render the dashboard when --interval is set. Use 0 to loop indefinitely.",
    )
    parser.add_argument(
        "--custom-endpoints",
        help=(
            "Override custom endpoint behaviour. Provide a JSON file path to reuse the same "
            "proxy configuration as the trading system, 'auto' to enable auto-discovery, or 'none' to "
            "disable overrides."
        ),
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenario_names",
        help=(
            "Name or ID of a configured scenario to evaluate. Repeat to include multiple scenarios."
        ),
    )
    parser.add_argument(
        "--shock",
        action="append",
        dest="shocks",
        metavar="SYMBOL:PCT",
        help=(
            "Ad-hoc price shock expressed as a decimal percentage. Example: --shock BTCUSDT:-0.10"
        ),
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.interval < 0:
        parser.error("--interval must be >= 0")
    if args.iterations < 0:
        parser.error("--iterations must be >= 0")

    try:
        return asyncio.run(_run_cli(args))
    except FileNotFoundError:
        parser.error(f"Snapshot file not found: {args.config}")
    except json.JSONDecodeError as exc:
        parser.error(f"Snapshot file is not valid JSON: {exc}")
    except ValueError as exc:
        parser.error(str(exc))
    return 1


async def _run_cli(args: argparse.Namespace) -> int:
    realtime_service: Optional[RiskService] = None
    configured_scenarios: Sequence[Scenario] = []
    if args.realtime_config:
        realtime_config = load_realtime_config(Path(args.realtime_config))
        override = args.custom_endpoints
        if override is not None:
            override_normalized = override.strip()
            if not override_normalized:
                realtime_config.custom_endpoints = None
            else:
                lowered = override_normalized.lower()
                if lowered in {"none", "off", "disable"}:
                    realtime_config.custom_endpoints = CustomEndpointSettings(
                        path=None, autodiscover=False
                    )
                elif lowered in {"auto", "autodiscover", "default"}:
                    realtime_config.custom_endpoints = CustomEndpointSettings(
                        path=None, autodiscover=True
                    )
                else:
                    realtime_config.custom_endpoints = CustomEndpointSettings(
                        path=override_normalized, autodiscover=False
                    )
        configured_scenarios = list(realtime_config.scenarios)
        realtime_service = RiskService.from_config(realtime_config)
        logger.info("Starting realtime dashboard using %s", args.realtime_config)

    try:
        selected_configured = _select_configured_scenarios(
            configured_scenarios,
            getattr(args, "scenario_names", None),
        )
        ad_hoc = _parse_ad_hoc_scenario(getattr(args, "shocks", None))
        scenarios_to_run: List[Scenario] = list(selected_configured)
        if ad_hoc is not None:
            scenarios_to_run.append(ad_hoc)

        iteration = 0
        while True:
            if realtime_service is not None:
                snapshot = await realtime_service.fetch_snapshot()
            else:
                snapshot = load_snapshot(Path(args.config))
            scenario_results: Sequence[ScenarioResult] = []
            if scenarios_to_run:
                scenario_results = simulate_scenarios(snapshot, scenarios_to_run)
            dashboard = build_dashboard(snapshot, scenario_results=scenario_results)
            print(dashboard)
            iteration += 1
            if args.iterations and iteration >= args.iterations:
                break
            if args.interval == 0:
                break
            await asyncio.sleep(args.interval)
        return 0
    finally:
        if realtime_service is not None:
            await realtime_service.close()


if __name__ == "__main__":  # pragma: no cover - manual invocation hook
    sys.exit(main())
