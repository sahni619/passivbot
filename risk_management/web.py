"""FastAPI powered web dashboard for live risk management.

The application exposes REST endpoints and templated views backed by the
RealtimeDataFetcher utilities.
"""

from __future__ import annotations
from collections.abc import Iterable as IterableABC
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.sessions import SessionMiddleware
from urllib.parse import quote, urljoin

from .configuration import RealtimeConfig
from .realtime import RealtimeDataFetcher
from .reporting import ReportManager
from .history import PortfolioHistoryStore
from .api_keys import load_api_keys, save_api_keys, validate_api_key_entry
from .snapshot_utils import build_presentable_snapshot

logger = logging.getLogger(__name__)


class AuthManager:
    """Handle authentication for the dashboard."""

    def __init__(
        self,
        secret_key: str,
        users: Mapping[str, str],
        session_cookie_name: str = "risk_dashboard_session",
        https_only: bool = True,
    ) -> None:
        if not secret_key:
            raise ValueError("Authentication requires a non-empty secret key.")
        if not users:
            raise ValueError("At least one dashboard user must be configured.")
        self.secret_key = secret_key
        self.users = dict(users)
        self.session_cookie_name = session_cookie_name
        self.https_only = https_only
        self._password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def authenticate(self, username: str, password: str) -> bool:
        hashed = self.users.get(username)
        if not hashed:
            return False
        return self._password_context.verify(password, hashed)


class RiskDashboardService:
    """Wrap a realtime fetcher to expose snapshot data."""

    def __init__(self, fetcher: RealtimeDataFetcher) -> None:
        self._fetcher = fetcher

    async def fetch_snapshot(self) -> Dict[str, Any]:
        return await self._fetcher.fetch_snapshot()

    async def close(self) -> None:
        await self._fetcher.close()

    async def trigger_kill_switch(
        self, account_name: Optional[str] = None, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        return await self._fetcher.execute_kill_switch(account_name, symbol)

    async def place_order(
        self,
        account_name: str,
        *,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        return await self._fetcher.place_order(
            account_name,
            symbol=symbol,
            order_type=order_type,
            side=side,
            amount=amount,
            price=price,
            params=params,
        )

    async def cancel_order(
        self,
        account_name: str,
        order_id: str,
        *,
        symbol: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        return await self._fetcher.cancel_order(
            account_name, order_id, symbol=symbol, params=params
        )

    async def close_position(self, account_name: str, symbol: str) -> Mapping[str, Any]:
        return await self._fetcher.close_position(account_name, symbol)

    async def list_order_types(self, account_name: str) -> Sequence[str]:
        return await self._fetcher.list_order_types(account_name)

    def get_portfolio_stop_loss(self) -> Optional[Dict[str, Any]]:
        state = self._fetcher.get_portfolio_stop_loss()
        return dict(state) if state is not None else None

    async def set_portfolio_stop_loss(self, threshold_pct: float) -> Dict[str, Any]:
        return await self._fetcher.set_portfolio_stop_loss(threshold_pct)

    async def clear_portfolio_stop_loss(self) -> None:
        await self._fetcher.clear_portfolio_stop_loss()

    def list_conditional_stop_losses(self) -> list[Dict[str, Any]]:
        return self._fetcher.get_conditional_stop_losses()

    async def add_conditional_stop_loss(
        self, name: str, metric: str, threshold: float, operator: str = "lte"
    ) -> Dict[str, Any]:
        return await self._fetcher.add_conditional_stop_loss(name, metric, threshold, operator)

    async def clear_conditional_stop_losses(self, name: Optional[str] = None) -> None:
        await self._fetcher.clear_conditional_stop_losses(name=name)


def _format_kill_switch_failure(account: str, action: str, payload: Mapping[str, Any]) -> str:
    symbol = payload.get("symbol")
    side = payload.get("side")
    order_id = payload.get("order_id")
    target: Optional[str] = None
    if symbol and side:
        target = f"{symbol} ({side})"
    elif symbol:
        target = str(symbol)
    elif order_id:
        target = f"order {order_id}"
    error_message = payload.get("error") or "Unknown error"
    if target:
        return f"[{account}] Failed to {action} {target}: {error_message}"
    return f"[{account}] Failed to {action}: {error_message}"


def _collect_kill_switch_errors(results: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(results, Mapping):
        return errors

    def _extend_with_failures(account: str, failures: Iterable[Mapping[str, Any]], action: str) -> None:
        for failure in failures:
            if isinstance(failure, Mapping):
                errors.append(_format_kill_switch_failure(account, action, failure))
            else:
                errors.append(f"[{account}] Failed to {action}: {failure}")

    for account, details in results.items():
        if not isinstance(details, Mapping):
            continue
        top_level_error = details.get("error")
        if top_level_error:
            errors.append(f"[{account}] {top_level_error}")
        failed_order_cancellations = details.get("failed_order_cancellations")
        if isinstance(failed_order_cancellations, IterableABC) and not isinstance(
            failed_order_cancellations, (str, bytes)
        ):
            _extend_with_failures(account, failed_order_cancellations, "cancel order")
        failed_position_closures = details.get("failed_position_closures")
        if isinstance(failed_position_closures, IterableABC) and not isinstance(
            failed_position_closures, (str, bytes)
        ):
            _extend_with_failures(account, failed_position_closures, "close position")
    return errors


def _build_kill_switch_response(results: Any) -> Dict[str, Any]:
    errors = _collect_kill_switch_errors(results)
    payload: Dict[str, Any] = {"success": not errors, "results": results}
    if errors:
        payload["errors"] = errors
    return payload


def create_app(
    config: RealtimeConfig,
    *,
    service: Optional[RiskDashboardService] = None,
    auth_manager: Optional[AuthManager] = None,
    templates_dir: Optional[Path] = None,
    letsencrypt_challenge_dir: Optional[Path] = None,
) -> FastAPI:
    if service is None:
        service = RiskDashboardService(RealtimeDataFetcher(config))
    if config.auth is None and auth_manager is None:
        raise ValueError("Realtime configuration must include authentication details for the web dashboard.")
    if auth_manager is None and config.auth is not None:
        auth_manager = AuthManager(
            config.auth.secret_key,
            config.auth.users,
            session_cookie_name=config.auth.session_cookie_name,
            https_only=config.auth.https_only,
        )
    assert auth_manager is not None  # for mypy/static tools

    app = FastAPI(title="Risk Management Dashboard")
    app.state.service = service
    app.state.auth_manager = auth_manager
    reports_dir = config.reports_dir
    if reports_dir is None:
        base_root = config.config_root or Path.cwd()
        reports_dir = base_root / "reports"
    app.state.report_manager = ReportManager(reports_dir)
    history_root = (reports_dir or config.config_root or Path.cwd()) / "history"
    app.state.history_store = PortfolioHistoryStore(history_root)

    def resolve_grafana_context() -> dict[str, Any]:
        grafana_cfg = config.grafana
        if grafana_cfg is None:
            return {"dashboards": [], "theme": None}

        def resolve_url(raw_url: str) -> str:
            url = raw_url.strip()
            if grafana_cfg.base_url and not url.lower().startswith(("http://", "https://")):
                base = grafana_cfg.base_url.rstrip("/") + "/"
                return urljoin(base, url.lstrip("/"))
            return url

        dashboards: list[dict[str, Any]] = []
        for dashboard in grafana_cfg.dashboards:
            dashboards.append(
                {
                    "title": dashboard.title,
                    "url": resolve_url(dashboard.url),
                    "description": dashboard.description,
                    "height": dashboard.height or grafana_cfg.default_height,
                }
            )

        account_dashboards: list[dict[str, Any]] = []
        if grafana_cfg.account_equity_template:
            template = grafana_cfg.account_equity_template
            for account in config.accounts:
                url = template.replace("{account}", quote(account.name))
                account_dashboards.append(
                    {
                        "title": f"{account.name} equity",
                        "url": resolve_url(url),
                        "description": f"Equity history for {account.name}",
                        "height": grafana_cfg.default_height,
                    }
                )

        return {
            "dashboards": dashboards,
            "account_dashboards": account_dashboards,
            "theme": grafana_cfg.theme,
        }

    app.state.grafana_context = resolve_grafana_context()

    templates_path = templates_dir or Path(__file__).with_name("templates")
    templates = Jinja2Templates(directory=str(templates_path))

    def currency_filter(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "-"
        return f"${number:,.2f}"

    def pct_filter(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "0.00%"
        return f"{number * 100:.2f}%"

    templates.env.filters.setdefault("currency", currency_filter)
    templates.env.filters.setdefault("pct", pct_filter)

    if auth_manager.https_only:
        app.add_middleware(HTTPSRedirectMiddleware)

    app.add_middleware(
        SessionMiddleware,
        secret_key=auth_manager.secret_key,
        session_cookie=auth_manager.session_cookie_name,
        https_only=auth_manager.https_only,
        same_site="lax",
    )

    if letsencrypt_challenge_dir is not None:
        challenge_dir = Path(letsencrypt_challenge_dir)
        challenge_dir.mkdir(parents=True, exist_ok=True)
        app.mount(
            "/.well-known/acme-challenge",
            StaticFiles(directory=str(challenge_dir), check_dir=False),
            name="acme-challenge",
        )

    def get_service(request: Request) -> RiskDashboardService:
        return request.app.state.service

    def require_user(request: Request) -> str:
        user = request.session.get("user")
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return str(user)

    def get_report_manager(request: Request) -> ReportManager:
        return request.app.state.report_manager

    def get_history_store(request: Request) -> PortfolioHistoryStore:
        return request.app.state.history_store

    def _load_config_payload() -> Dict[str, Any]:
        if config.config_path is None:
            raise RuntimeError("Realtime configuration path is not available for editing")
        try:
            return json.loads(config.config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(f"Realtime configuration file not found: {exc}") from exc

    def _save_config_payload(payload: Mapping[str, Any]) -> None:
        if config.config_path is None:
            raise RuntimeError("Realtime configuration path is not available for editing")
        config.config_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @app.get("/login", response_class=HTMLResponse)
    async def login_form(request: Request) -> HTMLResponse:
        if request.session.get("user"):
            return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        return templates.TemplateResponse("login.html", {"request": request, "error": None})

    @app.post("/login", response_class=HTMLResponse)
    async def login_submit(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ) -> HTMLResponse:
        if not auth_manager.authenticate(username, password):
            context = {"request": request, "error": "Invalid username or password."}
            return templates.TemplateResponse("login.html", context, status_code=status.HTTP_401_UNAUTHORIZED)
        request.session["user"] = username
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    @app.post("/logout")
    async def logout(request: Request) -> RedirectResponse:
        request.session.pop("user", None)
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(
        request: Request,
        service: RiskDashboardService = Depends(get_service),
        history: PortfolioHistoryStore = Depends(get_history_store),
    ) -> HTMLResponse:
        user = request.session.get("user")
        if not user:
            return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        snapshot = await service.fetch_snapshot()
        view_model = build_presentable_snapshot(snapshot)
        try:
            await history.record_async(view_model)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to persist snapshot history: %s", exc)
        grafana_context: dict[str, Any] = request.app.state.grafana_context
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "user": user,
                "snapshot": view_model,
                "grafana_dashboards": grafana_context.get("dashboards", []),
                "grafana_account_dashboards": grafana_context.get("account_dashboards", []),
                "grafana_theme": grafana_context.get("theme"),
            },
        )

    @app.get("/trading-panel", response_class=HTMLResponse)
    async def trading_panel(
        request: Request,
        service: RiskDashboardService = Depends(get_service),
        history: PortfolioHistoryStore = Depends(get_history_store),
    ) -> HTMLResponse:
        user = request.session.get("user")
        if not user:
            return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        snapshot = await service.fetch_snapshot()
        view_model = build_presentable_snapshot(snapshot)
        try:
            await history.record_async(view_model)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to persist snapshot history: %s", exc)
        grafana_context: dict[str, Any] = request.app.state.grafana_context
        return templates.TemplateResponse(
            "trading_panel.html",
            {
                "request": request,
                "user": user,
                "snapshot": view_model,
                "grafana_dashboards": grafana_context.get("dashboards", []),
                "grafana_account_dashboards": grafana_context.get("account_dashboards", []),
                "grafana_theme": grafana_context.get("theme"),
            },
        )

    @app.get("/api-keys", response_class=HTMLResponse)
    async def api_keys_page(request: Request) -> HTMLResponse:
        user = request.session.get("user")
        if not user:
            return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        if config.api_keys_path is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="API key file is not configured")
        try:
            api_keys_payload = load_api_keys(config.api_keys_path)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

        config_payload = _load_config_payload()
        accounts_payload = config_payload.get("accounts", []) if isinstance(config_payload, Mapping) else []
        grafana_context: dict[str, Any] = request.app.state.grafana_context

        return templates.TemplateResponse(
            "api_keys.html",
            {
                "request": request,
                "user": user,
                "api_keys": api_keys_payload,
                "accounts": accounts_payload,
                "config_path": str(config.config_path) if config.config_path else None,
                "api_keys_path": str(config.api_keys_path),
                "grafana_dashboards": grafana_context.get("dashboards", []),
                "grafana_theme": grafana_context.get("theme"),
            },
        )

    @app.get("/api/snapshot", response_class=JSONResponse)
    async def api_snapshot(
        request: Request,
        service: RiskDashboardService = Depends(get_service),
        history: PortfolioHistoryStore = Depends(get_history_store),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        snapshot = await service.fetch_snapshot()
        view_model = build_presentable_snapshot(snapshot)
        try:
            await history.record_async(view_model)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to persist snapshot history: %s", exc)
        return JSONResponse(view_model)

    @app.get(
        "/api/trading/accounts/{account_name}/order-types",
        response_class=JSONResponse,
    )
    async def api_list_order_types(
        account_name: str,
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        try:
            order_types = await service.list_order_types(account_name)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return JSONResponse({"account": account_name, "order_types": list(order_types)})

    @app.post(
        "/api/trading/accounts/{account_name}/orders",
        response_class=JSONResponse,
    )
    async def api_place_order(
        account_name: str,
        request: Request,
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception as exc:  # pragma: no cover - invalid JSON yields 400
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Order payload must be an object")
        symbol = str(payload.get("symbol", "")).strip()
        order_type = str(payload.get("order_type", "")).strip()
        side = str(payload.get("side", "")).strip().lower()
        if not symbol or not order_type or side not in {"buy", "sell"}:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid order parameters")
        try:
            amount = float(payload.get("amount"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Amount must be numeric")
        if amount <= 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Amount must be greater than zero")
        price_raw = payload.get("price")
        price_value: Optional[float]
        if price_raw in (None, ""):
            price_value = None
        else:
            try:
                price_value = float(price_raw)
            except (TypeError, ValueError):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Price must be numeric")
            if price_value <= 0:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Price must be greater than zero")
        params = payload.get("params")
        if not isinstance(params, Mapping):
            params = None
        try:
            result = await service.place_order(
                account_name,
                symbol=symbol,
                order_type=order_type,
                side=side,
                amount=amount,
                price=price_value,
                params=params,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.delete(
        "/api/trading/accounts/{account_name}/orders/{order_id}",
        response_class=JSONResponse,
    )
    async def api_cancel_order(
        account_name: str,
        order_id: str,
        request: Request,
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        params: Optional[Mapping[str, Any]] = None
        symbol: Optional[str] = None
        if request.headers.get("content-length") not in (None, "0"):
            try:
                payload = await request.json()
            except Exception as exc:  # pragma: no cover - invalid JSON yields 400
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload") from exc
            if isinstance(payload, Mapping):
                raw_symbol = payload.get("symbol")
                symbol = str(raw_symbol).strip() if raw_symbol is not None else None
                params_candidate = payload.get("params")
                if isinstance(params_candidate, Mapping):
                    params = params_candidate
        try:
            result = await service.cancel_order(account_name, order_id, symbol=symbol, params=params)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.post(
        "/api/trading/accounts/{account_name}/positions/{symbol:path}/close",
        response_class=JSONResponse,
    )
    async def api_close_position(
        account_name: str,
        symbol: str,
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        if not symbol:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Symbol is required")
        try:
            result = await service.close_position(account_name, symbol)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.get("/api/trading/portfolio/stop-loss", response_class=JSONResponse)
    async def api_get_portfolio_stop_loss(
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        state = service.get_portfolio_stop_loss()
        return JSONResponse({"stop_loss": state})

    @app.post("/api/trading/portfolio/stop-loss", response_class=JSONResponse)
    async def api_set_portfolio_stop_loss(
        request: Request,
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Payload must be an object")
        try:
            threshold = float(payload.get("threshold_pct"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="threshold_pct must be numeric")
        try:
            state = await service.set_portfolio_stop_loss(threshold)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(state)

    @app.delete("/api/trading/portfolio/stop-loss", response_class=JSONResponse)
    async def api_clear_portfolio_stop_loss(
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        await service.clear_portfolio_stop_loss()
        return JSONResponse({"status": "cleared"})

    @app.get("/api/trading/portfolio/conditional-stop-loss", response_class=JSONResponse)
    async def api_list_conditional_stop_losses(
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        return JSONResponse({"items": service.list_conditional_stop_losses()})

    @app.post("/api/trading/portfolio/conditional-stop-loss", response_class=JSONResponse)
    async def api_add_conditional_stop_loss(
        request: Request,
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Payload must be an object")
        name = str(payload.get("name", "")).strip() or "conditional"
        metric = str(payload.get("metric", "")).strip()
        operator = str(payload.get("operator", "lte")).strip() or "lte"
        try:
            threshold = float(payload.get("threshold"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="threshold must be numeric")
        try:
            condition = await service.add_conditional_stop_loss(name, metric, threshold, operator)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(condition, status_code=status.HTTP_201_CREATED)

    @app.delete("/api/trading/portfolio/conditional-stop-loss", response_class=JSONResponse)
    async def api_clear_conditional_stop_loss(
        request: Request,
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        target = None
        if request.query_params.get("name"):
            target = request.query_params.get("name")
        await service.clear_conditional_stop_losses(name=target)
        return JSONResponse({"status": "cleared", "name": target})

    @app.post("/api/kill-switch", response_class=JSONResponse)
    async def api_global_kill_switch(
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        results = await service.trigger_kill_switch()
        payload = _build_kill_switch_response(results)
        return JSONResponse(payload)

    @app.post("/api/accounts/{account_name}/kill-switch", response_class=JSONResponse)
    async def api_kill_switch(
        request: Request,
        account_name: str,
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        target = account_name.strip()
        symbol = request.query_params.get("symbol")
        if symbol:
            symbol = symbol.strip()
            if symbol.lower() == "all":
                symbol = None
        try:
            if not target or target.lower() == "all":
                results = await service.trigger_kill_switch(symbol=symbol)
            else:
                results = await service.trigger_kill_switch(target, symbol=symbol)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        payload = _build_kill_switch_response(results)
        return JSONResponse(payload)

    @app.post(
        "/api/accounts/{account_name}/positions/{symbol:path}/kill-switch",
        response_class=JSONResponse,
    )
    async def api_position_kill_switch(
        account_name: str,
        symbol: str,
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        target_symbol = symbol.strip()
        if not target_symbol:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Symbol is required")
        target_account = account_name.strip()
        if not target_account or target_account.lower() == "all":
            target_account = None
        try:
            results = await service.trigger_kill_switch(target_account, symbol=target_symbol)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        payload = _build_kill_switch_response(results)
        return JSONResponse(payload)

    @app.get("/api/accounts/{account_name}/reports", response_class=JSONResponse)
    async def api_list_reports(
        account_name: str,
        manager: ReportManager = Depends(get_report_manager),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        reports = await manager.list_reports(account_name)
        items = []
        for report in reports:
            data = report.to_view()
            data["download_url"] = (
                f"/api/accounts/{quote(account_name, safe='')}/reports/{quote(report.report_id, safe='')}"
            )
            items.append(data)
        return JSONResponse({"account": account_name, "reports": items})

    @app.post("/api/accounts/{account_name}/reports", response_class=JSONResponse)
    async def api_generate_report(
        account_name: str,
        service: RiskDashboardService = Depends(get_service),
        manager: ReportManager = Depends(get_report_manager),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        snapshot = await service.fetch_snapshot()
        view_model = build_presentable_snapshot(snapshot)
        try:
            report = await manager.create_account_report(account_name, view_model)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        data = report.to_view()
        data["download_url"] = (
            f"/api/accounts/{quote(account_name, safe='')}/reports/{quote(report.report_id, safe='')}"
        )
        return JSONResponse(data)

    @app.get("/api/accounts/{account_name}/reports/{report_id}")
    async def api_download_report(
        account_name: str,
        report_id: str,
        manager: ReportManager = Depends(get_report_manager),
        _: str = Depends(require_user),
    ) -> FileResponse:
        path = await manager.get_report_path(account_name, report_id)
        if path is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
        return FileResponse(path, media_type="text/csv", filename=path.name)

    @app.get("/api/cashflows", response_class=JSONResponse)
    async def api_list_cashflows(
        limit: int = 100,
        history: PortfolioHistoryStore = Depends(get_history_store),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        try:
            records = await history.list_cashflows_async(limit=limit)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        return JSONResponse({"items": records})

    @app.post("/api/cashflows", response_class=JSONResponse)
    async def api_add_cashflow(
        request: Request,
        history: PortfolioHistoryStore = Depends(get_history_store),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception as exc:  # pragma: no cover - invalid json yields 400
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Payload must be an object")
        try:
            result = await history.add_cashflow_async(
                flow_type=str(payload.get("type", "")).strip(),
                amount=float(payload.get("amount")),
                currency=str(payload.get("currency", "")).strip() or "USDT",
                account=str(payload.get("account", "")).strip() or None,
                note=str(payload.get("note", "")).strip() or None,
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        return JSONResponse(result, status_code=status.HTTP_201_CREATED)

    @app.get("/api/admin/api-keys", response_class=JSONResponse)
    async def api_list_api_keys(_: str = Depends(require_user)) -> JSONResponse:
        if config.api_keys_path is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="API key file is not configured")
        try:
            payload = load_api_keys(config.api_keys_path)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        return JSONResponse({"path": str(config.api_keys_path), "keys": payload})

    @app.post("/api/admin/api-keys", response_class=JSONResponse)
    async def api_upsert_api_key(request: Request, _: str = Depends(require_user)) -> JSONResponse:
        if config.api_keys_path is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="API key file is not configured")
        try:
            payload = await request.json()
        except Exception as exc:  # pragma: no cover - invalid json yields 400
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Payload must be an object")
        key_id_raw = payload.get("id")
        if not key_id_raw:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Key id is required")
        key_id = str(key_id_raw).strip()
        entry = payload.get("entry")
        try:
            normalized = validate_api_key_entry(entry)
            existing = load_api_keys(config.api_keys_path)
            existing[key_id] = normalized
            save_api_keys(config.api_keys_path, existing)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        return JSONResponse({"id": key_id, "entry": normalized}, status_code=status.HTTP_201_CREATED)

    @app.delete("/api/admin/api-keys/{key_id}", response_class=JSONResponse)
    async def api_delete_api_key(key_id: str, _: str = Depends(require_user)) -> JSONResponse:
        if config.api_keys_path is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="API key file is not configured")
        try:
            existing = load_api_keys(config.api_keys_path)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        if key_id in existing:
            existing.pop(key_id, None)
            try:
                save_api_keys(config.api_keys_path, existing)
            except Exception as exc:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        return JSONResponse({"deleted": key_id})

    @app.get("/api/admin/accounts", response_class=JSONResponse)
    async def api_list_accounts(_: str = Depends(require_user)) -> JSONResponse:
        try:
            payload = _load_config_payload()
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        accounts = payload.get("accounts", []) if isinstance(payload, Mapping) else []
        if not isinstance(accounts, list):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Accounts section must be a list")
        return JSONResponse({"accounts": accounts})

    @app.post("/api/admin/accounts", response_class=JSONResponse)
    async def api_upsert_account(request: Request, _: str = Depends(require_user)) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Payload must be an object")
        account_name = str(payload.get("name", "")).strip()
        if not account_name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Account name is required")
        try:
            config_payload = _load_config_payload()
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        accounts = config_payload.get("accounts") if isinstance(config_payload, Mapping) else None
        if not isinstance(accounts, list):
            accounts = []
        updated_accounts: list[Mapping[str, Any]] = []
        for entry in accounts:
            if isinstance(entry, Mapping) and entry.get("name") == account_name:
                continue
            updated_accounts.append(entry)
        clean_entry = {
            "name": account_name,
            "exchange": str(payload.get("exchange", "")).strip() or payload.get("exchange"),
            "api_key_id": str(payload.get("api_key_id", "")).strip() or None,
            "settle_currency": str(payload.get("settle_currency", "USDT")).strip() or "USDT",
            "symbols": payload.get("symbols") or None,
            "params": payload.get("params") or {},
            "enabled": payload.get("enabled", True),
        }
        updated_accounts.append(clean_entry)
        config_payload["accounts"] = updated_accounts
        try:
            _save_config_payload(config_payload)
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        return JSONResponse({"account": clean_entry}, status_code=status.HTTP_201_CREATED)

    @app.delete("/api/admin/accounts/{account_name}", response_class=JSONResponse)
    async def api_delete_account(account_name: str, _: str = Depends(require_user)) -> JSONResponse:
        try:
            config_payload = _load_config_payload()
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

        accounts = config_payload.get("accounts") if isinstance(config_payload, Mapping) else None
        if not isinstance(accounts, list):
            accounts = []

        updated_accounts: list[Mapping[str, Any]] = []
        removed = False
        for entry in accounts:
            if isinstance(entry, Mapping) and entry.get("name") == account_name:
                removed = True
                continue
            updated_accounts.append(entry)

        if not removed:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Account '{account_name}' not found")

        config_payload["accounts"] = updated_accounts
        try:
            _save_config_payload(config_payload)
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

        return JSONResponse({"deleted": account_name})

    @app.on_event("shutdown")
    async def shutdown() -> None:  # pragma: no cover - FastAPI lifecycle
        await service.close()

    return app
