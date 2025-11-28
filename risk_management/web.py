"""FastAPI powered web dashboard for live risk management.

The application exposes REST endpoints and templated views backed by the
``RiskService`` orchestration helpers.
"""

from __future__ import annotations
import json
import logging
from collections.abc import Iterable as IterableABC
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.sessions import SessionMiddleware
from urllib.parse import quote, urlencode, urljoin

from .audit import AuditLogWriter, AuditSettings, get_audit_logger, read_audit_entries
from .configuration import RealtimeConfig, load_realtime_config
from .domain.models import Scenario
from .services.risk_service import RiskService, RiskServiceProtocol
from .reporting import ReportManager
from .services import PerformanceRepository
from .performance_metrics import build_performance_metrics
from .snapshot_utils import (
    ACCOUNT_SORT_FIELDS,
    DEFAULT_ACCOUNT_SORT_KEY,
    DEFAULT_ACCOUNT_SORT_ORDER,
    DEFAULT_ACCOUNTS_PAGE_SIZE,
    EXPOSURE_FILTERS,
    MAX_ACCOUNTS_PAGE_SIZE,
    build_presentable_snapshot,
)
from .stress import scenario_results_to_dict, simulate_scenarios


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
    """Thin wrapper that exposes :class:`RiskServiceProtocol` to FastAPI handlers."""

    def __init__(self, service: RiskServiceProtocol) -> None:
        self._service = service

    async def fetch_snapshot(self) -> Dict[str, Any]:
        return await self._service.fetch_snapshot()

    async def close(self) -> None:
        await self._service.close()


class AutoReloadingRiskDashboardService(RiskDashboardService):
    """Wrap :class:`RiskDashboardService` to reload config changes on the fly.

    The reload logic is intentionally lightweight: it monitors the realtime
    configuration file's modification time and, when it detects a change,
    rebuilds the underlying :class:`RiskDashboardService` with the updated
    configuration. This keeps the dashboard and trading controls aligned with
    newly added accounts without requiring a process restart.
    """

    def __init__(
        self,
        service: RiskDashboardService,
        *,
        config_path: Optional[Path],
    ) -> None:
        super().__init__(service._service)
        self._config_path = config_path.resolve() if config_path else None
        self._config_mtime = self._get_config_mtime()
        self._reload_hooks: list[Callable[[RealtimeConfig], None]] = []

    def _get_config_mtime(self) -> Optional[float]:
        if not self._config_path:
            return None
        try:
            return self._config_path.stat().st_mtime
        except FileNotFoundError:
            return None

    def add_reload_hook(self, hook: Callable[[RealtimeConfig], None]) -> None:
        """Register a callback invoked after a successful config reload."""

        self._reload_hooks.append(hook)

    async def _reload_if_changed(self) -> None:
        if not self._config_path:
            return
        current_mtime = self._get_config_mtime()
        if current_mtime is None or (
            self._config_mtime is not None and current_mtime <= self._config_mtime
        ):
            return

        try:
            new_config = load_realtime_config(self._config_path)
            new_service = RiskDashboardService(RiskService.from_config(new_config))
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("Failed to reload realtime config from %s: %s", self._config_path, exc)
            return

        try:
            await self._service.close()
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Failed to close previous realtime service during reload", exc_info=True)

        self._service = new_service._service
        self._config_mtime = current_mtime
        for hook in list(self._reload_hooks):
            try:
                hook(new_config)
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Reload hook failed", exc_info=True)

    async def fetch_snapshot(self) -> Dict[str, Any]:
        await self._reload_if_changed()
        return await super().fetch_snapshot()

    async def trigger_kill_switch(
        self, account_name: Optional[str] = None, symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        await self._reload_if_changed()
        return await super().trigger_kill_switch(account_name, symbol)

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
        await self._reload_if_changed()
        return await super().place_order(
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
        await self._reload_if_changed()
        return await super().cancel_order(
            account_name,
            order_id,
            symbol=symbol,
            params=params,
        )

    async def close_position(self, account_name: str, symbol: str) -> Mapping[str, Any]:
        await self._reload_if_changed()
        return await super().close_position(account_name, symbol)

    async def list_order_types(self, account_name: str) -> Sequence[str]:
        await self._reload_if_changed()
        return await super().list_order_types(account_name)

    async def set_portfolio_stop_loss(self, threshold_pct: float) -> Dict[str, Any]:
        await self._reload_if_changed()
        return await super().set_portfolio_stop_loss(threshold_pct)

    async def clear_portfolio_stop_loss(self) -> None:
        await self._reload_if_changed()
        await super().clear_portfolio_stop_loss()

    async def set_account_stop_loss(self, account_name: str, threshold_pct: float) -> Dict[str, Any]:
        await self._reload_if_changed()
        return await super().set_account_stop_loss(account_name, threshold_pct)

    async def clear_account_stop_loss(self, account_name: str) -> None:
        await self._reload_if_changed()
        await super().clear_account_stop_loss(account_name)

    async def cancel_all_orders(
        self, account_name: str, symbol: Optional[str] = None
    ) -> Mapping[str, Any]:
        await self._reload_if_changed()
        return await super().cancel_all_orders(account_name, symbol)

    async def close_all_positions(
        self, account_name: str, symbol: Optional[str] = None
    ) -> Mapping[str, Any]:
        await self._reload_if_changed()
        return await super().close_all_positions(account_name, symbol)


def _parse_positive_int(value: Optional[str], name: str, *, maximum: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{name} must be an integer",
        ) from exc
    if parsed <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{name} must be greater than zero",
        )
    if maximum is not None and parsed > maximum:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{name} cannot exceed {maximum}",
        )
    return parsed


def _normalise_sort_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    key = value.lower()
    if key not in ACCOUNT_SORT_FIELDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported sort key '{value}'",
        )
    return key


def _normalise_sort_order(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    order = value.lower()
    if order not in {"asc", "desc"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="sort_order must be 'asc' or 'desc'",
        )
    return order


def _normalise_exposure_filter(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    mode = value.lower()
    if mode not in EXPOSURE_FILTERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported exposure filter '{value}'",
        )
    return mode


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
    config_path: Optional[Path] = None,
    service: Optional[RiskDashboardService] = None,
    auth_manager: Optional[AuthManager] = None,
    templates_dir: Optional[Path] = None,
    letsencrypt_challenge_dir: Optional[Path] = None,
    performance_repository: Optional[PerformanceRepository] = None,
) -> FastAPI:
    if service is None:
        base_service = RiskDashboardService(RiskService.from_config(config))
        if config_path:
            service = AutoReloadingRiskDashboardService(
                base_service, config_path=config_path
            )
        else:
            service = base_service
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
    if performance_repository is None:
        performance_repository = PerformanceRepository(Path(reports_dir))
    app.state.performance_repository = performance_repository

    audit_logger = get_audit_logger(config.audit)
    app.state.audit_logger = audit_logger
    app.state.audit_settings = config.audit
    audit_log = logging.getLogger("risk_management.web.audit")

    def emit_audit(
        request: Request,
        action: str,
        details: Mapping[str, Any],
        *,
        actor: Optional[str] = None,
    ) -> None:
        logger_instance: Optional[AuditLogWriter] = getattr(
            request.app.state, "audit_logger", None
        )
        if not logger_instance:
            return
        resolved_actor = actor or request.session.get("user") or "unknown"
        try:
            logger_instance.log(action=action, actor=str(resolved_actor), details=dict(details))
        except Exception as exc:  # pragma: no cover - defensive guard
            audit_log.warning("Failed to write audit entry '%s': %s", action, exc)

    def resolve_audit_settings(request: Request) -> AuditSettings:
        settings = getattr(request.app.state, "audit_settings", None)
        if settings is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Audit logging is not enabled.",
            )
        return settings

    def resolve_audit_filters(
        request: Request,
        *,
        default_limit: Optional[int] = None,
    ) -> tuple[Optional[str], Optional[str], Optional[int]]:
        params = request.query_params
        action_filter = params.get("action") or None
        actor_filter = params.get("actor") or None
        limit_value: Optional[int] = None
        limit_raw = params.get("limit")
        if limit_raw is not None:
            parsed_limit = _parse_positive_int(limit_raw, "limit", maximum=5000)
            if parsed_limit is not None:
                limit_value = parsed_limit
        elif default_limit is not None:
            limit_value = default_limit
        return action_filter, actor_filter, limit_value

    def _build_grafana_context(conf: RealtimeConfig) -> dict[str, Any]:
        grafana_cfg = conf.grafana
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

        return {"dashboards": dashboards, "theme": grafana_cfg.theme}

    def _apply_config_state(conf: RealtimeConfig) -> None:
        app.state.scenarios = list(conf.scenarios)
        app.state.grafana_context = _build_grafana_context(conf)
        app.state.audit_settings = conf.audit

    _apply_config_state(config)
    if isinstance(service, AutoReloadingRiskDashboardService):
        service.add_reload_hook(_apply_config_state)

    templates_path = templates_dir or Path(__file__).with_name("templates")
    templates = Jinja2Templates(directory=str(templates_path))

    static_path = Path(__file__).with_name("static")
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

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

    def get_performance_repository(request: Request) -> PerformanceRepository:
        return request.app.state.performance_repository

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
    async def dashboard(request: Request, service: RiskDashboardService = Depends(get_service)) -> HTMLResponse:
        user = request.session.get("user")
        if not user:
            return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        snapshot = await service.fetch_snapshot()
        view_model = build_presentable_snapshot(
            snapshot,
            page=1,
            page_size=DEFAULT_ACCOUNTS_PAGE_SIZE,
            sort_key=DEFAULT_ACCOUNT_SORT_KEY,
            sort_order=DEFAULT_ACCOUNT_SORT_ORDER,
        )
        configured_scenarios: Sequence[Scenario] = request.app.state.scenarios
        scenario_results = simulate_scenarios(snapshot, configured_scenarios)
        view_model["scenarios"] = (
            scenario_results_to_dict(scenario_results) if scenario_results else []
        )
        grafana_context: dict[str, Any] = request.app.state.grafana_context
        return templates.TemplateResponse(
            "dashboard/index.html",
            {
                "request": request,
                "user": user,
                "snapshot": view_model,
                "grafana_dashboards": grafana_context.get("dashboards", []),
                "grafana_theme": grafana_context.get("theme"),
            },
        )

    @app.get("/api/snapshot", response_class=JSONResponse)
    async def api_snapshot(
        request: Request,
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        snapshot = await service.fetch_snapshot()
        params = request.query_params
        account_param = params.get("account")
        search_param = params.get("search")
        exposure_param = _normalise_exposure_filter(params.get("exposure"))
        sort_param = _normalise_sort_key(params.get("sort"))
        sort_order_param = _normalise_sort_order(params.get("sort_order"))
        page_param = _parse_positive_int(params.get("page"), "page")
        page_size_param = _parse_positive_int(
            params.get("page_size"),
            "page_size",
            maximum=MAX_ACCOUNTS_PAGE_SIZE,
        )
        view_model = build_presentable_snapshot(
            snapshot,
            account_name=account_param,
            search=search_param,
            exposure_filter=exposure_param,
            page=page_param,
            page_size=page_size_param,
            sort_key=sort_param,
            sort_order=sort_order_param,
        )
        configured_scenarios: Sequence[Scenario] = request.app.state.scenarios
        scenario_results = simulate_scenarios(snapshot, configured_scenarios)
        view_model["scenarios"] = (
            scenario_results_to_dict(scenario_results) if scenario_results else []
        )
        return JSONResponse(view_model)

    @app.get("/api/performance/portfolio", response_class=JSONResponse)
    async def api_portfolio_performance(
        start: Optional[str] = None,
        end: Optional[str] = None,
        repository: PerformanceRepository = Depends(get_performance_repository),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        try:
            series = repository.get_portfolio_series(start=start, end=end)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse({"series": series})

    @app.get("/api/performance/accounts/{account_name}", response_class=JSONResponse)
    async def api_account_performance(
        account_name: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        repository: PerformanceRepository = Depends(get_performance_repository),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        try:
            series = repository.get_account_series(account_name, start=start, end=end)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Account '{account_name}' not found",
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse({"account": account_name, "series": series})

    @app.get("/api/performance/metrics", response_class=JSONResponse)
    async def api_performance_metrics(
        start: Optional[str] = None,
        end: Optional[str] = None,
        account: Optional[str] = None,
        repository: PerformanceRepository = Depends(get_performance_repository),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        try:
            portfolio_series = repository.get_portfolio_series(start=start, end=end)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        payload: dict[str, object] = {
            "portfolio": build_performance_metrics(portfolio_series),
            "accounts": {},
            "filters": {"start": start, "end": end},
        }

        accounts: List[str]
        if account:
            accounts = [account]
        else:
            accounts = repository.list_accounts()

        accounts_metrics: Dict[str, object] = {}
        for account_name in accounts:
            try:
                series = repository.get_account_series(account_name, start=start, end=end)
            except KeyError:
                if account:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Account '{account_name}' not found",
                    ) from None
                continue
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            accounts_metrics[account_name] = build_performance_metrics(series)

        payload["accounts"] = accounts_metrics
        return JSONResponse(payload)

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

    @app.get(
        "/api/trading/accounts/{account_name}/stop-loss",
        response_class=JSONResponse,
    )
    async def api_get_account_stop_loss(
        account_name: str,
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        try:
            state = service.get_account_stop_loss(account_name)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return JSONResponse({"account": account_name, "stop_loss": state})

    @app.post(
        "/api/trading/accounts/{account_name}/stop-loss",
        response_class=JSONResponse,
    )
    async def api_set_account_stop_loss(
        account_name: str,
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
            state = await service.set_account_stop_loss(account_name, threshold)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(state)

    @app.delete(
        "/api/trading/accounts/{account_name}/stop-loss",
        response_class=JSONResponse,
    )
    async def api_clear_account_stop_loss(
        account_name: str,
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        try:
            await service.clear_account_stop_loss(account_name)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return JSONResponse({"status": "cleared"})

    @app.post(
        "/api/trading/accounts/{account_name}/orders/cancel-all",
        response_class=JSONResponse,
    )
    async def api_cancel_all_orders(
        account_name: str,
        request: Request,
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        symbol: Optional[str] = None
        if request.headers.get("content-length") not in (None, "0"):
            try:
                payload = await request.json()
            except Exception as exc:  # pragma: no cover
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload") from exc
            if isinstance(payload, Mapping):
                raw_symbol = payload.get("symbol")
                symbol = str(raw_symbol).strip() if raw_symbol is not None else None
        try:
            result = await service.cancel_all_orders(account_name, symbol=symbol)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.post(
        "/api/trading/accounts/{account_name}/positions/close-all",
        response_class=JSONResponse,
    )
    async def api_close_all_positions(
        account_name: str,
        request: Request,
        service: RiskDashboardService = Depends(get_service),
        _: str = Depends(require_user),
    ) -> JSONResponse:
        symbol: Optional[str] = None
        if request.headers.get("content-length") not in (None, "0"):
            try:
                payload = await request.json()
            except Exception as exc:  # pragma: no cover
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload") from exc
            if isinstance(payload, Mapping):
                raw_symbol = payload.get("symbol")
                symbol = str(raw_symbol).strip() if raw_symbol is not None else None
        try:
            result = await service.close_all_positions(account_name, symbol=symbol)
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

    @app.post("/api/kill-switch", response_class=JSONResponse)
    async def api_global_kill_switch(
        request: Request,
        service: RiskDashboardService = Depends(get_service),
        user: str = Depends(require_user),
    ) -> JSONResponse:
        results = await service.trigger_kill_switch()
        payload = _build_kill_switch_response(results)
        emit_audit(
            request,
            "kill_switch.global",
            {
                "success": payload.get("success"),
                "error_count": len(payload.get("errors", [])),
            },
            actor=user,
        )
        return JSONResponse(payload)

    @app.post("/api/accounts/{account_name}/kill-switch", response_class=JSONResponse)
    async def api_kill_switch(
        request: Request,
        account_name: str,
        service: RiskDashboardService = Depends(get_service),
        user: str = Depends(require_user),
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
        emit_audit(
            request,
            "kill_switch.account",
            {
                "account": target or "all",
                "symbol": symbol or "all",
                "success": payload.get("success"),
                "error_count": len(payload.get("errors", [])),
            },
            actor=user,
        )
        return JSONResponse(payload)

    @app.post(
        "/api/accounts/{account_name}/positions/{symbol:path}/kill-switch",
        response_class=JSONResponse,
    )
    async def api_position_kill_switch(
        request: Request,
        account_name: str,
        symbol: str,
        service: RiskDashboardService = Depends(get_service),
        user: str = Depends(require_user),
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
        emit_audit(
            request,
            "kill_switch.position",
            {
                "account": target_account or "all",
                "symbol": target_symbol,
                "success": payload.get("success"),
                "error_count": len(payload.get("errors", [])),
            },
            actor=user,
        )
        return JSONResponse(payload)

    @app.get("/audit", response_class=HTMLResponse)
    async def audit_dashboard_view(
        request: Request,
        _: str = Depends(require_user),
    ) -> HTMLResponse:
        settings = resolve_audit_settings(request)
        action_filter, actor_filter, limit_value = resolve_audit_filters(
            request, default_limit=200
        )
        entries = read_audit_entries(
            settings.log_path,
            action=action_filter,
            actor=actor_filter,
            limit=limit_value,
        )
        ordered = list(reversed(entries))
        download_params: dict[str, Any] = {}
        if action_filter:
            download_params["action"] = action_filter
        if actor_filter:
            download_params["actor"] = actor_filter
        if limit_value:
            download_params["limit"] = limit_value
        download_url = request.url_for("audit_download")
        if download_params:
            download_url = f"{download_url}?{urlencode(download_params)}"
        return templates.TemplateResponse(
            "audit/index.html",
            {
                "request": request,
                "entries": ordered,
                "filters": {
                    "action": action_filter or "",
                    "actor": actor_filter or "",
                    "limit": limit_value,
                },
                "download_url": download_url,
            },
        )

    @app.get("/api/audit", response_class=JSONResponse)
    async def api_audit_entries(
        request: Request,
        _: str = Depends(require_user),
    ) -> JSONResponse:
        settings = resolve_audit_settings(request)
        action_filter, actor_filter, limit_value = resolve_audit_filters(
            request, default_limit=200
        )
        entries = read_audit_entries(
            settings.log_path,
            action=action_filter,
            actor=actor_filter,
            limit=limit_value,
        )
        payload = {
            "entries": list(reversed(entries)),
            "filters": {
                "action": action_filter,
                "actor": actor_filter,
                "limit": limit_value,
            },
        }
        return JSONResponse(payload)

    @app.get("/audit/download", response_class=PlainTextResponse)
    async def audit_download(
        request: Request,
        _: str = Depends(require_user),
    ) -> PlainTextResponse:
        settings = resolve_audit_settings(request)
        action_filter, actor_filter, limit_value = resolve_audit_filters(request)
        entries = read_audit_entries(
            settings.log_path,
            action=action_filter,
            actor=actor_filter,
            limit=limit_value,
        )
        lines = [json.dumps(entry, sort_keys=True) for entry in entries]
        content = "\n".join(lines)
        if content:
            content += "\n"
        response = PlainTextResponse(content, media_type="application/jsonl")
        response.headers["Content-Disposition"] = "attachment; filename=audit-log.jsonl"
        return response

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

    @app.on_event("shutdown")
    async def shutdown() -> None:  # pragma: no cover - FastAPI lifecycle
        await service.close()

    return app
