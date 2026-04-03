"""Backbone run ledger read handlers."""

from __future__ import annotations

import logging
from typing import Any

from aragora.pipeline.backbone_contracts import RunLedger, RunStageEvent
from aragora.server.handlers.base import BaseHandler, HandlerResult, error_response, json_response
from aragora.server.handlers.utils.routing import RouteDispatcher
from aragora.server.validation.query_params import safe_query_int

logger = logging.getLogger(__name__)

_ROUTES = [
    "/api/v1/runs",
    "/api/runs",
]
_RUN_PREFIX = "/api/v1/runs/"
_RUN_PREFIX_UNVERSIONED = "/api/runs/"


def _get_plan_store() -> Any:
    from aragora.pipeline.plan_store import get_plan_store

    return get_plan_store()


def _get_backbone_run(store: Any, run_id: str) -> RunLedger | None:
    getter = getattr(store, "get_backbone_run", None)
    if callable(getter):
        return getter(run_id)

    fallback = getattr(store, "get_run", None)
    if callable(fallback):
        return fallback(run_id)

    raise AttributeError("PlanStore does not expose a backbone run getter")


def _list_backbone_runs(
    store: Any,
    *,
    status: str | None = None,
    execution_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[RunLedger]:
    getter = getattr(store, "list_backbone_runs", None)
    if callable(getter):
        return list(
            getter(
                status=status,
                execution_id=execution_id,
                limit=limit,
                offset=offset,
            )
        )

    fallback = getattr(store, "list_runs", None)
    if callable(fallback):
        return list(
            fallback(
                status=status,
                execution_id=execution_id,
                limit=limit,
                offset=offset,
            )
        )

    raise AttributeError("PlanStore does not expose a backbone run lister")


def _count_backbone_runs(
    store: Any,
    *,
    status: str | None = None,
    execution_id: str | None = None,
    fallback_count: int,
) -> int:
    counter = getattr(store, "count_backbone_runs", None)
    if callable(counter):
        return int(counter(status=status, execution_id=execution_id))

    fallback = getattr(store, "count_runs", None)
    if callable(fallback):
        return int(fallback(status=status, execution_id=execution_id))

    return fallback_count


def _coerce_stage_payload(event: RunStageEvent) -> dict[str, str]:
    return {
        "stage": str(event.stage or "").strip(),
        "status": str(event.status or "").strip(),
    }


def _extract_safety_mode(run: RunLedger) -> str:
    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    for key in ("safety_mode", "execution_mode"):
        value = str(metadata.get(key, "") or "").strip()
        if value:
            return value

    for event in reversed(list(run.stage_events or [])):
        details = event.details if isinstance(event.details, dict) else {}
        for key in ("safety_mode", "execution_mode"):
            value = str(details.get(key, "") or "").strip()
            if value:
                return value

    return ""


def serialize_run_summary(run: RunLedger) -> dict[str, Any]:
    return {
        "run_id": str(run.run_id or "").strip(),
        "status": str(run.status or "").strip(),
        "stages": [_coerce_stage_payload(event) for event in list(run.stage_events or [])],
        "execution_id": str(run.execution_id or "").strip(),
        "receipt_id": str(run.receipt_id or "").strip(),
        "safety_mode": _extract_safety_mode(run),
    }


def serialize_run_detail(run: RunLedger) -> dict[str, Any]:
    payload = serialize_run_summary(run)
    payload.update(
        {
            "entrypoint": str(run.entrypoint or "").strip(),
            "plan_id": str(run.plan_id or "").strip(),
            "debate_id": str(run.debate_id or "").strip(),
            "created_at": str(run.created_at or "").strip(),
            "updated_at": str(run.updated_at or "").strip(),
        }
    )
    return payload


class RunsHandler(BaseHandler):
    """Read-only handler for persisted backbone run ledgers."""

    def __init__(self, ctx: dict[str, Any]) -> None:
        super().__init__(ctx)
        self._get_dispatcher = RouteDispatcher()
        self._get_dispatcher.add_route("/api/v1/runs", self._list_runs)
        self._get_dispatcher.add_route("/api/runs", self._list_runs)
        self._get_dispatcher.add_route("/api/v1/runs/{run_id}", self._get_run)
        self._get_dispatcher.add_route("/api/runs/{run_id}", self._get_run)

    def can_handle(self, path: str) -> bool:
        if path in _ROUTES:
            return True
        if path.startswith(_RUN_PREFIX) or path.startswith(_RUN_PREFIX_UNVERSIONED):
            return True
        return False

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        self.set_request_context(handler, query_params)
        result = self._get_dispatcher.dispatch(path, query_params)
        if result is not None:
            return result
        return self._try_get_by_id(path, query_params)

    def _list_runs(self, query_params: dict[str, Any]) -> HandlerResult:
        store = _get_plan_store()
        status = str(query_params.get("status", "") or "").strip() or None
        execution_id = str(query_params.get("execution_id", "") or "").strip() or None
        limit = safe_query_int(query_params, "limit", default=50, min_val=1, max_val=200)
        offset = safe_query_int(query_params, "offset", default=0, min_val=0, max_val=100000)

        try:
            runs = _list_backbone_runs(
                store,
                status=status,
                execution_id=execution_id,
                limit=limit,
                offset=offset,
            )
        except AttributeError as exc:
            logger.warning("Run ledger listing unavailable: %s", exc)
            return error_response("Run ledger storage unavailable", 500)

        total = _count_backbone_runs(
            store,
            status=status,
            execution_id=execution_id,
            fallback_count=len(runs),
        )

        return json_response(
            {
                "runs": [serialize_run_summary(run) for run in runs],
                "total": total,
                "limit": limit,
                "offset": offset,
            }
        )

    def _get_run(self, params: dict[str, str], query_params: dict[str, Any]) -> HandlerResult:
        del query_params
        run_id = params["run_id"]
        store = _get_plan_store()
        try:
            run = _get_backbone_run(store, run_id)
        except AttributeError as exc:
            logger.warning("Run ledger lookup unavailable: %s", exc)
            return error_response("Run ledger storage unavailable", 500)

        if run is None:
            return error_response(f"Run not found: {run_id}", 404)

        return json_response(serialize_run_detail(run))

    def _try_get_by_id(self, path: str, query_params: dict[str, Any]) -> HandlerResult | None:
        for prefix in (_RUN_PREFIX, _RUN_PREFIX_UNVERSIONED):
            if path.startswith(prefix):
                remainder = path[len(prefix) :]
                if "/" not in remainder and remainder:
                    return self._get_run({"run_id": remainder}, query_params)
        return None


__all__ = [
    "RunsHandler",
    "serialize_run_detail",
    "serialize_run_summary",
]
