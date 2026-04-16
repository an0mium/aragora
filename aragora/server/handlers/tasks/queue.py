"""Public REST surface for the developer task queue and lease lifecycle."""

from __future__ import annotations

import inspect
import logging
import sqlite3
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, cast

from aragora.server.http_utils import run_async as _run_async
from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from aragora.server.handlers.utils.rate_limit import rate_limit
from aragora.worktree.fleet import resolve_repo_root

logger = logging.getLogger(__name__)


def _await_if_needed(result: Any) -> Any:
    if inspect.isawaitable(result):
        return _run_async(cast(Coroutine[Any, Any, Any], result))
    return result


class TaskQueueHandler(BaseHandler):
    """Expose queue inspection plus lease-scoped lifecycle endpoints."""

    ROUTES = [
        "/api/v1/tasks/queue",
        "/api/v1/tasks/queue/stats",
        "/api/v1/tasks/queue/sync",
        "/api/v1/tasks/queue/{task_id}",
        "/api/v1/tasks/queue/{task_id}/claim",
        "/api/v1/tasks/leases",
        "/api/v1/tasks/leases/{lease_id}/heartbeat",
        "/api/v1/tasks/leases/{lease_id}/release",
        "/api/v1/tasks/leases/{lease_id}/complete",
        "/api/v1/tasks/salvage",
    ]

    def can_handle(self, path: str, method: str = "GET") -> bool:
        return (
            path.startswith("/api/v1/tasks/queue")
            or path.startswith("/api/v1/tasks/leases")
            or path == "/api/v1/tasks/salvage"
        )

    def _repo_root(self) -> Path:
        repo_hint = Path(str(self.ctx.get("repo_root", ".")))
        return resolve_repo_root(repo_hint)

    def _get_store(self):  # type: ignore[return]
        from aragora.nomic.dev_coordination import DevCoordinationStore

        return DevCoordinationStore(repo_root=self._repo_root())

    def _get_queue(self):  # type: ignore[return]
        from aragora.nomic.global_work_queue import GlobalWorkQueue

        queue = GlobalWorkQueue(storage_dir=self._repo_root() / ".work_queue")
        _run_async(queue.initialize())
        return queue

    @staticmethod
    def _serialize(item: Any) -> dict[str, Any]:
        if hasattr(item, "to_dict"):
            return item.to_dict()
        if isinstance(item, dict):
            return item
        raise TypeError(f"Unsupported payload type: {type(item)!r}")

    @staticmethod
    def _active_lease(store: Any, lease_id: str) -> Any | None:
        for lease in store.list_active_leases():
            if getattr(lease, "lease_id", "") == lease_id:
                return lease
        return None

    @staticmethod
    def _parse_queue_filters(query_params: dict[str, Any]) -> tuple[Any, Any, int] | HandlerResult:
        from aragora.nomic.global_work_queue import WorkStatus, WorkType

        status = query_params.get("status")
        work_type = query_params.get("work_type")
        limit_raw = query_params.get("limit", 20)
        try:
            limit = int(limit_raw)
        except (ValueError, TypeError):
            return error_response("limit must be an integer", 400)
        limit = max(1, min(limit, 100))

        try:
            parsed_status = WorkStatus(str(status)) if status else None
        except ValueError:
            return error_response(f"Invalid status: {status}", 400)
        try:
            parsed_type = WorkType(str(work_type)) if work_type else None
        except ValueError:
            return error_response(f"Invalid work_type: {work_type}", 400)
        return parsed_status, parsed_type, limit

    @rate_limit(requests_per_minute=60)
    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> HandlerResult | None:
        if not self.can_handle(path):
            return None

        user, err = self.require_auth_or_error(handler)
        if err:
            return err
        _, perm_err = self.require_permission_or_error(handler, "tasks:read")
        if perm_err:
            return perm_err
        _ = user

        if path == "/api/v1/tasks/leases":
            return self._handle_list_leases()
        if path == "/api/v1/tasks/salvage":
            return self._handle_list_salvage()
        if path == "/api/v1/tasks/queue/stats":
            return self._handle_stats()
        if path == "/api/v1/tasks/queue":
            return self._handle_list_queue(query_params)

        parts = path.rstrip("/").split("/")
        if len(parts) == 6 and parts[4] == "queue":
            return self._handle_get_task(parts[5])
        return error_response("Not found", 404)

    @handle_errors("task queue")
    @rate_limit(requests_per_minute=30)
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        if not self.can_handle(path):
            return None

        user, err = self.require_auth_or_error(handler)
        if err:
            return err
        _, perm_err = self.require_permission_or_error(handler, "tasks:execute")
        if perm_err:
            return perm_err
        _ = user

        body = self.read_json_body(handler) or {}
        parts = path.rstrip("/").split("/")

        if path == "/api/v1/tasks/queue/sync":
            return self._handle_sync(body)
        if len(parts) == 7 and parts[4] == "queue" and parts[6] == "claim":
            return self._handle_claim(parts[5], body)
        if len(parts) == 7 and parts[4] == "leases":
            lease_id = parts[5]
            action = parts[6]
            if action == "heartbeat":
                return self._handle_heartbeat(lease_id, body)
            if action == "release":
                return self._handle_release(lease_id)
            if action == "complete":
                return self._handle_complete(lease_id, body)
        return error_response("Not found", 404)

    def _handle_list_queue(self, query_params: dict[str, Any]) -> HandlerResult:
        parsed = self._parse_queue_filters(query_params)
        if isinstance(parsed, HandlerResult):
            return parsed
        status, work_type, limit = parsed
        queue = self._get_queue()
        items = _await_if_needed(queue.list_items(status=status, work_type=work_type, limit=limit))
        payload = [self._serialize(item) for item in items]
        return json_response({"data": payload, "count": len(payload)})

    def _handle_get_task(self, task_id: str) -> HandlerResult:
        queue = self._get_queue()
        item = _await_if_needed(queue.get(task_id))
        if item is None:
            return error_response(f"Task {task_id} not found", 404)
        return json_response({"data": self._serialize(item)})

    def _handle_list_leases(self) -> HandlerResult:
        store = self._get_store()
        payload = [lease.to_dict() for lease in store.list_active_leases()]
        return json_response({"data": payload, "count": len(payload)})

    def _handle_list_salvage(self) -> HandlerResult:
        store = self._get_store()
        payload = [candidate.to_dict() for candidate in store.list_salvage_candidates()]
        return json_response({"data": payload, "count": len(payload)})

    def _handle_stats(self) -> HandlerResult:
        queue = self._get_queue()
        return json_response({"data": _await_if_needed(queue.get_statistics())})

    def _handle_claim(self, task_id: str, body: dict[str, Any]) -> HandlerResult:
        from aragora.nomic.dev_coordination import LeaseConflictError

        store = self._get_store()
        queue = self._get_queue()
        title = str(body.get("title") or task_id).strip() or task_id
        expected_tests = [str(item) for item in body.get("expected_tests", []) if str(item).strip()]
        allowed_globs = [str(item) for item in body.get("allowed_globs", []) if str(item).strip()]
        claimed_paths = [str(item) for item in body.get("claimed_paths", []) if str(item).strip()]

        item = _await_if_needed(queue.get(task_id))
        developer_task = None
        if item is not None:
            payload = self._serialize(item)
            if title == task_id:
                title = str(payload.get("title") or title)
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            if isinstance(metadata, dict):
                if not allowed_globs:
                    allowed_globs = [
                        str(path) for path in metadata.get("allowed_paths", []) if str(path).strip()
                    ]
                if not expected_tests:
                    expected_tests = [
                        str(test)
                        for test in metadata.get("acceptance_checks", [])
                        if str(test).strip()
                    ]
        elif task_id.startswith("task:"):
            developer_task = store.get_developer_task(task_id.split("task:", 1)[1])
            if developer_task is not None:
                if title == task_id:
                    title = developer_task.title or title
                if not allowed_globs:
                    allowed_globs = [
                        str(path) for path in developer_task.allowed_paths if str(path).strip()
                    ]
                if not expected_tests:
                    expected_tests = [
                        str(test) for test in developer_task.acceptance_checks if str(test).strip()
                    ]

        if item is None and developer_task is None:
            return error_response(f"Task {task_id} not found", 404)

        ttl_hours = float(body.get("ttl_hours", 8.0))
        try:
            lease = store.claim_lease(
                task_id=task_id,
                title=title,
                owner_agent=str(body.get("owner_agent", "unknown")).strip() or "unknown",
                owner_session_id=str(body.get("owner_session_id", "public-api")).strip()
                or "public-api",
                branch=str(body.get("branch", "")).strip(),
                worktree_path=str(body.get("worktree_path", self._repo_root())).strip()
                or str(self._repo_root()),
                allowed_globs=allowed_globs,
                claimed_paths=claimed_paths,
                expected_tests=expected_tests,
                ttl_hours=ttl_hours,
                metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else None,
                allow_overlap=bool(body.get("allow_overlap", False)),
            )
        except LeaseConflictError as exc:
            return json_response({"error": str(exc), "conflicts": exc.conflicts}, status=409)
        return json_response({"data": lease.to_dict()}, status=201)

    def _handle_heartbeat(self, lease_id: str, body: dict[str, Any]) -> HandlerResult:
        store = self._get_store()
        ttl_hours = body.get("ttl_hours")
        try:
            lease = store.heartbeat_lease(
                lease_id, ttl_hours=float(ttl_hours) if ttl_hours is not None else None
            )
        except KeyError as exc:
            return error_response(str(exc), 404)
        return json_response({"data": lease.to_dict()})

    def _handle_release(self, lease_id: str) -> HandlerResult:
        store = self._get_store()
        try:
            lease = store.release_lease(lease_id)
        except KeyError as exc:
            return error_response(str(exc), 404)
        return json_response({"data": lease.to_dict()})

    def _handle_complete(self, lease_id: str, body: dict[str, Any]) -> HandlerResult:
        from aragora.nomic.dev_coordination import FileScopeViolationError, LeaseConflictError

        store = self._get_store()
        active_lease = self._active_lease(store, lease_id)
        try:
            receipt = store.record_completion(
                lease_id=lease_id,
                owner_agent=str(
                    body.get("owner_agent") or getattr(active_lease, "owner_agent", "") or "unknown"
                ),
                owner_session_id=str(
                    body.get("owner_session_id")
                    or getattr(active_lease, "owner_session_id", "")
                    or "public-api"
                ),
                branch=str(body.get("branch") or getattr(active_lease, "branch", "") or ""),
                worktree_path=str(
                    body.get("worktree_path")
                    or getattr(active_lease, "worktree_path", "")
                    or self._repo_root()
                ),
                base_sha=str(body.get("base_sha") or "").strip() or None,
                head_sha=str(body.get("head_sha") or "").strip() or None,
                commit_shas=[
                    str(item) for item in body.get("commit_shas", []) if str(item).strip()
                ],
                changed_paths=[
                    str(item) for item in body.get("changed_paths", []) if str(item).strip()
                ],
                tests_run=[str(item) for item in body.get("tests_run", []) if str(item).strip()],
                validations_run=[
                    str(item) for item in body.get("validations_run", []) if str(item).strip()
                ],
                assumptions=[
                    str(item) for item in body.get("assumptions", []) if str(item).strip()
                ],
                blockers=[str(item) for item in body.get("blockers", []) if str(item).strip()],
                outcome=str(body.get("outcome", "completed")).strip() or "completed",
                risks=[str(item) for item in body.get("risks", []) if str(item).strip()],
                pr_url=str(body.get("pr_url") or "").strip() or None,
                pr_number=body.get("pr_number"),
                confidence=float(body.get("confidence", 0.0)),
                metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else None,
            )
        except KeyError as exc:
            return error_response(str(exc), 404)
        except FileScopeViolationError as exc:
            return json_response({"error": str(exc), "violations": exc.violations}, status=409)
        except LeaseConflictError as exc:
            return json_response({"error": str(exc), "conflicts": exc.conflicts}, status=409)
        return json_response({"data": receipt.to_dict()})

    def _handle_sync(self, body: dict[str, Any]) -> HandlerResult:
        store = self._get_store()
        queue = self._get_queue()
        include_pending = bool(body.get("include_pending", True))
        include_developer_tasks = bool(body.get("include_developer_tasks", True))
        complete_missing = bool(body.get("complete_missing", True))
        payload: dict[str, Any] = {}
        try:
            if include_developer_tasks:
                payload["developer_tasks"] = _await_if_needed(
                    store.sync_developer_task_queue(queue, complete_missing=complete_missing)
                )
            if include_pending:
                payload["pending"] = _await_if_needed(
                    store.sync_pending_work_queue(queue, complete_missing=complete_missing)
                )
        except (RuntimeError, OSError, ValueError, sqlite3.Error) as exc:
            logger.warning("Task queue sync failed: %s", exc)
            return error_response(str(exc), 500)
        return json_response({"data": payload})
