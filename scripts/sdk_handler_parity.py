#!/usr/bin/env python3
"""
Generate SDK ↔ handler parity report.

Compares SDK endpoints (TS + Python) to the live handler registry to
identify which SDK methods have no corresponding server handler.
"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path
from typing import Iterable

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from sdk_parity_audit import Endpoint, iter_files, parse_python_sdk, parse_ts_sdk

from aragora.server.handler_registry import HandlerRegistryMixin, get_route_index


class _RegistryProbe(HandlerRegistryMixin):
    """Minimal registry host for building the route index."""

    storage = None
    stream_emitter = None
    control_plane_stream = None
    nomic_loop_stream = None
    elo_system = None
    debate_embeddings = None
    critique_store = None
    document_store = None
    persona_manager = None
    position_ledger = None
    user_store = None
    nomic_state_file = None
    wfile = io.BytesIO()

    def _add_cors_headers(self) -> None:
        return None

    def _add_security_headers(self) -> None:
        return None

    def send_response(self, status: int) -> None:  # pragma: no cover - no network use
        self._status = status

    def send_header(self, name: str, value: str) -> None:  # pragma: no cover - no network use
        return None

    def end_headers(self) -> None:  # pragma: no cover - no network use
        return None


def handler_supports_method(handler: object, method: str) -> bool:
    method = method.upper()
    if method == "GET":
        return hasattr(handler, "handle_get") or hasattr(handler, "handle")
    if method == "POST":
        return hasattr(handler, "handle_post") or hasattr(handler, "handle")
    if method == "PUT":
        return hasattr(handler, "handle_put") or hasattr(handler, "handle")
    if method == "PATCH":
        return hasattr(handler, "handle_patch") or hasattr(handler, "handle")
    if method == "DELETE":
        return hasattr(handler, "handle_delete") or hasattr(handler, "handle")
    return hasattr(handler, "handle")


def is_handled(endpoint: Endpoint) -> bool:
    registry = _RegistryProbe()
    registry._init_handlers()
    route_index = get_route_index()
    handler_entry = route_index.get_handler(endpoint.path)
    if not handler_entry:
        return False
    _, handler = handler_entry
    return handler_supports_method(handler, endpoint.method)


def load_sdk_endpoints() -> tuple[set[Endpoint], set[Endpoint]]:
    ts_paths = list(iter_files(Path("sdk/typescript/src"), ".ts"))
    py_paths = list(iter_files(Path("sdk/python/aragora/namespaces"), ".py"))
    return parse_ts_sdk(ts_paths), parse_python_sdk(py_paths)


def render_report(
    missing_ts: list[Endpoint],
    missing_py: list[Endpoint],
) -> str:
    lines = [
        "# SDK ↔ Handler Parity Report",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "## Summary",
        f"- SDK TS endpoints not implemented by handlers: {len(missing_ts)}",
        f"- SDK PY endpoints not implemented by handlers: {len(missing_py)}",
        "",
        "These endpoints likely represent stale SDK methods or legacy routes. They should be removed,",
        "deprecated, or reintroduced in the server if still required.",
        "",
        "## TypeScript SDK endpoints not in handlers",
        "",
        "```text",
    ]
    for ep in missing_ts:
        lines.append(ep.display())
    lines.extend(["```", "", "## Python SDK endpoints not in handlers", "", "```text"])
    for ep in missing_py:
        lines.append(ep.display())
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ts_endpoints, py_endpoints = load_sdk_endpoints()

    missing_ts = sorted(
        [ep for ep in ts_endpoints if not is_handled(ep)],
        key=lambda e: (e.method, e.path),
    )
    missing_py = sorted(
        [ep for ep in py_endpoints if not is_handled(ep)],
        key=lambda e: (e.method, e.path),
    )

    report = render_report(missing_ts, missing_py)
    Path("docs/SDK_HANDLER_PARITY.md").write_text(report)


if __name__ == "__main__":
    main()
