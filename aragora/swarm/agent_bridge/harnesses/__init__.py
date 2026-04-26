from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any

from .base import BinaryResolver
from .base import Runner
from .base import Transport
from .claude import ClaudeTransport
from .codex import CodexTransport
from .droid import DroidTransport

TransportClass = type[Transport]
_REGISTRY: dict[str, TransportClass] = {}


def register(harness_name: str, harness_class: TransportClass) -> None:
    _REGISTRY[harness_name] = harness_class


def get_transport_class(harness_name: str) -> TransportClass:
    return _REGISTRY[harness_name]


def create_transport(
    harness_name: str,
    *,
    cwd: Path,
    model: str | None = None,
    harness_options: dict[str, Any] | None = None,
    runner: Runner | None = None,
    binary_resolver: BinaryResolver | None = None,
) -> Transport:
    harness_class = get_transport_class(harness_name)
    return harness_class(
        cwd=cwd,
        model=model,
        harness_options=harness_options,
        runner=runner or subprocess.run,
        binary_resolver=binary_resolver or shutil.which,
    )


register("claude", ClaudeTransport)
register("claude_code", ClaudeTransport)
register("codex", CodexTransport)
register("droid", DroidTransport)

__all__ = [
    "ClaudeTransport",
    "CodexTransport",
    "DroidTransport",
    "Transport",
    "create_transport",
    "get_transport_class",
    "register",
]
