"""Lazy GitHub App auth helpers for automation scripts.

Importing ``aragora.swarm.github_app_auth`` through the package path eagerly
loads the heavy ``aragora.swarm`` facade. These scripts only need the auth
helpers when a ``gh`` command runs, so load the helper module directly.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
_AUTH_MODULE: ModuleType | None = None


def _load_auth_module() -> ModuleType:
    global _AUTH_MODULE
    if _AUTH_MODULE is not None:
        return _AUTH_MODULE

    auth_path = REPO_ROOT / "aragora" / "swarm" / "github_app_auth.py"
    spec = importlib.util.spec_from_file_location("_aragora_github_app_auth", auth_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load GitHub App auth helper from {auth_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    _AUTH_MODULE = module
    return module


def github_cli_env(
    base_env: Mapping[str, str] | None = None,
    *,
    prefer_app: bool = True,
) -> dict[str, str]:
    try:
        module = _load_auth_module()
        helper = getattr(module, "github_cli_env")
    except Exception:
        return dict(os.environ if base_env is None else base_env)
    return helper(base_env, prefer_app=prefer_app)


def gh_subprocess_run(
    args: Sequence[str],
    *,
    timeout: float = 30.0,
    prefer_app: bool = True,
    write_op: bool = False,
    env: Mapping[str, str] | None = None,
    max_retries: int = 3,
    base_backoff: float = 5.0,
    max_backoff: float = 600.0,
    sleep: Callable[[float], None] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        module = _load_auth_module()
        runner = getattr(module, "gh_subprocess_run")
    except Exception:
        return subprocess.run(
            ["gh", *list(args)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(os.environ if env is None else env),
            check=False,
        )
    return runner(
        args,
        timeout=timeout,
        prefer_app=prefer_app,
        write_op=write_op,
        env=env,
        max_retries=max_retries,
        base_backoff=base_backoff,
        max_backoff=max_backoff,
        sleep=sleep,
    )


def gh_subprocess_iter_buckets(
    env: Mapping[str, str] | None = None,
) -> dict[str, dict[str, int]]:
    try:
        module = _load_auth_module()
        helper = getattr(module, "gh_subprocess_iter_buckets")
    except Exception:
        return {}
    result: Any = helper(env)
    return result if isinstance(result, dict) else {}
