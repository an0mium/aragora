from __future__ import annotations

import os
from typing import Mapping


_GITHUB_TOKEN_KEYS = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GH_ENTERPRISE_TOKEN",
    "GITHUB_ENTERPRISE_TOKEN",
)


def git_safe_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an environment without GitHub API tokens for git subprocesses."""
    env = dict(base_env or os.environ)
    for key in _GITHUB_TOKEN_KEYS:
        env.pop(key, None)
    return env
