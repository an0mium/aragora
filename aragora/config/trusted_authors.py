"""Configurable trusted-author / automation-identity allowlists.

The committed defaults intentionally contain only generic automation identities
(bot accounts, automation name fragments) and never a personal GitHub login, so
a public fork does not auto-trust an operator's handle for auto-merge or triage
decisions. Operators add their own logins via ``ARAGORA_TRUSTED_AUTHORS`` (comma
separated); individual call sites may additionally honor a context-specific
environment variable.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

TRUSTED_AUTHORS_ENV = "ARAGORA_TRUSTED_AUTHORS"


def _tokens(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {token.strip() for token in raw.split(",") if token.strip()}


def resolve_trusted_authors(
    defaults: Iterable[str] = (),
    *,
    env_vars: Iterable[str] = (TRUSTED_AUTHORS_ENV,),
    env: Mapping[str, str] | None = None,
) -> frozenset[str]:
    """Return ``defaults`` unioned with comma-separated logins from ``env_vars``.

    ``env`` defaults to ``os.environ``. Surrounding whitespace is trimmed and
    empty entries are dropped.
    """
    source = os.environ if env is None else env
    result: set[str] = {entry.strip() for entry in defaults if entry and entry.strip()}
    for var in env_vars:
        result |= _tokens(source.get(var))
    return frozenset(result)
