"""
Canonical frontier-model pin registry.

All code that needs a "best available" model for a given role should import
constants from this module instead of hardcoding IDs. The goal is:

1. One place to bump the frontier (Opus 4.7 -> 4.8, GPT 5.5 -> 5.6, etc.)
2. OpenRouter aliases are the default transport so a missing direct-provider
   key never blocks functionality. Set ARAGORA_ROUTE_THROUGH_OPENROUTER=true
   to force every call through OpenRouter even if a direct key is present.
3. Direct-provider IDs are still exposed for code paths that prefer to hit
   the native API when a key is available and the router allows it.

Naming convention:
- ``*_VIA_OPENROUTER`` -> the alias you pass to ``OpenRouterAgent``
  (e.g. ``anthropic/claude-opus-4.7``).
- ``*_DIRECT``         -> the raw model ID the native provider expects
  (e.g. ``claude-opus-4-7``).

Role-keyed helpers (``frontier_model_for_role``, ``openrouter_alias_for_role``)
return the best pin for a debate role (proposer, critic, synthesizer, etc.).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Final, Literal

from aragora.config.secrets import get_secret_presence

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Frontier pins (user-requested floor: Opus 4.7 / GPT 5.5 / Gemini 3.1 Pro)
# -----------------------------------------------------------------------------

# Anthropic Claude Opus 4.7 - top-tier reasoning, debate, synthesis
OPUS_47_DIRECT: Final = "claude-opus-4-7"
OPUS_47_VIA_OPENROUTER: Final = "anthropic/claude-opus-4.7"

# OpenAI GPT-5.5 - top-tier general reasoning
GPT55_DIRECT: Final = "gpt-5.5"
GPT55_VIA_OPENROUTER: Final = "openai/gpt-5.5"
# Backwards-compatible constant names for callers that have not migrated yet.
GPT54_DIRECT: Final = GPT55_DIRECT
GPT54_VIA_OPENROUTER: Final = GPT55_VIA_OPENROUTER

# Google Gemini 3.1 Pro - top-tier long-context + multimodal
GEMINI_31_PRO_DIRECT: Final = "gemini-3.1-pro"
GEMINI_31_PRO_VIA_OPENROUTER: Final = "google/gemini-3.1-pro"

# xAI Grok 4 (latest) - contrarian / contrarian-by-design agent
GROK_4_DIRECT: Final = "grok-4-latest"
GROK_4_VIA_OPENROUTER: Final = "x-ai/grok-4"

# Mistral Large (latest) - European provider diversity
MISTRAL_LARGE_DIRECT: Final = "mistral-large-2512"
MISTRAL_LARGE_VIA_OPENROUTER: Final = "mistralai/mistral-large"


# -----------------------------------------------------------------------------
# Canonical-metrics + legacy underscored aliases
# -----------------------------------------------------------------------------
#
# ``docs/status/claims/canonical_metrics.yaml`` and
# ``scripts/check_canonical_metrics.py`` look for the underscored
# frontier names (``OPUS_4_7``, ``GPT_5_4``, ``GEMINI_3_1_PRO``).
# These map to the same direct-provider IDs as the ``*_DIRECT``
# constants above; expose them at module scope so the security
# canonical-metrics gate can see that the frontier floor is honored.
OPUS_4_7: Final = OPUS_47_DIRECT
GPT_5_4: Final = GPT55_DIRECT
GEMINI_3_1_PRO: Final = GEMINI_31_PRO_DIRECT


# -----------------------------------------------------------------------------
# Frontier bundle per debate role
# -----------------------------------------------------------------------------

Role = Literal[
    "proposer",
    "critic",
    "synthesizer",
    "devils_advocate",
    "researcher",
    "reviewer",
    "quality_reviewer",
    "security_auditor",
    "compliance_auditor",
    "judge",
    "default",
]


@dataclass(frozen=True)
class _RolePin:
    """Preferred frontier pin for a role, expressed both as direct and OpenRouter IDs."""

    direct: str
    openrouter: str


_ROLE_TO_PIN: Final[dict[Role, _RolePin]] = {
    # Anthropic leads on adversarial reasoning, nuance, and long-form synthesis,
    # so it is the default for the core debate roles.
    "proposer": _RolePin(OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "critic": _RolePin(OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "synthesizer": _RolePin(OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "devils_advocate": _RolePin(GROK_4_DIRECT, GROK_4_VIA_OPENROUTER),
    "researcher": _RolePin(GEMINI_31_PRO_DIRECT, GEMINI_31_PRO_VIA_OPENROUTER),
    "reviewer": _RolePin(GPT55_DIRECT, GPT55_VIA_OPENROUTER),
    "quality_reviewer": _RolePin(OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "security_auditor": _RolePin(OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "compliance_auditor": _RolePin(OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "judge": _RolePin(OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "default": _RolePin(OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
}


# -----------------------------------------------------------------------------
# Routing policy
# -----------------------------------------------------------------------------


def route_through_openrouter() -> bool:
    """Force every frontier call through OpenRouter regardless of direct keys.

    Enabled when ``ARAGORA_ROUTE_THROUGH_OPENROUTER`` is truthy OR when no
    direct Anthropic key is set (so the benchmark never blocks on a missing
    provider key).
    """
    forced = os.environ.get("ARAGORA_ROUTE_THROUGH_OPENROUTER", "").strip().lower()
    if forced in {"1", "true", "yes", "on"}:
        return True

    # Auto-fallback: no direct Anthropic key -> OpenRouter becomes primary.
    if get_secret_presence("ANTHROPIC_API_KEY").source not in {"aws", "env"}:
        return True

    return False


def frontier_model_for_role(role: Role = "default") -> str:
    """Return the best frontier model ID for a role.

    If OpenRouter routing is forced (see :func:`route_through_openrouter`),
    returns the OpenRouter alias so callers can pass it straight to
    ``OpenRouterAgent``. Otherwise returns the direct-provider ID.
    """
    pin = _ROLE_TO_PIN.get(role, _ROLE_TO_PIN["default"])
    return pin.openrouter if route_through_openrouter() else pin.direct


def openrouter_alias_for_role(role: Role = "default") -> str:
    """Return the OpenRouter alias for a role, regardless of routing policy."""
    pin = _ROLE_TO_PIN.get(role, _ROLE_TO_PIN["default"])
    return pin.openrouter


def direct_model_for_role(role: Role = "default") -> str:
    """Return the direct-provider model ID for a role, regardless of routing policy."""
    pin = _ROLE_TO_PIN.get(role, _ROLE_TO_PIN["default"])
    return pin.direct


# -----------------------------------------------------------------------------
# Legacy aliases mapped to the new frontier
# -----------------------------------------------------------------------------
#
# Any code that still references an older Claude/GPT/Gemini ID can pass it
# through :func:`upgrade_legacy_pin` to transparently get the frontier.
# This is the migration handle for the ~400 hardcoded IDs across the codebase
# without doing a risky global sed.

_LEGACY_UPGRADES: Final[dict[str, tuple[str, str]]] = {
    # Claude family -> Opus 4.7
    "claude-opus-4-5-20251101": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "claude-opus-4-5": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "claude-opus-4": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "claude-sonnet-4-6": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "claude-sonnet-4.6": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "claude-sonnet-4-20250514": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "claude-sonnet-4": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "claude-haiku-4-5-20251001": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "claude-haiku-4.5": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "claude-haiku-4-20250514": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "claude-haiku-4": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "claude-3-5-sonnet-20241022": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "claude-3-5-sonnet": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "claude-3-opus-20240229": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "claude-3-opus": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "claude-3-haiku-20240307": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "claude-3-haiku": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    # GPT family -> GPT-5.5
    "gpt-4.1": (GPT55_DIRECT, GPT55_VIA_OPENROUTER),
    "gpt-4.1-mini": (GPT55_DIRECT, GPT55_VIA_OPENROUTER),
    "gpt-4.1-nano": (GPT55_DIRECT, GPT55_VIA_OPENROUTER),
    "gpt-4o": (GPT55_DIRECT, GPT55_VIA_OPENROUTER),
    "gpt-4o-mini": (GPT55_DIRECT, GPT55_VIA_OPENROUTER),
    "gpt-4": (GPT55_DIRECT, GPT55_VIA_OPENROUTER),
    "gpt-4-turbo": (GPT55_DIRECT, GPT55_VIA_OPENROUTER),
    "gpt-5": (GPT55_DIRECT, GPT55_VIA_OPENROUTER),
    "gpt-5.3": (GPT55_DIRECT, GPT55_VIA_OPENROUTER),
    "gpt-5.3-codex": (GPT55_DIRECT, GPT55_VIA_OPENROUTER),
    "gpt-5.4": (GPT55_DIRECT, GPT55_VIA_OPENROUTER),
    "gpt-5.4-pro": (GPT55_DIRECT, GPT55_VIA_OPENROUTER),
    # OpenRouter-style legacy -> OpenRouter-style frontier
    "anthropic/claude-opus-4.5": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "anthropic/claude-sonnet-4": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "anthropic/claude-sonnet-4.6": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "anthropic/claude-haiku-4.5": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "anthropic/claude-3.5-sonnet": (OPUS_47_DIRECT, OPUS_47_VIA_OPENROUTER),
    "openai/gpt-4o": (GPT55_DIRECT, GPT55_VIA_OPENROUTER),
    "openai/gpt-4-turbo": (GPT55_DIRECT, GPT55_VIA_OPENROUTER),
    "openai/gpt-5.4": (GPT55_DIRECT, GPT55_VIA_OPENROUTER),
    # Gemini family -> Gemini 3.1 Pro
    "gemini-2.5-pro": (GEMINI_31_PRO_DIRECT, GEMINI_31_PRO_VIA_OPENROUTER),
    "gemini-2.5-flash": (GEMINI_31_PRO_DIRECT, GEMINI_31_PRO_VIA_OPENROUTER),
    "gemini-1.5-pro": (GEMINI_31_PRO_DIRECT, GEMINI_31_PRO_VIA_OPENROUTER),
    "gemini-1.5-flash": (GEMINI_31_PRO_DIRECT, GEMINI_31_PRO_VIA_OPENROUTER),
    "gemini-3.1-pro-preview": (GEMINI_31_PRO_DIRECT, GEMINI_31_PRO_VIA_OPENROUTER),
    "openrouter/google/gemini-3.1-pro-preview": (
        GEMINI_31_PRO_DIRECT,
        GEMINI_31_PRO_VIA_OPENROUTER,
    ),
    "google/gemini-2.5-pro": (GEMINI_31_PRO_DIRECT, GEMINI_31_PRO_VIA_OPENROUTER),
    "google/gemini-2.5-flash": (GEMINI_31_PRO_DIRECT, GEMINI_31_PRO_VIA_OPENROUTER),
}


def upgrade_legacy_pin(model_id: str) -> str:
    """Upgrade a legacy model ID to the current frontier.

    Returns the OpenRouter alias when OpenRouter routing is active, otherwise
    the direct-provider ID. Unknown IDs are returned unchanged so this can be
    called on any string without risk.
    """
    hit = _LEGACY_UPGRADES.get(model_id)
    if hit is None:
        return model_id
    direct, via_or = hit
    return via_or if route_through_openrouter() else direct


__all__ = [
    "OPUS_47_DIRECT",
    "OPUS_47_VIA_OPENROUTER",
    "GPT55_DIRECT",
    "GPT55_VIA_OPENROUTER",
    "GPT54_DIRECT",
    "GPT54_VIA_OPENROUTER",
    "GEMINI_31_PRO_DIRECT",
    "GEMINI_31_PRO_VIA_OPENROUTER",
    "GROK_4_DIRECT",
    "GROK_4_VIA_OPENROUTER",
    "MISTRAL_LARGE_DIRECT",
    "MISTRAL_LARGE_VIA_OPENROUTER",
    "OPUS_4_7",
    "GPT_5_4",
    "GEMINI_3_1_PRO",
    "Role",
    "route_through_openrouter",
    "frontier_model_for_role",
    "openrouter_alias_for_role",
    "direct_model_for_role",
    "upgrade_legacy_pin",
]
