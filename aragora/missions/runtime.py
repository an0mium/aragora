"""Runtime configuration for native missions."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "xai": ("XAI_API_KEY", "GROK_API_KEY"),
    "mistral": ("MISTRAL_API_KEY",),
}


@dataclass(frozen=True)
class MissionRuntimeConfig:
    """Local/headless runtime selection without changing feature-flag defaults."""

    mode: str = "local-cli"
    available_provider_env_vars: dict[str, str] = field(default_factory=dict)
    enables_native_mission_flag: bool = False

    @classmethod
    def from_env(cls) -> MissionRuntimeConfig:
        mode = os.getenv("ARAGORA_MISSION_RUNTIME", "local-cli")
        available: dict[str, str] = {}
        for provider, env_vars in _PROVIDER_ENV_VARS.items():
            present = next((name for name in env_vars if os.getenv(name)), None)
            if present:
                available[provider] = present
        enables_flag = os.getenv("ARAGORA_ENABLE_NATIVE_MISSION", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return cls(
            mode=mode,
            available_provider_env_vars=available,
            enables_native_mission_flag=enables_flag,
        )
