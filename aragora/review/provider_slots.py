from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from typing import Any

from aragora.agents.credential_validator import get_credential_status
from aragora.agents.registry import AgentRegistry, register_all_agents

_CLI_TO_BINARY: dict[str, str] = {
    "claude": "claude",
    "codex": "codex",
    "openai": "openai",
    "gemini-cli": "gemini",
    "grok-cli": "grok",
    "qwen-cli": "qwen",
    "deepseek-cli": "deepseek",
    "kilocode": "kilocode",
}


@dataclass(frozen=True, slots=True)
class ProviderSlotDefinition:
    slot_id: str
    review_role: str
    lens: str
    family: str
    candidates: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProviderCandidateCheck:
    provider: str
    registered: bool
    allowlisted: bool
    available: bool
    status: str
    detail: str
    default_model: str | None = None
    requires: str | None = None
    env_vars: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProviderSlotResolution:
    slot_id: str
    review_role: str
    lens: str
    family: str
    selected_provider: str | None
    status: str
    detail: str
    candidates: list[str] = field(default_factory=list)
    candidate_checks: list[ProviderCandidateCheck] = field(default_factory=list)
    selected_allowlisted: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_packet_dict(self) -> dict[str, Any]:
        """Return the stable public packet shape mirrored by the UI types.

        Keep richer resolver-only diagnostics (`candidate_checks`,
        `selected_allowlisted`) available to Python callers without
        widening the serialized review-packet contract unnecessarily.
        """
        return {
            "slot_id": self.slot_id,
            "review_role": self.review_role,
            "lens": self.lens,
            "family": self.family,
            "selected_provider": self.selected_provider,
            "status": self.status,
            "detail": self.detail,
            "candidates": list(self.candidates),
        }


@dataclass(slots=True)
class ProviderSlotAvailabilitySummary:
    total_slots: int
    resolved_slots: int
    unresolved_slots: list[str] = field(default_factory=list)
    core_slots_total: int = 0
    core_slots_resolved: int = 0
    available_families: list[str] = field(default_factory=list)
    unresolved_families: list[str] = field(default_factory=list)
    opt_in_slots: list[str] = field(default_factory=list)
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProviderSlotResolver:
    """Resolve provider slots to available candidates and summarize readiness."""

    def __init__(self) -> None:
        register_all_agents()

    def resolve_slots(
        self, slot_definitions: tuple[ProviderSlotDefinition, ...] | list[ProviderSlotDefinition]
    ) -> list[ProviderSlotResolution]:
        return [self.resolve_slot(slot) for slot in slot_definitions]

    def summarize(
        self, resolutions: tuple[ProviderSlotResolution, ...] | list[ProviderSlotResolution]
    ) -> ProviderSlotAvailabilitySummary:
        resolved_slots = [slot for slot in resolutions if slot.selected_provider]
        unresolved_slots = [slot for slot in resolutions if not slot.selected_provider]
        core_slots = [slot for slot in resolutions if slot.lens == "core"]
        core_resolved = [slot for slot in core_slots if slot.selected_provider]

        available_families = sorted({slot.family for slot in resolved_slots})
        unresolved_families = sorted({slot.family for slot in unresolved_slots})
        opt_in_slots = sorted(
            slot.slot_id for slot in resolved_slots if slot.selected_allowlisted is False
        )

        return ProviderSlotAvailabilitySummary(
            total_slots=len(resolutions),
            resolved_slots=len(resolved_slots),
            unresolved_slots=[slot.slot_id for slot in unresolved_slots],
            core_slots_total=len(core_slots),
            core_slots_resolved=len(core_resolved),
            available_families=available_families,
            unresolved_families=unresolved_families,
            opt_in_slots=opt_in_slots,
            degraded=bool(unresolved_slots),
        )

    def resolve_slot(self, slot: ProviderSlotDefinition) -> ProviderSlotResolution:
        candidate_checks = [self._evaluate_candidate(candidate) for candidate in slot.candidates]
        selected = next((check for check in candidate_checks if check.available), None)
        if selected:
            return ProviderSlotResolution(
                slot_id=slot.slot_id,
                review_role=slot.review_role,
                lens=slot.lens,
                family=slot.family,
                selected_provider=selected.provider,
                status="available",
                detail=selected.detail,
                candidates=list(slot.candidates),
                candidate_checks=candidate_checks,
                selected_allowlisted=selected.allowlisted,
            )

        detail = "; ".join(f"{check.provider}: {check.detail}" for check in candidate_checks)
        return ProviderSlotResolution(
            slot_id=slot.slot_id,
            review_role=slot.review_role,
            lens=slot.lens,
            family=slot.family,
            selected_provider=None,
            status="unavailable",
            detail=f"No configured provider available for {slot.family}; {detail}",
            candidates=list(slot.candidates),
            candidate_checks=candidate_checks,
            selected_allowlisted=None,
        )

    def _evaluate_candidate(self, provider: str) -> ProviderCandidateCheck:
        spec = AgentRegistry.get_spec(provider)
        if spec is None:
            return ProviderCandidateCheck(
                provider=provider,
                registered=False,
                allowlisted=False,
                available=False,
                status="unregistered",
                detail="provider candidate is not registered in the agent registry",
            )

        allowlisted = AgentRegistry.validate_allowed(provider)
        opt_in_suffix = (
            "; provider is registered but not allowlisted by default" if not allowlisted else ""
        )
        binary = _CLI_TO_BINARY.get(provider)
        if binary:
            cli_path = shutil.which(binary)
            if cli_path:
                return ProviderCandidateCheck(
                    provider=provider,
                    registered=True,
                    allowlisted=allowlisted,
                    available=True,
                    status="available" if allowlisted else "available_opt_in",
                    detail=f"{binary} CLI available on PATH{opt_in_suffix}",
                    default_model=spec.default_model,
                    requires=spec.requires,
                    env_vars=spec.env_vars,
                )
            return ProviderCandidateCheck(
                provider=provider,
                registered=True,
                allowlisted=allowlisted,
                available=False,
                status="missing_cli" if allowlisted else "missing_cli_opt_in",
                detail=f"{binary} CLI not found on PATH{opt_in_suffix}",
                default_model=spec.default_model,
                requires=spec.requires,
                env_vars=spec.env_vars,
            )

        credential = get_credential_status(provider)
        if credential.is_available:
            availability_via = credential.available_via or "credentials configured"
            detail = f"{availability_via} configured"
            if not credential.live_ready:
                detail += " (connectivity not yet verified)"
            detail += opt_in_suffix
            return ProviderCandidateCheck(
                provider=provider,
                registered=True,
                allowlisted=allowlisted,
                available=True,
                status="available" if allowlisted else "available_opt_in",
                detail=detail,
                default_model=spec.default_model,
                requires=spec.requires,
                env_vars=spec.env_vars,
            )

        next_action = credential.next_action or "credentials not configured"
        return ProviderCandidateCheck(
            provider=provider,
            registered=True,
            allowlisted=allowlisted,
            available=False,
            status="missing_credentials" if allowlisted else "missing_credentials_opt_in",
            detail=f"{next_action}{opt_in_suffix}",
            default_model=spec.default_model,
            requires=spec.requires,
            env_vars=spec.env_vars,
        )


__all__ = [
    "ProviderCandidateCheck",
    "ProviderSlotAvailabilitySummary",
    "ProviderSlotDefinition",
    "ProviderSlotResolution",
    "ProviderSlotResolver",
]
