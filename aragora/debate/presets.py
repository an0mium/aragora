"""SME-friendly configuration presets for ArenaConfig.

Provides one-line configuration shortcuts so SMEs don't need to discover
18+ boolean flags individually.

Usage:
    from aragora.debate.presets import get_preset, apply_preset

    # Get raw preset dict
    preset = get_preset("sme")

    # Apply with overrides
    merged = apply_preset("sme", overrides={"enable_telemetry": True})

    # Use with ArenaConfig
    config = ArenaConfig(**apply_preset("sme"))
"""

from __future__ import annotations

from typing import Any


_PRESETS: dict[str, dict[str, Any]] = {
    "sme": {
        # Audit trail
        "enable_receipt_generation": True,
        "enable_receipt_auto_sign": True,
        "enable_provenance": True,
        "enable_bead_tracking": True,
        # Knowledge
        "enable_knowledge_extraction": True,
        "enable_auto_revalidation": True,
        # Budget
        "budget_downgrade_models": True,
        # Memory
        "enable_supermemory": True,
        # Convergence
        "enable_stability_detection": True,
        # Tracking
        "enable_position_ledger": True,
        # Compliance
        "enable_compliance_artifacts": True,
    },
    "enterprise": {
        # Everything in SME
        "enable_receipt_generation": True,
        "enable_receipt_auto_sign": True,
        "enable_provenance": True,
        "enable_bead_tracking": True,
        "enable_knowledge_extraction": True,
        "enable_auto_revalidation": True,
        "budget_downgrade_models": True,
        "enable_supermemory": True,
        "enable_stability_detection": True,
        "enable_position_ledger": True,
        "enable_compliance_artifacts": True,
        # Enterprise extras
        "enable_telemetry": True,
        "use_airlock": True,
        "enable_performance_monitor": True,
        # Advanced debate features
        "enable_debate_forking": True,
        "enable_unified_voting": True,
        # Knowledge flywheel (Receipt -> KM -> Next Debate)
        "enable_adaptive_consensus": True,
        "enable_synthesis": True,
        "enable_knowledge_injection": True,
        "enable_meta_learning": True,
        # Post-debate pipeline (converted to PostDebateConfig in get_preset)
        "_post_debate_preset": {
            "auto_explain": True,
            "auto_create_plan": True,
            "auto_notify": True,
            "auto_persist_receipt": True,
            "auto_gauntlet_validate": True,
            "auto_queue_improvement": True,
        },
    },
    "minimal": {
        # Cheap & fast
        "enable_stability_detection": True,
        "budget_downgrade_models": True,
    },
    "audit": {
        # Maximum traceability
        "enable_receipt_generation": True,
        "enable_receipt_auto_sign": True,
        "enable_provenance": True,
        "enable_bead_tracking": True,
        "enable_compliance_artifacts": True,
        "enable_position_ledger": True,
        "enable_telemetry": True,
        # Post-debate pipeline (converted to PostDebateConfig in get_preset)
        "_post_debate_preset": {
            "auto_explain": True,
            "auto_create_plan": True,
            "auto_persist_receipt": True,
            "auto_gauntlet_validate": True,
        },
    },
    "healthcare": {
        # HIPAA compliance
        "enable_receipt_generation": True,
        "enable_receipt_auto_sign": True,
        "enable_provenance": True,
        "enable_bead_tracking": True,
        "enable_compliance_artifacts": True,
        "enable_position_ledger": True,
        "enable_telemetry": True,
        # Privacy
        "enable_privacy_anonymization": True,
        "anonymization_method": "redact",
        # Knowledge
        "enable_knowledge_extraction": True,
        "enable_auto_revalidation": True,
        # Vertical
        "vertical": "healthcare_hipaa",
        # Budget
        "budget_downgrade_models": True,
    },
    "visual": {
        # Full observability: see the debate unfold
        "enable_cartographer": True,
        "enable_spectator": True,
        "enable_position_ledger": True,
        "enable_introspection": True,
        # Minimal audit overhead
        "enable_stability_detection": True,
    },
    "compliance": {
        # EU AI Act and regulatory compliance
        "enable_receipt_generation": True,
        "enable_receipt_auto_sign": True,
        "enable_provenance": True,
        "enable_bead_tracking": True,
        "enable_compliance_artifacts": True,
        "enable_position_ledger": True,
        "enable_telemetry": True,
        # Knowledge extraction for audit trails
        "enable_knowledge_extraction": True,
        "enable_auto_revalidation": True,
        # Privacy
        "enable_privacy_anonymization": True,
    },
    "research": {
        # Deep analysis: all cognitive features enabled
        "enable_knowledge_extraction": True,
        "enable_auto_revalidation": True,
        "enable_supermemory": True,
        "enable_stability_detection": True,
        "enable_position_ledger": True,
        "enable_introspection": True,
        "enable_cartographer": True,
        # Power sampling for better consensus
        "enable_power_sampling": True,
        # Forking for exploring alternatives
        "enable_debate_forking": True,
    },
    "financial": {
        # SOX compliance and financial risk
        "enable_receipt_generation": True,
        "enable_receipt_auto_sign": True,
        "enable_provenance": True,
        "enable_bead_tracking": True,
        "enable_compliance_artifacts": True,
        "enable_position_ledger": True,
        "enable_telemetry": True,
        # Knowledge
        "enable_knowledge_extraction": True,
        "enable_auto_revalidation": True,
        # Vertical
        "vertical": "financial_audit",
        # Budget
        "budget_downgrade_models": True,
    },
}

_PRESET_DESCRIPTIONS: dict[str, str] = {
    "sme": "Balanced preset for small/medium businesses: receipts, knowledge, budget controls, compliance",
    "enterprise": "Full-featured preset: everything in SME plus telemetry, airlock, performance monitoring, forking, unified voting",
    "minimal": "Lightweight preset: stability detection and budget controls only (fast & cheap)",
    "audit": "Maximum traceability: receipts, provenance, beads, compliance, position ledger, telemetry",
    "healthcare": "HIPAA-compliant preset: full audit trail, privacy anonymization, healthcare vertical weight profiles",
    "visual": "Full observability: argument cartography, spectator streaming, position tracking, introspection",
    "compliance": "EU AI Act and regulatory: receipts, provenance, compliance artifacts, privacy anonymization",
    "research": "Deep analysis: knowledge extraction, supermemory, power sampling, debate forking, cartography",
    "financial": "SOX-compliant: full audit trail, financial vertical weight profiles, budget controls",
}


def get_preset(name: str) -> dict[str, Any]:
    """Get a preset configuration dict by name.

    Args:
        name: Preset name (sme, enterprise, minimal, audit).

    Returns:
        Dict of ArenaConfig kwargs.

    Raises:
        KeyError: If preset name is not found.
    """
    if name not in _PRESETS:
        available = ", ".join(sorted(_PRESETS))
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    result = dict(_PRESETS[name])
    # Convert post-debate preset shorthand to actual config
    pdc_preset = result.pop("_post_debate_preset", None)
    if pdc_preset:
        try:
            from aragora.debate.post_debate_coordinator import PostDebateConfig
            result["post_debate_config"] = PostDebateConfig(**pdc_preset)
        except ImportError:
            pass
    return result


def list_presets() -> list[str]:
    """List available preset names.

    Returns:
        Sorted list of preset names.
    """
    return sorted(_PRESETS)


def get_preset_info(name: str) -> dict[str, Any]:
    """Get preset metadata including description and flags.

    Args:
        name: Preset name.

    Returns:
        Dict with 'name', 'description', 'flags' keys.

    Raises:
        KeyError: If preset name is not found.
    """
    if name not in _PRESETS:
        available = ", ".join(sorted(_PRESETS))
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return {
        "name": name,
        "description": _PRESET_DESCRIPTIONS.get(name, ""),
        "flags": dict(_PRESETS[name]),
    }


def apply_preset(
    name: str,
    overrides: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Get a preset and merge with overrides.

    Overrides take precedence over preset values.

    Args:
        name: Preset name.
        overrides: Optional dict of overrides.
        **kwargs: Additional overrides as keyword arguments.

    Returns:
        Merged configuration dict.

    Raises:
        KeyError: If preset name is not found.
    """
    merged = get_preset(name)
    if overrides:
        merged.update(overrides)
    if kwargs:
        merged.update(kwargs)
    return merged
