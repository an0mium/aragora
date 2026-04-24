from __future__ import annotations

import pytest

from aragora.review.provider_slots import (
    ProviderSlotDefinition,
    ProviderSlotResolution,
    ProviderSlotResolver,
)


def test_provider_slot_resolver_reports_candidate_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    def fake_which(binary: str) -> str | None:
        return {
            "claude": "/usr/bin/claude",
            "gemini": "/usr/bin/gemini",
        }.get(binary)

    monkeypatch.setattr("aragora.review.provider_slots.shutil.which", fake_which)

    resolver = ProviderSlotResolver()
    resolutions = resolver.resolve_slots(
        (
            ProviderSlotDefinition(
                slot_id="logic",
                review_role="logic_reviewer",
                lens="core",
                family="claude",
                candidates=("claude", "anthropic-api"),
            ),
            ProviderSlotDefinition(
                slot_id="regulatory",
                review_role="skeptic",
                lens="regulatory",
                family="mistral",
                candidates=("mistral-api", "mistral"),
            ),
        )
    )

    assert [slot.selected_provider for slot in resolutions] == ["claude", "mistral-api"]
    assert resolutions[0].candidate_checks[0].provider == "claude"
    assert resolutions[0].candidate_checks[0].available is True
    assert resolutions[1].status == "available"
    assert resolutions[1].candidate_checks[0].allowlisted is False
    assert "not allowlisted by default" in resolutions[1].detail


def test_provider_slot_resolver_explains_unavailable_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.setattr("aragora.review.provider_slots.shutil.which", lambda binary: None)

    resolver = ProviderSlotResolver()
    resolution = resolver.resolve_slot(
        ProviderSlotDefinition(
            slot_id="skeptic",
            review_role="skeptic",
            lens="heterodox",
            family="grok",
            candidates=("grok-cli", "grok"),
        )
    )

    assert resolution.selected_provider is None
    assert resolution.status == "unavailable"
    assert "grok-cli" in resolution.detail
    assert "grok CLI not found on PATH" in resolution.detail
    assert "grok:" in resolution.detail


def test_provider_slot_summary_counts_core_and_opt_in_slots() -> None:
    resolver = ProviderSlotResolver()
    summary = resolver.summarize(
        [
            ProviderSlotResolution(
                slot_id="logic",
                review_role="logic_reviewer",
                lens="core",
                family="claude",
                selected_provider="claude",
                status="available",
                detail="claude CLI available on PATH",
                candidates=["claude", "anthropic-api"],
                selected_allowlisted=True,
            ),
            ProviderSlotResolution(
                slot_id="regulatory",
                review_role="skeptic",
                lens="regulatory",
                family="mistral",
                selected_provider="mistral-api",
                status="available",
                detail="MISTRAL_API_KEY configured; provider is registered but not allowlisted by default",
                candidates=["mistral-api", "mistral"],
                selected_allowlisted=False,
            ),
            ProviderSlotResolution(
                slot_id="skeptic",
                review_role="skeptic",
                lens="heterodox",
                family="grok",
                selected_provider=None,
                status="unavailable",
                detail="No configured provider available for grok",
                candidates=["grok-cli", "grok"],
                selected_allowlisted=None,
            ),
        ]
    )

    assert summary.total_slots == 3
    assert summary.resolved_slots == 2
    assert summary.unresolved_slots == ["skeptic"]
    assert summary.core_slots_total == 1
    assert summary.core_slots_resolved == 1
    assert summary.available_families == ["claude", "mistral"]
    assert summary.unresolved_families == ["grok"]
    assert summary.opt_in_slots == ["regulatory"]
    assert summary.degraded is True
