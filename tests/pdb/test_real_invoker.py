"""Tests for :mod:`aragora.pdb.real_invoker`.

Covers:

- happy-path findings / critique / synthesis dispatch for Claude + GPT
- heterodox families raise :class:`ProviderUnavailableError`
- cost + latency recorded from the agent's ``last_tokens_*`` counters
- malformed provider responses still produce a valid dataclass
- end-to-end integration with :func:`aragora.pdb.protocol.run_protocol_b`
  using mocked agent clients — no real network calls
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
from unittest.mock import MagicMock

import pytest

from aragora.pdb.budget import PDBBudgetLedger
from aragora.pdb.panel_config import (
    PDBBudgetConfig,
    PDBPanelConfig,
    PDBPanelDefinition,
    PDBPanelSlot,
    PDBPromptSet,
)
from aragora.pdb.protocol import (
    PDBExecutionInput,
    PDBExecutionStatus,
    SlotCritiqueResponse,
    SlotFindingsResponse,
    SynthesisResponse,
    run_protocol_b,
)
from aragora.pdb.real_invoker import (
    FAMILY_CLAUDE,
    FAMILY_GPT,
    HETERODOX_FAMILIES,
    ProviderUnavailableError,
    RealProviderInvoker,
    estimate_cost_usd,
)
from aragora.review.builder import PanelVote
from aragora.review.policy import ReviewBudget, ReviewPolicy
from aragora.review.protocol import DissentPosition, ReviewBrief, ReviewRole, RoleFinding
from aragora.review.provider_slots import (
    ProviderSlotAvailabilitySummary,
    ProviderSlotDefinition,
    ProviderSlotResolution,
    ProviderSlotResolver,
)
from aragora.swarm.pr_review_protocol import (
    PRReviewBinding,
    PRReviewProtocol,
    PRReviewProtocolPacket,
    PROTOCOL_STATUS,
    PROTOCOL_VERSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slot(
    slot_id: str,
    *,
    family: str,
    review_role: str = "logic_reviewer",
    lens: str = "core",
    required: bool = False,
) -> PDBPanelSlot:
    return PDBPanelSlot(
        slot_id=slot_id,
        review_role=review_role,
        lens=lens,
        family=family,
        candidates=(f"{family}-cli",),
        required=required,
    )


def _binding() -> PRReviewBinding:
    return PRReviewBinding(
        repo="synaptent/aragora",
        pr_number=4242,
        base_sha="base0000",
        head_sha="head1111",
    )


def _make_mock_agent(
    *,
    model: str = "claude-sonnet-4-6",
    response_text: str = '{"recommendation": "approve", "confidence": 0.8}',
    tokens_in: int = 1000,
    tokens_out: int = 500,
) -> MagicMock:
    """Build a MagicMock that mimics the :class:`_AgentLike` contract.

    We assign ``last_tokens_in`` / ``last_tokens_out`` directly as
    attributes (not as Mock auto-children) so ``int(...)`` on them
    produces the expected numeric value.
    """
    mock = MagicMock()
    mock.model = model
    mock.last_tokens_in = tokens_in
    mock.last_tokens_out = tokens_out

    async def _generate(prompt: str, context: Any = None, **kwargs: Any) -> str:
        return response_text

    mock.generate.side_effect = _generate
    return mock


def _findings_payload_json(position: str = "approve", slot_id: str = "claude_core") -> str:
    return (
        f'{{"recommendation": "{position}", "confidence": 0.85, '
        f'"top_findings": [{{"finding_id": "{slot_id}-F1", "category": "logic", '
        f'"severity": "medium", "summary": "Looks fine.", "evidence": []}}], '
        f'"contested_finding_ids": [], '
        f'"reason": "Code reads correct."}}'
    )


def _critique_payload_json(position: str = "approve") -> str:
    return (
        f'{{"recommendation": "{position}", "confidence": 0.7, '
        f'"agrees_with": [], "disagrees_with": [], '
        f'"contested_finding_ids": [], "reason": "I stand by {position}."}}'
    )


def _synthesis_payload_json() -> str:
    return (
        '{"top_line": "Panel consensus: approve with minor notes.", '
        '"validation_summary": "CI green; no dissent.", '
        '"preserved_dissent": []}'
    )


# ---------------------------------------------------------------------------
# Cost estimator
# ---------------------------------------------------------------------------


class TestEstimateCostUsd:
    def test_known_anthropic_model(self) -> None:
        # 1M in + 1M out at (3, 15) → 18 USD
        cost = estimate_cost_usd(
            model="claude-sonnet-4-6", tokens_in=1_000_000, tokens_out=1_000_000
        )
        assert cost == pytest.approx(18.0)

    def test_known_openai_model(self) -> None:
        cost = estimate_cost_usd(model="gpt-5.4", tokens_in=1_000_000, tokens_out=1_000_000)
        # (2.50, 10.00) → 12.50
        assert cost == pytest.approx(12.5)

    def test_unknown_model_returns_zero(self) -> None:
        cost = estimate_cost_usd(model="unknown-model", tokens_in=1000, tokens_out=500)
        assert cost == 0.0

    def test_handles_provider_prefix(self) -> None:
        cost = estimate_cost_usd(
            model="anthropic/claude-sonnet-4-6",
            tokens_in=1_000_000,
            tokens_out=0,
        )
        assert cost == pytest.approx(3.0)

    def test_negative_tokens_clamped(self) -> None:
        cost = estimate_cost_usd(model="gpt-5.4", tokens_in=-100, tokens_out=-100)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# Unavailability
# ---------------------------------------------------------------------------


class TestProviderUnavailable:
    def test_heterodox_families_raise(self) -> None:
        invoker = RealProviderInvoker(
            claude=_make_mock_agent(),
            gpt=_make_mock_agent(model="gpt-5.4"),
        )
        for family in HETERODOX_FAMILIES:
            slot = _slot(f"{family}_slot", family=family, lens="heterodox")
            with pytest.raises(ProviderUnavailableError) as exc:
                invoker.findings(
                    slot=slot,
                    provider="some-provider",
                    prompt="p",
                    binding=_binding(),
                )
            assert exc.value.family == family
            assert exc.value.slot_id == f"{family}_slot"

    def test_missing_claude_agent_raises_on_claude_slot(self) -> None:
        invoker = RealProviderInvoker(claude=None, gpt=_make_mock_agent())
        slot = _slot("claude_core", family=FAMILY_CLAUDE, required=True)
        with pytest.raises(ProviderUnavailableError):
            invoker.findings(slot=slot, provider="claude", prompt="p", binding=_binding())

    def test_unavailable_slots_override_respected(self) -> None:
        invoker = RealProviderInvoker(
            claude=_make_mock_agent(),
            gpt=_make_mock_agent(),
            unavailable_slots=frozenset({"claude_core"}),
        )
        slot = _slot("claude_core", family=FAMILY_CLAUDE, required=True)
        with pytest.raises(ProviderUnavailableError) as exc:
            invoker.findings(slot=slot, provider="claude", prompt="p", binding=_binding())
        assert "no API key" in exc.value.reason


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


class TestFindings:
    def test_claude_findings_parses_and_dispatches(self) -> None:
        agent = _make_mock_agent(
            model="claude-sonnet-4-6",
            response_text=_findings_payload_json(slot_id="claude_core"),
            tokens_in=2000,
            tokens_out=500,
        )
        invoker = RealProviderInvoker(claude=agent, gpt=_make_mock_agent())
        slot = _slot("claude_core", family=FAMILY_CLAUDE, required=True)

        result = invoker.findings(
            slot=slot,
            provider="claude",
            prompt="findings prompt here",
            binding=_binding(),
        )

        assert isinstance(result, SlotFindingsResponse)
        assert result.slot_id == "claude_core"
        assert result.provider == "claude"
        assert result.model == "claude-sonnet-4-6"
        assert result.position is DissentPosition.APPROVE
        assert result.confidence == 0.85
        assert result.summary  # non-empty
        assert len(result.top_findings) == 1
        assert result.top_findings[0].finding_id == "claude_core-F1"
        # cost & latency recorded
        assert result.cost_usd > 0
        assert result.latency_ms >= 0
        # The mock agent was actually called once with the prompt.
        agent.generate.assert_called_once()

    def test_gpt_findings_parses_and_dispatches(self) -> None:
        agent = _make_mock_agent(
            model="gpt-5.4",
            response_text=_findings_payload_json("request_changes", slot_id="gpt_core"),
            tokens_in=1500,
            tokens_out=400,
        )
        invoker = RealProviderInvoker(claude=_make_mock_agent(), gpt=agent)
        slot = _slot("gpt_core", family=FAMILY_GPT, review_role="security_reviewer", required=True)

        result = invoker.findings(slot=slot, provider="openai-api", prompt="p", binding=_binding())
        assert result.position is DissentPosition.REQUEST_CHANGES
        assert result.cost_usd > 0

    def test_malformed_response_yields_safe_dataclass(self) -> None:
        agent = _make_mock_agent(
            model="claude-sonnet-4-6",
            response_text="I refuse to output JSON.",
        )
        invoker = RealProviderInvoker(claude=agent, gpt=_make_mock_agent())
        slot = _slot("claude_core", family=FAMILY_CLAUDE, required=True)

        result = invoker.findings(slot=slot, provider="claude", prompt="p", binding=_binding())
        assert isinstance(result, SlotFindingsResponse)
        assert result.confidence == 0.0  # zero when parse fails
        assert result.position is DissentPosition.DEFER
        assert result.top_findings == ()

    def test_agent_exception_propagates(self) -> None:
        agent = MagicMock()
        agent.model = "claude-sonnet-4-6"
        agent.last_tokens_in = 0
        agent.last_tokens_out = 0

        async def _boom(prompt: str, context: Any = None, **kwargs: Any) -> str:
            raise RuntimeError("upstream timeout")

        agent.generate.side_effect = _boom
        invoker = RealProviderInvoker(claude=agent, gpt=_make_mock_agent())
        slot = _slot("claude_core", family=FAMILY_CLAUDE, required=True)
        with pytest.raises(RuntimeError, match="upstream timeout"):
            invoker.findings(slot=slot, provider="claude", prompt="p", binding=_binding())


# ---------------------------------------------------------------------------
# Critique
# ---------------------------------------------------------------------------


class TestCritique:
    def test_critique_parses(self) -> None:
        agent = _make_mock_agent(
            model="claude-sonnet-4-6",
            response_text=_critique_payload_json("request_changes"),
            tokens_in=1000,
            tokens_out=300,
        )
        invoker = RealProviderInvoker(claude=agent, gpt=_make_mock_agent())
        slot = _slot("claude_core", family=FAMILY_CLAUDE, required=True)

        result = invoker.critique(
            slot=slot,
            provider="claude",
            prompt="critique prompt",
            peer_findings={},
            binding=_binding(),
        )
        assert isinstance(result, SlotCritiqueResponse)
        assert result.position is DissentPosition.REQUEST_CHANGES
        assert result.confidence == 0.7


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


class TestSynthesize:
    def test_synthesize_parses(self) -> None:
        agent = _make_mock_agent(
            model="claude-sonnet-4-6",
            response_text=_synthesis_payload_json(),
            tokens_in=3000,
            tokens_out=700,
        )
        invoker = RealProviderInvoker(claude=agent, gpt=_make_mock_agent())
        slot = _slot("claude_core", family=FAMILY_CLAUDE, required=True)

        # Build a votes tuple with majority APPROVE
        votes = (
            PanelVote(
                finding=RoleFinding(
                    role=ReviewRole.LOGIC,
                    agent="claude_core:claude",
                    model="claude-sonnet-4-6",
                    confidence=0.8,
                    finding_text="",
                ),
                position=DissentPosition.APPROVE,
                reason="ok",
            ),
            PanelVote(
                finding=RoleFinding(
                    role=ReviewRole.SECURITY,
                    agent="gpt_core:openai",
                    model="gpt-5.4",
                    confidence=0.7,
                    finding_text="",
                ),
                position=DissentPosition.APPROVE,
                reason="ok",
            ),
        )
        result = invoker.synthesize(
            synthesizer_slot=slot,
            provider="claude",
            prompt="synth prompt",
            votes=votes,
            binding=_binding(),
        )
        assert isinstance(result, SynthesisResponse)
        assert "consensus" in result.top_line.lower() or "approve" in result.top_line.lower()
        assert result.position is DissentPosition.APPROVE
        assert result.cost_usd > 0


# ---------------------------------------------------------------------------
# End-to-end: run_protocol_b + RealProviderInvoker with mocked agents
# ---------------------------------------------------------------------------


class _StubResolver(ProviderSlotResolver):
    """Resolver that returns canned resolutions without touching real agents."""

    def __init__(self, resolved: set[str], all_slots: Sequence[PDBPanelSlot]) -> None:
        self._resolved = set(resolved)
        self._slots = {s.slot_id: s for s in all_slots}

    def resolve_slot(self, slot: ProviderSlotDefinition) -> ProviderSlotResolution:
        meta = self._slots.get(slot.slot_id)
        if slot.slot_id in self._resolved and meta is not None:
            return ProviderSlotResolution(
                slot_id=slot.slot_id,
                review_role=slot.review_role,
                lens=slot.lens,
                family=slot.family,
                selected_provider=slot.candidates[0],
                status="available",
                detail="stub",
                candidates=list(slot.candidates),
            )
        return ProviderSlotResolution(
            slot_id=slot.slot_id,
            review_role=slot.review_role,
            lens=slot.lens,
            family=slot.family,
            selected_provider=None,
            status="unavailable",
            detail="stub",
            candidates=list(slot.candidates),
        )

    def resolve_slots(self, slot_definitions: Sequence[ProviderSlotDefinition]):
        return [self.resolve_slot(s) for s in slot_definitions]


def _mini_config(slots: Sequence[PDBPanelSlot], synthesizer: str) -> PDBPanelConfig:
    slot_map = {s.slot_id: s for s in slots}
    panel = PDBPanelDefinition(
        panel_id="p",
        findings_slots=tuple(s.slot_id for s in slots),
        critique_slots=tuple(s.slot_id for s in slots),
        synthesizer_slot=synthesizer,
    )
    return PDBPanelConfig(
        version=1,
        default_panel="p",
        default_prompt_set="ps",
        budget=PDBBudgetConfig(
            per_brief_usd=20.0,
            per_day_usd=200.0,
            reserve_for_manual_escalation_usd=10.0,
        ),
        slots=slot_map,
        panels={"p": panel},
        prompt_sets={
            "ps": PDBPromptSet(
                prompt_set_id="ps",
                findings_prompt="f",
                critique_prompt="c",
                synthesis_prompt="s",
            ),
        },
    )


def _heuristic_packet(binding: PRReviewBinding) -> PRReviewProtocolPacket:
    return PRReviewProtocolPacket(
        protocol_version=PROTOCOL_VERSION,
        status=PROTOCOL_STATUS,
        binding=binding,
        review_roles=list(PRReviewProtocol.default().review_roles),
        provider_slots=[],
        availability_summary=ProviderSlotAvailabilitySummary(total_slots=0, resolved_slots=0),
        recommendation_class="needs_human_attention",
        recommendation_reason="pre-execution",
        confidence=0.5,
        confidence_basis=PROTOCOL_STATUS,
        dissent_summary="",
        dissenting_views=[],
        validation_summary={},
        top_findings=[],
        cost_estimate={},
    )


def _execution_input() -> PDBExecutionInput:
    b = _binding()
    return PDBExecutionInput(
        binding=b,
        packet=_heuristic_packet(b),
        packet_sha="",
        pr_title="Tighten rate limiter",
        pr_body="",
        labels=("backend",),
        changed_files=("aragora/server/rate_limit.py",),
        diff_excerpt="diff --git a b\n+ pass\n",
        validation_summary={"checks_summary": "green"},
        panel_id="p",
        policy=ReviewPolicy(budget=ReviewBudget(per_pr_usd_cap=0.0)),
    )


class TestEndToEndWithRealInvoker:
    def test_run_protocol_b_with_mocked_agents_produces_brief(self) -> None:
        # Stateful mocks that return the right payload per call phase.
        claude_mock = MagicMock()
        claude_mock.model = "claude-sonnet-4-6"
        claude_mock.last_tokens_in = 1000
        claude_mock.last_tokens_out = 500

        call_counter = {"n": 0}

        async def _claude_generate(prompt: str, context: Any = None, **kwargs: Any) -> str:
            # Phase detection via prompt markers the templates use.
            call_counter["n"] += 1
            if "findings round" in prompt:
                return _findings_payload_json("approve", slot_id="claude_core")
            if "critique round" in prompt:
                return _critique_payload_json("approve")
            if "synthesis" in prompt:
                return _synthesis_payload_json()
            return "{}"

        claude_mock.generate.side_effect = _claude_generate

        gpt_mock = MagicMock()
        gpt_mock.model = "gpt-5.4"
        gpt_mock.last_tokens_in = 900
        gpt_mock.last_tokens_out = 400

        async def _gpt_generate(prompt: str, context: Any = None, **kwargs: Any) -> str:
            if "findings round" in prompt:
                return _findings_payload_json("approve", slot_id="gpt_core")
            if "critique round" in prompt:
                return _critique_payload_json("approve")
            return "{}"

        gpt_mock.generate.side_effect = _gpt_generate

        invoker = RealProviderInvoker(claude=claude_mock, gpt=gpt_mock)

        slots = [
            _slot(
                "claude_core",
                family=FAMILY_CLAUDE,
                review_role="logic_reviewer",
                lens="core",
                required=True,
            ),
            _slot(
                "gpt_core",
                family=FAMILY_GPT,
                review_role="security_reviewer",
                lens="core",
                required=True,
            ),
            # Heterodox slot — resolver says "available" (stub) but the
            # RealProviderInvoker will raise ProviderUnavailableError
            # which the executor records in degrade_reasons.
            _slot(
                "grok_heterodox",
                family="grok",
                review_role="skeptic",
                lens="heterodox",
                required=False,
            ),
        ]
        config = _mini_config(slots, synthesizer="claude_core")
        resolver = _StubResolver({"claude_core", "gpt_core", "grok_heterodox"}, slots)
        ledger = PDBBudgetLedger(daily_cap_usd=config.budget.per_day_usd)

        result = run_protocol_b(
            _execution_input(),
            invoker=invoker,
            config=config,
            ledger=ledger,
            resolver=resolver,
        )

        # Heterodox slot was attempted and degraded; core slots produced a brief.
        assert result.status in (PDBExecutionStatus.SUCCESS, PDBExecutionStatus.DEGRADED)
        assert isinstance(result.brief, ReviewBrief)
        assert "claude_core" in result.active_roster
        assert "gpt_core" in result.active_roster
        assert "grok_heterodox" not in result.active_roster
        # Degrade reason mentions the grok slot
        assert any("grok" in reason for reason in result.degrade_reasons)

    def test_heterogeneity_floor_fails_closed_with_only_one_family(self) -> None:
        # Only claude wired + only claude_core in the panel would fail
        # validation (≥2 core required); we use resolver-level collapse
        # to exercise the runtime heterogeneity check.
        claude_mock = _make_mock_agent(
            model="claude-sonnet-4-6",
            response_text=_findings_payload_json(),
        )
        invoker = RealProviderInvoker(claude=claude_mock, gpt=None)

        slots = [
            _slot("claude_core", family=FAMILY_CLAUDE, lens="core", required=True),
            _slot(
                "gpt_core",
                family=FAMILY_GPT,
                review_role="security_reviewer",
                lens="core",
                required=True,
            ),
            _slot(
                "grok_heterodox",
                family="grok",
                review_role="skeptic",
                lens="heterodox",
            ),
        ]
        config = _mini_config(slots, synthesizer="claude_core")
        # Resolver: only claude resolves. gpt_core is required + missing →
        # executor fails closed before any agent is called.
        resolver = _StubResolver({"claude_core"}, slots)

        result = run_protocol_b(
            _execution_input(),
            invoker=invoker,
            config=config,
            resolver=resolver,
        )
        assert result.status is PDBExecutionStatus.FAILED_CLOSED
        assert result.brief is None
