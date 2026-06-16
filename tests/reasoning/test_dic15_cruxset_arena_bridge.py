"""DIC-15 (#6025): Arena-path bridge tests — CruxFinderResult → CruxSet.

Verifies :func:`~aragora.reasoning.cruxset_emission.maybe_emit_cruxset_from_finder_result`:

DIC-15 acceptance criteria exercised here:
- Flag-gated (default off, no live queue effect)
- Ranked by load_bearing_score desc (same ordering as CruxSet.build)
- ``decision`` is always None (crux_finder never produces a verdict)
- Provenance links back to the source debate and accepts extra_provenance
- Checksum valid and receipt_id threaded through

All tests use deterministic mocked inputs (no Arena or API calls).

Note: ``aragora.debate.*`` imports may be blocked by a broken Rust/cffi
``cryptography`` backend in some containers.  We try the real import first;
only when it fails do we inject a lightweight stub into ``sys.modules`` so
the lazy import inside the function under test can resolve.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import pytest

from aragora.reasoning import cruxset_emission as mod
from aragora.reasoning.crux_detector import CruxAnalysisResult, CruxClaim
from aragora.reasoning.cruxset import CRUXSET_SCHEMA_VERSION, CruxSet


# ---------------------------------------------------------------------------
# Real import or stub — prefer real when available
# ---------------------------------------------------------------------------

try:
    from aragora.debate.crux_mode import CruxFinderResult as _ResultClass  # type: ignore[assignment]

    _USE_STUB = False
except BaseException:  # noqa: BLE001 - pyo3_runtime.PanicException is not a subclass of Exception
    _USE_STUB = True

    @dataclass  # type: ignore[no-redef]
    class _ResultClass:  # type: ignore[no-redef]
        """Lightweight stand-in for aragora.debate.crux_mode.CruxFinderResult."""

        debate_id: str
        question: str
        analysis: Any
        counterfactuals: list[dict[str, Any]] = field(default_factory=list)
        agents: list[str] = field(default_factory=list)
        rounds: int = 0
        raw_claims: list[dict[str, Any]] = field(default_factory=list)
        metadata: dict[str, Any] = field(default_factory=dict)


@pytest.fixture(autouse=True)
def _inject_crux_mode_stub(monkeypatch):
    """Inject stub only when the real import failed (broken cryptography backend)."""
    if not _USE_STUB:
        return
    stub = ModuleType("aragora.debate.crux_mode")
    stub.CruxFinderResult = _ResultClass  # type: ignore[attr-defined]
    if "aragora.debate" not in sys.modules:
        monkeypatch.setitem(sys.modules, "aragora.debate", ModuleType("aragora.debate"))
    monkeypatch.setitem(sys.modules, "aragora.debate.crux_mode", stub)


@pytest.fixture(autouse=True)
def _reset_emission_flag(monkeypatch):
    monkeypatch.delenv(mod.CRUXSET_EMISSION_ENV_VAR, raising=False)
    yield
    monkeypatch.delenv(mod.CRUXSET_EMISSION_ENV_VAR, raising=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _claim(claim_id: str, statement: str, score: float) -> CruxClaim:
    return CruxClaim(
        claim_id=claim_id,
        statement=statement,
        author="agent-alpha",
        crux_score=score,
        influence_score=score * 0.9,
        disagreement_score=score * 0.8,
        uncertainty_score=score * 0.5,
        centrality_score=score * 0.7,
        affected_claims=[],
        contesting_agents=["agent-beta"] if score > 0.5 else [],
        resolution_impact=score * 0.4,
    )


def _analysis(*claims: CruxClaim, barrier: float = 0.5) -> CruxAnalysisResult:
    return CruxAnalysisResult(
        cruxes=list(claims),
        total_claims=len(claims) + 2,
        total_disagreements=len(claims),
        average_uncertainty=0.45,
        convergence_barrier=barrier,
        recommended_focus=[c.claim_id for c in claims],
    )


def _result(
    analysis: Any,
    *,
    debate_id: str = "debate-test-001",
    question: str = "Should we adopt X?",
    counterfactuals: list[dict[str, Any]] | None = None,
    agents: list[str] | None = None,
    rounds: int = 3,
) -> Any:
    return _ResultClass(
        debate_id=debate_id,
        question=question,
        analysis=analysis,
        counterfactuals=list(counterfactuals or []),
        agents=list(agents or ["agent-alpha", "agent-beta"]),
        rounds=rounds,
        raw_claims=[],
        metadata={"mode": "crux_finder", "approach": "A"},
    )


# ---------------------------------------------------------------------------
# 1. Feature flag
# ---------------------------------------------------------------------------


def test_returns_none_when_disabled() -> None:
    cs = mod.maybe_emit_cruxset_from_finder_result(_result(_analysis(_claim("c1", "S", 0.8))))
    assert cs is None


def test_returns_cruxset_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")
    cs = mod.maybe_emit_cruxset_from_finder_result(
        _result(_analysis(_claim("c1", "X load-bearing", 0.8), _claim("c2", "Y dep X", 0.5)))
    )
    assert isinstance(cs, CruxSet)


# ---------------------------------------------------------------------------
# 2. No-verdict invariant (DIC-15 core guardrail)
# ---------------------------------------------------------------------------


def test_decision_is_always_none(monkeypatch) -> None:
    """crux_finder never produces a verdict; decision must be None."""
    monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")
    cs = mod.maybe_emit_cruxset_from_finder_result(_result(_analysis(_claim("c1", "S", 0.9))))
    assert cs is not None
    assert cs.decision is None


# ---------------------------------------------------------------------------
# 3. Ranking
# ---------------------------------------------------------------------------


def test_cruxes_sorted_by_load_bearing_score_desc(monkeypatch) -> None:
    monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")
    a = _analysis(_claim("c1", "Low", 0.3), _claim("c2", "High", 0.9), _claim("c3", "Mid", 0.6))
    cs = mod.maybe_emit_cruxset_from_finder_result(_result(a))
    assert cs is not None
    scores = [c.load_bearing_score for c in cs.cruxes]
    assert scores == sorted(scores, reverse=True)
    assert cs.cruxes[0].crux_id == "c2"


# ---------------------------------------------------------------------------
# 4. Provenance links
# ---------------------------------------------------------------------------


def test_provenance_carries_debate_id_and_mode(monkeypatch) -> None:
    monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")
    cs = mod.maybe_emit_cruxset_from_finder_result(
        _result(_analysis(_claim("c1", "S", 0.7)), debate_id="debate-xyz-42")
    )
    assert cs is not None
    assert cs.provenance.get("debate_id") == "debate-xyz-42"
    assert cs.provenance.get("mode") == "crux_finder"
    assert cs.provenance.get("approach") == "A"


def test_extra_provenance_merged(monkeypatch) -> None:
    monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")
    cs = mod.maybe_emit_cruxset_from_finder_result(
        _result(_analysis(_claim("c1", "S", 0.7))),
        extra_provenance={"pipeline": "nightly-audit", "corpus_rev": 6},
    )
    assert cs is not None
    assert cs.provenance.get("pipeline") == "nightly-audit"
    assert cs.provenance.get("corpus_rev") == 6
    assert cs.provenance.get("debate_id") is not None  # base provenance preserved


def test_counterfactuals_preserved_in_provenance(monkeypatch) -> None:
    monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")
    counterfactuals = [
        {
            "claim_id": "c1",
            "assumption": "X is false",
            "would_flip": True,
        }
    ]
    cs = mod.maybe_emit_cruxset_from_finder_result(
        _result(_analysis(_claim("c1", "S", 0.7)), counterfactuals=counterfactuals)
    )
    assert cs is not None
    assert cs.provenance.get("counterfactuals") == counterfactuals


def test_receipt_id_threaded_through(monkeypatch) -> None:
    monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")
    cs = mod.maybe_emit_cruxset_from_finder_result(
        _result(_analysis(_claim("c1", "S", 0.7))), receipt_id="rcpt_abc123"
    )
    assert cs is not None
    assert cs.receipt_id == "rcpt_abc123"


# ---------------------------------------------------------------------------
# 5. Checksum and schema
# ---------------------------------------------------------------------------


def test_checksum_valid_and_schema_version_set(monkeypatch) -> None:
    monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")
    cs = mod.maybe_emit_cruxset_from_finder_result(
        _result(_analysis(_claim("c1", "Load-bearing", 0.8), _claim("c2", "Secondary", 0.4)))
    )
    assert cs is not None
    assert cs.verify_checksum() is True
    assert cs.schema_version == CRUXSET_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


def test_returns_none_when_analysis_has_no_cruxes(monkeypatch) -> None:
    monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")
    empty = CruxAnalysisResult(
        cruxes=[],
        total_claims=0,
        total_disagreements=0,
        average_uncertainty=0.0,
        convergence_barrier=0.0,
        recommended_focus=[],
    )
    assert mod.maybe_emit_cruxset_from_finder_result(_result(empty)) is None


def test_returns_none_for_wrong_type(monkeypatch) -> None:
    monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")
    assert mod.maybe_emit_cruxset_from_finder_result("not-a-result") is None  # type: ignore[arg-type]


def test_malformed_finder_analysis_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")
    malformed = _result(_analysis(_claim("c1", "S", 0.7)))
    malformed.analysis = None
    assert mod.maybe_emit_cruxset_from_finder_result(malformed) is None


def test_analysis_conversion_failure_fails_closed(monkeypatch) -> None:
    class BrokenAnalysis:
        cruxes = [_claim("c1", "S", 0.7)]

        def to_dict(self) -> dict[str, Any]:
            raise ValueError("bad analysis payload")

    monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")
    assert mod.maybe_emit_cruxset_from_finder_result(_result(BrokenAnalysis())) is None


def test_convergence_barrier_in_counterfactual_notes(monkeypatch) -> None:
    """build_cruxset_from_analysis must embed the convergence_barrier signal."""
    monkeypatch.setenv(mod.CRUXSET_EMISSION_ENV_VAR, "1")
    cs = mod.maybe_emit_cruxset_from_finder_result(
        _result(_analysis(_claim("c1", "Key assumption", 0.75), barrier=0.66))
    )
    assert cs is not None
    assert any("convergence_barrier=0.66" in n for n in cs.counterfactual_notes)
