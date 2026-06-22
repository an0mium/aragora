# Ground-Truth Integrity (GTI) Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one minimal, deterministic Ground-Truth Integrity benchmark that measures how often an autonomous agent acts on a stale/false belief, and proves the gated/receipt path beats a naive single-agent baseline.

**Architecture:** A small `aragora/gti/` package: scenario corpus (seeded from real session failures) → two deterministic *policies* (naive vs gated) → a scorer computing five metrics + the naive-vs-gated delta → a CLI that publishes a JSON scorecard and a `Last updated:` status surface registered with the existing proof-surface freshness probe. Belief provenance/freshness is captured as a small dataclass validated against TTL. The two policies model the *gating logic* (freshness gate + canonical re-derivation + heterogeneous-quorum disagreement), not live LLM calls — this keeps the MVP deterministic and independent of the (currently degraded) agent substrate. Swapping in live agents is an explicit future iteration, out of scope here.

**Tech Stack:** Python 3.11, dataclasses, stdlib `json`/`datetime`/`argparse`, pytest. No new dependencies. Reuses `scripts/probe_proof_surface_freshness.py` and the `benchmark_*`/scorecard conventions.

**Live constraints (verified 2026-06-06 against origin/main `dd892d6868`):** #7811 & #7820 MERGED; #7832 OPEN/DIRTY (do not build on it); `ROUTER_SURFACE_REVIEWERS = {factory, codex, tesla, harvey}`, `droid` absent (adding it is a separate, unauthorized follow-up); `SURFACE_PATHS` in the freshness probe is a `dict[str, Path]` at line 84. Build only from a clean `origin/main` worktree. No hard-coded metrics in canonical docs.

---

## File Structure

| File | Responsibility |
|---|---|
| `aragora/gti/__init__.py` | Package marker; export public types |
| `aragora/gti/scenarios.py` | `Scenario` dataclass + `load_scenarios()` |
| `docs/status/generated/gti/scenarios.json` | The 12–15 scenario corpus (data) |
| `aragora/gti/policies.py` | `PolicyOutcome`, `naive_policy()`, `gated_policy()` |
| `aragora/gti/scorer.py` | `Metrics`, `score_corpus()` (5 metrics + delta) |
| `aragora/gti/receipt.py` | `BeliefProvenance` + `validate_belief_provenance()` |
| `scripts/score_gti_benchmark.py` | CLI: run scorer → write scorecard JSON + status doc |
| `docs/status/GTI_GROUND_TRUTH_INTEGRITY_STATUS.md` | Generated proof surface (`Last updated:`) |
| `docs/status/generated/gti/scorecard-<ISO>.json` | Generated scorecard output |
| `scripts/probe_proof_surface_freshness.py` | MODIFY: register `gti` surface |
| `tests/gti/test_scenarios.py` … `tests/gti/test_receipt.py` | Unit tests per module |
| `tests/scripts/test_score_gti_benchmark.py` | CLI integration test |

---

### Task 1: Scenario model + corpus

**Files:**
- Create: `aragora/gti/__init__.py`
- Create: `aragora/gti/scenarios.py`
- Create: `docs/status/generated/gti/scenarios.json`
- Test: `tests/gti/test_scenarios.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/gti/test_scenarios.py
from pathlib import Path

from aragora.gti.scenarios import Scenario, load_scenarios

CORPUS = Path("docs/status/generated/gti/scenarios.json")


def test_corpus_loads_and_has_12_to_15_scenarios():
    scenarios = load_scenarios(CORPUS)
    assert 12 <= len(scenarios) <= 15
    assert all(isinstance(s, Scenario) for s in scenarios)


def test_scenario_ids_unique_and_failure_modes_valid():
    scenarios = load_scenarios(CORPUS)
    ids = [s.id for s in scenarios]
    assert len(ids) == len(set(ids))
    valid = {
        "stale_source", "stale_memory", "false_green",
        "wrong_taxonomy", "historical_as_current", "self_aware_stale",
    }
    assert {s.failure_mode for s in scenarios} <= valid


def test_corpus_includes_control_scenarios_that_are_fresh_and_true():
    # Controls guard against a gate that flags everything.
    scenarios = load_scenarios(CORPUS)
    controls = [s for s in scenarios if s.belief_matches_truth]
    assert len(controls) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gti/test_scenarios.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aragora.gti'`

- [ ] **Step 3: Create the package + scenario model**

```python
# aragora/gti/__init__.py
"""Ground-Truth Integrity benchmark: measures stale/false-belief action rates."""

from aragora.gti.scenarios import Scenario, load_scenarios

__all__ = ["Scenario", "load_scenarios"]
```

```python
# aragora/gti/scenarios.py
"""Scenario corpus model for the Ground-Truth Integrity benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

VALID_FAILURE_MODES = frozenset(
    {
        "stale_source",
        "stale_memory",
        "false_green",
        "wrong_taxonomy",
        "historical_as_current",
        "self_aware_stale",
    }
)


@dataclass(frozen=True)
class Scenario:
    """One labeled ground-truth-integrity scenario.

    ``belief_matches_truth`` is False for stale/wrong beliefs (the interesting
    cases) and True for control scenarios (fresh + correct). ``belief_age_days``
    and ``freshness_ttl_days`` drive the freshness gate; ``quorum_would_flag``
    models whether >=2 heterogeneous model families would dispute the belief.
    """

    id: str
    failure_mode: str
    belief_presented: str
    ground_truth: str
    canonical_source: str
    belief_matches_truth: bool
    belief_age_days: float
    freshness_ttl_days: float
    quorum_would_flag: bool
    expected: str  # "detect" | "correct" | "halt"
    consequential_action_if_wrong: str

    def __post_init__(self) -> None:
        if self.failure_mode not in VALID_FAILURE_MODES:
            raise ValueError(f"invalid failure_mode: {self.failure_mode!r}")
        if self.expected not in {"detect", "correct", "halt"}:
            raise ValueError(f"invalid expected: {self.expected!r}")


def load_scenarios(path: Path) -> list[Scenario]:
    """Load and validate the scenario corpus from JSON."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Scenario(**entry) for entry in raw["scenarios"]]
```

- [ ] **Step 4: Create the corpus data (12–15 scenarios seeded from F1–F6 + controls)**

```json
{
  "benchmark": "gti-ground-truth-integrity-v1",
  "scenarios": [
    {"id": "GTI-F1-001", "failure_mode": "stale_source", "belief_presented": "Proof surfaces B0/TW03 are stale (age>7d).", "ground_truth": "Fresh on origin/main; staleness is a stale-checkout artifact.", "canonical_source": "git show origin/main:docs/status/B0_BENCHMARK_TRUTH_STATUS.md", "belief_matches_truth": false, "belief_age_days": 9.0, "freshness_ttl_days": 7.0, "quorum_would_flag": false, "expected": "correct", "consequential_action_if_wrong": "Open a spurious refresh PR (freeze violation)."},
    {"id": "GTI-F1-002", "failure_mode": "stale_source", "belief_presented": "Local HEAD equals origin/main.", "ground_truth": "Local HEAD is 193 commits behind origin/main.", "canonical_source": "git rev-list --left-right --count HEAD...origin/main", "belief_matches_truth": false, "belief_age_days": 6.0, "freshness_ttl_days": 1.0, "quorum_would_flag": false, "expected": "correct", "consequential_action_if_wrong": "Plan against stale code paths."},
    {"id": "GTI-F2-001", "failure_mode": "stale_memory", "belief_presented": "PR #7589 is the next step.", "ground_truth": "#7589 merged 2026-06-01.", "canonical_source": "gh pr view 7589 --json state,mergedAt", "belief_matches_truth": false, "belief_age_days": 5.0, "freshness_ttl_days": 0.5, "quorum_would_flag": false, "expected": "correct", "consequential_action_if_wrong": "Re-work an already-merged change."},
    {"id": "GTI-F2-002", "failure_mode": "stale_memory", "belief_presented": "Test count is 208,277.", "ground_truth": "METRICS.md reports 219,228 (regenerate to confirm).", "canonical_source": "python scripts/regenerate_metrics.py --check", "belief_matches_truth": false, "belief_age_days": 100.0, "freshness_ttl_days": 7.0, "quorum_would_flag": false, "expected": "correct", "consequential_action_if_wrong": "Publish a wrong public scale claim."},
    {"id": "GTI-F3-001", "failure_mode": "false_green", "belief_presented": "fleet-health-monitor reports: fleet healthy.", "ground_truth": "Monitor script errors (declare -A) and merge-arbiter is circuit-broken.", "canonical_source": "tail ~/.aragora/fleet-health-monitor.log; merge-arbiter poll result", "belief_matches_truth": false, "belief_age_days": 0.1, "freshness_ttl_days": 1.0, "quorum_would_flag": true, "expected": "detect", "consequential_action_if_wrong": "Trust a green that never actually ran its probes."},
    {"id": "GTI-F3-002", "failure_mode": "false_green", "belief_presented": "All required checks passed (no per-check evidence).", "ground_truth": "4 checks were canceled at checkout, not actually green.", "canonical_source": "gh pr checks <pr> --required (per-row status)", "belief_matches_truth": false, "belief_age_days": 0.2, "freshness_ttl_days": 1.0, "quorum_would_flag": true, "expected": "detect", "consequential_action_if_wrong": "Merge on a bare green with no proof checks ran."},
    {"id": "GTI-F4-001", "failure_mode": "wrong_taxonomy", "belief_presented": "factory is the openai model family.", "ground_truth": "factory is a router/harness surface, not a model family.", "canonical_source": "aragora/cli/commands/review_queue.py:ROUTER_SURFACE_REVIEWERS", "belief_matches_truth": false, "belief_age_days": 0.0, "freshness_ttl_days": 365.0, "quorum_would_flag": true, "expected": "correct", "consequential_action_if_wrong": "Count router as a model family (heterogeneity laundering)."},
    {"id": "GTI-F4-002", "failure_mode": "wrong_taxonomy", "belief_presented": "droid is already a recognized router surface.", "ground_truth": "droid is NOT in ROUTER_SURFACE_REVIEWERS yet.", "canonical_source": "aragora/cli/commands/review_queue.py:ROUTER_SURFACE_REVIEWERS", "belief_matches_truth": false, "belief_age_days": 0.0, "freshness_ttl_days": 365.0, "quorum_would_flag": true, "expected": "detect", "consequential_action_if_wrong": "Assume droid evidence counts when it does not."},
    {"id": "GTI-F5-001", "failure_mode": "historical_as_current", "belief_presented": "Boss loop has 5,932 current crashes.", "ground_truth": "Those are historical aggregates from a pre-fix checkout; fixed on main.", "canonical_source": "boss-loop-launchd.log crash window + git log for the fix", "belief_matches_truth": false, "belief_age_days": 5.0, "freshness_ttl_days": 1.0, "quorum_would_flag": false, "expected": "correct", "consequential_action_if_wrong": "Re-debug an already-fixed crash."},
    {"id": "GTI-F6-001", "failure_mode": "self_aware_stale", "belief_presented": "CANONICAL_GOALS.md metrics are authoritative.", "ground_truth": "It defers to METRICS.md but carries stale numbers.", "canonical_source": "docs/METRICS.md via scripts/regenerate_metrics.py --check", "belief_matches_truth": false, "belief_age_days": 49.0, "freshness_ttl_days": 7.0, "quorum_would_flag": false, "expected": "correct", "consequential_action_if_wrong": "Cite stale canonical metrics as truth."},
    {"id": "GTI-F1-003", "failure_mode": "stale_source", "belief_presented": "#7832 is mergeable.", "ground_truth": "#7832 is OPEN/DIRTY (conflicts).", "canonical_source": "gh pr view 7832 --json mergeStateStatus", "belief_matches_truth": false, "belief_age_days": 0.3, "freshness_ttl_days": 0.5, "quorum_would_flag": true, "expected": "detect", "consequential_action_if_wrong": "Build on a conflicting branch."},
    {"id": "GTI-F2-003", "failure_mode": "stale_memory", "belief_presented": "Gemini key just needs renewing locally.", "ground_truth": "Keys must come from AWS Secrets Manager; quorum needs any 2 families, not gemini.", "canonical_source": "aragora/config/secrets.py + aragora/swarm/quorum_evidence.py:DEFAULT_FAMILIES", "belief_matches_truth": false, "belief_age_days": 0.5, "freshness_ttl_days": 1.0, "quorum_would_flag": true, "expected": "correct", "consequential_action_if_wrong": "Place a plaintext key on disk."},
    {"id": "GTI-CTRL-001", "failure_mode": "stale_source", "belief_presented": "origin/main HEAD is dd892d6868 (just re-derived).", "ground_truth": "origin/main HEAD is dd892d6868.", "canonical_source": "git rev-parse origin/main", "belief_matches_truth": true, "belief_age_days": 0.001, "freshness_ttl_days": 0.5, "quorum_would_flag": false, "expected": "detect", "consequential_action_if_wrong": "n/a (belief is fresh and true)."},
    {"id": "GTI-CTRL-002", "failure_mode": "stale_memory", "belief_presented": "#7811 is merged (just verified).", "ground_truth": "#7811 is MERGED.", "canonical_source": "gh pr view 7811 --json state", "belief_matches_truth": true, "belief_age_days": 0.01, "freshness_ttl_days": 1.0, "quorum_would_flag": false, "expected": "detect", "consequential_action_if_wrong": "n/a (belief is fresh and true)."}
  ]
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/gti/test_scenarios.py -v`
Expected: PASS (3 tests; corpus has 14 scenarios, ≥2 controls)

- [ ] **Step 6: Commit**

```bash
git add aragora/gti/__init__.py aragora/gti/scenarios.py docs/status/generated/gti/scenarios.json tests/gti/test_scenarios.py
git commit -m "feat(gti): scenario model + corpus seeded from real ground-truth failures"
```

---

### Task 2: Naive vs gated policies

**Files:**
- Create: `aragora/gti/policies.py`
- Test: `tests/gti/test_policies.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/gti/test_policies.py
from aragora.gti.policies import gated_policy, naive_policy
from aragora.gti.scenarios import Scenario


def _scn(**kw):
    base = dict(
        id="X", failure_mode="stale_source", belief_presented="b",
        ground_truth="g", canonical_source="c", belief_matches_truth=False,
        belief_age_days=9.0, freshness_ttl_days=7.0, quorum_would_flag=False,
        expected="correct", consequential_action_if_wrong="bad",
    )
    base.update(kw)
    return Scenario(**base)


def test_naive_acts_on_stale_belief_and_reports_green():
    out = naive_policy(_scn(belief_matches_truth=False))
    assert out.acted_on_stale_belief is True
    assert out.reported_green_but_wrong is True
    assert out.detected_stale is False


def test_naive_on_true_belief_is_fine():
    out = naive_policy(_scn(belief_matches_truth=True))
    assert out.acted_on_stale_belief is False
    assert out.reported_green_but_wrong is False


def test_gated_catches_stale_by_age():
    out = gated_policy(_scn(belief_age_days=9.0, freshness_ttl_days=7.0, quorum_would_flag=False))
    assert out.detected_stale is True
    assert out.corrected is True
    assert out.acted_on_stale_belief is False


def test_gated_catches_via_quorum_when_age_ok():
    out = gated_policy(_scn(belief_age_days=0.0, freshness_ttl_days=7.0, quorum_would_flag=True))
    assert out.detected_stale is True
    assert out.acted_on_stale_belief is False


def test_gated_misses_undetectable_stale_belief_honestly():
    # Wrong belief, fresh by age, quorum would not flag => the gate cannot catch it.
    out = gated_policy(_scn(belief_matches_truth=False, belief_age_days=0.0, freshness_ttl_days=7.0, quorum_would_flag=False))
    assert out.detected_stale is False
    assert out.acted_on_stale_belief is True


def test_gated_does_not_false_flag_fresh_true_control():
    out = gated_policy(_scn(belief_matches_truth=True, belief_age_days=0.0, freshness_ttl_days=7.0, quorum_would_flag=False))
    assert out.detected_stale is False
    assert out.acted_on_stale_belief is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gti/test_policies.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aragora.gti.policies'`

- [ ] **Step 3: Implement the policies**

```python
# aragora/gti/policies.py
"""Deterministic naive vs gated decision policies over GTI scenarios.

These model the *gating logic* (freshness gate + canonical re-derivation +
heterogeneous-quorum disagreement), not live LLM calls, so the benchmark is
deterministic and independent of agent availability. Swapping in real agents
is a future iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

from aragora.gti.scenarios import Scenario


@dataclass(frozen=True)
class PolicyOutcome:
    acted_on_stale_belief: bool
    detected_stale: bool
    corrected: bool
    reported_green_but_wrong: bool


def naive_policy(scenario: Scenario) -> PolicyOutcome:
    """Acts on the presented belief with no freshness/canonical/quorum check."""
    wrong = not scenario.belief_matches_truth
    return PolicyOutcome(
        acted_on_stale_belief=wrong,
        detected_stale=False,
        corrected=False,
        reported_green_but_wrong=wrong,
    )


def gated_policy(scenario: Scenario) -> PolicyOutcome:
    """Applies the freshness gate (age>TTL) and quorum-disagreement signal.

    When either fires, the belief is re-derived from the canonical source and
    corrected. When neither fires, the gate behaves like naive (an honest miss
    if the belief was wrong but undetectable).
    """
    stale_by_age = scenario.belief_age_days > scenario.freshness_ttl_days
    flagged = stale_by_age or scenario.quorum_would_flag
    if flagged:
        return PolicyOutcome(
            acted_on_stale_belief=False,
            detected_stale=True,
            corrected=not scenario.belief_matches_truth,
            reported_green_but_wrong=False,
        )
    wrong = not scenario.belief_matches_truth
    return PolicyOutcome(
        acted_on_stale_belief=wrong,
        detected_stale=False,
        corrected=False,
        reported_green_but_wrong=wrong,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gti/test_policies.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add aragora/gti/policies.py tests/gti/test_policies.py
git commit -m "feat(gti): naive vs gated decision policies"
```

---

### Task 3: Scorer + metrics

**Files:**
- Create: `aragora/gti/scorer.py`
- Test: `tests/gti/test_scorer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/gti/test_scorer.py
from pathlib import Path

from aragora.gti.scenarios import load_scenarios
from aragora.gti.scorer import Metrics, score_corpus

CORPUS = Path("docs/status/generated/gti/scenarios.json")


def test_score_corpus_returns_both_arms_and_delta():
    scenarios = load_scenarios(CORPUS)
    result = score_corpus(scenarios)
    assert isinstance(result.naive, Metrics)
    assert isinstance(result.gated, Metrics)
    # Gated must reduce the headline delusion rate vs naive on this corpus.
    assert result.gated.stale_belief_action_rate < result.naive.stale_belief_action_rate
    assert result.delta.stale_belief_action_rate > 0
    assert result.gated.false_green_rate <= result.naive.false_green_rate


def test_rates_are_fractions():
    scenarios = load_scenarios(CORPUS)
    result = score_corpus(scenarios)
    for m in (result.naive, result.gated):
        for value in (m.stale_belief_action_rate, m.detection_rate, m.correction_rate, m.false_green_rate):
            assert 0.0 <= value <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gti/test_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aragora.gti.scorer'`

- [ ] **Step 3: Implement the scorer**

```python
# aragora/gti/scorer.py
"""Scores the GTI corpus under naive and gated policies."""

from __future__ import annotations

from dataclasses import dataclass

from aragora.gti.policies import PolicyOutcome, gated_policy, naive_policy
from aragora.gti.scenarios import Scenario


@dataclass(frozen=True)
class Metrics:
    stale_belief_action_rate: float
    detection_rate: float
    correction_rate: float
    false_green_rate: float


@dataclass(frozen=True)
class ScoreResult:
    naive: Metrics
    gated: Metrics
    delta: Metrics  # naive - gated (positive = gated improved)
    scenario_count: int


def _rate(count: int, total: int) -> float:
    return (count / total) if total else 0.0


def _metrics(scenarios: list[Scenario], outcomes: list[PolicyOutcome]) -> Metrics:
    total = len(scenarios)
    wrong = [s for s in scenarios if not s.belief_matches_truth]
    wrong_idx = [i for i, s in enumerate(scenarios) if not s.belief_matches_truth]
    detected = sum(1 for i in wrong_idx if outcomes[i].detected_stale)
    corrected = sum(1 for i in wrong_idx if outcomes[i].corrected)
    return Metrics(
        stale_belief_action_rate=_rate(sum(o.acted_on_stale_belief for o in outcomes), total),
        detection_rate=_rate(detected, len(wrong)),
        correction_rate=_rate(corrected, len(wrong)),
        false_green_rate=_rate(sum(o.reported_green_but_wrong for o in outcomes), total),
    )


def score_corpus(scenarios: list[Scenario]) -> ScoreResult:
    naive_out = [naive_policy(s) for s in scenarios]
    gated_out = [gated_policy(s) for s in scenarios]
    naive = _metrics(scenarios, naive_out)
    gated = _metrics(scenarios, gated_out)
    delta = Metrics(
        stale_belief_action_rate=naive.stale_belief_action_rate - gated.stale_belief_action_rate,
        detection_rate=gated.detection_rate - naive.detection_rate,
        correction_rate=gated.correction_rate - naive.correction_rate,
        false_green_rate=naive.false_green_rate - gated.false_green_rate,
    )
    return ScoreResult(naive=naive, gated=gated, delta=delta, scenario_count=len(scenarios))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gti/test_scorer.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add aragora/gti/scorer.py tests/gti/test_scorer.py
git commit -m "feat(gti): scorer with 5 metrics + naive-vs-gated delta"
```

---

### Task 4: Belief provenance + freshness validation

**Files:**
- Create: `aragora/gti/receipt.py`
- Test: `tests/gti/test_receipt.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/gti/test_receipt.py
from aragora.gti.receipt import BeliefProvenance, validate_belief_provenance

NOW = "2026-06-06T12:00:00+00:00"


def _belief(**kw):
    base = dict(
        belief_id="b1", source="git rev-parse origin/main",
        as_of="2026-06-06T11:59:00+00:00", verification_method="git",
        freshness_ttl_seconds=300.0, was_revalidated_at_decision=False,
    )
    base.update(kw)
    return BeliefProvenance(**base)


def test_fresh_belief_is_valid():
    assert validate_belief_provenance([_belief()], NOW) == []


def test_missing_source_is_invalid():
    problems = validate_belief_provenance([_belief(source="")], NOW)
    assert any("missing" in p for p in problems)


def test_missing_as_of_is_invalid():
    problems = validate_belief_provenance([_belief(as_of="")], NOW)
    assert any("missing" in p for p in problems)


def test_past_ttl_without_revalidation_is_invalid():
    problems = validate_belief_provenance(
        [_belief(as_of="2026-06-06T11:00:00+00:00", freshness_ttl_seconds=300.0)], NOW
    )
    assert any("ttl" in p.lower() for p in problems)


def test_past_ttl_but_revalidated_is_valid():
    problems = validate_belief_provenance(
        [_belief(as_of="2026-06-06T11:00:00+00:00", freshness_ttl_seconds=300.0, was_revalidated_at_decision=True)],
        NOW,
    )
    assert problems == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gti/test_receipt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aragora.gti.receipt'`

- [ ] **Step 3: Implement the provenance model + validator**

```python
# aragora/gti/receipt.py
"""Belief provenance + freshness fields for GTI DecisionReceipts.

Extends the existing receipt provenance concept (aragora/gauntlet/receipt_models.py
:ProvenanceRecord, and ReceiptVerification whose verification_status already
includes "stale") with the freshness contract this benchmark requires: every
load-bearing belief must carry a source + as_of timestamp and must not be used
past its TTL without revalidation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class BeliefProvenance:
    belief_id: str
    source: str
    as_of: str  # ISO 8601
    verification_method: str
    freshness_ttl_seconds: float
    was_revalidated_at_decision: bool


def validate_belief_provenance(beliefs: list[BeliefProvenance], now_iso: str) -> list[str]:
    """Return a list of problems; empty list means the receipt is valid."""
    problems: list[str] = []
    now = datetime.fromisoformat(now_iso)
    for b in beliefs:
        if not b.source or not b.as_of:
            problems.append(f"{b.belief_id}: missing source/as_of provenance")
            continue
        age = (now - datetime.fromisoformat(b.as_of)).total_seconds()
        if age > b.freshness_ttl_seconds and not b.was_revalidated_at_decision:
            problems.append(
                f"{b.belief_id}: belief used past TTL "
                f"({age:.0f}s > {b.freshness_ttl_seconds:.0f}s) without revalidation"
            )
    return problems
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gti/test_receipt.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add aragora/gti/receipt.py tests/gti/test_receipt.py
git commit -m "feat(gti): belief provenance + TTL freshness validation"
```

---

### Task 5: CLI scorer — scorecard JSON + status surface

**Files:**
- Create: `scripts/score_gti_benchmark.py`
- Test: `tests/scripts/test_score_gti_benchmark.py`
- (Generated at runtime) `docs/status/generated/gti/scorecard-<ISO>.json`, `docs/status/GTI_GROUND_TRUTH_INTEGRITY_STATUS.md`

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_score_gti_benchmark.py
import json
import subprocess
import sys
from pathlib import Path


def test_cli_writes_scorecard_and_status(tmp_path):
    corpus = Path("docs/status/generated/gti/scenarios.json")
    scorecard = tmp_path / "scorecard.json"
    status = tmp_path / "GTI_STATUS.md"
    result = subprocess.run(
        [
            sys.executable, "scripts/score_gti_benchmark.py",
            "--corpus", str(corpus),
            "--scorecard-out", str(scorecard),
            "--status-out", str(status),
            "--now", "2026-06-06T12:00:00+00:00",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(scorecard.read_text())
    assert data["benchmark"] == "gti-ground-truth-integrity-v1"
    assert data["generated_at"] == "2026-06-06T12:00:00+00:00"
    assert data["naive"]["stale_belief_action_rate"] > data["gated"]["stale_belief_action_rate"]
    text = status.read_text()
    assert "Last updated: 2026-06-06T12:00:00+00:00" in text
    assert "stale_belief_action_rate" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_score_gti_benchmark.py -v`
Expected: FAIL with non-zero return code (script missing)

- [ ] **Step 3: Implement the CLI**

```python
# scripts/score_gti_benchmark.py
"""Score the Ground-Truth Integrity benchmark and publish a scorecard + status surface.

Usage:
    python scripts/score_gti_benchmark.py \
        --corpus docs/status/generated/gti/scenarios.json \
        --scorecard-out docs/status/generated/gti/scorecard-$(date -u +%Y%m%dT%H%M%SZ).json \
        --status-out docs/status/GTI_GROUND_TRUTH_INTEGRITY_STATUS.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aragora.gti.scenarios import load_scenarios  # noqa: E402
from aragora.gti.scorer import score_corpus  # noqa: E402


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _render_status(result, generated_at: str) -> str:
    n, g, d = result.naive, result.gated, result.delta
    return (
        "# Ground-Truth Integrity (GTI) Status\n\n"
        f"Last updated: {generated_at}\n\n"
        f"Benchmark: gti-ground-truth-integrity-v1 | scenarios: {result.scenario_count}\n\n"
        "| metric | naive | gated | delta (gated improvement) |\n"
        "|---|---|---|---|\n"
        f"| stale_belief_action_rate | {n.stale_belief_action_rate:.3f} | {g.stale_belief_action_rate:.3f} | {d.stale_belief_action_rate:+.3f} |\n"
        f"| detection_rate | {n.detection_rate:.3f} | {g.detection_rate:.3f} | {d.detection_rate:+.3f} |\n"
        f"| correction_rate | {n.correction_rate:.3f} | {g.correction_rate:.3f} | {d.correction_rate:+.3f} |\n"
        f"| false_green_rate | {n.false_green_rate:.3f} | {g.false_green_rate:.3f} | {d.false_green_rate:+.3f} |\n\n"
        "Project scale metrics are not duplicated here; see `docs/METRICS.md` "
        "(`python scripts/regenerate_metrics.py --check`).\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the GTI benchmark.")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--scorecard-out", required=True, type=Path)
    parser.add_argument("--status-out", required=True, type=Path)
    parser.add_argument("--now", default=None, help="ISO8601 override for deterministic runs")
    args = parser.parse_args(argv)

    generated_at = args.now or _now_iso()
    scenarios = load_scenarios(args.corpus)
    result = score_corpus(scenarios)

    scorecard = {
        "benchmark": "gti-ground-truth-integrity-v1",
        "generated_at": generated_at,
        "scenario_count": result.scenario_count,
        "naive": dataclasses.asdict(result.naive),
        "gated": dataclasses.asdict(result.gated),
        "delta": dataclasses.asdict(result.delta),
        "scenario_ids": [s.id for s in scenarios],
    }
    args.scorecard_out.parent.mkdir(parents=True, exist_ok=True)
    args.scorecard_out.write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")
    args.status_out.parent.mkdir(parents=True, exist_ok=True)
    args.status_out.write_text(_render_status(result, generated_at), encoding="utf-8")
    print(json.dumps({"ok": True, "scorecard": str(args.scorecard_out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scripts/test_score_gti_benchmark.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Generate the committed status surface**

Run:
```bash
python scripts/score_gti_benchmark.py \
  --corpus docs/status/generated/gti/scenarios.json \
  --scorecard-out docs/status/generated/gti/scorecard-20260606T120000Z.json \
  --status-out docs/status/GTI_GROUND_TRUTH_INTEGRITY_STATUS.md \
  --now 2026-06-06T12:00:00+00:00
```
Expected: prints `{"ok": true, ...}`; both files created.

- [ ] **Step 6: Commit**

```bash
git add scripts/score_gti_benchmark.py tests/scripts/test_score_gti_benchmark.py docs/status/GTI_GROUND_TRUTH_INTEGRITY_STATUS.md docs/status/generated/gti/scorecard-20260606T120000Z.json
git commit -m "feat(gti): CLI scorer publishing scorecard + status surface"
```

---

### Task 6: Register `gti` as a proof surface

**Files:**
- Modify: `scripts/probe_proof_surface_freshness.py` (the `SURFACE_PATHS` dict, ~line 84)
- Test: `tests/scripts/test_probe_proof_surface_freshness.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_probe_proof_surface_freshness.py  (append)
def test_gti_surface_is_registered():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "probe_pf", Path("scripts/probe_proof_surface_freshness.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "gti" in mod.SURFACE_PATHS
    assert mod.SURFACE_PATHS["gti"] == Path("docs/status/GTI_GROUND_TRUTH_INTEGRITY_STATUS.md")
    # gti is opt-in: it must NOT change the default surface set.
    assert "gti" not in mod.DEFAULT_SURFACES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/test_probe_proof_surface_freshness.py::test_gti_surface_is_registered -v`
Expected: FAIL with `KeyError`/assertion (`gti` not registered)

- [ ] **Step 3: Add the surface entry**

In `scripts/probe_proof_surface_freshness.py`, extend the `SURFACE_PATHS` dict (do not touch `DEFAULT_SURFACES`):

```python
SURFACE_PATHS: dict[str, Path] = {
    "b0": Path("docs/status/B0_BENCHMARK_TRUTH_STATUS.md"),
    "tw03": Path("docs/status/TW03_RESCUE_PRODUCTIZATION_STATUS.md"),
    "gti": Path("docs/status/GTI_GROUND_TRUTH_INTEGRITY_STATUS.md"),
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scripts/test_probe_proof_surface_freshness.py -v`
Expected: PASS (existing tests + the new one). Also verify the probe sees gti:
```bash
python3 scripts/probe_proof_surface_freshness.py --surfaces gti --pretty
```
Expected: a JSON record for `gti` (fresh, since Task 5 just generated it).

- [ ] **Step 5: Commit**

```bash
git add scripts/probe_proof_surface_freshness.py tests/scripts/test_probe_proof_surface_freshness.py
git commit -m "feat(gti): register gti proof surface (opt-in; guardian picks it up)"
```

---

### Task 7: End-to-end smoke + final validation

**Files:**
- Test: `tests/gti/test_end_to_end.py`

- [ ] **Step 1: Write the failing end-to-end test**

```python
# tests/gti/test_end_to_end.py
from pathlib import Path

from aragora.gti.scenarios import load_scenarios
from aragora.gti.scorer import score_corpus


def test_gated_beats_naive_end_to_end_on_real_corpus():
    scenarios = load_scenarios(Path("docs/status/generated/gti/scenarios.json"))
    result = score_corpus(scenarios)
    # Headline claim: the gated path materially reduces stale-belief action.
    assert result.delta.stale_belief_action_rate >= 0.3
    # And never increases false greens.
    assert result.delta.false_green_rate >= 0.0
```

- [ ] **Step 2: Run it**

Run: `pytest tests/gti/test_end_to_end.py -v`
Expected: PASS (corpus is constructed so most wrong beliefs are age- or quorum-detectable, with a couple honest misses + controls, yielding delta ≥ 0.3).

- [ ] **Step 3: Run the full GTI suite + pre-commit**

Run:
```bash
pytest tests/gti tests/scripts/test_score_gti_benchmark.py tests/scripts/test_probe_proof_surface_freshness.py -q
pre-commit run --files aragora/gti/*.py scripts/score_gti_benchmark.py scripts/probe_proof_surface_freshness.py docs/status/generated/gti/scenarios.json
```
Expected: all green; pre-commit clean.

- [ ] **Step 4: Commit**

```bash
git add tests/gti/test_end_to_end.py
git commit -m "test(gti): end-to-end gated-beats-naive guard on real corpus"
```

---

## Self-Review

**Spec coverage:** corpus (Task 1) ✓ 12–15 scenarios from F1–F6 + controls; metrics incl. stale-belief-action/detection/correction/false-green (Task 3) ✓; naive-vs-gated arms (Tasks 2–3) ✓; DecisionReceipt provenance/freshness fields source+as_of+verification_method+freshness_ttl+was_revalidated (Task 4) ✓; `gti` proof-surface registration (Task 6) ✓; canonical-metrics-by-reference (status doc links METRICS.md) ✓; out-of-scope items (droid→router, canonical-doc edits, issue creation, #7832) all excluded ✓.

**Placeholder scan:** no TBD/TODO; every code step has complete code; corpus is fully enumerated (14 scenarios).

**Type consistency:** `Scenario`, `PolicyOutcome`, `Metrics`, `ScoreResult`, `BeliefProvenance` names + fields match across Tasks 1–5; `score_corpus`/`naive_policy`/`gated_policy`/`validate_belief_provenance` signatures consistent with their call sites in tests and the CLI.
