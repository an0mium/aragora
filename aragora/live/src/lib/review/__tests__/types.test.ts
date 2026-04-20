/**
 * Tests for aragora/live/src/lib/review/types.ts.
 *
 * The critical property is CANONICAL-STRING DISCIPLINE: every enum string
 * in this module must match exactly the ``to_dict()`` output of the Python
 * dataclasses in ``aragora/review/{protocol,receipt,policy}.py``.  Drift
 * breaks JSON interop silently.  Each test below hard-codes the Python
 * string value, so a refactor that touches one side will fail loudly if
 * the other is not updated.
 *
 * Python counterpart values are drawn from the canonical test suites in
 * ``tests/review/test_protocol.py``, ``tests/review/test_receipt.py``,
 * and ``tests/review/test_policy.py`` — which are themselves the source
 * of truth.
 */

import {
  ADVISORY_NOTE,
  BudgetScope,
  DissentPosition,
  EvidenceKind,
  Recommendation,
  ReviewDepth,
  ReviewPolicyDecision,
  ReviewRole,
  RiskClass,
  SettlementAction,
  SynthesisPolicy,
  ValidationKind,
  ValidationResult,
  type BriefReceipt,
  type CostMeter,
  type DissentingView,
  type EvidenceRef,
  type ReviewBrief,
  type ReviewBudget,
  type ReviewPolicy,
  type RoleFinding,
  type SettlementLinkage,
  type ValidationRef,
} from "../types";

// ---------------------------------------------------------------------------
// Enum canonical-string discipline (each value must match Python exactly)
// ---------------------------------------------------------------------------

describe("canonical string discipline — enums match aragora/review/*.py", () => {
  test("ReviewRole values", () => {
    expect(ReviewRole.LOGIC).toBe("logic_reviewer");
    expect(ReviewRole.SECURITY).toBe("security_reviewer");
    expect(ReviewRole.MAINTAINABILITY).toBe("maintainability_reviewer");
    expect(ReviewRole.SKEPTIC).toBe("skeptic");
    expect(ReviewRole.SYNTHESIZER).toBe("synthesizer");
  });

  test("Recommendation values", () => {
    expect(Recommendation.APPROVE_CANDIDATE).toBe("approve_candidate");
    expect(Recommendation.NEEDS_HUMAN_ATTENTION).toBe("needs_human_attention");
    expect(Recommendation.REPAIR_FIRST).toBe("repair_first");
  });

  test("DissentPosition values", () => {
    expect(DissentPosition.APPROVE).toBe("approve");
    expect(DissentPosition.REQUEST_CHANGES).toBe("request_changes");
    expect(DissentPosition.DEFER).toBe("defer");
  });

  test("SynthesisPolicy values", () => {
    expect(SynthesisPolicy.MAJORITY).toBe("majority");
    expect(SynthesisPolicy.WEIGHTED).toBe("weighted");
    expect(SynthesisPolicy.SYNTHESIZER_AGENT).toBe("synthesizer");
    expect(SynthesisPolicy.UNANIMOUS_OR_ESCALATE).toBe("unanimous_or_escalate");
  });

  test("EvidenceKind values", () => {
    expect(EvidenceKind.FILE).toBe("file");
    expect(EvidenceKind.TEST).toBe("test");
    expect(EvidenceKind.COMMIT).toBe("commit");
    expect(EvidenceKind.ARTIFACT).toBe("artifact");
    expect(EvidenceKind.ISSUE).toBe("issue");
    expect(EvidenceKind.PR).toBe("pr");
    expect(EvidenceKind.EXTERNAL).toBe("external");
  });

  test("ValidationKind values", () => {
    expect(ValidationKind.CI_CHECK).toBe("ci_check");
    expect(ValidationKind.TEST_SUITE).toBe("test_suite");
    expect(ValidationKind.RECEIPT).toBe("receipt");
    expect(ValidationKind.BENCHMARK).toBe("benchmark");
    expect(ValidationKind.MANUAL_REVIEW).toBe("manual_review");
  });

  test("ValidationResult values", () => {
    expect(ValidationResult.SUCCESS).toBe("success");
    expect(ValidationResult.FAILURE).toBe("failure");
    expect(ValidationResult.SKIPPED).toBe("skipped");
    expect(ValidationResult.CANCELLED).toBe("cancelled");
    expect(ValidationResult.PENDING).toBe("pending");
  });

  test("SettlementAction values", () => {
    expect(SettlementAction.APPROVE).toBe("approve");
    expect(SettlementAction.REQUEST_CHANGES).toBe("request_changes");
    expect(SettlementAction.DEFER).toBe("defer");
  });

  test("ReviewDepth values", () => {
    expect(ReviewDepth.TRIVIAL).toBe("trivial");
    expect(ReviewDepth.STANDARD).toBe("standard");
    expect(ReviewDepth.DEEP).toBe("deep");
  });

  test("RiskClass values", () => {
    expect(RiskClass.LOW).toBe("low");
    expect(RiskClass.MEDIUM).toBe("medium");
    expect(RiskClass.HIGH).toBe("high");
    expect(RiskClass.CRITICAL).toBe("critical");
  });

  test("ReviewPolicyDecision values", () => {
    expect(ReviewPolicyDecision.ALLOW).toBe("allow");
    expect(ReviewPolicyDecision.DEGRADE).toBe("degrade");
    expect(ReviewPolicyDecision.DENY).toBe("deny");
    expect(ReviewPolicyDecision.ESCALATE).toBe("escalate");
  });

  test("BudgetScope values", () => {
    expect(BudgetScope.PER_PR).toBe("per_pr");
    expect(BudgetScope.PER_REPO_DAILY).toBe("per_repo_daily");
    expect(BudgetScope.PER_ORG_DAILY).toBe("per_org_daily");
  });
});

// ---------------------------------------------------------------------------
// ADVISORY_NOTE contract
// ---------------------------------------------------------------------------

describe("ADVISORY_NOTE", () => {
  test("mentions advisory-only and human-settlement semantics", () => {
    expect(ADVISORY_NOTE.toLowerCase()).toContain("advisory");
    expect(ADVISORY_NOTE.toLowerCase()).toContain("human settlement");
  });
});

// ---------------------------------------------------------------------------
// JSON-payload compatibility — a Python to_dict() output must parse cleanly
// as the corresponding TS interface.  Fixtures below mirror the exact
// field set produced by the Python dataclasses.
// ---------------------------------------------------------------------------

describe("python-json payloads parse as TS types", () => {
  test("RoleFinding shape", () => {
    const payload = {
      role: "logic_reviewer",
      agent: "claude-opus-4-7",
      model: "claude-opus-4-7-1m",
      confidence: 0.9,
      finding_text: "No regressions found.",
      latency_ms: 1200,
      cost_usd: 0.045,
    };
    const finding: RoleFinding = payload as RoleFinding;
    expect(finding.role).toBe(ReviewRole.LOGIC);
    expect(finding.confidence).toBe(0.9);
  });

  test("DissentingView shape (optional role)", () => {
    const payload = {
      agent: "grok-3",
      position: "request_changes",
      reason: "Security concern.",
    };
    const view: DissentingView = payload as DissentingView;
    expect(view.position).toBe(DissentPosition.REQUEST_CHANGES);
    expect(view.role).toBeUndefined();
  });

  test("ReviewBrief shape", () => {
    const payload: ReviewBrief = {
      pr_number: 6304,
      repo: "synaptent/aragora",
      head_sha: "abc123",
      base_sha: "def456",
      packet_sha: "hash789",
      recommendation: "approve_candidate" as Recommendation,
      top_line: "Bounded docs PR.",
      role_findings: [],
      dissent: [],
      validation_summary: "pre-commit clean",
      overall_confidence: 0.88,
      disagreement_score: 0.05,
      total_cost_usd: 0.18,
      total_wall_clock_ms: 4200,
      agent_roster: ["claude-opus-4-7", "gpt-5-4"],
      generated_at: "2026-04-20T15:00:00+00:00",
      advisory_only: true,
      settlement_note: ADVISORY_NOTE,
    };
    expect(payload.advisory_only).toBe(true);
    expect(payload.recommendation).toBe(Recommendation.APPROVE_CANDIDATE);
  });

  test("EvidenceRef shape", () => {
    const payload: EvidenceRef = {
      kind: "file" as EvidenceKind,
      path: "aragora/review/protocol.py",
      sha: "",
      line_range: [42, 58],
      quote: "def to_dict(self) -> dict[str, Any]:",
    };
    expect(payload.kind).toBe(EvidenceKind.FILE);
    expect(payload.line_range).toEqual([42, 58]);
  });

  test("ValidationRef shape", () => {
    const payload: ValidationRef = {
      kind: "ci_check" as ValidationKind,
      name: "Version Alignment",
      result: "success" as ValidationResult,
      url: "https://github.com/synaptent/aragora/actions/runs/12345",
    };
    expect(payload.kind).toBe(ValidationKind.CI_CHECK);
    expect(payload.result).toBe(ValidationResult.SUCCESS);
  });

  test("BriefReceipt advisory invariant", () => {
    const brief: ReviewBrief = {
      pr_number: 6304,
      repo: "synaptent/aragora",
      head_sha: "abc",
      base_sha: "def",
      packet_sha: "h",
      recommendation: "approve_candidate" as Recommendation,
      top_line: "",
      role_findings: [],
      dissent: [],
      validation_summary: "",
      overall_confidence: 0.9,
      disagreement_score: 0,
      total_cost_usd: 0,
      total_wall_clock_ms: 0,
      agent_roster: [],
      generated_at: "",
      advisory_only: true,
      settlement_note: ADVISORY_NOTE,
    };
    const receipt: BriefReceipt = {
      brief,
      evidence_refs: [],
      validation_refs: [],
      receipt_id: "receipt-sha",
      created_at: "2026-04-20T15:00:00+00:00",
      advisory_only: true,
      settlement_note: ADVISORY_NOTE,
    };
    expect(receipt.advisory_only).toBe(true);
    expect(receipt.brief.advisory_only).toBe(true);
  });

  test("SettlementLinkage shape (human settlement is not advisory)", () => {
    const linkage: SettlementLinkage = {
      brief_receipt_id: "brief-001",
      settlement_receipt_id: "settlement-001",
      settlement_receipt_path: ".aragora/review-queue/settlements/pr-6304.json",
      head_sha: "abc",
      packet_sha: "h",
      pr_number: 6304,
      repo: "synaptent/aragora",
      action: "approve" as SettlementAction,
      settled_at: "2026-04-20T15:00:00+00:00",
      repair_receipt_ids: [],
      repair_receipt_paths: [],
      advisory_only: false,
    };
    expect(linkage.advisory_only).toBe(false);
    expect(linkage.action).toBe(SettlementAction.APPROVE);
  });

  test("ReviewBudget default-shape assumptions", () => {
    const budget: ReviewBudget = {
      per_pr_usd_cap: 25.0,
      per_repo_usd_daily_cap: 0.0,
      per_org_usd_daily_cap: 0.0,
      daily_caps_apply_at_or_above_depth: "standard" as ReviewDepth,
      alert_threshold_pct: 80.0,
      hard_limit: true,
    };
    expect(budget.per_pr_usd_cap).toBe(25.0);
    expect(budget.daily_caps_apply_at_or_above_depth).toBe(ReviewDepth.STANDARD);
  });

  test("ReviewPolicy nests budget and tuple of rules", () => {
    const policy: ReviewPolicy = {
      budget: {
        per_pr_usd_cap: 25.0,
        per_repo_usd_daily_cap: 0.0,
        per_org_usd_daily_cap: 0.0,
        daily_caps_apply_at_or_above_depth: "standard" as ReviewDepth,
        alert_threshold_pct: 80.0,
        hard_limit: true,
      },
      depth_rules: [
        {
          target_depth: "deep" as ReviewDepth,
          min_additions_plus_deletions: 500,
          subsystem_prefixes: ["aragora/security/"],
          min_risk_class: "high" as RiskClass,
        },
      ],
      default_depth: "standard" as ReviewDepth,
    };
    expect(policy.depth_rules).toHaveLength(1);
    expect(policy.depth_rules[0].target_depth).toBe(ReviewDepth.DEEP);
  });

  test("CostMeter with multi-pool headroom and binding_scope", () => {
    const meter: CostMeter = {
      depth_chosen: "standard" as ReviewDepth,
      decision: "degrade" as ReviewPolicyDecision,
      estimated_cost_usd: 8.0,
      actual_cost_usd: 7.5,
      headroom_by_scope: [
        {
          scope: "per_pr" as BudgetScope,
          cap_usd: 25.0,
          remaining_usd: 24.0,
        },
        {
          scope: "per_repo_daily" as BudgetScope,
          cap_usd: 50.0,
          remaining_usd: 2.0,
          applies_at_or_above_depth: "standard" as ReviewDepth,
        },
      ],
      binding_scope: "per_repo_daily" as BudgetScope,
      alert_triggered: true,
    };
    expect(meter.binding_scope).toBe(BudgetScope.PER_REPO_DAILY);
    expect(meter.headroom_by_scope).toHaveLength(2);
    expect(meter.headroom_by_scope[1].remaining_usd).toBe(2.0);
  });
});

// ---------------------------------------------------------------------------
// Type-level contract: readonly discipline.  These tests pass at runtime;
// the value they add is compile-time: the TS compiler rejects attempts to
// mutate readonly fields, which guarantees the schema behaves like the
// Python frozen-dataclass + tuple pattern.
// ---------------------------------------------------------------------------

describe("readonly discipline (compile-time)", () => {
  test("readonly tuples reject mutation (TS compile-time check)", () => {
    const brief: ReviewBrief = {
      pr_number: 1,
      repo: "",
      head_sha: "",
      base_sha: "",
      packet_sha: "",
      recommendation: "approve_candidate" as Recommendation,
      top_line: "",
      role_findings: [],
      dissent: [],
      validation_summary: "",
      overall_confidence: 0,
      disagreement_score: 0,
      total_cost_usd: 0,
      total_wall_clock_ms: 0,
      agent_roster: ["model-a"],
      generated_at: "",
      advisory_only: true,
      settlement_note: ADVISORY_NOTE,
    };
    // @ts-expect-error — agent_roster is readonly; push must fail type-check.
    brief.agent_roster.push("model-b");
    // @ts-expect-error — dissent is readonly.
    brief.dissent.push({ agent: "x", position: "defer", reason: "" });
    // @ts-expect-error — pr_number is readonly.
    brief.pr_number = 2;
  });
});
