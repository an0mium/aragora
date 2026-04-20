/**
 * PR intelligence brief — TypeScript contracts for the Next.js UI (#6304 foundation).
 *
 * Mirrors the Python schema in ``aragora/review/{protocol,receipt,policy}.py``
 * that landed in #6334, #6353, and #6359. Consumers are future UI components
 * (the `aragora/live/src/app/(app)/reviews/` route) plus any TS SDK surfaces.
 *
 * This module is **schema only**. No components, no data fetching, no state
 * machines. Behavior ships in successor PRs that import these types.
 *
 * Canonical-string discipline (critical): every enum string must match the
 * Python ``to_dict()`` output exactly. Drift breaks JSON interop silently.
 * Tests guard every string value against its Python counterpart.
 *
 * Immutability: sequence fields are typed as ``readonly T[]``. Python tuples
 * give runtime immutability; TS readonly gives compile-time immutability only
 * (a caller who casts can still mutate). That asymmetry is unavoidable and
 * documented; downstream components should not rely on runtime freezing.
 */

// ---------------------------------------------------------------------------
// Enums (as const objects for `Foo.BAR` access + string-literal union types)
// ---------------------------------------------------------------------------

export const ReviewRole = {
  LOGIC: "logic_reviewer",
  SECURITY: "security_reviewer",
  MAINTAINABILITY: "maintainability_reviewer",
  SKEPTIC: "skeptic",
  SYNTHESIZER: "synthesizer",
} as const;
export type ReviewRole = (typeof ReviewRole)[keyof typeof ReviewRole];

export const Recommendation = {
  APPROVE_CANDIDATE: "approve_candidate",
  NEEDS_HUMAN_ATTENTION: "needs_human_attention",
  REPAIR_FIRST: "repair_first",
} as const;
export type Recommendation = (typeof Recommendation)[keyof typeof Recommendation];

export const DissentPosition = {
  APPROVE: "approve",
  REQUEST_CHANGES: "request_changes",
  DEFER: "defer",
} as const;
export type DissentPosition = (typeof DissentPosition)[keyof typeof DissentPosition];

export const SynthesisPolicy = {
  MAJORITY: "majority",
  WEIGHTED: "weighted",
  SYNTHESIZER_AGENT: "synthesizer",
  UNANIMOUS_OR_ESCALATE: "unanimous_or_escalate",
} as const;
export type SynthesisPolicy = (typeof SynthesisPolicy)[keyof typeof SynthesisPolicy];

export const EvidenceKind = {
  FILE: "file",
  TEST: "test",
  COMMIT: "commit",
  ARTIFACT: "artifact",
  ISSUE: "issue",
  PR: "pr",
  EXTERNAL: "external",
} as const;
export type EvidenceKind = (typeof EvidenceKind)[keyof typeof EvidenceKind];

export const ValidationKind = {
  CI_CHECK: "ci_check",
  TEST_SUITE: "test_suite",
  RECEIPT: "receipt",
  BENCHMARK: "benchmark",
  MANUAL_REVIEW: "manual_review",
} as const;
export type ValidationKind = (typeof ValidationKind)[keyof typeof ValidationKind];

export const ValidationResult = {
  SUCCESS: "success",
  FAILURE: "failure",
  SKIPPED: "skipped",
  CANCELLED: "cancelled",
  PENDING: "pending",
} as const;
export type ValidationResult = (typeof ValidationResult)[keyof typeof ValidationResult];

export const SettlementAction = {
  APPROVE: "approve",
  REQUEST_CHANGES: "request_changes",
  DEFER: "defer",
} as const;
export type SettlementAction = (typeof SettlementAction)[keyof typeof SettlementAction];

export const ReviewDepth = {
  TRIVIAL: "trivial",
  STANDARD: "standard",
  DEEP: "deep",
} as const;
export type ReviewDepth = (typeof ReviewDepth)[keyof typeof ReviewDepth];

export const RiskClass = {
  LOW: "low",
  MEDIUM: "medium",
  HIGH: "high",
  CRITICAL: "critical",
} as const;
export type RiskClass = (typeof RiskClass)[keyof typeof RiskClass];

export const ReviewPolicyDecision = {
  ALLOW: "allow",
  DEGRADE: "degrade",
  DENY: "deny",
  ESCALATE: "escalate",
} as const;
export type ReviewPolicyDecision =
  (typeof ReviewPolicyDecision)[keyof typeof ReviewPolicyDecision];

export const BudgetScope = {
  PER_PR: "per_pr",
  PER_REPO_DAILY: "per_repo_daily",
  PER_ORG_DAILY: "per_org_daily",
} as const;
export type BudgetScope = (typeof BudgetScope)[keyof typeof BudgetScope];

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

export const ADVISORY_NOTE =
  "Aragora review brief is advisory only. It does not approve or block merge. " +
  "Human settlement required.";

// ---------------------------------------------------------------------------
// Brief + debate shapes (mirror aragora/review/protocol.py)
// ---------------------------------------------------------------------------

export interface RoleFinding {
  readonly role: ReviewRole;
  readonly agent: string;
  readonly model: string;
  readonly confidence: number;
  readonly finding_text: string;
  readonly latency_ms: number;
  readonly cost_usd: number;
}

export interface DissentingView {
  readonly agent: string;
  readonly position: DissentPosition;
  readonly reason: string;
  readonly role?: ReviewRole | null;
}

export interface ReviewBrief {
  readonly pr_number: number;
  readonly repo: string;
  readonly head_sha: string;
  readonly base_sha: string;
  readonly packet_sha: string;
  readonly recommendation: Recommendation;
  readonly top_line: string;
  readonly role_findings: readonly RoleFinding[];
  readonly dissent: readonly DissentingView[];
  readonly validation_summary: string;
  readonly overall_confidence: number;
  readonly disagreement_score: number;
  readonly total_cost_usd: number;
  readonly total_wall_clock_ms: number;
  readonly agent_roster: readonly string[];
  readonly generated_at: string;
  readonly advisory_only: boolean;
  readonly settlement_note: string;
}

export interface PRReviewProtocolConfig {
  readonly model_panel: readonly string[];
  readonly output_roles: readonly ReviewRole[];
  readonly rounds: number;
  readonly synthesis_policy: SynthesisPolicy;
  readonly require_heterogeneous_models: boolean;
  readonly advisory_only: boolean;
}

// ---------------------------------------------------------------------------
// Receipt + linkage shapes (mirror aragora/review/receipt.py)
// ---------------------------------------------------------------------------

export interface EvidenceRef {
  readonly kind: EvidenceKind;
  readonly path: string;
  readonly sha: string;
  readonly line_range: readonly [number, number] | null;
  readonly quote: string;
}

export interface ValidationRef {
  readonly kind: ValidationKind;
  readonly name: string;
  readonly result: ValidationResult;
  readonly url: string;
}

export interface BriefReceipt {
  readonly brief: ReviewBrief;
  readonly evidence_refs: readonly EvidenceRef[];
  readonly validation_refs: readonly ValidationRef[];
  readonly receipt_id: string;
  readonly created_at: string;
  readonly advisory_only: boolean;
  readonly settlement_note: string;
}

export interface SettlementLinkage {
  readonly brief_receipt_id: string;
  readonly settlement_receipt_id: string;
  readonly settlement_receipt_path: string;
  readonly head_sha: string;
  readonly packet_sha: string;
  readonly pr_number: number;
  readonly repo: string;
  readonly action: SettlementAction;
  readonly settled_at: string;
  readonly repair_receipt_ids: readonly string[];
  readonly repair_receipt_paths: readonly string[];
  readonly advisory_only: boolean;
}

// ---------------------------------------------------------------------------
// Policy + budget + cost-meter shapes (mirror aragora/review/policy.py)
// ---------------------------------------------------------------------------

export interface DepthTrigger {
  readonly target_depth: ReviewDepth;
  readonly min_additions_plus_deletions: number;
  readonly subsystem_prefixes: readonly string[];
  readonly min_risk_class: RiskClass | null;
}

export interface ReviewBudget {
  readonly per_pr_usd_cap: number;
  readonly per_repo_usd_daily_cap: number;
  readonly per_org_usd_daily_cap: number;
  readonly daily_caps_apply_at_or_above_depth: ReviewDepth;
  readonly alert_threshold_pct: number;
  readonly hard_limit: boolean;
}

export interface ReviewPolicy {
  readonly budget: ReviewBudget;
  readonly depth_rules: readonly DepthTrigger[];
  readonly default_depth: ReviewDepth;
}

export interface BudgetHeadroom {
  readonly scope: BudgetScope;
  readonly cap_usd: number;
  readonly remaining_usd: number;
  readonly applies_at_or_above_depth?: ReviewDepth | null;
}

export interface CostMeter {
  readonly depth_chosen: ReviewDepth;
  readonly decision: ReviewPolicyDecision;
  readonly estimated_cost_usd: number;
  readonly actual_cost_usd: number;
  readonly headroom_by_scope: readonly BudgetHeadroom[];
  readonly binding_scope?: BudgetScope | null;
  readonly alert_triggered: boolean;
}
