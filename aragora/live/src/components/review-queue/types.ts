export type ReviewQueueVerdict =
  | 'approve_candidate'
  | 'needs_human_attention'
  | 'repair_first'
  | null;

export interface ReviewQueueBrief {
  pr_number: number;
  title?: string | null;
  source?: string | null;
  source_path?: string | null;
  head_sha?: string | null;
  verdict: ReviewQueueVerdict;
  raw_verdict?: string | null;
  confidence?: number | null;
  scope?: string | null;
  logic?: string | null;
  security?: string | null;
  maintainability?: string | null;
  skeptic?: string | null;
  recommended_action?: string | null;
}

export interface ReviewQueueStatusCounts {
  success: number;
  failure: number;
  pending: number;
  cancelled: number;
  total: number;
}

export interface ReviewQueueItem {
  number: number;
  title: string;
  url: string;
  diff_url: string;
  head_sha: string;
  author: string;
  is_draft: boolean;
  mergeable: string;
  review_decision: string;
  labels: string[];
  additions: number;
  deletions: number;
  changed_files: number;
  checks_summary: string;
  lane: string;
  lane_reason: string;
  created_at?: string | null;
  updated_at?: string | null;
  status_counts: ReviewQueueStatusCounts;
  touched_subsystems: string[];
  high_risk_paths_touched: string[];
  machine_recommendation?: ReviewQueueVerdict;
  machine_recommendation_reason?: string | null;
  brief?: ReviewQueueBrief | null;
  brief_available: boolean;
  deferred_until?: string | null;
  deferred_reason?: string | null;
}

export interface ReviewQueueDetail {
  pr: ReviewQueueItem;
  packet: {
    pr_number: number;
    title: string;
    url: string;
    head_sha: string;
    base_sha: string;
    author: string;
    is_draft: boolean;
    additions: number;
    deletions: number;
    changed_files: number;
    queue_bucket: string;
    touched_subsystems: string[];
    high_risk_paths_touched: string[];
    validation: string[];
    checks_summary: string;
    risk_flags: string[];
    machine_recommendation: ReviewQueueVerdict;
    machine_recommendation_reason: string;
    packet_sha: string;
    generated_at: string;
    advisory_only: boolean;
    settlement_note: string;
  };
  brief?: ReviewQueueBrief | null;
  checks: Array<{
    name: string;
    status: string;
    conclusion: string;
    details_url?: string | null;
  }>;
  files: Array<{
    path: string;
    additions: number;
    deletions: number;
  }>;
  diff_url: string;
}

export interface ReviewQueueListResponse {
  prs: ReviewQueueItem[];
  count: number;
  generated_at: string;
  source: string;
}

export interface ReviewQueueStats {
  decisions_today: number;
  approvals_today: number;
  median_decision_seconds: number;
  streak: number;
  source: string;
}
