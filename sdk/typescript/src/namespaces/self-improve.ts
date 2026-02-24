/**
 * Self-Improve Namespace API
 *
 * Provides REST API endpoints for autonomous self-improvement run management:
 * - Start new improvement runs (with optional dry-run preview)
 * - List and query run history
 * - Get run status and progress
 * - Cancel active runs
 * - Manage git worktrees used during execution
 */

/**
 * Self-improvement run status.
 */
export type SelfImproveRunStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

/**
 * Execution mode.
 */
export type SelfImproveMode = 'flat' | 'hierarchical';

/**
 * Self-improvement run record.
 */
export interface SelfImproveRun {
  run_id: string;
  goal: string;
  tracks: string[];
  mode: SelfImproveMode;
  status: SelfImproveRunStatus;
  dry_run: boolean;
  max_cycles: number;
  budget_limit_usd?: number;
  total_subtasks?: number;
  completed_subtasks?: number;
  failed_subtasks?: number;
  summary?: string;
  error?: string;
  plan?: Record<string, unknown>;
  created_at?: string;
  started_at?: string;
  completed_at?: string;
}

/**
 * Worktree entry.
 */
export interface Worktree {
  branch_name: string;
  worktree_path: string;
  track: string;
  created_at: string;
  assignment_id: string;
}

/**
 * Start run request parameters.
 */
export interface StartRunRequest {
  goal: string;
  tracks?: string[];
  mode?: SelfImproveMode;
  budget_limit_usd?: number;
  max_cycles?: number;
  dry_run?: boolean;
}

/**
 * Client interface for self-improve operations.
 */
interface SelfImproveClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Self-Improve API for autonomous improvement run management.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Start a dry-run to preview a plan
 * const preview = await client.selfImprove.start({
 *   goal: 'Improve test coverage',
 *   dry_run: true,
 * });
 *
 * // Start an actual run
 * const run = await client.selfImprove.start({
 *   goal: 'Improve test coverage',
 *   tracks: ['developer', 'qa'],
 *   max_cycles: 3,
 * });
 *
 * // Check run status
 * const status = await client.selfImprove.getRun(run.run_id);
 * ```
 */
export class SelfImproveAPI {
  constructor(private client: SelfImproveClientInterface) {}

  /**
   * Start a new self-improvement run.
   */
  async start(request: StartRunRequest): Promise<{
    run_id: string;
    status: string;
    plan?: Record<string, unknown>;
  }> {
    return this.client.request('POST', '/api/v1/self-improve/start', {
      json: request as unknown as Record<string, unknown>,
    });
  }

  /**
   * List self-improvement runs with optional filtering and pagination.
   */
  async listRuns(options?: {
    limit?: number;
    offset?: number;
    status?: SelfImproveRunStatus;
  }): Promise<{
    runs: SelfImproveRun[];
    total: number;
    limit: number;
    offset: number;
  }> {
    return this.client.request('GET', '/api/v1/self-improve/runs', {
      params: options as Record<string, unknown>,
    });
  }

  /**
   * Get a specific run's status and progress.
   */
  async getRun(runId: string): Promise<SelfImproveRun> {
    return this.client.request('GET', `/api/v1/self-improve/runs/${runId}`);
  }

  /**
   * Cancel a running self-improvement run.
   */
  async cancelRun(runId: string): Promise<{ run_id: string; status: string }> {
    return this.client.request('POST', `/api/v1/self-improve/runs/${runId}/cancel`);
  }

  /**
   * Get run history (alias for listRuns).
   */
  async getHistory(options?: {
    limit?: number;
    offset?: number;
    status?: SelfImproveRunStatus;
  }): Promise<{
    runs: SelfImproveRun[];
    total: number;
    limit: number;
    offset: number;
  }> {
    return this.client.request('GET', '/api/v1/self-improve/history', {
      params: options as Record<string, unknown>,
    });
  }

  /**
   * List active git worktrees managed by the branch coordinator.
   */
  async listWorktrees(): Promise<{
    worktrees: Worktree[];
    total: number;
  }> {
    return this.client.request('GET', '/api/v1/self-improve/worktrees');
  }

  /**
   * Clean up all managed worktrees.
   */
  async cleanupWorktrees(): Promise<{
    removed: number;
    status: string;
  }> {
    return this.client.request('POST', '/api/v1/self-improve/worktrees/cleanup');
  }

  /**
   * Start a new self-improvement run (canonical endpoint).
   *
   * This is the primary endpoint for starting runs. Accepts all configuration
   * options including scan_mode, quick_mode, and require_approval.
   */
  async run(request: StartRunRequest & {
    scan_mode?: boolean;
    quick_mode?: boolean;
    require_approval?: boolean;
  }): Promise<{
    run_id: string;
    status: string;
    plan?: Record<string, unknown>;
  }> {
    return this.client.request('POST', '/api/v1/self-improve/run', {
      json: request as unknown as Record<string, unknown>,
    });
  }

  /**
   * Get current self-improvement cycle status.
   *
   * Returns whether a cycle is running or idle, along with active run details.
   */
  async getStatus(): Promise<{
    state: 'idle' | 'running';
    active_runs: number;
    runs: SelfImproveRun[];
  }> {
    return this.client.request('GET', '/api/v1/self-improve/status');
  }

  /**
   * Start a hierarchical planner/worker/judge coordination cycle.
   *
   * Uses HierarchicalCoordinator for structured goal execution with
   * automatic decomposition, parallel workers, and judge review.
   */
  async coordinate(request: {
    goal: string;
    tracks?: string[];
    max_cycles?: number;
    quality_threshold?: number;
    max_parallel_workers?: number;
  }): Promise<{
    run_id: string;
    status: string;
    mode: string;
  }> {
    return this.client.request('POST', '/api/v1/self-improve/coordinate', {
      json: request as unknown as Record<string, unknown>,
    });
  }

  /**
   * Submit self-improvement feedback.
   */
  async submitFeedback(feedback: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/self-improve/feedback', {
      json: feedback,
    });
  }

  /**
   * Get self-improvement feedback summary.
   */
  async getFeedbackSummary(query?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/self-improve/feedback-summary', {
      json: query ?? {},
    });
  }

  /**
   * Create or update self-improvement goals.
   */
  async upsertGoals(goals: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/self-improve/goals', {
      json: goals,
    });
  }

  /**
   * Get self-improvement metrics summary.
   */
  async getMetricsSummary(query?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/self-improve/metrics/summary', {
      json: query ?? {},
    });
  }

  /**
   * Get self-improvement regression history.
   */
  async getRegressionHistory(query?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/self-improve/regression-history', {
      json: query ?? {},
    });
  }
}
