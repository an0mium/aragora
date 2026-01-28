/**
 * Nomic Namespace API
 *
 * Provides a namespaced interface for nomic loop control and monitoring.
 * The nomic loop is the autonomous self-improvement cycle that enables
 * the system to propose, design, implement, and verify its own improvements.
 */

/**
 * Nomic loop health status
 */
export type NomicHealthStatus = 'healthy' | 'stalled' | 'not_running' | 'error';

/**
 * Nomic loop phase
 */
export type NomicPhase = 'context' | 'debate' | 'design' | 'implement' | 'verify';

/**
 * Nomic loop state
 */
export interface NomicState {
  running: boolean;
  cycle: number;
  phase: NomicPhase;
  paused: boolean;
  started_at?: string;
  last_update?: string;
  target_cycles?: number;
  auto_approve?: boolean;
}

/**
 * Nomic loop health
 */
export interface NomicHealth {
  status: NomicHealthStatus;
  cycle: number;
  phase: NomicPhase | null;
  last_activity: string | null;
  stall_duration_seconds: number | null;
  warnings: string[];
}

/**
 * Nomic loop metrics summary
 */
export interface NomicMetrics {
  summary: Record<string, unknown>;
  stuck_detection: {
    is_stuck: boolean;
    phase?: string;
    idle_seconds?: number;
  };
  status: string;
}

/**
 * Nomic log response
 */
export interface NomicLogResponse {
  lines: string[];
  total: number;
  showing: number;
}

/**
 * Risk register entry
 */
export interface RiskEntry {
  id: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  description: string;
  category: string;
  created_at: string;
  resolved_at?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Risk register response
 */
export interface RiskRegisterResponse {
  risks: RiskEntry[];
  total: number;
  critical_count: number;
  high_count: number;
}

/**
 * Improvement proposal
 */
export interface Proposal {
  id: string;
  title: string;
  description: string;
  status: 'pending' | 'approved' | 'rejected';
  submitted_at: string;
  approved_at?: string;
  rejected_at?: string;
  approved_by?: string;
  rejected_by?: string;
  rejection_reason?: string;
  changes?: Record<string, unknown>;
}

/**
 * Proposals list response
 */
export interface ProposalsResponse {
  proposals: Proposal[];
  total: number;
  all_proposals: number;
}

/**
 * Start loop request
 */
export interface StartLoopRequest {
  cycles?: number;
  max_cycles?: number;
  auto_approve?: boolean;
  dry_run?: boolean;
}

/**
 * Stop loop request
 */
export interface StopLoopRequest {
  graceful?: boolean;
}

/**
 * Approve proposal request
 */
export interface ApproveProposalRequest {
  proposal_id: string;
  approved_by?: string;
}

/**
 * Reject proposal request
 */
export interface RejectProposalRequest {
  proposal_id: string;
  rejected_by?: string;
  reason?: string;
}

/**
 * Operational mode
 */
export interface OperationalMode {
  name: string;
  type: 'builtin' | 'custom';
  description: string;
  slug?: string;
}

/**
 * Gas Town witness status
 */
export interface WitnessStatus {
  active: boolean;
  patrol_count: number;
  last_patrol: string | null;
  violations_detected: number;
  current_focus?: string;
  witnesses: Array<{
    id: string;
    name: string;
    status: 'active' | 'idle' | 'investigating';
    last_report: string | null;
  }>;
}

/**
 * Gas Town mayor information
 */
export interface MayorInfo {
  current_mayor: string | null;
  elected_at: string | null;
  term_ends: string | null;
  approval_rating: number | null;
  policies_enacted: number;
  vetoes: number;
  emergency_powers_active: boolean;
}

/**
 * Modes response
 */
export interface ModesResponse {
  modes: OperationalMode[];
  total: number;
}

/**
 * Interface for the internal client used by NomicAPI.
 */
interface NomicClientInterface {
  get<T>(path: string): Promise<T>;
  post<T>(path: string, body?: unknown): Promise<T>;
  request<T>(method: string, path: string, options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }): Promise<T>;
}

/**
 * Nomic API namespace.
 *
 * Provides methods for nomic loop control and monitoring:
 * - Get loop state, health, and metrics
 * - Start, stop, pause, and resume the loop
 * - Manage improvement proposals
 * - Access risk register and logs
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Check nomic loop health
 * const health = await client.nomic.getHealth();
 * console.log(`Status: ${health.status}, Phase: ${health.phase}`);
 *
 * // Start nomic loop
 * await client.nomic.start({ cycles: 3, auto_approve: false });
 *
 * // Get pending proposals
 * const { proposals } = await client.nomic.getProposals();
 *
 * // Approve a proposal
 * await client.nomic.approveProposal({ proposal_id: 'prop-123' });
 *
 * // Get risk register
 * const risks = await client.nomic.getRiskRegister({ limit: 50 });
 * ```
 */
export class NomicAPI {
  constructor(private client: NomicClientInterface) {}

  // ===========================================================================
  // State & Monitoring
  // ===========================================================================

  /**
   * Get current nomic loop state.
   */
  async getState(): Promise<NomicState> {
    return this.client.get('/api/v1/nomic/state');
  }

  /**
   * Get nomic loop health with stall detection.
   */
  async getHealth(): Promise<NomicHealth> {
    return this.client.get('/api/v1/nomic/health');
  }

  /**
   * Get nomic loop metrics summary.
   */
  async getMetrics(): Promise<NomicMetrics> {
    return this.client.get('/api/v1/nomic/metrics');
  }

  /**
   * Get recent nomic loop log lines.
   * @param lines - Number of log lines to return (default: 100, max: 1000)
   */
  async getLogs(lines: number = 100): Promise<NomicLogResponse> {
    return this.client.request('GET', '/api/v1/nomic/log', { params: { lines } });
  }

  /**
   * Get risk register entries.
   * @param options - Filter options
   */
  async getRiskRegister(options?: { limit?: number }): Promise<RiskRegisterResponse> {
    return this.client.request('GET', '/api/v1/nomic/risk-register', { params: options });
  }

  // ===========================================================================
  // Loop Control
  // ===========================================================================

  /**
   * Start the nomic loop.
   * @param body - Start configuration
   */
  async start(body?: StartLoopRequest): Promise<{ status: string; pid: number; target_cycles: number }> {
    return this.client.post('/api/v1/nomic/control/start', body);
  }

  /**
   * Stop the running nomic loop.
   * @param body - Stop configuration
   */
  async stop(body?: StopLoopRequest): Promise<{ status: string; pid: number }> {
    return this.client.post('/api/v1/nomic/control/stop', body);
  }

  /**
   * Pause the nomic loop at the current phase.
   */
  async pause(): Promise<{ status: string; cycle: number; phase: string }> {
    return this.client.post('/api/v1/nomic/control/pause');
  }

  /**
   * Resume a paused nomic loop.
   */
  async resume(): Promise<{ status: string; cycle: number; phase: string }> {
    return this.client.post('/api/v1/nomic/control/resume');
  }

  /**
   * Skip the current phase and move to the next.
   */
  async skipPhase(): Promise<{ status: string; previous_phase: string; next_phase: string; cycle: number }> {
    return this.client.post('/api/v1/nomic/control/skip-phase');
  }

  // ===========================================================================
  // Proposal Management
  // ===========================================================================

  /**
   * Get pending improvement proposals.
   */
  async getProposals(): Promise<ProposalsResponse> {
    return this.client.get('/api/v1/nomic/proposals');
  }

  /**
   * Approve a pending proposal.
   * @param body - Approval details
   */
  async approveProposal(body: ApproveProposalRequest): Promise<{ status: string; proposal_id: string }> {
    return this.client.post('/api/v1/nomic/proposals/approve', body);
  }

  /**
   * Reject a pending proposal.
   * @param body - Rejection details
   */
  async rejectProposal(body: RejectProposalRequest): Promise<{ status: string; proposal_id: string }> {
    return this.client.post('/api/v1/nomic/proposals/reject', body);
  }

  // ===========================================================================
  // Operational Modes
  // ===========================================================================

  /**
   * Get available operational modes (builtin + custom).
   */
  async getModes(): Promise<ModesResponse> {
    return this.client.get('/api/v1/modes');
  }

  // ===========================================================================
  // Gas Town
  // ===========================================================================

  /**
   * Get Gas Town witness patrol status.
   *
   * Witnesses monitor the nomic loop for violations and irregularities.
   */
  async getWitnessStatus(): Promise<WitnessStatus> {
    return this.client.get('/api/v1/nomic/witness/status');
  }

  /**
   * Get current Gas Town mayor information.
   *
   * The mayor is the elected leader who can enact policies and use emergency powers.
   */
  async getMayorCurrent(): Promise<MayorInfo> {
    return this.client.get('/api/v1/nomic/mayor/current');
  }

  // ===========================================================================
  // Convenience Aliases
  // ===========================================================================

  /**
   * Alias for getState().
   */
  async state(): Promise<NomicState> {
    return this.getState();
  }

  /**
   * Alias for getHealth().
   */
  async health(): Promise<NomicHealth> {
    return this.getHealth();
  }

  /**
   * Check if the nomic loop is currently running.
   */
  async isRunning(): Promise<boolean> {
    const state = await this.getState();
    return state.running && !state.paused;
  }

  /**
   * Get proposal by ID from the proposals list.
   *
   * @param proposalId - The proposal ID to find
   */
  async getProposal(proposalId: string): Promise<Proposal | null> {
    const { proposals } = await this.getProposals();
    return proposals.find(p => p.id === proposalId) || null;
  }
}
