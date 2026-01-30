/**
 * Disaster Recovery Namespace API
 *
 * Provides a namespaced interface for disaster recovery operations.
 * This wraps the flat client methods for DR drills, status, and validation.
 */

/**
 * DR status response.
 */
export interface DRStatus {
  ready: boolean;
  last_backup_at: string | null;
  last_drill_at: string | null;
  rpo_met: boolean;
  rto_met: boolean;
  issues: DRIssue[];
  overall_health: 'healthy' | 'degraded' | 'critical';
}

/**
 * DR issue detail.
 */
export interface DRIssue {
  severity: 'low' | 'medium' | 'high' | 'critical';
  component: string;
  message: string;
  remediation: string;
}

/**
 * Recovery objectives.
 */
export interface DRObjectives {
  rpo_minutes: number;
  rto_minutes: number;
  current_rpo_minutes: number;
  current_rto_minutes: number;
  rpo_compliant: boolean;
  rto_compliant: boolean;
  last_measured_at: string;
}

/**
 * DR drill request.
 */
export interface DRDrillRequest {
  type: 'tabletop' | 'simulation' | 'full';
  components?: string[];
  notify_team?: boolean;
  dry_run?: boolean;
}

/**
 * DR drill result.
 */
export interface DRDrillResult {
  drill_id: string;
  type: string;
  started_at: string;
  completed_at: string;
  duration_seconds: number;
  success: boolean;
  recovery_time_seconds: number;
  data_loss_seconds: number;
  steps: DRDrillStep[];
  recommendations: string[];
}

/**
 * DR drill step detail.
 */
export interface DRDrillStep {
  name: string;
  status: 'passed' | 'failed' | 'skipped';
  duration_seconds: number;
  message: string;
}

/**
 * DR validation request.
 */
export interface DRValidateRequest {
  check_backups?: boolean;
  check_replication?: boolean;
  check_failover?: boolean;
  check_dns?: boolean;
}

/**
 * DR validation result.
 */
export interface DRValidationResult {
  valid: boolean;
  checks: DRValidationCheck[];
  overall_score: number;
  last_validated_at: string;
}

/**
 * Individual validation check.
 */
export interface DRValidationCheck {
  name: string;
  passed: boolean;
  message: string;
  details: Record<string, unknown>;
}

/**
 * Interface for the internal client methods used by DisasterRecoveryAPI.
 */
interface DRClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: unknown }
  ): Promise<T>;
}

/**
 * Disaster Recovery API namespace.
 *
 * Provides methods for enterprise disaster recovery operations:
 * - Checking DR readiness status
 * - Running DR drills (tabletop, simulation, full)
 * - Getting and monitoring RPO/RTO objectives
 * - Validating DR configuration
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Check DR status
 * const status = await client.dr.getStatus();
 * console.log(`DR ready: ${status.ready}, Health: ${status.overall_health}`);
 *
 * // Run a tabletop drill
 * const drill = await client.dr.runDrill({
 *   type: 'tabletop',
 *   notify_team: true,
 * });
 *
 * // Check recovery objectives
 * const objectives = await client.dr.getObjectives();
 * console.log(`RPO: ${objectives.rpo_minutes}min, RTO: ${objectives.rto_minutes}min`);
 * ```
 */
export class DisasterRecoveryAPI {
  constructor(private client: DRClientInterface) {}

  /**
   * Get current DR readiness status.
   */
  async getStatus(): Promise<DRStatus> {
    return this.client.request<DRStatus>('GET', '/api/v2/dr/status');
  }

  /**
   * Run a DR drill.
   *
   * @param request - Drill configuration
   * @returns Drill results including recovery time and data loss metrics
   */
  async runDrill(request: DRDrillRequest): Promise<DRDrillResult> {
    return this.client.request<DRDrillResult>('POST', '/api/v2/dr/drill', {
      json: request,
    });
  }

  /**
   * Get current RPO/RTO objectives and compliance status.
   */
  async getObjectives(): Promise<DRObjectives> {
    return this.client.request<DRObjectives>('GET', '/api/v2/dr/objectives');
  }

  /**
   * Validate DR configuration.
   *
   * @param request - Optional validation scope
   * @returns Validation results with detailed check status
   */
  async validate(request?: DRValidateRequest): Promise<DRValidationResult> {
    return this.client.request<DRValidationResult>('POST', '/api/v2/dr/validate', {
      json: request || {},
    });
  }

  /**
   * Check if DR is ready for production.
   * Convenience method that checks status and validates configuration.
   */
  async isReady(): Promise<boolean> {
    const [status, validation] = await Promise.all([
      this.getStatus(),
      this.validate(),
    ]);
    return status.ready && validation.valid;
  }

  /**
   * Get a summary of DR health.
   */
  async getHealthSummary(): Promise<{
    ready: boolean;
    health: string;
    rpo_compliant: boolean;
    rto_compliant: boolean;
    issues_count: number;
  }> {
    const [status, objectives] = await Promise.all([
      this.getStatus(),
      this.getObjectives(),
    ]);
    return {
      ready: status.ready,
      health: status.overall_health,
      rpo_compliant: objectives.rpo_compliant,
      rto_compliant: objectives.rto_compliant,
      issues_count: status.issues.length,
    };
  }
}

export default DisasterRecoveryAPI;
