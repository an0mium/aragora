/**
 * SLO (Service Level Objective) Namespace API
 *
 * Provides endpoints for monitoring SLO compliance, error budgets, and violations.
 */

import type { AragoraClient } from '../client';

/**
 * SLO target configuration
 */
export interface SLOTarget {
  name: string;
  target_percent: number;
  window_days: number;
  description?: string;
}

/**
 * Current SLO status
 */
export interface SLOStatus {
  name: string;
  current_percent: number;
  target_percent: number;
  is_meeting: boolean;
  window_start: string;
  window_end: string;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
}

/**
 * Error budget information
 */
export interface ErrorBudget {
  slo_name: string;
  budget_percent: number;
  consumed_percent: number;
  remaining_percent: number;
  is_exhausted: boolean;
  burn_rate: number;
  projected_exhaustion?: string;
  window_days: number;
}

/**
 * SLO violation record
 */
export interface SLOViolation {
  slo_name: string;
  timestamp: string;
  actual_percent: number;
  target_percent: number;
  duration_seconds: number;
  severity: 'warning' | 'critical';
  resolved: boolean;
  resolved_at?: string;
}

/**
 * Overall SLO compliance status
 */
export interface OverallSLOStatus {
  status: 'healthy' | 'degraded' | 'critical';
  timestamp: string;
  slos: Record<string, SLOStatus>;
  alerts: SLOAlert[];
  summary: {
    total: number;
    meeting: number;
    not_meeting: number;
    compliance_percent: number;
  };
}

/**
 * SLO alert information
 */
export interface SLOAlert {
  slo_name: string;
  severity: 'warning' | 'critical';
  message: string;
  triggered_at: string;
  acknowledged: boolean;
}

/**
 * SLO namespace for service level objective monitoring.
 *
 * @example
 * ```typescript
 * // Check overall SLO status
 * const status = await client.slo.getStatus();
 * console.log(`SLO compliance: ${status.summary.compliance_percent}%`);
 *
 * // Get error budget for availability
 * const budget = await client.slo.getErrorBudget('availability');
 * console.log(`Error budget remaining: ${budget.remaining_percent}%`);
 *
 * // List recent violations
 * const violations = await client.slo.getViolations({ limit: 10 });
 * violations.forEach(v => console.log(`${v.slo_name}: ${v.severity}`));
 * ```
 */
export class SLONamespace {
  constructor(private client: AragoraClient) {}

  /**
   * Get overall SLO compliance status.
   *
   * Returns status of all SLOs including compliance percentages and alerts.
   */
  async getStatus(): Promise<OverallSLOStatus> {
    return this.client.request<OverallSLOStatus>('GET', '/api/v1/slos/status');
  }

  /**
   * Get individual SLO details.
   *
   * @param sloName - Name of the SLO (e.g., 'availability', 'latency')
   */
  async getSLO(sloName: string): Promise<SLOStatus> {
    return this.client.request<SLOStatus>('GET', `/api/v1/slos/${sloName}`);
  }

  /**
   * Get availability SLO status.
   */
  async getAvailability(): Promise<SLOStatus> {
    return this.getSLO('availability');
  }

  /**
   * Get latency SLO status.
   */
  async getLatency(): Promise<SLOStatus> {
    return this.getSLO('latency');
  }

  /**
   * Get debate success SLO status.
   */
  async getDebateSuccess(): Promise<SLOStatus> {
    return this.getSLO('debate-success');
  }

  /**
   * Get error budget information.
   *
   * @param sloName - Optional SLO name to get specific budget
   */
  async getErrorBudget(sloName?: string): Promise<ErrorBudget | ErrorBudget[]> {
    const path = sloName
      ? `/api/v1/slos/error-budget?slo=${sloName}`
      : '/api/v1/slos/error-budget';
    return this.client.request<ErrorBudget | ErrorBudget[]>('GET', path);
  }

  /**
   * Get SLO violations.
   *
   * @param options.limit - Maximum violations to return (default: 50)
   * @param options.sloName - Filter by specific SLO
   * @param options.severity - Filter by severity
   * @param options.unresolved - Only show unresolved violations
   */
  async getViolations(options?: {
    limit?: number;
    sloName?: string;
    severity?: 'warning' | 'critical';
    unresolved?: boolean;
  }): Promise<SLOViolation[]> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', options.limit.toString());
    if (options?.sloName) params.set('slo', options.sloName);
    if (options?.severity) params.set('severity', options.severity);
    if (options?.unresolved) params.set('unresolved', 'true');

    const query = params.toString();
    const path = query ? `/api/v1/slos/violations?${query}` : '/api/v1/slos/violations';
    return this.client.request<SLOViolation[]>('GET', path);
  }

  /**
   * Get configured SLO targets.
   */
  async getTargets(): Promise<SLOTarget[]> {
    return this.client.request<SLOTarget[]>('GET', '/api/v1/slos/targets');
  }

  /**
   * Check if all SLOs are meeting their targets.
   */
  async isCompliant(): Promise<boolean> {
    const status = await this.getStatus();
    return status.status === 'healthy';
  }

  /**
   * Get current alerts for SLO violations.
   */
  async getAlerts(): Promise<SLOAlert[]> {
    const status = await this.getStatus();
    return status.alerts;
  }

  /**
   * Get debate SLO health for operational windows.
   *
   * @param options.window - Window to evaluate ("1h", "24h", "7d")
   * @param options.allWindows - Include all windows when true
   */
  async getDebateHealth(options?: {
    window?: '1h' | '24h' | '7d';
    allWindows?: boolean;
  }): Promise<Record<string, unknown>> {
    const params: Record<string, string> = {};
    if (options?.window) params.window = options.window;
    if (options?.allWindows !== undefined) {
      params.all_windows = String(options.allWindows);
    }

    return this.client.request<Record<string, unknown>>(
      'GET',
      '/api/health/slos',
      Object.keys(params).length > 0 ? { params } : {}
    );
  }

  /**
   * Get SLO enforcer error budget information.
   */
  async getEnforcerBudget(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/slo/budget') as Promise<Record<string, unknown>>;
  }

  /**
   * Get SLO operational status.
   */
  async getSloStatus(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/slo/status') as Promise<Record<string, unknown>>;
  }

}
