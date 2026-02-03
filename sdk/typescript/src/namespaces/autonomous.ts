/**
 * Autonomous Namespace API
 *
 * Provides endpoints for autonomous learning operations including
 * approval flows, scheduled triggers, alert management, and monitoring.
 */

import type { AragoraClient } from '../client';

/** Approval request for autonomous actions */
export interface ApprovalRequest {
  id: string;
  action_type: string;
  description: string;
  status: 'pending' | 'approved' | 'rejected';
  requested_at: string;
  resolved_at?: string;
  resolved_by?: string;
}

/** Scheduled trigger definition */
export interface Trigger {
  id: string;
  name: string;
  schedule: string;
  action: string;
  enabled: boolean;
  last_run?: string;
  next_run?: string;
}

/** Autonomous alert */
export interface AutonomousAlert {
  id: string;
  severity: 'info' | 'warning' | 'critical';
  message: string;
  source: string;
  acknowledged: boolean;
  created_at: string;
}

/** Monitoring metrics for autonomous operations */
export interface AutonomousMetrics {
  total_actions: number;
  approved_actions: number;
  rejected_actions: number;
  active_triggers: number;
  active_alerts: number;
}

/**
 * Autonomous namespace for self-learning and automated operations.
 *
 * @example
 * ```typescript
 * const approvals = await client.autonomous.listApprovals();
 * await client.autonomous.approveAction(approvals[0].id);
 * ```
 */
export class AutonomousNamespace {
  constructor(private client: AragoraClient) {}

  /** List pending approval requests. */
  async listApprovals(options?: {
    status?: string;
    limit?: number;
  }): Promise<ApprovalRequest[]> {
    const response = await this.client.request<{ approvals: ApprovalRequest[] }>(
      'GET',
      '/api/v1/autonomous/approvals',
      { params: options }
    );
    return response.approvals;
  }

  /** Approve an autonomous action. */
  async approveAction(approvalId: string): Promise<ApprovalRequest> {
    return this.client.request<ApprovalRequest>(
      'POST',
      `/api/v1/autonomous/approvals/${encodeURIComponent(approvalId)}/approve`
    );
  }

  /** Reject an autonomous action. */
  async rejectAction(approvalId: string, reason?: string): Promise<ApprovalRequest> {
    return this.client.request<ApprovalRequest>(
      'POST',
      `/api/v1/autonomous/approvals/${encodeURIComponent(approvalId)}/reject`,
      { body: { reason } }
    );
  }

  /** List active alerts. */
  async listAlerts(options?: { severity?: string }): Promise<AutonomousAlert[]> {
    const response = await this.client.request<{ alerts: AutonomousAlert[] }>(
      'GET',
      '/api/v1/autonomous/alerts',
      { params: options }
    );
    return response.alerts;
  }

  /** Get autonomous operation metrics. */
  async getMetrics(): Promise<AutonomousMetrics> {
    return this.client.request<AutonomousMetrics>(
      'GET',
      '/api/v1/autonomous/metrics'
    );
  }
}
