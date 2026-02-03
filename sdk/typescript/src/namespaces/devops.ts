/**
 * DevOps Namespace API
 *
 * Provides endpoints for DevOps incident management (PagerDuty),
 * including incidents, on-call schedules, and service management.
 */

import type { AragoraClient } from '../client';

/** Incident from PagerDuty */
export interface Incident {
  id: string;
  title: string;
  status: 'triggered' | 'acknowledged' | 'resolved';
  urgency: 'high' | 'low';
  service_id: string;
  service_name?: string;
  incident_number?: number;
  description?: string;
  assignees?: string[];
  created_at: string;
  html_url?: string;
}

/** On-call schedule entry */
export interface OnCallEntry {
  schedule_id: string;
  schedule_name: string;
  user: { id: string; name: string; email: string } | null;
  start: string;
  end: string;
  escalation_level?: number;
}

/** PagerDuty service */
export interface DevOpsService {
  id: string;
  name: string;
  description?: string;
  status: string;
  html_url?: string;
  escalation_policy_id?: string;
  created_at?: string;
}

/** Create incident request */
export interface CreateIncidentRequest {
  title: string;
  service_id: string;
  urgency?: 'high' | 'low';
  description?: string;
  priority_id?: string;
}

/** DevOps connector status */
export interface DevOpsStatus {
  configured: boolean;
  api_key_set: boolean;
  email_set: boolean;
  webhook_secret_set: boolean;
  circuit_breaker: Record<string, unknown>;
}

/**
 * DevOps namespace for incident management and on-call.
 *
 * @example
 * ```typescript
 * const incidents = await client.devops.listIncidents();
 * await client.devops.acknowledgeIncident(incidents[0].id);
 * ```
 */
export class DevOpsNamespace {
  constructor(private client: AragoraClient) {}

  /** Get DevOps integration status. */
  async getStatus(): Promise<DevOpsStatus> {
    const response = await this.client.request<{ data: DevOpsStatus }>(
      'GET',
      '/api/v1/devops/status'
    );
    return response.data;
  }

  /** List incidents. */
  async listIncidents(options?: {
    status?: string;
    urgency?: string;
    limit?: number;
    offset?: number;
  }): Promise<Incident[]> {
    const response = await this.client.request<{ data: { incidents: Incident[] } }>(
      'GET',
      '/api/v1/incidents',
      { params: options }
    );
    return response.data.incidents;
  }

  /** Create an incident. */
  async createIncident(request: CreateIncidentRequest): Promise<Incident> {
    const response = await this.client.request<{ incident: Incident }>(
      'POST',
      '/api/v1/incidents',
      { body: request }
    );
    return response.incident;
  }

  /** Acknowledge an incident. */
  async acknowledgeIncident(incidentId: string): Promise<Incident> {
    const response = await this.client.request<{ data: { incident: Incident } }>(
      'POST',
      `/api/v1/incidents/${encodeURIComponent(incidentId)}/acknowledge`
    );
    return response.data.incident;
  }

  /** Resolve an incident. */
  async resolveIncident(
    incidentId: string,
    resolution?: string
  ): Promise<Incident> {
    const response = await this.client.request<{ data: { incident: Incident } }>(
      'POST',
      `/api/v1/incidents/${encodeURIComponent(incidentId)}/resolve`,
      { body: { resolution } }
    );
    return response.data.incident;
  }
}
