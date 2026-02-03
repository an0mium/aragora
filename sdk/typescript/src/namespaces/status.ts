/**
 * Status Namespace API
 *
 * Provides endpoints for the platform status page including
 * service health, incident history, and maintenance windows.
 */

import type { AragoraClient } from '../client';

/** Overall platform status */
export type PlatformStatus = 'operational' | 'degraded' | 'partial_outage' | 'major_outage';

/** Service component status */
export interface ServiceComponent {
  id: string;
  name: string;
  status: PlatformStatus;
  description?: string;
  updated_at: string;
}

/** Status page incident */
export interface StatusIncident {
  id: string;
  title: string;
  status: 'investigating' | 'identified' | 'monitoring' | 'resolved';
  impact: PlatformStatus;
  message: string;
  components: string[];
  created_at: string;
  resolved_at?: string;
}

/** Maintenance window */
export interface MaintenanceWindow {
  id: string;
  title: string;
  description: string;
  scheduled_start: string;
  scheduled_end: string;
  components: string[];
  status: 'scheduled' | 'in_progress' | 'completed';
}

/** Status page summary */
export interface StatusSummary {
  status: PlatformStatus;
  components: ServiceComponent[];
  active_incidents: StatusIncident[];
  upcoming_maintenance: MaintenanceWindow[];
  updated_at: string;
}

/**
 * Status namespace for platform health monitoring.
 *
 * @example
 * ```typescript
 * const summary = await client.status.getSummary();
 * console.log(`Platform is ${summary.status}`);
 * ```
 */
export class StatusNamespace {
  constructor(private client: AragoraClient) {}

  /** Get overall status summary. */
  async getSummary(): Promise<StatusSummary> {
    return this.client.request<StatusSummary>('GET', '/api/v1/status');
  }

  /** List service components and their status. */
  async listComponents(): Promise<ServiceComponent[]> {
    const response = await this.client.request<{ components: ServiceComponent[] }>(
      'GET',
      '/api/v1/status/components'
    );
    return response.components;
  }

  /** List status incidents. */
  async listIncidents(options?: {
    limit?: number;
    status?: string;
  }): Promise<StatusIncident[]> {
    const response = await this.client.request<{ incidents: StatusIncident[] }>(
      'GET',
      '/api/v1/status/incidents',
      { params: options }
    );
    return response.incidents;
  }

  /** List upcoming and active maintenance windows. */
  async listMaintenance(): Promise<MaintenanceWindow[]> {
    const response = await this.client.request<{ maintenance: MaintenanceWindow[] }>(
      'GET',
      '/api/v1/status/maintenance'
    );
    return response.maintenance;
  }
}
