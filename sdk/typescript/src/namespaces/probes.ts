/**
 * Probes Namespace API
 *
 * Provides capability probing and probe report management.
 */

export interface CapabilityProbeRequest {
  agent: string;
  task: string;
  config?: Record<string, unknown>;
}

export interface CapabilityProbeResponse {
  probe_id?: string;
  agent: string;
  result?: Record<string, unknown>;
  score?: number;
  capabilities?: Record<string, unknown>;
  timestamp?: string;
}

export interface ProbeReport {
  id: string;
  agent: string;
  task: string;
  score: number;
  capabilities: Record<string, unknown>;
  weaknesses?: string[];
  created_at: string;
  metadata?: Record<string, unknown>;
}

interface ProbesClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; body?: unknown }
  ): Promise<T>;
}

export class ProbesAPI {
  constructor(private client: ProbesClientInterface) {}

  /**
   * Run a capability probe on an agent.
   */
  async probeCapability(body: CapabilityProbeRequest): Promise<CapabilityProbeResponse> {
    return this.client.request('POST', '/api/v1/probes/capability', { body });
  }

  /**
   * List all stored probe reports.
   */
  async listReports(params?: {
    limit?: number;
    offset?: number;
    agent?: string;
  }): Promise<{ reports: ProbeReport[]; total?: number }> {
    return this.client.request('GET', '/api/v1/probes/reports', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get a specific probe report by ID.
   */
  async getReport(reportId: string): Promise<ProbeReport> {
    return this.client.request('GET', `/api/v1/probes/reports/${reportId}`);
  }
}
