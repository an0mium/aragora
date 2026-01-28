/**
 * Probes Namespace API
 *
 * Provides capability probing endpoints.
 */

export interface CapabilityProbeRequest {
  agent: string;
  task: string;
  config?: Record<string, unknown>;
}

export interface CapabilityProbeResponse {
  result?: Record<string, unknown>;
  score?: number;
}

interface ProbesClientInterface {
  post<T>(path: string, body?: unknown): Promise<T>;
}

export class ProbesAPI {
  constructor(private client: ProbesClientInterface) {}

  async probeCapability(body: CapabilityProbeRequest): Promise<CapabilityProbeResponse> {
    return this.client.post('/api/v1/probes/capability', body);
  }
}
