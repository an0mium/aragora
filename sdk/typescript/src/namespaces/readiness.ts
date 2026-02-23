/**
 * Readiness Namespace API
 *
 * Provides methods for system readiness and health checks.
 */

interface ReadinessClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
}

export class ReadinessAPI {
  constructor(private client: ReadinessClientInterface) {}

  /** Check system readiness status. */
  async check(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/readiness');
  }

  /** Get health check status (lightweight liveness probe). */
  async health(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/health');
  }

  /** Get detailed health with subsystem diagnostics. */
  async detailed(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/health/detailed');
  }
}
