/**
 * Breakpoints Namespace API
 *
 * Provides methods for managing debug breakpoints in debate execution.
 */

interface BreakpointsClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
}

export class BreakpointsAPI {
  constructor(private client: BreakpointsClientInterface) {}

  /** Create a new breakpoint. */
  async create(data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/breakpoints', { body: data });
  }

  /** List all pending breakpoints. */
  async listPending(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/breakpoints/pending');
  }

  /** Get the status of a specific breakpoint. */
  async getStatus(breakpointId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/breakpoints/${breakpointId}/status`);
  }

  /** Resolve a breakpoint (continue or abort). */
  async resolve(breakpointId: string, data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', `/api/v1/breakpoints/${breakpointId}/resolve`, { body: data });
  }
}
