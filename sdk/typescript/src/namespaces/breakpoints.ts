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
}
