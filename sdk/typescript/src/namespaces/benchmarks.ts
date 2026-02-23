/**
 * Benchmarks Namespace API
 *
 * Provides methods for benchmark management and comparison.
 */

interface BenchmarksClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
}

export class BenchmarksAPI {
  constructor(private client: BenchmarksClientInterface) {}

  /** List available benchmarks. */
  async list(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/benchmarks', { params });
  }

  /** Compare benchmark results. */
  async compare(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/benchmarks/compare', { params });
  }
}
