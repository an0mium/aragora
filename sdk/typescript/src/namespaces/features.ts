/**
 * Features Namespace API
 *
 * Provides access to feature flags and feature discovery.
 */

interface FeaturesClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Features namespace for feature flag management.
 *
 * @example
 * ```typescript
 * const flags = await client.features.list();
 * const isEnabled = await client.features.isEnabled('new-debate-ui');
 * ```
 */
export class FeaturesAPI {
  constructor(private client: FeaturesClientInterface) {}

  /** List all feature flags. */
  async list(): Promise<{ features: Record<string, unknown>[] }> {
    return this.client.request('GET', '/api/v1/features');
  }

  /** Check if a feature flag is enabled. */
  async isEnabled(featureName: string): Promise<{ enabled: boolean }> {
    return this.client.request('GET', `/api/v1/features/${featureName}`);
  }
}
