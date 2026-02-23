/**
 * Users Namespace API
 *
 * Provides methods for user self-service operations.
 */

interface UsersClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
}

export class UsersAPI {
  constructor(private client: UsersClientInterface) {}

  /** Request account deletion. */
  async requestDeletion(data?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/users/self/deletion-request', { body: data });
  }

  /** Cancel a pending account deletion request. */
  async cancelDeletion(): Promise<Record<string, unknown>> {
    return this.client.request('DELETE', '/api/v1/users/self/deletion-request');
  }
}
