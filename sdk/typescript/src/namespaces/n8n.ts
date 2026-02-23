/**
 * n8n Integration Namespace API
 *
 * Provides methods for n8n workflow automation integration:
 * - Credential management
 * - Node definitions
 * - Trigger/webhook management
 */

interface N8nClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
}

export class N8nAPI {
  constructor(private client: N8nClientInterface) {}

  // Credentials
  async listCredentials(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/n8n/credentials');
  }

  async createCredential(data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/n8n/credentials', { body: data });
  }

  async deleteCredential(data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('DELETE', '/api/v1/n8n/credentials', { body: data });
  }

  // Node
  async getNode(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/n8n/node');
  }

  async createNode(data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/n8n/node', { body: data });
  }

  async deleteNode(data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('DELETE', '/api/v1/n8n/node', { body: data });
  }

  // Trigger
  async getTrigger(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/n8n/trigger');
  }

  async createTrigger(data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/n8n/trigger', { body: data });
  }

  async deleteTrigger(data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('DELETE', '/api/v1/n8n/trigger', { body: data });
  }
}
