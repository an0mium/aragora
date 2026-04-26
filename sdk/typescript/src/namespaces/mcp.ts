/**
 * MCP Namespace API
 *
 * Provides access to MCP tool discovery routes.
 */

interface MCPClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; body?: unknown; json?: Record<string, unknown> }
  ): Promise<T>;
}

export class MCPAPI {
  constructor(private client: MCPClientInterface) {}

  async listTools(category?: string): Promise<Record<string, unknown>> {
    const params = category ? { category } : undefined;
    return this.client.request('GET', '/api/v1/mcp/tools', { params });
  }

  async getTool(name: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/mcp/tools/${encodeURIComponent(name)}`);
  }
}
