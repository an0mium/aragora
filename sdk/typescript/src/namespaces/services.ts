/**
 * Services Namespace API
 *
 * Provides endpoints for service discovery and detail lookup.
 */

import type { AragoraClient } from '../client';

/** Service health status */
export type ServiceHealthStatus = 'healthy' | 'degraded' | 'unhealthy' | 'unknown';

/** Registered service */
export interface Service {
  id: string;
  name: string;
  version: string;
  status: ServiceHealthStatus;
  endpoint: string;
  tags: string[];
  metadata: Record<string, unknown>;
  last_heartbeat: string;
  registered_at: string;
}

/**
 * Services namespace for service discovery.
 *
 * @example
 * ```typescript
 * const services = await client.services.list();
 * const healthy = services.filter(s => s.status === 'healthy');
 * ```
 */
export class ServicesNamespace {
  constructor(private client: AragoraClient) {}

  /** List all registered services. */
  async list(options?: { status?: string; tag?: string }): Promise<Service[]> {
    const response = await this.client.request<{ services: Service[] }>(
      'GET',
      '/api/v1/services',
      { params: options }
    );
    return response.services;
  }

  /** Get a service by ID. */
  async get(serviceId: string): Promise<Service> {
    return this.client.request<Service>(
      'GET',
      `/api/v1/services/${encodeURIComponent(serviceId)}`
    );
  }
}
