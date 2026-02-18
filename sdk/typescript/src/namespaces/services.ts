/**
 * Services Namespace API
 *
 * Provides endpoints for service discovery including
 * service registration, health checks, and dependency mapping.
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

/** Service dependency */
export interface ServiceDependency {
  service_id: string;
  depends_on: string;
  type: 'required' | 'optional';
  health: ServiceHealthStatus;
}

/** Service registration request */
export interface RegisterServiceRequest {
  name: string;
  version: string;
  endpoint: string;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

/**
 * Services namespace for service discovery and registration.
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

  /** Register a new service. */
  async register(request: RegisterServiceRequest): Promise<Service> {
    return this.client.request<Service>('POST', '/api/v1/services', {
      body: request,
    });
  }

  /** Deregister a service. */
  async deregister(serviceId: string): Promise<{ success: boolean }> {
    return this.client.request<{ success: boolean }>(
      'DELETE',
      `/api/v1/services/${encodeURIComponent(serviceId)}`
    );
  }

  /** Get service dependencies. */
  async getDependencies(serviceId: string): Promise<ServiceDependency[]> {
    const response = await this.client.request<{ dependencies: ServiceDependency[] }>(
      'GET',
      `/api/v1/services/${encodeURIComponent(serviceId)}/dependencies`
    );
    return response.dependencies;
  }
}
