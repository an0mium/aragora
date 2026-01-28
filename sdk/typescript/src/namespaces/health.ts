/**
 * Health Namespace API
 *
 * Provides endpoints for system health checks and status monitoring.
 */

import type { AragoraClient } from '../client';

/**
 * Basic health status
 */
export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  version?: string;
  uptime_seconds?: number;
}

/**
 * Detailed health check result
 */
export interface DetailedHealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  version: string;
  uptime_seconds: number;
  checks: HealthCheck[];
  metrics: {
    requests_per_minute: number;
    average_latency_ms: number;
    error_rate: number;
    active_connections: number;
    memory_usage_mb: number;
    cpu_usage_percent: number;
  };
}

/**
 * Individual health check
 */
export interface HealthCheck {
  name: string;
  status: 'pass' | 'warn' | 'fail';
  message?: string;
  latency_ms?: number;
  last_check?: string;
  details?: Record<string, unknown>;
}

/**
 * Component health status
 */
export interface ComponentHealth {
  component: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  checks: HealthCheck[];
  dependencies: string[];
  last_updated: string;
}

/**
 * Health namespace for system monitoring.
 *
 * @example
 * ```typescript
 * // Quick health check
 * const health = await client.health.check();
 * console.log(`System is ${health.status}`);
 *
 * // Detailed health with all checks
 * const detailed = await client.health.getDetailed();
 * detailed.checks.forEach(check => {
 *   console.log(`${check.name}: ${check.status}`);
 * });
 * ```
 */
export class HealthNamespace {
  constructor(private client: AragoraClient) {}

  /**
   * Perform a basic health check.
   *
   * Returns a simple status indicating if the system is operational.
   */
  async check(): Promise<HealthStatus> {
    return this.client.request<HealthStatus>('GET', '/api/v1/health');
  }

  /**
   * Get detailed health status with all checks.
   *
   * Includes individual component checks, metrics, and dependency status.
   */
  async getDetailed(): Promise<DetailedHealthStatus> {
    return this.client.request<DetailedHealthStatus>('GET', '/api/v1/health/detailed');
  }

  /**
   * Check if the system is healthy (convenience method).
   *
   * @returns true if status is 'healthy', false otherwise
   */
  async isHealthy(): Promise<boolean> {
    try {
      const status = await this.check();
      return status.status === 'healthy';
    } catch {
      return false;
    }
  }

  /**
   * Get health status for the nomic loop system.
   */
  async getNomicHealth(): Promise<HealthStatus & { phase?: string; last_cycle?: string }> {
    return this.client.request<HealthStatus & { phase?: string; last_cycle?: string }>(
      'GET',
      '/api/v1/nomic/health'
    );
  }

  /**
   * Get metrics health status.
   */
  async getMetricsHealth(): Promise<HealthStatus & { metrics_available: boolean }> {
    return this.client.request<HealthStatus & { metrics_available: boolean }>(
      'GET',
      '/api/v1/metrics/health'
    );
  }

  /**
   * Wait for the system to become healthy.
   *
   * Polls the health endpoint until healthy or timeout.
   *
   * @param options.timeout - Maximum time to wait in ms (default: 30000)
   * @param options.interval - Polling interval in ms (default: 1000)
   * @returns true if healthy, false if timeout reached
   */
  async waitUntilHealthy(options?: { timeout?: number; interval?: number }): Promise<boolean> {
    const timeout = options?.timeout ?? 30000;
    const interval = options?.interval ?? 1000;
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      if (await this.isHealthy()) {
        return true;
      }
      await new Promise((resolve) => setTimeout(resolve, interval));
    }

    return false;
  }
}
