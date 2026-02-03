/**
 * Health Namespace Tests
 *
 * Comprehensive tests for the health namespace API including:
 * - Basic health checks
 * - Detailed health status
 * - Component health
 * - Nomic and metrics health
 * - Wait for healthy functionality
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { HealthNamespace } from '../health';

interface MockClient {
  request: Mock;
}

describe('HealthNamespace', () => {
  let api: HealthNamespace;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new HealthNamespace(mockClient as any);
  });

  // ===========================================================================
  // Basic Health Check
  // ===========================================================================

  describe('Basic Health Check', () => {
    it('should perform basic health check', async () => {
      const mockStatus = {
        status: 'healthy',
        timestamp: '2024-01-20T10:00:00Z',
        version: '1.5.0',
        uptime_seconds: 86400,
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.check();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/health');
      expect(result.status).toBe('healthy');
      expect(result.version).toBe('1.5.0');
    });

    it('should return degraded status', async () => {
      const mockStatus = {
        status: 'degraded',
        timestamp: '2024-01-20T10:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.check();

      expect(result.status).toBe('degraded');
    });

    it('should return unhealthy status', async () => {
      const mockStatus = {
        status: 'unhealthy',
        timestamp: '2024-01-20T10:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.check();

      expect(result.status).toBe('unhealthy');
    });
  });

  // ===========================================================================
  // Detailed Health Status
  // ===========================================================================

  describe('Detailed Health Status', () => {
    it('should get detailed health status', async () => {
      const mockDetailed = {
        status: 'healthy',
        timestamp: '2024-01-20T10:00:00Z',
        version: '1.5.0',
        uptime_seconds: 86400,
        checks: [
          { name: 'database', status: 'pass', latency_ms: 5 },
          { name: 'redis', status: 'pass', latency_ms: 2 },
          { name: 'external_apis', status: 'pass', latency_ms: 150 },
        ],
        metrics: {
          requests_per_minute: 450,
          average_latency_ms: 85,
          error_rate: 0.01,
          active_connections: 250,
          memory_usage_mb: 512,
          cpu_usage_percent: 35.5,
        },
      };
      mockClient.request.mockResolvedValue(mockDetailed);

      const result = await api.getDetailed();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/health/detailed');
      expect(result.checks).toHaveLength(3);
      expect(result.metrics.requests_per_minute).toBe(450);
    });

    it('should show failed checks in detailed status', async () => {
      const mockDetailed = {
        status: 'degraded',
        timestamp: '2024-01-20T10:00:00Z',
        version: '1.5.0',
        uptime_seconds: 86400,
        checks: [
          { name: 'database', status: 'pass', latency_ms: 5 },
          { name: 'redis', status: 'fail', message: 'Connection timeout', latency_ms: 5000 },
          { name: 'external_apis', status: 'warn', message: 'High latency', latency_ms: 2500 },
        ],
        metrics: {
          requests_per_minute: 200,
          average_latency_ms: 500,
          error_rate: 0.15,
          active_connections: 50,
          memory_usage_mb: 1024,
          cpu_usage_percent: 85.0,
        },
      };
      mockClient.request.mockResolvedValue(mockDetailed);

      const result = await api.getDetailed();

      expect(result.status).toBe('degraded');
      expect(result.checks.find((c) => c.name === 'redis')?.status).toBe('fail');
      expect(result.metrics.error_rate).toBe(0.15);
    });
  });

  // ===========================================================================
  // isHealthy Convenience Method
  // ===========================================================================

  describe('isHealthy', () => {
    it('should return true when healthy', async () => {
      mockClient.request.mockResolvedValue({ status: 'healthy' });

      const result = await api.isHealthy();

      expect(result).toBe(true);
    });

    it('should return false when degraded', async () => {
      mockClient.request.mockResolvedValue({ status: 'degraded' });

      const result = await api.isHealthy();

      expect(result).toBe(false);
    });

    it('should return false when unhealthy', async () => {
      mockClient.request.mockResolvedValue({ status: 'unhealthy' });

      const result = await api.isHealthy();

      expect(result).toBe(false);
    });

    it('should return false on error', async () => {
      mockClient.request.mockRejectedValue(new Error('Connection refused'));

      const result = await api.isHealthy();

      expect(result).toBe(false);
    });
  });

  // ===========================================================================
  // Nomic Health
  // ===========================================================================

  describe('Nomic Health', () => {
    it('should get nomic health status', async () => {
      const mockNomic = {
        status: 'healthy',
        timestamp: '2024-01-20T10:00:00Z',
        phase: 'implement',
        last_cycle: '2024-01-20T09:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockNomic);

      const result = await api.getNomicHealth();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/nomic/health');
      expect(result.phase).toBe('implement');
      expect(result.status).toBe('healthy');
    });

    it('should handle nomic unhealthy status', async () => {
      const mockNomic = {
        status: 'unhealthy',
        timestamp: '2024-01-20T10:00:00Z',
        phase: null,
        last_cycle: '2024-01-19T10:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockNomic);

      const result = await api.getNomicHealth();

      expect(result.status).toBe('unhealthy');
    });
  });

  // ===========================================================================
  // Metrics Health
  // ===========================================================================

  describe('Metrics Health', () => {
    it('should get metrics health status', async () => {
      const mockMetrics = {
        status: 'healthy',
        timestamp: '2024-01-20T10:00:00Z',
        metrics_available: true,
      };
      mockClient.request.mockResolvedValue(mockMetrics);

      const result = await api.getMetricsHealth();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/metrics/health');
      expect(result.metrics_available).toBe(true);
    });

    it('should handle metrics unavailable', async () => {
      const mockMetrics = {
        status: 'degraded',
        timestamp: '2024-01-20T10:00:00Z',
        metrics_available: false,
      };
      mockClient.request.mockResolvedValue(mockMetrics);

      const result = await api.getMetricsHealth();

      expect(result.metrics_available).toBe(false);
      expect(result.status).toBe('degraded');
    });
  });

  // ===========================================================================
  // Wait Until Healthy
  // ===========================================================================

  describe('waitUntilHealthy', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should return true immediately when healthy', async () => {
      mockClient.request.mockResolvedValue({ status: 'healthy' });

      const promise = api.waitUntilHealthy();
      await vi.runAllTimersAsync();
      const result = await promise;

      expect(result).toBe(true);
    });

    it('should poll until healthy', async () => {
      mockClient.request
        .mockResolvedValueOnce({ status: 'unhealthy' })
        .mockResolvedValueOnce({ status: 'unhealthy' })
        .mockResolvedValueOnce({ status: 'healthy' });

      const promise = api.waitUntilHealthy({ timeout: 10000, interval: 1000 });

      // First check - unhealthy
      await vi.advanceTimersByTimeAsync(0);
      // Wait for interval
      await vi.advanceTimersByTimeAsync(1000);
      // Second check - unhealthy
      await vi.advanceTimersByTimeAsync(1000);
      // Third check - healthy

      const result = await promise;

      expect(result).toBe(true);
      expect(mockClient.request).toHaveBeenCalledTimes(3);
    });

    it('should return false on timeout', async () => {
      mockClient.request.mockResolvedValue({ status: 'unhealthy' });

      const promise = api.waitUntilHealthy({ timeout: 3000, interval: 1000 });

      // Run all timers until timeout
      await vi.advanceTimersByTimeAsync(3500);

      const result = await promise;

      expect(result).toBe(false);
    });

    it('should use default timeout and interval', async () => {
      mockClient.request.mockResolvedValue({ status: 'healthy' });

      const promise = api.waitUntilHealthy();
      await vi.runAllTimersAsync();

      const result = await promise;

      expect(result).toBe(true);
    });
  });
});
