/**
 * Metrics Namespace Tests
 *
 * Comprehensive tests for the metrics namespace API including:
 * - General metrics
 * - Health metrics
 * - Cache statistics
 * - System metrics
 * - Prometheus export
 * - Debate metrics
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { MetricsAPI } from '../metrics';

interface MockClient {
  get: Mock;
}

describe('MetricsAPI Namespace', () => {
  let api: MetricsAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      get: vi.fn(),
    };
    api = new MetricsAPI(mockClient as any);
  });

  // ===========================================================================
  // General Metrics
  // ===========================================================================

  describe('General Metrics', () => {
    it('should get application metrics', async () => {
      const mockMetrics = {
        requests: {
          total: 1500000,
          per_second: 250,
          by_status: {
            '200': 1450000,
            '400': 30000,
            '500': 20000,
          },
        },
        latency: {
          p50_ms: 45,
          p95_ms: 150,
          p99_ms: 350,
          avg_ms: 65,
        },
        debates: {
          active: 25,
          completed_today: 150,
          avg_duration_ms: 180000,
        },
        agents: {
          total: 15,
          available: 12,
          busy: 3,
        },
      };
      mockClient.get.mockResolvedValue(mockMetrics);

      const result = await api.get();

      expect(mockClient.get).toHaveBeenCalledWith('/api/metrics');
      expect(result.requests.total).toBe(1500000);
      expect(result.latency.p99_ms).toBe(350);
      expect(result.debates.active).toBe(25);
    });
  });

  // ===========================================================================
  // Health Metrics
  // ===========================================================================

  describe('Health Metrics', () => {
    it('should get healthy status', async () => {
      const mockHealth = {
        status: 'healthy',
        checks: {
          database: { status: 'pass', latency_ms: 5 },
          redis: { status: 'pass', latency_ms: 2 },
          openai: { status: 'pass', latency_ms: 150 },
          anthropic: { status: 'pass', latency_ms: 120 },
        },
        uptime_seconds: 86400,
        version: '2.5.0',
      };
      mockClient.get.mockResolvedValue(mockHealth);

      const result = await api.getHealth();

      expect(mockClient.get).toHaveBeenCalledWith('/api/metrics/health');
      expect(result.status).toBe('healthy');
      expect(result.uptime_seconds).toBe(86400);
      expect(result.checks.database.status).toBe('pass');
    });

    it('should get degraded status', async () => {
      const mockHealth = {
        status: 'degraded',
        checks: {
          database: { status: 'pass', latency_ms: 5 },
          redis: { status: 'warn', message: 'High memory usage', latency_ms: 15 },
          openai: { status: 'pass', latency_ms: 200 },
        },
        uptime_seconds: 172800,
        version: '2.5.0',
      };
      mockClient.get.mockResolvedValue(mockHealth);

      const result = await api.getHealth();

      expect(result.status).toBe('degraded');
      expect(result.checks.redis.status).toBe('warn');
      expect(result.checks.redis.message).toBe('High memory usage');
    });

    it('should get unhealthy status', async () => {
      const mockHealth = {
        status: 'unhealthy',
        checks: {
          database: { status: 'fail', message: 'Connection timeout' },
          redis: { status: 'pass', latency_ms: 2 },
        },
        uptime_seconds: 3600,
        version: '2.5.0',
      };
      mockClient.get.mockResolvedValue(mockHealth);

      const result = await api.getHealth();

      expect(result.status).toBe('unhealthy');
      expect(result.checks.database.status).toBe('fail');
    });
  });

  // ===========================================================================
  // Cache Metrics
  // ===========================================================================

  describe('Cache Metrics', () => {
    it('should get cache statistics', async () => {
      const mockCache = {
        hits: 950000,
        misses: 50000,
        hit_rate: 0.95,
        size_bytes: 524288000,
        max_size_bytes: 1073741824,
        evictions: 15000,
        entries: 250000,
      };
      mockClient.get.mockResolvedValue(mockCache);

      const result = await api.getCache();

      expect(mockClient.get).toHaveBeenCalledWith('/api/metrics/cache');
      expect(result.hit_rate).toBe(0.95);
      expect(result.entries).toBe(250000);
      expect(result.evictions).toBe(15000);
    });

    it('should show low hit rate', async () => {
      const mockCache = {
        hits: 30000,
        misses: 70000,
        hit_rate: 0.3,
        size_bytes: 104857600,
        max_size_bytes: 1073741824,
        evictions: 5000,
        entries: 50000,
      };
      mockClient.get.mockResolvedValue(mockCache);

      const result = await api.getCache();

      expect(result.hit_rate).toBe(0.3);
    });
  });

  // ===========================================================================
  // System Metrics
  // ===========================================================================

  describe('System Metrics', () => {
    it('should get system metrics', async () => {
      const mockSystem = {
        cpu: {
          usage_percent: 45.5,
          cores: 8,
          load_average: [2.1, 2.5, 2.3],
        },
        memory: {
          total_bytes: 17179869184,
          used_bytes: 12884901888,
          free_bytes: 4294967296,
          usage_percent: 75.0,
        },
        disk: {
          total_bytes: 1099511627776,
          used_bytes: 549755813888,
          free_bytes: 549755813888,
          usage_percent: 50.0,
        },
        network: {
          bytes_sent: 1073741824,
          bytes_received: 5368709120,
          connections: 250,
        },
      };
      mockClient.get.mockResolvedValue(mockSystem);

      const result = await api.getSystem();

      expect(mockClient.get).toHaveBeenCalledWith('/api/metrics/system');
      expect(result.cpu.usage_percent).toBe(45.5);
      expect(result.cpu.cores).toBe(8);
      expect(result.memory.usage_percent).toBe(75.0);
      expect(result.disk.usage_percent).toBe(50.0);
      expect(result.network.connections).toBe(250);
    });

    it('should show high resource usage', async () => {
      const mockSystem = {
        cpu: {
          usage_percent: 92.5,
          cores: 4,
          load_average: [4.5, 4.2, 4.0],
        },
        memory: {
          total_bytes: 8589934592,
          used_bytes: 8053063680,
          free_bytes: 536870912,
          usage_percent: 93.75,
        },
        disk: {
          total_bytes: 107374182400,
          used_bytes: 96636764160,
          free_bytes: 10737418240,
          usage_percent: 90.0,
        },
        network: {
          bytes_sent: 5368709120,
          bytes_received: 10737418240,
          connections: 1000,
        },
      };
      mockClient.get.mockResolvedValue(mockSystem);

      const result = await api.getSystem();

      expect(result.cpu.usage_percent).toBeGreaterThan(90);
      expect(result.memory.usage_percent).toBeGreaterThan(90);
      expect(result.disk.usage_percent).toBe(90.0);
    });
  });

  // ===========================================================================
  // Prometheus Export
  // ===========================================================================

  describe('Prometheus Export', () => {
    it('should get Prometheus metrics', async () => {
      const mockPrometheus = `# HELP aragora_requests_total Total HTTP requests
# TYPE aragora_requests_total counter
aragora_requests_total{method="GET",status="200"} 1500000
aragora_requests_total{method="POST",status="200"} 500000
aragora_requests_total{method="GET",status="404"} 5000

# HELP aragora_latency_seconds Request latency histogram
# TYPE aragora_latency_seconds histogram
aragora_latency_seconds_bucket{le="0.05"} 900000
aragora_latency_seconds_bucket{le="0.1"} 1400000
aragora_latency_seconds_bucket{le="0.5"} 1990000
aragora_latency_seconds_bucket{le="+Inf"} 2000000

# HELP aragora_debates_active Current active debates
# TYPE aragora_debates_active gauge
aragora_debates_active 25
`;
      mockClient.get.mockResolvedValue(mockPrometheus);

      const result = await api.getPrometheus();

      expect(mockClient.get).toHaveBeenCalledWith('/metrics');
      expect(result).toContain('aragora_requests_total');
      expect(result).toContain('aragora_debates_active 25');
    });
  });

  // ===========================================================================
  // Debate Metrics
  // ===========================================================================

  describe('Debate Metrics', () => {
    it('should get debate metrics', async () => {
      const mockDebateMetrics = {
        active_debates: 25,
        completed_debates: 1500,
        avg_rounds: 4.5,
        avg_duration_ms: 180000,
        consensus_rate: 0.85,
        throughput_per_hour: 62.5,
      };
      mockClient.get.mockResolvedValue(mockDebateMetrics);

      const result = await api.getDebates();

      expect(mockClient.get).toHaveBeenCalledWith('/api/metrics/debate');
      expect(result.active_debates).toBe(25);
      expect(result.consensus_rate).toBe(0.85);
      expect(result.throughput_per_hour).toBe(62.5);
    });

    it('should show low consensus rate', async () => {
      const mockDebateMetrics = {
        active_debates: 10,
        completed_debates: 500,
        avg_rounds: 6.0,
        avg_duration_ms: 300000,
        consensus_rate: 0.45,
        throughput_per_hour: 20.8,
      };
      mockClient.get.mockResolvedValue(mockDebateMetrics);

      const result = await api.getDebates();

      expect(result.consensus_rate).toBe(0.45);
      expect(result.avg_rounds).toBe(6.0);
    });
  });
});
