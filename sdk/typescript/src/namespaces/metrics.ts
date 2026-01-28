/**
 * Metrics Namespace API
 *
 * Provides methods for system and application metrics:
 * - System health and performance
 * - Cache statistics
 * - Application metrics
 * - Prometheus metrics export
 */

/**
 * Health check result
 */
export interface HealthMetrics {
  status: 'healthy' | 'degraded' | 'unhealthy';
  checks: Record<string, {
    status: 'pass' | 'fail' | 'warn';
    message?: string;
    latency_ms?: number;
  }>;
  uptime_seconds: number;
  version: string;
}

/**
 * Cache statistics
 */
export interface CacheMetrics {
  hits: number;
  misses: number;
  hit_rate: number;
  size_bytes: number;
  max_size_bytes: number;
  evictions: number;
  entries: number;
}

/**
 * System metrics
 */
export interface SystemMetrics {
  cpu: {
    usage_percent: number;
    cores: number;
    load_average: number[];
  };
  memory: {
    total_bytes: number;
    used_bytes: number;
    free_bytes: number;
    usage_percent: number;
  };
  disk: {
    total_bytes: number;
    used_bytes: number;
    free_bytes: number;
    usage_percent: number;
  };
  network: {
    bytes_sent: number;
    bytes_received: number;
    connections: number;
  };
}

/**
 * Application metrics
 */
export interface ApplicationMetrics {
  requests: {
    total: number;
    per_second: number;
    by_status: Record<string, number>;
  };
  latency: {
    p50_ms: number;
    p95_ms: number;
    p99_ms: number;
    avg_ms: number;
  };
  debates: {
    active: number;
    completed_today: number;
    avg_duration_ms: number;
  };
  agents: {
    total: number;
    available: number;
    busy: number;
  };
}

/**
 * Debate metrics
 */
export interface DebateMetrics {
  active_debates: number;
  completed_debates: number;
  avg_rounds: number;
  avg_duration_ms: number;
  consensus_rate: number;
  throughput_per_hour: number;
}

/**
 * Interface for the internal client used by MetricsAPI.
 */
interface MetricsClientInterface {
  get<T>(path: string): Promise<T>;
}

/**
 * Metrics API namespace.
 *
 * Provides methods for accessing system and application metrics.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Check system health
 * const health = await client.metrics.getHealth();
 * console.log(`Status: ${health.status}, Uptime: ${health.uptime_seconds}s`);
 *
 * // Get system metrics
 * const system = await client.metrics.getSystem();
 * console.log(`CPU: ${system.cpu.usage_percent}%, Memory: ${system.memory.usage_percent}%`);
 *
 * // Get cache stats
 * const cache = await client.metrics.getCache();
 * console.log(`Cache hit rate: ${cache.hit_rate * 100}%`);
 *
 * // Get Prometheus metrics
 * const prometheus = await client.metrics.getPrometheus();
 * ```
 */
export class MetricsAPI {
  constructor(private client: MetricsClientInterface) {}

  // ===========================================================================
  // General Metrics
  // ===========================================================================

  /**
   * Get general application metrics.
   */
  async get(): Promise<ApplicationMetrics> {
    return this.client.get('/api/metrics');
  }

  /**
   * Get health metrics.
   */
  async getHealth(): Promise<HealthMetrics> {
    return this.client.get('/api/metrics/health');
  }

  /**
   * Get cache statistics.
   */
  async getCache(): Promise<CacheMetrics> {
    return this.client.get('/api/metrics/cache');
  }

  /**
   * Get system metrics.
   */
  async getSystem(): Promise<SystemMetrics> {
    return this.client.get('/api/metrics/system');
  }

  /**
   * Get Prometheus-format metrics.
   */
  async getPrometheus(): Promise<string> {
    return this.client.get('/metrics');
  }

  // ===========================================================================
  // Specialized Metrics
  // ===========================================================================

  /**
   * Get debate metrics.
   */
  async getDebates(): Promise<DebateMetrics> {
    return this.client.get('/api/metrics/debate');
  }
}
