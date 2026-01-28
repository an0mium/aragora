/**
 * Monitoring Namespace API
 *
 * Provides a namespaced interface for monitoring and anomaly detection.
 * Tracks metrics, trends, and detects anomalies in system behavior.
 */

/**
 * Trend direction.
 */
export type TrendDirection = 'up' | 'down' | 'stable';

/**
 * Anomaly severity level.
 */
export type AnomalySeverity = 'low' | 'medium' | 'high' | 'critical';

/**
 * Metric trend data.
 */
export interface MetricTrend {
  metric_name: string;
  direction: TrendDirection;
  current_value: number;
  previous_value: number;
  change_percent: number;
  period_start: string;
  period_end: string;
  data_points: number;
  confidence: number;
}

/**
 * Trend summary (without period details).
 */
export interface TrendSummary {
  direction: TrendDirection;
  current_value: number;
  change_percent: number;
  data_points: number;
  confidence: number;
}

/**
 * Detected anomaly.
 */
export interface Anomaly {
  id: string;
  metric_name: string;
  value: number;
  expected_value: number;
  deviation: number;
  timestamp: string;
  severity: AnomalySeverity;
  description: string;
}

/**
 * Baseline statistics for a metric.
 */
export interface BaselineStats {
  mean: number;
  stdev: number;
  min: number;
  max: number;
  median: number;
}

/**
 * Response from recording a metric.
 */
export interface RecordMetricResponse {
  success: boolean;
  metric_name: string;
  value: number;
  anomaly_detected: boolean;
  anomaly?: {
    id: string;
    severity: AnomalySeverity;
    deviation: number;
    expected_value: number;
    description: string;
  };
}

/**
 * Response for getting a single trend.
 */
export interface GetTrendResponse {
  success: boolean;
  trend: MetricTrend;
}

/**
 * Response for getting all trends.
 */
export interface GetAllTrendsResponse {
  success: boolean;
  trends: Record<string, TrendSummary>;
  count: number;
}

/**
 * Response for getting anomalies.
 */
export interface GetAnomaliesResponse {
  success: boolean;
  anomalies: Anomaly[];
  count: number;
}

/**
 * Response for getting baseline stats.
 */
export interface GetBaselineResponse {
  success: boolean;
  metric_name: string;
  stats: BaselineStats;
}

/**
 * Options for recording a metric.
 */
export interface RecordMetricOptions {
  metric_name: string;
  value: number;
}

/**
 * Options for getting anomalies.
 */
export interface GetAnomaliesOptions {
  hours?: number;
  metric_name?: string;
}

/**
 * Interface for the internal client methods used by MonitoringAPI.
 */
interface MonitoringClientInterface {
  request<T>(method: string, path: string, options?: { params?: Record<string, unknown>; body?: unknown }): Promise<T>;
}

/**
 * Monitoring API namespace.
 *
 * Provides methods for monitoring metrics and detecting anomalies:
 * - Recording metric values
 * - Getting metric trends
 * - Detecting and listing anomalies
 * - Retrieving baseline statistics
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Record a metric
 * const result = await client.monitoring.record({
 *   metric_name: 'api_latency_ms',
 *   value: 150
 * });
 * if (result.anomaly_detected) {
 *   console.log('Anomaly detected:', result.anomaly);
 * }
 *
 * // Get trend for a metric
 * const { trend } = await client.monitoring.getTrend('api_latency_ms', {
 *   period_seconds: 3600 // Last hour
 * });
 *
 * // Get recent anomalies
 * const { anomalies } = await client.monitoring.getAnomalies({ hours: 24 });
 * ```
 */
export class MonitoringAPI {
  constructor(private client: MonitoringClientInterface) {}

  /**
   * Record a metric value.
   * Automatically checks for anomalies.
   */
  async record(options: RecordMetricOptions): Promise<RecordMetricResponse> {
    return this.client.request<RecordMetricResponse>('POST', '/api/v1/autonomous/monitoring/record', {
      body: {
        metric_name: options.metric_name,
        value: options.value,
      },
    });
  }

  /**
   * Get all metric trends.
   * Returns trends for all tracked metrics.
   */
  async getAllTrends(): Promise<GetAllTrendsResponse> {
    return this.client.request<GetAllTrendsResponse>('GET', '/api/v1/autonomous/monitoring/trends');
  }

  /**
   * Get trend for a specific metric.
   * @param metricName - Name of the metric
   * @param options - Optional period configuration
   */
  async getTrend(metricName: string, options?: { period_seconds?: number }): Promise<GetTrendResponse> {
    return this.client.request<GetTrendResponse>('GET', `/api/v1/autonomous/monitoring/trends/${encodeURIComponent(metricName)}`, {
      params: {
        period_seconds: options?.period_seconds,
      },
    });
  }

  /**
   * Get recent anomalies.
   * @param options - Filter by hours lookback and/or specific metric
   */
  async getAnomalies(options?: GetAnomaliesOptions): Promise<GetAnomaliesResponse> {
    return this.client.request<GetAnomaliesResponse>('GET', '/api/v1/autonomous/monitoring/anomalies', {
      params: {
        hours: options?.hours ?? 24,
        metric_name: options?.metric_name,
      },
    });
  }

  /**
   * Get baseline statistics for a metric.
   * Returns mean, stdev, min, max, and median.
   */
  async getBaseline(metricName: string): Promise<GetBaselineResponse> {
    return this.client.request<GetBaselineResponse>('GET', `/api/v1/autonomous/monitoring/baseline/${encodeURIComponent(metricName)}`);
  }
}
