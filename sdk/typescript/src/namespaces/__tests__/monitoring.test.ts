/**
 * Monitoring Namespace Tests
 *
 * Comprehensive tests for the monitoring namespace API including:
 * - Recording metrics
 * - Getting trends
 * - Anomaly detection
 * - Baseline statistics
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { MonitoringAPI } from '../monitoring';

interface MockClient {
  request: Mock;
}

describe('MonitoringAPI Namespace', () => {
  let api: MonitoringAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new MonitoringAPI(mockClient as any);
  });

  // ===========================================================================
  // Recording Metrics
  // ===========================================================================

  describe('Recording Metrics', () => {
    it('should record metric without anomaly', async () => {
      const mockResponse = {
        success: true,
        metric_name: 'api_latency_ms',
        value: 150,
        anomaly_detected: false,
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.record({
        metric_name: 'api_latency_ms',
        value: 150,
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/autonomous/monitoring/record', {
        body: { metric_name: 'api_latency_ms', value: 150 },
      });
      expect(result.success).toBe(true);
      expect(result.anomaly_detected).toBe(false);
    });

    it('should record metric with anomaly detected', async () => {
      const mockResponse = {
        success: true,
        metric_name: 'api_latency_ms',
        value: 5000,
        anomaly_detected: true,
        anomaly: {
          id: 'anom_123',
          severity: 'high',
          deviation: 3.5,
          expected_value: 200,
          description: 'Latency 25x higher than expected',
        },
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.record({
        metric_name: 'api_latency_ms',
        value: 5000,
      });

      expect(result.anomaly_detected).toBe(true);
      expect(result.anomaly?.severity).toBe('high');
      expect(result.anomaly?.deviation).toBe(3.5);
    });
  });

  // ===========================================================================
  // Getting Trends
  // ===========================================================================

  describe('Getting Trends', () => {
    it('should get all trends', async () => {
      const mockResponse = {
        success: true,
        trends: {
          api_latency_ms: {
            direction: 'stable',
            current_value: 150,
            change_percent: 2.5,
            data_points: 1000,
            confidence: 0.95,
          },
          error_rate: {
            direction: 'down',
            current_value: 0.01,
            change_percent: -15.0,
            data_points: 1000,
            confidence: 0.90,
          },
          request_count: {
            direction: 'up',
            current_value: 5000,
            change_percent: 25.0,
            data_points: 1000,
            confidence: 0.98,
          },
        },
        count: 3,
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.getAllTrends();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/autonomous/monitoring/trends');
      expect(result.count).toBe(3);
      expect(result.trends.api_latency_ms.direction).toBe('stable');
    });

    it('should get specific metric trend', async () => {
      const mockResponse = {
        success: true,
        trend: {
          metric_name: 'api_latency_ms',
          direction: 'up',
          current_value: 200,
          previous_value: 150,
          change_percent: 33.3,
          period_start: '2024-01-20T09:00:00Z',
          period_end: '2024-01-20T10:00:00Z',
          data_points: 60,
          confidence: 0.92,
        },
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.getTrend('api_latency_ms');

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/autonomous/monitoring/trends/api_latency_ms',
        { params: { period_seconds: undefined } }
      );
      expect(result.trend.direction).toBe('up');
      expect(result.trend.change_percent).toBe(33.3);
    });

    it('should get trend with custom period', async () => {
      const mockResponse = {
        success: true,
        trend: {
          metric_name: 'error_rate',
          direction: 'down',
          current_value: 0.005,
          previous_value: 0.02,
          change_percent: -75.0,
        },
      };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.getTrend('error_rate', { period_seconds: 86400 });

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/autonomous/monitoring/trends/error_rate',
        { params: { period_seconds: 86400 } }
      );
    });

    it('should encode metric name in URL', async () => {
      const mockResponse = { success: true, trend: {} };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.getTrend('my/custom/metric');

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/autonomous/monitoring/trends/my%2Fcustom%2Fmetric',
        { params: { period_seconds: undefined } }
      );
    });
  });

  // ===========================================================================
  // Anomaly Detection
  // ===========================================================================

  describe('Anomaly Detection', () => {
    it('should get anomalies with default hours', async () => {
      const mockResponse = {
        success: true,
        anomalies: [
          {
            id: 'anom_1',
            metric_name: 'api_latency_ms',
            value: 5000,
            expected_value: 200,
            deviation: 24.0,
            timestamp: '2024-01-20T09:30:00Z',
            severity: 'critical',
            description: 'Extreme latency spike',
          },
          {
            id: 'anom_2',
            metric_name: 'error_rate',
            value: 0.15,
            expected_value: 0.01,
            deviation: 14.0,
            timestamp: '2024-01-20T09:35:00Z',
            severity: 'high',
            description: 'Error rate significantly elevated',
          },
        ],
        count: 2,
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.getAnomalies();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/autonomous/monitoring/anomalies', {
        params: { hours: 24, metric_name: undefined },
      });
      expect(result.anomalies).toHaveLength(2);
      expect(result.anomalies[0].severity).toBe('critical');
    });

    it('should get anomalies with custom hours', async () => {
      const mockResponse = { success: true, anomalies: [], count: 0 };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.getAnomalies({ hours: 72 });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/autonomous/monitoring/anomalies', {
        params: { hours: 72, metric_name: undefined },
      });
    });

    it('should get anomalies for specific metric', async () => {
      const mockResponse = {
        success: true,
        anomalies: [
          {
            id: 'anom_3',
            metric_name: 'memory_usage_mb',
            value: 8000,
            expected_value: 2000,
            deviation: 3.0,
            timestamp: '2024-01-20T10:00:00Z',
            severity: 'medium',
            description: 'Memory usage above normal',
          },
        ],
        count: 1,
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.getAnomalies({ metric_name: 'memory_usage_mb' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/autonomous/monitoring/anomalies', {
        params: { hours: 24, metric_name: 'memory_usage_mb' },
      });
      expect(result.anomalies[0].metric_name).toBe('memory_usage_mb');
    });

    it('should get anomalies with both filters', async () => {
      const mockResponse = { success: true, anomalies: [], count: 0 };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.getAnomalies({ hours: 48, metric_name: 'cpu_percent' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/autonomous/monitoring/anomalies', {
        params: { hours: 48, metric_name: 'cpu_percent' },
      });
    });
  });

  // ===========================================================================
  // Baseline Statistics
  // ===========================================================================

  describe('Baseline Statistics', () => {
    it('should get baseline for metric', async () => {
      const mockResponse = {
        success: true,
        metric_name: 'api_latency_ms',
        stats: {
          mean: 150.5,
          stdev: 25.3,
          min: 50,
          max: 500,
          median: 145,
        },
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.getBaseline('api_latency_ms');

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/autonomous/monitoring/baseline/api_latency_ms'
      );
      expect(result.stats.mean).toBe(150.5);
      expect(result.stats.stdev).toBe(25.3);
    });

    it('should encode metric name in baseline URL', async () => {
      const mockResponse = { success: true, metric_name: 'my/metric', stats: {} };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.getBaseline('my/metric');

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/autonomous/monitoring/baseline/my%2Fmetric'
      );
    });
  });
});
