/**
 * Gateway Namespace Tests
 *
 * Comprehensive tests for the gateway namespace API including:
 * - Device management
 * - Route configuration
 * - Gateway status
 * - Connection management
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { GatewayNamespace } from '../gateway';

interface MockClient {
  request: Mock;
}

describe('GatewayNamespace', () => {
  let api: GatewayNamespace;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new GatewayNamespace(mockClient as any);
  });

  // ===========================================================================
  // Device Management
  // ===========================================================================

  describe('Device Management', () => {
    it('should list devices', async () => {
      const mockDevices = {
        devices: [
          {
            device_id: 'dev_1',
            name: 'Edge Server 1',
            type: 'edge',
            status: 'online',
            last_seen: '2024-01-20T10:00:00Z',
            ip_address: '192.168.1.100',
          },
          {
            device_id: 'dev_2',
            name: 'IoT Gateway',
            type: 'iot',
            status: 'online',
            last_seen: '2024-01-20T09:55:00Z',
          },
        ],
        total: 2,
      };
      mockClient.request.mockResolvedValue(mockDevices);

      const result = await api.listDevices();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/devices', {
        params: undefined,
      });
      expect(result.devices).toHaveLength(2);
    });

    it('should list devices with filters', async () => {
      const mockDevices = { devices: [], total: 0 };
      mockClient.request.mockResolvedValue(mockDevices);

      await api.listDevices({ status: 'online', type: 'edge' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/devices', {
        params: { status: 'online', type: 'edge' },
      });
    });

    it('should get device by ID', async () => {
      const mockDevice = {
        device_id: 'dev_1',
        name: 'Edge Server 1',
        type: 'edge',
        status: 'online',
        config: {
          max_connections: 100,
          timeout_ms: 5000,
        },
        metrics: {
          requests_per_minute: 150,
          error_rate: 0.01,
        },
      };
      mockClient.request.mockResolvedValue(mockDevice);

      const result = await api.getDevice('dev_1');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/devices/dev_1');
      expect(result.config.max_connections).toBe(100);
    });

    it('should register device', async () => {
      const mockDevice = {
        device_id: 'dev_new',
        name: 'New Edge Server',
        type: 'edge',
        status: 'pending',
        token: 'auth_token_123',
      };
      mockClient.request.mockResolvedValue(mockDevice);

      const result = await api.registerDevice({
        name: 'New Edge Server',
        type: 'edge',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/gateway/devices', {
        json: { name: 'New Edge Server', type: 'edge' },
      });
      expect(result.token).toBe('auth_token_123');
    });

    it('should update device', async () => {
      const mockDevice = {
        device_id: 'dev_1',
        name: 'Updated Server',
        status: 'online',
      };
      mockClient.request.mockResolvedValue(mockDevice);

      const result = await api.updateDevice('dev_1', { name: 'Updated Server' });

      expect(mockClient.request).toHaveBeenCalledWith('PATCH', '/api/v1/gateway/devices/dev_1', {
        json: { name: 'Updated Server' },
      });
      expect(result.name).toBe('Updated Server');
    });

    it('should delete device', async () => {
      mockClient.request.mockResolvedValue({ success: true });

      await api.deleteDevice('dev_1');

      expect(mockClient.request).toHaveBeenCalledWith('DELETE', '/api/v1/gateway/devices/dev_1');
    });
  });

  // ===========================================================================
  // Route Configuration
  // ===========================================================================

  describe('Route Configuration', () => {
    it('should list routes', async () => {
      const mockRoutes = {
        routes: [
          {
            route_id: 'route_1',
            path: '/api/debates/*',
            target: 'debate-service',
            methods: ['GET', 'POST'],
            enabled: true,
          },
          {
            route_id: 'route_2',
            path: '/api/agents/*',
            target: 'agent-service',
            methods: ['GET'],
            enabled: true,
          },
        ],
      };
      mockClient.request.mockResolvedValue(mockRoutes);

      const result = await api.listRoutes();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/routes');
      expect(result.routes).toHaveLength(2);
    });

    it('should create route', async () => {
      const mockRoute = {
        route_id: 'route_new',
        path: '/api/custom/*',
        target: 'custom-service',
        enabled: true,
      };
      mockClient.request.mockResolvedValue(mockRoute);

      const result = await api.createRoute({
        path: '/api/custom/*',
        target: 'custom-service',
        methods: ['GET', 'POST'],
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/gateway/routes', {
        json: { path: '/api/custom/*', target: 'custom-service', methods: ['GET', 'POST'] },
      });
      expect(result.route_id).toBe('route_new');
    });

    it('should update route', async () => {
      const mockRoute = {
        route_id: 'route_1',
        enabled: false,
      };
      mockClient.request.mockResolvedValue(mockRoute);

      const result = await api.updateRoute('route_1', { enabled: false });

      expect(mockClient.request).toHaveBeenCalledWith('PATCH', '/api/v1/gateway/routes/route_1', {
        json: { enabled: false },
      });
      expect(result.enabled).toBe(false);
    });

    it('should delete route', async () => {
      mockClient.request.mockResolvedValue({ success: true });

      await api.deleteRoute('route_1');

      expect(mockClient.request).toHaveBeenCalledWith('DELETE', '/api/v1/gateway/routes/route_1');
    });
  });

  // ===========================================================================
  // Gateway Status
  // ===========================================================================

  describe('Gateway Status', () => {
    it('should get gateway status', async () => {
      const mockStatus = {
        status: 'healthy',
        uptime_seconds: 86400,
        version: '2.1.0',
        active_connections: 150,
        total_requests_24h: 50000,
        error_rate: 0.005,
        latency_p50_ms: 25,
        latency_p99_ms: 150,
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.getStatus();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/status');
      expect(result.status).toBe('healthy');
      expect(result.active_connections).toBe(150);
    });

    it('should get gateway metrics', async () => {
      const mockMetrics = {
        requests: {
          total: 50000,
          success: 49750,
          error: 250,
        },
        latency: {
          p50: 25,
          p90: 80,
          p99: 150,
        },
        bandwidth: {
          inbound_mb: 500,
          outbound_mb: 1500,
        },
      };
      mockClient.request.mockResolvedValue(mockMetrics);

      const result = await api.getMetrics();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/metrics', {
        params: undefined,
      });
      expect(result.requests.success).toBe(49750);
    });

    it('should get metrics with time range', async () => {
      const mockMetrics = { requests: { total: 1000 } };
      mockClient.request.mockResolvedValue(mockMetrics);

      await api.getMetrics({
        start_time: '2024-01-20T00:00:00Z',
        end_time: '2024-01-20T12:00:00Z',
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/metrics', {
        params: {
          start_time: '2024-01-20T00:00:00Z',
          end_time: '2024-01-20T12:00:00Z',
        },
      });
    });
  });

  // ===========================================================================
  // Connection Management
  // ===========================================================================

  describe('Connection Management', () => {
    it('should list active connections', async () => {
      const mockConnections = {
        connections: [
          {
            connection_id: 'conn_1',
            device_id: 'dev_1',
            client_ip: '10.0.0.50',
            connected_at: '2024-01-20T09:00:00Z',
            requests: 150,
          },
          {
            connection_id: 'conn_2',
            device_id: 'dev_2',
            client_ip: '10.0.0.51',
            connected_at: '2024-01-20T09:30:00Z',
            requests: 80,
          },
        ],
        total: 2,
      };
      mockClient.request.mockResolvedValue(mockConnections);

      const result = await api.listConnections();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/connections', {
        params: undefined,
      });
      expect(result.connections).toHaveLength(2);
    });

    it('should list connections by device', async () => {
      const mockConnections = { connections: [], total: 0 };
      mockClient.request.mockResolvedValue(mockConnections);

      await api.listConnections({ device_id: 'dev_1' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/connections', {
        params: { device_id: 'dev_1' },
      });
    });

    it('should disconnect connection', async () => {
      mockClient.request.mockResolvedValue({ success: true });

      await api.disconnect('conn_1');

      expect(mockClient.request).toHaveBeenCalledWith(
        'POST',
        '/api/v1/gateway/connections/conn_1/disconnect'
      );
    });

    it('should disconnect all device connections', async () => {
      mockClient.request.mockResolvedValue({ disconnected: 5 });

      const result = await api.disconnectDevice('dev_1');

      expect(mockClient.request).toHaveBeenCalledWith(
        'POST',
        '/api/v1/gateway/devices/dev_1/disconnect'
      );
      expect(result.disconnected).toBe(5);
    });
  });

  // ===========================================================================
  // Rate Limiting
  // ===========================================================================

  describe('Rate Limiting', () => {
    it('should get rate limit config', async () => {
      const mockConfig = {
        enabled: true,
        default_limit: 1000,
        default_window_seconds: 60,
        rules: [
          { path: '/api/debates', limit: 100, window_seconds: 60 },
          { path: '/api/agents', limit: 500, window_seconds: 60 },
        ],
      };
      mockClient.request.mockResolvedValue(mockConfig);

      const result = await api.getRateLimitConfig();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/rate-limits');
      expect(result.default_limit).toBe(1000);
    });

    it('should update rate limit config', async () => {
      const mockConfig = { enabled: true, default_limit: 2000 };
      mockClient.request.mockResolvedValue(mockConfig);

      const result = await api.updateRateLimitConfig({ default_limit: 2000 });

      expect(mockClient.request).toHaveBeenCalledWith('PATCH', '/api/v1/gateway/rate-limits', {
        json: { default_limit: 2000 },
      });
      expect(result.default_limit).toBe(2000);
    });
  });

  // ===========================================================================
  // Health Checks
  // ===========================================================================

  describe('Health Checks', () => {
    it('should run health check', async () => {
      const mockHealth = {
        healthy: true,
        checks: [
          { name: 'database', status: 'healthy', latency_ms: 5 },
          { name: 'redis', status: 'healthy', latency_ms: 2 },
          { name: 'upstream', status: 'healthy', latency_ms: 15 },
        ],
        timestamp: '2024-01-20T10:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockHealth);

      const result = await api.healthCheck();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/health');
      expect(result.healthy).toBe(true);
      expect(result.checks).toHaveLength(3);
    });

    it('should check device health', async () => {
      const mockHealth = {
        device_id: 'dev_1',
        healthy: true,
        last_heartbeat: '2024-01-20T09:59:55Z',
        metrics: {
          cpu_percent: 45,
          memory_percent: 60,
        },
      };
      mockClient.request.mockResolvedValue(mockHealth);

      const result = await api.checkDeviceHealth('dev_1');

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/api/v1/gateway/devices/dev_1/health'
      );
      expect(result.healthy).toBe(true);
    });
  });
});
