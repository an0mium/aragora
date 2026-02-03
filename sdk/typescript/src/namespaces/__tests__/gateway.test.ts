/**
 * Gateway Namespace Tests
 *
 * Comprehensive tests for the GatewayAPI namespace class.
 * Tests all methods including:
 * - Device management (list, get, register, unregister)
 * - Device heartbeat
 * - Channel listing
 * - Routing statistics and rules
 * - Message routing
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { GatewayAPI } from '../gateway';

interface MockClient {
  request: Mock;
}

describe('GatewayAPI', () => {
  let api: GatewayAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    vi.clearAllMocks();
    mockClient = {
      request: vi.fn(),
    };
    api = new GatewayAPI(mockClient as any);
  });

  // ===========================================================================
  // Device Management
  // ===========================================================================

  describe('Device Management', () => {
    it('should list devices without filters', async () => {
      const mockDevices = {
        devices: [
          {
            device_id: 'dev_1',
            name: 'Edge Server 1',
            device_type: 'edge',
            status: 'online',
            last_heartbeat: '2024-01-20T10:00:00Z',
            created_at: '2024-01-01T00:00:00Z',
          },
          {
            device_id: 'dev_2',
            name: 'IoT Gateway',
            device_type: 'iot',
            status: 'online',
            created_at: '2024-01-02T00:00:00Z',
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

    it('should list devices with status filter', async () => {
      const mockDevices = { devices: [], total: 0 };
      mockClient.request.mockResolvedValue(mockDevices);

      await api.listDevices({ status: 'online' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/devices', {
        params: { status: 'online' },
      });
    });

    it('should list devices with device type filter', async () => {
      const mockDevices = { devices: [], total: 0 };
      mockClient.request.mockResolvedValue(mockDevices);

      await api.listDevices({ deviceType: 'edge' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/devices', {
        params: { type: 'edge' },
      });
    });

    it('should list devices with combined filters', async () => {
      const mockDevices = { devices: [], total: 0 };
      mockClient.request.mockResolvedValue(mockDevices);

      await api.listDevices({ status: 'online', deviceType: 'edge' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/devices', {
        params: { status: 'online', type: 'edge' },
      });
    });

    it('should get device by ID', async () => {
      const mockDevice = {
        device: {
          device_id: 'dev_1',
          name: 'Edge Server 1',
          device_type: 'edge',
          status: 'online',
          created_at: '2024-01-01T00:00:00Z',
        },
      };
      mockClient.request.mockResolvedValue(mockDevice);

      const result = await api.getDevice('dev_1');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/devices/dev_1');
      expect(result.device.device_id).toBe('dev_1');
    });

    it('should register a new device', async () => {
      const mockResult = {
        device_id: 'dev_new',
        message: 'Device registered successfully',
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.registerDevice({
        name: 'New Edge Server',
        deviceType: 'edge',
        capabilities: ['voice', 'tts'],
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/gateway/devices', {
        json: {
          name: 'New Edge Server',
          device_type: 'edge',
          capabilities: ['voice', 'tts'],
        },
      });
      expect(result.device_id).toBe('dev_new');
    });

    it('should register a device with minimal options', async () => {
      const mockResult = { device_id: 'dev_min', message: 'Registered' };
      mockClient.request.mockResolvedValue(mockResult);

      await api.registerDevice({ name: 'Basic Device' });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/gateway/devices', {
        json: {
          name: 'Basic Device',
          device_type: 'unknown',
        },
      });
    });

    it('should unregister a device', async () => {
      const mockResult = { success: true, message: 'Device unregistered' };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.unregisterDevice('dev_1');

      expect(mockClient.request).toHaveBeenCalledWith('DELETE', '/api/v1/gateway/devices/dev_1');
      expect(result.success).toBe(true);
    });
  });

  // ===========================================================================
  // Device Heartbeat
  // ===========================================================================

  describe('Device Heartbeat', () => {
    it('should send device heartbeat', async () => {
      const mockResult = {
        status: 'acknowledged',
        timestamp: '2024-01-20T10:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.heartbeat('dev_1');

      expect(mockClient.request).toHaveBeenCalledWith(
        'POST',
        '/api/v1/gateway/devices/dev_1/heartbeat'
      );
      expect(result.status).toBe('acknowledged');
    });
  });

  // ===========================================================================
  // Channels
  // ===========================================================================

  describe('Channels', () => {
    it('should list active channels', async () => {
      const mockChannels = {
        channels: [
          {
            channel_id: 'ch_1',
            name: 'Slack',
            type: 'chat',
            active: true,
            connected_devices: 5,
          },
          {
            channel_id: 'ch_2',
            name: 'Email',
            type: 'email',
            active: true,
            connected_devices: 3,
          },
        ],
        total: 2,
      };
      mockClient.request.mockResolvedValue(mockChannels);

      const result = await api.listChannels();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/channels');
      expect(result.channels).toHaveLength(2);
      expect(result.channels[0].name).toBe('Slack');
    });
  });

  // ===========================================================================
  // Routing
  // ===========================================================================

  describe('Routing', () => {
    it('should get routing statistics', async () => {
      const mockStats = {
        stats: {
          total_messages: 50000,
          messages_today: 1200,
          messages_by_channel: { slack: 30000, email: 15000 },
          messages_by_agent: { claude: 25000, gpt4: 20000 },
          average_latency_ms: 45,
        },
      };
      mockClient.request.mockResolvedValue(mockStats);

      const result = await api.getRoutingStats();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/routing/stats');
      expect(result.stats.total_messages).toBe(50000);
      expect(result.stats.average_latency_ms).toBe(45);
    });

    it('should list routing rules', async () => {
      const mockRules = {
        rules: [
          {
            rule_id: 'rule_1',
            name: 'Debate routing',
            channel: 'slack',
            agent_id: 'claude',
            priority: 1,
            active: true,
          },
          {
            rule_id: 'rule_2',
            name: 'Email routing',
            channel: 'email',
            agent_id: 'gpt4',
            priority: 2,
            active: true,
          },
        ],
        total: 2,
      };
      mockClient.request.mockResolvedValue(mockRules);

      const result = await api.listRoutingRules();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/gateway/routing/rules');
      expect(result.rules).toHaveLength(2);
      expect(result.rules[0].priority).toBe(1);
    });
  });

  // ===========================================================================
  // Message Routing
  // ===========================================================================

  describe('Message Routing', () => {
    it('should route a message', async () => {
      const mockResult = {
        routed: true,
        agent_id: 'claude',
        rule_id: 'rule_1',
        message_id: 'msg_123',
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.routeMessage({
        channel: 'slack',
        content: 'Hello from the gateway!',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/gateway/messages/route', {
        json: {
          channel: 'slack',
          content: 'Hello from the gateway!',
        },
      });
      expect(result.routed).toBe(true);
      expect(result.agent_id).toBe('claude');
      expect(result.message_id).toBe('msg_123');
    });

    it('should route a message to a different channel', async () => {
      const mockResult = {
        routed: true,
        agent_id: 'gpt4',
        message_id: 'msg_124',
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.routeMessage({
        channel: 'email',
        content: 'Important debate result',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/gateway/messages/route', {
        json: {
          channel: 'email',
          content: 'Important debate result',
        },
      });
      expect(result.routed).toBe(true);
    });
  });
});
