/**
 * Gateway Namespace API
 *
 * Provides methods for device gateway management:
 * - Device registration and management
 * - Channel listing
 * - Routing statistics and rules
 * - Message routing
 *
 * Endpoints:
 *   GET    /api/v1/gateway/devices           - List devices
 *   POST   /api/v1/gateway/devices           - Register device
 *   GET    /api/v1/gateway/devices/{id}      - Get device
 *   DELETE /api/v1/gateway/devices/{id}      - Unregister device
 *   POST   /api/v1/gateway/devices/{id}/heartbeat - Device heartbeat
 *   GET    /api/v1/gateway/channels          - List channels
 *   GET    /api/v1/gateway/routing/stats     - Routing statistics
 *   GET    /api/v1/gateway/routing/rules     - List routing rules
 *   POST   /api/v1/gateway/messages/route    - Route a message
 */

/**
 * Device status values.
 */
export type DeviceStatus = 'online' | 'offline' | 'unknown';

/**
 * Gateway device information.
 */
export interface GatewayDevice {
  device_id: string;
  name: string;
  device_type: string;
  status: DeviceStatus;
  capabilities?: string[];
  allowed_channels?: string[];
  metadata?: Record<string, unknown>;
  last_heartbeat?: string;
  created_at: string;
  updated_at?: string;
}

/**
 * Gateway channel information.
 */
export interface GatewayChannel {
  channel_id: string;
  name: string;
  type: string;
  active: boolean;
  connected_devices: number;
}

/**
 * Routing rule information.
 */
export interface RoutingRule {
  rule_id: string;
  name: string;
  description?: string;
  channel: string;
  agent_id: string;
  priority: number;
  conditions?: Record<string, unknown>;
  active: boolean;
}

/**
 * Routing statistics.
 */
export interface RoutingStats {
  total_messages: number;
  messages_today: number;
  messages_by_channel: Record<string, number>;
  messages_by_agent: Record<string, number>;
  average_latency_ms: number;
}

/**
 * Options for listing devices.
 */
export interface ListDevicesOptions {
  /** Filter by device status */
  status?: DeviceStatus;
  /** Filter by device type */
  deviceType?: string;
}

/**
 * Options for registering a device.
 */
export interface RegisterDeviceOptions {
  /** Device name */
  name: string;
  /** Device type (e.g., "alexa", "google_home") */
  deviceType?: string;
  /** Optional device ID (auto-generated if not provided) */
  deviceId?: string;
  /** List of device capabilities */
  capabilities?: string[];
  /** Channels the device can access */
  allowedChannels?: string[];
  /** Additional device metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Options for routing a message.
 */
export interface RouteMessageOptions {
  /** Target channel */
  channel: string;
  /** Message content */
  content: string;
}

/**
 * Client interface for making HTTP requests.
 */
interface GatewayClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Gateway API namespace.
 *
 * Provides methods for managing devices and routing messages
 * through the local gateway.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // List devices
 * const devices = await client.gateway.listDevices();
 *
 * // Register a device
 * const device = await client.gateway.registerDevice({
 *   name: 'Smart Speaker',
 *   deviceType: 'alexa',
 *   capabilities: ['voice', 'tts'],
 * });
 *
 * // Send heartbeat
 * await client.gateway.heartbeat(device.device_id);
 *
 * // Route a message
 * const result = await client.gateway.routeMessage({
 *   channel: 'slack',
 *   content: 'Hello from the gateway!',
 * });
 * ```
 */
export class GatewayAPI {
  constructor(private client: GatewayClientInterface) {}

  // =========================================================================
  // Devices
  // =========================================================================

  /**
   * List registered devices.
   *
   * @param options - Filtering options
   * @returns List of devices with total count
   */
  async listDevices(options?: ListDevicesOptions): Promise<{
    devices: GatewayDevice[];
    total: number;
  }> {
    const params: Record<string, unknown> = {};
    if (options?.status) params.status = options.status;
    if (options?.deviceType) params.type = options.deviceType;

    return this.client.request('GET', '/api/v1/gateway/devices', {
      params: Object.keys(params).length > 0 ? params : undefined,
    });
  }

  /**
   * Get device details.
   *
   * @param deviceId - Device ID
   * @returns Device information
   */
  async getDevice(deviceId: string): Promise<{
    device: GatewayDevice;
  }> {
    return this.client.request('GET', `/api/v1/gateway/devices/${deviceId}`);
  }

  /**
   * Register a new device.
   *
   * @param options - Device registration options
   * @returns Registered device information
   */
  async registerDevice(options: RegisterDeviceOptions): Promise<{
    device_id: string;
    message: string;
  }> {
    const data: Record<string, unknown> = {
      name: options.name,
      device_type: options.deviceType ?? 'unknown',
    };
    if (options.deviceId) data.device_id = options.deviceId;
    if (options.capabilities) data.capabilities = options.capabilities;
    if (options.allowedChannels) data.allowed_channels = options.allowedChannels;
    if (options.metadata) data.metadata = options.metadata;

    return this.client.request('POST', '/api/v1/gateway/devices', {
      json: data,
    });
  }

  /**
   * Unregister a device.
   *
   * @param deviceId - Device ID to unregister
   * @returns Success message
   */
  async unregisterDevice(deviceId: string): Promise<{
    success: boolean;
    message: string;
  }> {
    return this.client.request('DELETE', `/api/v1/gateway/devices/${deviceId}`);
  }

  /**
   * Send device heartbeat.
   *
   * @param deviceId - Device ID
   * @returns Heartbeat status
   */
  async heartbeat(deviceId: string): Promise<{
    status: string;
    timestamp: string;
  }> {
    return this.client.request('POST', `/api/v1/gateway/devices/${deviceId}/heartbeat`);
  }

  // =========================================================================
  // Channels
  // =========================================================================

  /**
   * List active channels.
   *
   * @returns List of channels with total count
   */
  async listChannels(): Promise<{
    channels: GatewayChannel[];
    total: number;
  }> {
    return this.client.request('GET', '/api/v1/gateway/channels');
  }

  // =========================================================================
  // Routing
  // =========================================================================

  /**
   * Get routing statistics.
   *
   * @returns Routing statistics
   */
  async getRoutingStats(): Promise<{
    stats: RoutingStats;
  }> {
    return this.client.request('GET', '/api/v1/gateway/routing/stats');
  }

  /**
   * List routing rules.
   *
   * @returns List of routing rules with total count
   */
  async listRoutingRules(): Promise<{
    rules: RoutingRule[];
    total: number;
  }> {
    return this.client.request('GET', '/api/v1/gateway/routing/rules');
  }

  // =========================================================================
  // Messages
  // =========================================================================

  /**
   * Route a message through the gateway.
   *
   * @param options - Message routing options
   * @returns Routing result with agent and rule information
   */
  async routeMessage(options: RouteMessageOptions): Promise<{
    routed: boolean;
    agent_id: string;
    rule_id?: string;
    message_id: string;
  }> {
    return this.client.request('POST', '/api/v1/gateway/messages/route', {
      json: {
        channel: options.channel,
        content: options.content,
      },
    });
  }
}
