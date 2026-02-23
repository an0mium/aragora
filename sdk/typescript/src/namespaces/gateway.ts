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
 * Gateway agent information.
 */
export interface GatewayAgent {
  /** Agent name */
  name: string;
  /** Agent type */
  type: string;
  /** Agent status */
  status: 'active' | 'inactive' | 'error';
  /** Supported channels */
  channels?: string[];
  /** Agent capabilities */
  capabilities?: string[];
  /** Last health check time */
  last_health_check?: string;
  /** Registration time */
  registered_at: string;
}

/**
 * Gateway credential metadata (no secrets).
 */
export interface GatewayCredential {
  /** Credential ID */
  id: string;
  /** Credential name */
  name: string;
  /** Credential type */
  type: string;
  /** Creation time */
  created_at: string;
  /** Last rotation time */
  last_rotated?: string;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Options for creating a credential.
 */
export interface CreateCredentialOptions {
  /** Credential name */
  name: string;
  /** Credential type (api_key, oauth_token, etc.) */
  type: string;
  /** Credential value (encrypted at rest) */
  value: string;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Options for creating a routing rule.
 */
export interface CreateRoutingRuleOptions {
  /** Channel to apply the rule to */
  channel: string;
  /** Message matching pattern (regex supported) */
  pattern: string;
  /** Target agent for matched messages */
  agentId: string;
  /** Rule evaluation priority (lower is higher) */
  priority?: number;
  /** Whether the rule is active */
  enabled?: boolean;
}

/**
 * Options for sending a message.
 */
export interface SendMessageOptions {
  /** Target channel for routing */
  channel: string;
  /** Message content to route */
  content: string;
  /** Additional message metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Gateway message information.
 */
export interface GatewayMessage {
  /** Message ID */
  message_id: string;
  /** Target channel */
  channel: string;
  /** Message content */
  content: string;
  /** Delivery status */
  status: 'pending' | 'routed' | 'delivered' | 'failed';
  /** Agent the message was routed to */
  agent_id?: string;
  /** Routing rule that matched */
  rule_id?: string;
  /** Creation time */
  created_at: string;
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

  // =========================================================================
  // Gateway Health
  // =========================================================================

  /**
   * Check gateway health status.
   *
   * @returns Gateway health status and component details
   */
  async health(): Promise<{
    status: string;
    components: Record<string, { status: string; latency_ms?: number }>;
  }> {
    return this.client.request('GET', '/api/v1/gateway/health');
  }

  // =========================================================================
  // Agents
  // =========================================================================

  /**
   * List registered gateway agents.
   *
   * @returns List of agents with total count
   */
  async listAgents(): Promise<{
    agents: GatewayAgent[];
    total: number;
  }> {
    return this.client.request('POST', '/api/v1/gateway/agents');
  }

  /**
   * Get details for a specific gateway agent.
   *
   * @param agentName - Agent name or identifier
   * @returns Agent information
   */
  async getAgent(agentName: string): Promise<{
    agent: GatewayAgent;
  }> {
    return this.client.request('GET', `/api/v1/gateway/agents/${agentName}`);
  }

  /**
   * Check health of a specific gateway agent.
   *
   * @param agentName - Agent name or identifier
   * @returns Agent health status
   */
  async getAgentHealth(agentName: string): Promise<{
    status: string;
    agent: string;
    latency_ms?: number;
    last_check: string;
  }> {
    return this.client.request('GET', `/api/v1/gateway/agents/${agentName}/health`);
  }

  // =========================================================================
  // Credentials
  // =========================================================================

  /**
   * List stored credentials (metadata only, no secrets).
   *
   * @returns List of credentials with total count
   */
  async listCredentials(): Promise<{
    credentials: GatewayCredential[];
    total: number;
  }> {
    return this.client.request('POST', '/api/v1/gateway/credentials');
  }

  /**
   * Store a new credential in the gateway.
   *
   * @param options - Credential creation options
   * @returns Created credential ID and success message
   */
  async createCredential(options: CreateCredentialOptions): Promise<{
    credential_id: string;
    message: string;
  }> {
    const data: Record<string, unknown> = {
      name: options.name,
      type: options.type,
      value: options.value,
    };
    if (options.metadata) data.metadata = options.metadata;

    return this.client.request('POST', '/api/v1/gateway/credentials', {
      json: data,
    });
  }

  /**
   * Delete a stored credential.
   *
   * @param credentialId - Credential ID to delete
   * @returns Success message
   */
  async deleteCredential(credentialId: string): Promise<{
    success: boolean;
    message: string;
  }> {
    return this.client.request('GET', `/api/v1/gateway/credentials/${credentialId}`);
  }

  // =========================================================================
  // Messages (extended)
  // =========================================================================

  /**
   * Submit a message to the gateway for routing.
   *
   * @param options - Message options including channel, content, and optional metadata
   * @returns Routing result with agent and rule information
   */
  async sendMessage(options: SendMessageOptions): Promise<{
    routed: boolean;
    agent_id: string;
    rule_id?: string;
    message_id: string;
  }> {
    const data: Record<string, unknown> = {
      channel: options.channel,
      content: options.content,
    };
    if (options.metadata) data.metadata = options.metadata;

    return this.client.request('POST', '/api/v1/gateway/messages', {
      json: data,
    });
  }

  /**
   * Get details for a specific routed message.
   *
   * @param messageId - Message ID
   * @returns Message details and routing information
   */
  async getMessage(messageId: string): Promise<{
    message: GatewayMessage;
  }> {
    return this.client.request('GET', `/api/v1/gateway/messages/${messageId}`);
  }

  // =========================================================================
  // Routing (extended)
  // =========================================================================

  /**
   * List routing rules with statistics.
   *
   * @returns List of routing rules with total count and aggregate stats
   */
  async listRouting(): Promise<{
    rules: RoutingRule[];
    total: number;
    stats: {
      total_rules: number;
      messages_routed: number;
      routing_errors: number;
    };
  }> {
    return this.client.request('GET', '/api/v1/gateway/routing');
  }

  /**
   * Create a new routing rule.
   *
   * @param options - Routing rule creation options
   * @returns Created rule and success message
   */
  async createRoutingRule(options: CreateRoutingRuleOptions): Promise<{
    rule: RoutingRule;
    message: string;
  }> {
    const data: Record<string, unknown> = {
      channel: options.channel,
      pattern: options.pattern,
      agent_id: options.agentId,
      enabled: options.enabled ?? true,
    };
    if (options.priority !== undefined) data.priority = options.priority;

    return this.client.request('POST', '/api/v1/gateway/routing', {
      json: data,
    });
  }

  /**
   * Get details for a specific routing rule.
   *
   * @param routeId - Routing rule ID
   * @returns Routing rule details
   */
  async getRoutingRule(routeId: string): Promise<{
    rule: RoutingRule;
  }> {
    return this.client.request('GET', `/api/v1/gateway/routing/${routeId}`);
  }

  // --- Gateway Config ---

  /** Create a gateway configuration. */
  async createConfig(data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/gateway/config', { json: data });
  }

  /** Create default gateway configuration. */
  async createConfigDefaults(data: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/gateway/config/defaults', { json: data });
  }
}
