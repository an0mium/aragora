/**
 * Devices Namespace API
 *
 * Provides a namespaced interface for device registration and push notifications.
 * Supports Android (FCM), iOS (APNS), Web Push, Alexa, and Google Home.
 */

// =============================================================================
// Types
// =============================================================================

/** Supported device types */
export type DeviceType = 'android' | 'ios' | 'web' | 'alexa' | 'google_home';

/** Notification delivery status */
export type NotificationStatus = 'sent' | 'delivered' | 'failed' | 'pending';

/** Device registration */
export interface DeviceRegistration {
  device_type: DeviceType;
  push_token: string;
  user_id?: string;
  device_name?: string;
  app_version?: string;
  os_version?: string;
  device_model?: string;
  timezone?: string;
  locale?: string;
  app_bundle_id?: string;
}

/** Registered device */
export interface Device {
  device_id: string;
  user_id: string;
  device_type: DeviceType;
  device_name: string | null;
  app_version: string | null;
  os_version?: string;
  last_active: string;
  notification_count: number;
  created_at: string;
}

/** Push notification message */
export interface NotificationMessage {
  title: string;
  body: string;
  data?: Record<string, unknown>;
  image_url?: string;
  action_url?: string;
  badge?: number;
  sound?: string;
}

/** Notification result */
export interface NotificationResult {
  success: boolean;
  device_id: string;
  message_id: string | null;
  status: NotificationStatus;
  error: string | null;
}

/** User notification result */
export interface UserNotificationResult {
  success: boolean;
  user_id: string;
  devices_notified: number;
  devices_failed: number;
  devices_removed: number;
  results: Array<{
    device_id: string;
    success: boolean;
    error: string | null;
  }>;
}

/** Device connector health */
export interface ConnectorHealth {
  status: 'healthy' | 'degraded' | 'unavailable';
  fcm?: {
    available: boolean;
    last_check: string;
  };
  apns?: {
    available: boolean;
    last_check: string;
  };
  web_push?: {
    available: boolean;
    last_check: string;
  };
  alexa?: {
    available: boolean;
    last_check: string;
  };
  google_home?: {
    available: boolean;
    last_check: string;
  };
  error?: string;
}

/** Alexa skill request */
export interface AlexaRequest {
  version: string;
  session: {
    sessionId: string;
    application: { applicationId: string };
    user: { userId: string };
    attributes: Record<string, unknown>;
  };
  request: {
    type: string;
    requestId: string;
    timestamp: string;
    locale: string;
    intent?: {
      name: string;
      slots?: Record<string, { name: string; value: string }>;
    };
  };
}

/** Alexa skill response */
export interface AlexaResponse {
  version: string;
  sessionAttributes?: Record<string, unknown>;
  response: {
    outputSpeech?: {
      type: 'PlainText' | 'SSML';
      text?: string;
      ssml?: string;
    };
    card?: {
      type: 'Simple' | 'Standard';
      title: string;
      content?: string;
      text?: string;
    };
    reprompt?: {
      outputSpeech: {
        type: 'PlainText' | 'SSML';
        text?: string;
        ssml?: string;
      };
    };
    shouldEndSession: boolean;
  };
}

/** Google Actions request */
export interface GoogleActionsRequest {
  requestId: string;
  inputs: Array<{
    intent: string;
    payload?: {
      commands?: Array<{
        devices: Array<{ id: string }>;
        execution: Array<{
          command: string;
          params?: Record<string, unknown>;
        }>;
      }>;
      devices?: Array<{ id: string }>;
    };
  }>;
  user?: { userId: string };
  session?: { params: Record<string, unknown> };
}

/** Google Actions response */
export interface GoogleActionsResponse {
  requestId: string;
  payload?: {
    agentUserId?: string;
    devices?: Array<{
      id: string;
      type: string;
      traits: string[];
      name: { name: string };
      willReportState: boolean;
    }>;
    commands?: Array<{
      ids: string[];
      status: string;
      states?: Record<string, unknown>;
    }>;
  };
  prompt?: {
    firstSimple?: { speech: string; text?: string };
    override?: boolean;
  };
  scene?: { name: string };
  session?: { params: Record<string, unknown> };
}

// =============================================================================
// Devices API
// =============================================================================

/**
 * Client interface for devices operations.
 */
interface DevicesClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Devices namespace API for push notification management.
 *
 * Provides comprehensive device management:
 * - Device registration for multiple platforms
 * - Push notification delivery
 * - Voice assistant webhook handling
 * - Health monitoring
 */
export class DevicesAPI {
  constructor(private client: DevicesClientInterface) {}

  // ===========================================================================
  // Device Registration
  // ===========================================================================

  /**
   * Register a device for push notifications.
   *
   * Supports Android (FCM), iOS (APNS), Web Push, Alexa, and Google Home.
   *
   * @param registration - Device registration details
   * @returns Registered device
   */
  async register(registration: DeviceRegistration): Promise<{
    success: boolean;
    device_id: string;
    device_type: DeviceType;
    registered_at: string;
  }> {
    return this.client.request('POST', '/devices/register', {
      json: registration as unknown as Record<string, unknown>,
    });
  }

  /**
   * Unregister a device.
   *
   * @param deviceId - Device ID to unregister
   * @returns Confirmation
   */
  async unregister(deviceId: string): Promise<{
    success: boolean;
    device_id: string;
    deleted_at: string;
  }> {
    return this.client.request('DELETE', `/devices/${deviceId}`);
  }

  /**
   * Get device information.
   *
   * @param deviceId - Device ID
   * @returns Device details
   */
  async get(deviceId: string): Promise<Device> {
    return this.client.request('GET', `/devices/${deviceId}`);
  }

  /**
   * List all devices for a user.
   *
   * @param userId - User ID
   * @returns User's devices
   */
  async listByUser(userId: string): Promise<{
    user_id: string;
    device_count: number;
    devices: Device[];
  }> {
    return this.client.request('GET', `/devices/user/${userId}`);
  }

  // ===========================================================================
  // Push Notifications
  // ===========================================================================

  /**
   * Send notification to a specific device.
   *
   * @param deviceId - Target device ID
   * @param message - Notification message
   * @returns Delivery result
   */
  async notify(deviceId: string, message: NotificationMessage): Promise<NotificationResult> {
    return this.client.request('POST', `/devices/${deviceId}/notify`, {
      json: message as unknown as Record<string, unknown>,
    });
  }

  /**
   * Send notification to all devices for a user.
   *
   * @param userId - Target user ID
   * @param message - Notification message
   * @returns Delivery results for all devices
   */
  async notifyUser(userId: string, message: NotificationMessage): Promise<UserNotificationResult> {
    return this.client.request('POST', `/devices/user/${userId}/notify`, {
      json: message as unknown as Record<string, unknown>,
    });
  }

  // ===========================================================================
  // Health Monitoring
  // ===========================================================================

  /**
   * Get device connector health status.
   *
   * Shows availability of FCM, APNS, Web Push, and voice connectors.
   *
   * @returns Connector health status
   */
  async getHealth(): Promise<ConnectorHealth> {
    return this.request('GET', '/devices/health') as Promise<ConnectorHealth>;
  }

  // ===========================================================================
  // Voice Assistant Webhooks
  // ===========================================================================

  /**
   * Handle Alexa skill webhook request.
   *
   * Processes voice commands and returns Alexa-formatted responses.
   * Note: This is typically called by Alexa's servers, not directly.
   *
   * @param request - Alexa skill request
   * @returns Alexa skill response
   */
  async handleAlexaWebhook(request: AlexaRequest): Promise<AlexaResponse> {
    return this.request('POST', '/devices/alexa/webhook', request) as Promise<AlexaResponse>;
  }

  /**
   * Handle Google Actions webhook request.
   *
   * Processes voice commands and Smart Home intents.
   * Note: This is typically called by Google's servers, not directly.
   *
   * @param request - Google Actions request
   * @returns Google Actions response
   */
  async handleGoogleWebhook(request: GoogleActionsRequest): Promise<GoogleActionsResponse> {
    return this.request('POST', '/devices/google/webhook', request) as Promise<GoogleActionsResponse>;
  }
}
