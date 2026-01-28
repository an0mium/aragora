/**
 * Bots Namespace API
 *
 * Provides webhook helpers for bot integrations (Teams, Discord, Telegram,
 * WhatsApp, Google Chat, Zoom, Slack).
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Get Teams integration status
 * const status = await client.bots.teamsStatus();
 *
 * // Get all bot integrations status
 * const allStatus = await client.bots.getAllStatus();
 * ```
 */

/**
 * Interface for bots client methods.
 */
interface BotsClientInterface {
  post<T>(path: string, body?: unknown): Promise<T>;
  get<T>(path: string): Promise<T>;
}

/**
 * Bot integration status.
 */
export interface BotStatus {
  connected: boolean;
  last_event?: string;
  events_processed?: number;
}

/**
 * Teams-specific status.
 */
export interface TeamsStatus extends BotStatus {
  bot_id?: string;
  tenant_count?: number;
}

/**
 * Discord-specific status.
 */
export interface DiscordStatus extends BotStatus {
  guilds?: number;
  users?: number;
}

/**
 * Telegram-specific status.
 */
export interface TelegramStatus extends BotStatus {
  bot_username?: string;
  chats?: number;
}

/**
 * WhatsApp-specific status.
 */
export interface WhatsAppStatus extends BotStatus {
  phone_number?: string;
  conversations?: number;
}

/**
 * Google Chat-specific status.
 */
export interface GoogleChatStatus extends BotStatus {
  spaces?: number;
}

/**
 * Zoom-specific status.
 */
export interface ZoomStatus extends BotStatus {
  account_id?: string;
}

/**
 * Slack-specific status.
 */
export interface SlackStatus extends BotStatus {
  workspaces?: number;
  channels?: number;
}

/**
 * All bot statuses combined.
 */
export interface AllBotStatus {
  teams?: TeamsStatus;
  discord?: DiscordStatus;
  telegram?: TelegramStatus;
  whatsapp?: WhatsAppStatus;
  google_chat?: GoogleChatStatus;
  zoom?: ZoomStatus;
  slack?: SlackStatus;
}

/**
 * Bots API.
 *
 * Provides methods for managing bot integrations across platforms:
 * - Microsoft Teams
 * - Discord
 * - Telegram
 * - WhatsApp
 * - Google Chat
 * - Zoom
 * - Slack
 */
export class BotsAPI {
  constructor(private client: BotsClientInterface) {}

  // ===========================================================================
  // Microsoft Teams
  // ===========================================================================

  /**
   * Send a message to Teams.
   */
  async teamsMessages(payload: Record<string, unknown>): Promise<unknown> {
    return this.client.post('/api/v1/bots/teams/messages', payload);
  }

  /**
   * Get Teams integration status.
   */
  async teamsStatus(): Promise<TeamsStatus> {
    return this.client.get('/api/v1/bots/teams/status');
  }

  // ===========================================================================
  // Discord
  // ===========================================================================

  /**
   * Handle Discord interaction.
   */
  async discordInteractions(payload: Record<string, unknown>): Promise<unknown> {
    return this.client.post('/api/v1/bots/discord/interactions', payload);
  }

  /**
   * Get Discord integration status.
   */
  async discordStatus(): Promise<DiscordStatus> {
    return this.client.get('/api/v1/bots/discord/status');
  }

  // ===========================================================================
  // Telegram
  // ===========================================================================

  /**
   * Handle Telegram webhook.
   */
  async telegramWebhook(payload: Record<string, unknown>): Promise<unknown> {
    return this.client.post('/api/v1/bots/telegram/webhook', payload);
  }

  /**
   * Handle Telegram webhook with token verification.
   */
  async telegramWebhookToken(token: string, payload: Record<string, unknown>): Promise<unknown> {
    return this.client.post(`/api/v1/bots/telegram/webhook/${token}`, payload);
  }

  /**
   * Get Telegram integration status.
   */
  async telegramStatus(): Promise<TelegramStatus> {
    return this.client.get('/api/v1/bots/telegram/status');
  }

  // ===========================================================================
  // WhatsApp
  // ===========================================================================

  /**
   * Handle WhatsApp webhook.
   */
  async whatsappWebhook(payload: Record<string, unknown>): Promise<unknown> {
    return this.client.post('/api/v1/bots/whatsapp/webhook', payload);
  }

  /**
   * Verify WhatsApp webhook (GET).
   */
  async whatsappWebhookVerify(): Promise<unknown> {
    return this.client.get('/api/v1/bots/whatsapp/webhook');
  }

  /**
   * Get WhatsApp integration status.
   */
  async whatsappStatus(): Promise<WhatsAppStatus> {
    return this.client.get('/api/v1/bots/whatsapp/status');
  }

  // ===========================================================================
  // Google Chat
  // ===========================================================================

  /**
   * Handle Google Chat webhook.
   */
  async googleChatWebhook(payload: Record<string, unknown>): Promise<unknown> {
    return this.client.post('/api/v1/bots/google-chat/webhook', payload);
  }

  /**
   * Get Google Chat integration status.
   */
  async googleChatStatus(): Promise<GoogleChatStatus> {
    return this.client.get('/api/v1/bots/google-chat/status');
  }

  // ===========================================================================
  // Zoom
  // ===========================================================================

  /**
   * Handle Zoom events.
   */
  async zoomEvents(payload: Record<string, unknown>): Promise<unknown> {
    return this.client.post('/api/v1/bots/zoom/events', payload);
  }

  /**
   * Get Zoom integration status.
   */
  async zoomStatus(): Promise<ZoomStatus> {
    return this.client.get('/api/v1/bots/zoom/status');
  }

  // ===========================================================================
  // Slack
  // ===========================================================================

  /**
   * Get Slack integration status.
   */
  async slackStatus(): Promise<SlackStatus> {
    return this.client.get('/api/v1/bots/slack/status');
  }

  // ===========================================================================
  // Utility Methods
  // ===========================================================================

  /**
   * Get status for all bot integrations.
   */
  async getAllStatus(): Promise<AllBotStatus> {
    const [teams, discord, telegram, whatsapp, googleChat, zoom, slack] = await Promise.allSettled([
      this.teamsStatus(),
      this.discordStatus(),
      this.telegramStatus(),
      this.whatsappStatus(),
      this.googleChatStatus(),
      this.zoomStatus(),
      this.slackStatus(),
    ]);

    return {
      teams: teams.status === 'fulfilled' ? teams.value : undefined,
      discord: discord.status === 'fulfilled' ? discord.value : undefined,
      telegram: telegram.status === 'fulfilled' ? telegram.value : undefined,
      whatsapp: whatsapp.status === 'fulfilled' ? whatsapp.value : undefined,
      google_chat: googleChat.status === 'fulfilled' ? googleChat.value : undefined,
      zoom: zoom.status === 'fulfilled' ? zoom.value : undefined,
      slack: slack.status === 'fulfilled' ? slack.value : undefined,
    };
  }
}
