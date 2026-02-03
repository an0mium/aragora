/**
 * Bots Namespace Tests
 *
 * Comprehensive tests for the bots namespace API including:
 * - Microsoft Teams integration
 * - Discord integration
 * - Telegram integration
 * - WhatsApp integration
 * - Google Chat integration
 * - Zoom integration
 * - Slack integration
 * - Combined status
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { BotsAPI } from '../bots';

interface MockClient {
  post: Mock;
  get: Mock;
}

describe('BotsAPI Namespace', () => {
  let api: BotsAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      post: vi.fn(),
      get: vi.fn(),
    };
    api = new BotsAPI(mockClient as any);
  });

  // ===========================================================================
  // Microsoft Teams
  // ===========================================================================

  describe('Microsoft Teams', () => {
    it('should send Teams message', async () => {
      const mockResponse = { success: true, message_id: 'msg_123' };
      mockClient.post.mockResolvedValue(mockResponse);

      const result = await api.teamsMessages({
        channel_id: 'channel_1',
        text: 'Hello from Aragora!',
      });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/bots/teams/messages', {
        channel_id: 'channel_1',
        text: 'Hello from Aragora!',
      });
      expect(result.success).toBe(true);
    });

    it('should get Teams status', async () => {
      const mockStatus = {
        connected: true,
        bot_id: 'teams_bot_123',
        tenant_count: 5,
        events_processed: 1250,
        last_event: '2024-01-20T10:00:00Z',
      };
      mockClient.get.mockResolvedValue(mockStatus);

      const result = await api.teamsStatus();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/bots/teams/status');
      expect(result.connected).toBe(true);
      expect(result.tenant_count).toBe(5);
    });
  });

  // ===========================================================================
  // Discord
  // ===========================================================================

  describe('Discord', () => {
    it('should handle Discord interaction', async () => {
      const mockResponse = { type: 4, data: { content: 'Response!' } };
      mockClient.post.mockResolvedValue(mockResponse);

      const result = await api.discordInteractions({
        type: 2,
        data: { name: 'debate' },
        member: { user: { id: 'user_123' } },
      });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/bots/discord/interactions', {
        type: 2,
        data: { name: 'debate' },
        member: { user: { id: 'user_123' } },
      });
      expect(result.type).toBe(4);
    });

    it('should get Discord status', async () => {
      const mockStatus = {
        connected: true,
        guilds: 15,
        users: 5000,
        events_processed: 25000,
      };
      mockClient.get.mockResolvedValue(mockStatus);

      const result = await api.discordStatus();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/bots/discord/status');
      expect(result.guilds).toBe(15);
      expect(result.users).toBe(5000);
    });
  });

  // ===========================================================================
  // Telegram
  // ===========================================================================

  describe('Telegram', () => {
    it('should handle Telegram webhook', async () => {
      const mockResponse = { ok: true };
      mockClient.post.mockResolvedValue(mockResponse);

      const result = await api.telegramWebhook({
        update_id: 123456,
        message: { chat: { id: 789 }, text: '/debate' },
      });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/bots/telegram/webhook', {
        update_id: 123456,
        message: { chat: { id: 789 }, text: '/debate' },
      });
      expect(result.ok).toBe(true);
    });

    it('should handle Telegram webhook with token', async () => {
      const mockResponse = { ok: true };
      mockClient.post.mockResolvedValue(mockResponse);

      await api.telegramWebhookToken('secret_token_123', {
        update_id: 789,
        message: { chat: { id: 123 }, text: 'Hello' },
      });

      expect(mockClient.post).toHaveBeenCalledWith(
        '/api/v1/bots/telegram/webhook/secret_token_123',
        { update_id: 789, message: { chat: { id: 123 }, text: 'Hello' } }
      );
    });

    it('should get Telegram status', async () => {
      const mockStatus = {
        connected: true,
        bot_username: 'aragora_bot',
        chats: 150,
        events_processed: 8000,
      };
      mockClient.get.mockResolvedValue(mockStatus);

      const result = await api.telegramStatus();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/bots/telegram/status');
      expect(result.bot_username).toBe('aragora_bot');
      expect(result.chats).toBe(150);
    });
  });

  // ===========================================================================
  // WhatsApp
  // ===========================================================================

  describe('WhatsApp', () => {
    it('should handle WhatsApp webhook', async () => {
      const mockResponse = { success: true };
      mockClient.post.mockResolvedValue(mockResponse);

      const result = await api.whatsappWebhook({
        entry: [{ changes: [{ value: { messages: [{ text: { body: 'Hi' } }] } }] }],
      });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/bots/whatsapp/webhook', {
        entry: [{ changes: [{ value: { messages: [{ text: { body: 'Hi' } }] } }] }],
      });
      expect(result.success).toBe(true);
    });

    it('should verify WhatsApp webhook', async () => {
      const mockResponse = { challenge: 'verification_challenge' };
      mockClient.get.mockResolvedValue(mockResponse);

      const result = await api.whatsappWebhookVerify();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/bots/whatsapp/webhook');
      expect(result.challenge).toBe('verification_challenge');
    });

    it('should get WhatsApp status', async () => {
      const mockStatus = {
        connected: true,
        phone_number: '+1234567890',
        conversations: 75,
        events_processed: 3000,
      };
      mockClient.get.mockResolvedValue(mockStatus);

      const result = await api.whatsappStatus();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/bots/whatsapp/status');
      expect(result.phone_number).toBe('+1234567890');
    });
  });

  // ===========================================================================
  // Google Chat
  // ===========================================================================

  describe('Google Chat', () => {
    it('should handle Google Chat webhook', async () => {
      const mockResponse = { text: 'Response message' };
      mockClient.post.mockResolvedValue(mockResponse);

      const result = await api.googleChatWebhook({
        type: 'MESSAGE',
        message: { text: '/debate topic', sender: { displayName: 'User' } },
        space: { name: 'spaces/123' },
      });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/bots/google-chat/webhook', {
        type: 'MESSAGE',
        message: { text: '/debate topic', sender: { displayName: 'User' } },
        space: { name: 'spaces/123' },
      });
      expect(result.text).toBe('Response message');
    });

    it('should get Google Chat status', async () => {
      const mockStatus = {
        connected: true,
        spaces: 25,
        events_processed: 5000,
      };
      mockClient.get.mockResolvedValue(mockStatus);

      const result = await api.googleChatStatus();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/bots/google-chat/status');
      expect(result.spaces).toBe(25);
    });
  });

  // ===========================================================================
  // Zoom
  // ===========================================================================

  describe('Zoom', () => {
    it('should handle Zoom events', async () => {
      const mockResponse = { success: true };
      mockClient.post.mockResolvedValue(mockResponse);

      const result = await api.zoomEvents({
        event: 'meeting.started',
        payload: { object: { id: 'meeting_123' } },
      });

      expect(mockClient.post).toHaveBeenCalledWith('/api/v1/bots/zoom/events', {
        event: 'meeting.started',
        payload: { object: { id: 'meeting_123' } },
      });
      expect(result.success).toBe(true);
    });

    it('should get Zoom status', async () => {
      const mockStatus = {
        connected: true,
        account_id: 'zoom_account_456',
        events_processed: 2000,
      };
      mockClient.get.mockResolvedValue(mockStatus);

      const result = await api.zoomStatus();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/bots/zoom/status');
      expect(result.account_id).toBe('zoom_account_456');
    });
  });

  // ===========================================================================
  // Slack
  // ===========================================================================

  describe('Slack', () => {
    it('should get Slack status', async () => {
      const mockStatus = {
        connected: true,
        workspaces: 10,
        channels: 150,
        events_processed: 50000,
      };
      mockClient.get.mockResolvedValue(mockStatus);

      const result = await api.slackStatus();

      expect(mockClient.get).toHaveBeenCalledWith('/api/v1/bots/slack/status');
      expect(result.workspaces).toBe(10);
      expect(result.channels).toBe(150);
    });
  });

  // ===========================================================================
  // Combined Status
  // ===========================================================================

  describe('Combined Status', () => {
    it('should get all bot statuses', async () => {
      mockClient.get
        .mockResolvedValueOnce({ connected: true, tenant_count: 5 }) // Teams
        .mockResolvedValueOnce({ connected: true, guilds: 15 }) // Discord
        .mockResolvedValueOnce({ connected: true, chats: 150 }) // Telegram
        .mockResolvedValueOnce({ connected: false }) // WhatsApp
        .mockResolvedValueOnce({ connected: true, spaces: 25 }) // Google Chat
        .mockResolvedValueOnce({ connected: true }) // Zoom
        .mockResolvedValueOnce({ connected: true, workspaces: 10 }); // Slack

      const result = await api.getAllStatus();

      expect(result.teams?.connected).toBe(true);
      expect(result.discord?.guilds).toBe(15);
      expect(result.telegram?.chats).toBe(150);
      expect(result.whatsapp?.connected).toBe(false);
      expect(result.google_chat?.spaces).toBe(25);
      expect(result.zoom?.connected).toBe(true);
      expect(result.slack?.workspaces).toBe(10);
    });

    it('should handle partial failures gracefully', async () => {
      mockClient.get
        .mockResolvedValueOnce({ connected: true }) // Teams
        .mockRejectedValueOnce(new Error('Discord unavailable')) // Discord fails
        .mockResolvedValueOnce({ connected: true }) // Telegram
        .mockRejectedValueOnce(new Error('WhatsApp unavailable')) // WhatsApp fails
        .mockResolvedValueOnce({ connected: true }) // Google Chat
        .mockResolvedValueOnce({ connected: true }) // Zoom
        .mockResolvedValueOnce({ connected: true }); // Slack

      const result = await api.getAllStatus();

      expect(result.teams?.connected).toBe(true);
      expect(result.discord).toBeUndefined();
      expect(result.telegram?.connected).toBe(true);
      expect(result.whatsapp).toBeUndefined();
      expect(result.google_chat?.connected).toBe(true);
    });
  });
});
