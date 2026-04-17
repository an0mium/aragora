import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import { ChatAPI } from '../chat';

interface MockClient {
  request: Mock;
  get: Mock;
  post: Mock;
}

describe('ChatAPI legacy integration routes', () => {
  let api: ChatAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn().mockResolvedValue({}),
      get: vi.fn().mockResolvedValue({}),
      post: vi.fn().mockResolvedValue({}),
    };
    api = new ChatAPI(mockClient as any);
  });

  it('maps chat status to the versioned route', async () => {
    await api.getStatus({ include_connectors: true });

    expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/chat/status', {
      params: { include_connectors: true },
    });
  });

  it('maps platform-specific and generic webhooks to versioned routes', async () => {
    await api.receiveWebhook({ event: 'message' });
    await api.receiveSlackWebhook({ type: 'event_callback' });
    await api.receiveGoogleChatWebhook({ type: 'MESSAGE' });

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'POST', '/api/v1/chat/webhook', {
      body: { event: 'message' },
      params: undefined,
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(2, 'POST', '/api/v1/chat/slack/webhook', {
      body: { type: 'event_callback' },
      params: undefined,
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(
      3,
      'POST',
      '/api/v1/chat/google_chat/webhook',
      {
        body: { type: 'MESSAGE' },
        params: undefined,
      }
    );
  });
});
