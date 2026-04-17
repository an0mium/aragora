import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { ReceiptsAPI } from '../receipts';

interface MockClient {
  request: Mock;
  listGauntletReceipts: Mock;
  getGauntletReceipt: Mock;
  verifyGauntletReceipt: Mock;
  exportGauntletReceipt: Mock;
}

function createMockClient(): MockClient {
  return {
    request: vi.fn(),
    listGauntletReceipts: vi.fn(),
    getGauntletReceipt: vi.fn(),
    verifyGauntletReceipt: vi.fn(),
    exportGauntletReceipt: vi.fn(),
  };
}

describe('ReceiptsAPI Namespace', () => {
  let api: ReceiptsAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = createMockClient();
    api = new ReceiptsAPI(mockClient as any);
  });

  it('lists receipt deliveries with filters', async () => {
    mockClient.request.mockResolvedValue({ deliveries: [], total: 0 });

    await api.listDeliveries({
      limit: 20,
      offset: 5,
      receiptId: 'r_123',
      channelType: 'slack',
      status: 'delivered',
    });

    expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/receipts/deliveries', {
      params: {
        limit: 20,
        offset: 5,
        receipt_id: 'r_123',
        channel_type: 'slack',
        status: 'delivered',
      },
    });
  });

  it('lists recent receipt anchors', async () => {
    mockClient.request.mockResolvedValue({ anchors: [], total: 0, limit: 7 });

    const response = await api.listRecentAnchors({ limit: 7 });

    expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/receipts/recent-anchors', {
      params: { limit: 7 },
    });
    expect(response).toEqual({ anchors: [], total: 0, limit: 7 });
  });

  it('gets receipt anchor status', async () => {
    mockClient.request.mockResolvedValue({ receipt_id: 'r/123', anchored: true });

    await api.getAnchorStatus('r/123');

    expect(mockClient.request).toHaveBeenCalledWith(
      'GET',
      '/api/v1/receipts/r%2F123/anchor-status'
    );
  });
});
