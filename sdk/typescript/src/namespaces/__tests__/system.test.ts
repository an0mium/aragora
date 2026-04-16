import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { SystemAPI } from '../system';

interface MockClient {
  request: Mock;
}

describe('SystemAPI', () => {
  let api: SystemAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new SystemAPI(mockClient as any);
  });

  it('maps system intelligence dashboard routes', async () => {
    mockClient.request.mockResolvedValue({ data: {} });

    await api.getSystemIntelligenceOverview();
    await api.getSystemIntelligenceAgentPerformance();
    await api.getSystemIntelligenceInstitutionalMemory();
    await api.getSystemIntelligenceImprovementQueue();
    await api.getSystemIntelligenceAnomalies();
    await api.getSystemIntelligenceEvents({ limit: 20 });
    await api.getSystemIntelligenceKmSync();
    await api.getSystemIntelligenceNomicStatus();
    await api.getSystemIntelligenceDebateQueue();

    expect(mockClient.request).toHaveBeenNthCalledWith(
      1,
      'GET',
      '/api/v1/system-intelligence/overview'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      2,
      'GET',
      '/api/v1/system-intelligence/agent-performance'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      3,
      'GET',
      '/api/v1/system-intelligence/institutional-memory'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      4,
      'GET',
      '/api/v1/system-intelligence/improvement-queue'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      5,
      'GET',
      '/api/v1/system-intelligence/anomalies'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      6,
      'GET',
      '/api/v1/system-intelligence/events',
      { params: { limit: 20 } }
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      7,
      'GET',
      '/api/v1/system-intelligence/km-sync'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      8,
      'GET',
      '/api/v1/system-intelligence/nomic-status'
    );
    expect(mockClient.request).toHaveBeenNthCalledWith(
      9,
      'GET',
      '/api/v1/system-intelligence/debate-queue'
    );
  });
});
