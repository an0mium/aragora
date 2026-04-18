/**
 * Settlements Namespace API
 *
 * Provides access to debate settlement and calibration routes.
 */

interface SettlementsClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; body?: unknown; json?: Record<string, unknown> }
  ): Promise<T>;
}

export interface SettlementBatchItem {
  settlement_id: string;
  outcome: 'correct' | 'incorrect' | 'partial';
  evidence?: string;
  [key: string]: unknown;
}

export class SettlementAPI {
  constructor(private client: SettlementsClientInterface) {}

  async list(params?: { debate_id?: string; domain?: string; limit?: number }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/settlements', {
      params: params as Record<string, unknown> | undefined,
    });
  }

  async getHistory(params?: { debate_id?: string; domain?: string; limit?: number }): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/settlements/history', {
      params: params as Record<string, unknown> | undefined,
    });
  }

  async getSummary(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/settlements/summary');
  }

  async get(settlementId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/v1/settlements/${encodeURIComponent(settlementId)}`);
  }

  async settle(
    settlementId: string,
    body: { outcome: 'correct' | 'incorrect' | 'partial'; evidence?: string; settled_by?: string }
  ): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/v1/settlements/${encodeURIComponent(settlementId)}/settle`,
      { body }
    );
  }

  async settleBatch(
    settlements: SettlementBatchItem[],
    settledBy = 'api'
  ): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/settlements/batch', {
      body: { settlements, settled_by: settledBy },
    });
  }

  async getAgentAccuracy(agentName: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/settlements/agent/${encodeURIComponent(agentName)}/accuracy`
    );
  }
}
