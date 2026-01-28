/**
 * Evolution Namespace API
 *
 * Provides access to agent evolution history.
 */

export interface EvolutionHistoryEntry {
  timestamp: string;
  score?: number;
  changes?: Record<string, unknown>;
}

export interface EvolutionHistoryResponse {
  agent: string;
  history: EvolutionHistoryEntry[];
}

interface EvolutionClientInterface {
  get<T>(path: string): Promise<T>;
}

export class EvolutionAPI {
  constructor(private client: EvolutionClientInterface) {}

  async getHistory(agent: string): Promise<EvolutionHistoryResponse> {
    return this.client.get(`/api/v1/evolution/${agent}/history`);
  }
}
