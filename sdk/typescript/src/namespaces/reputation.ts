/**
 * Reputation Namespace API
 *
 * Provides access to reputation listings.
 */

export interface ReputationEntry {
  agent: string;
  reputation: number | null;
  updated_at?: string;
}

interface ReputationClientInterface {
  get<T>(path: string): Promise<T>;
}

export class ReputationAPI {
  constructor(private client: ReputationClientInterface) {}

  async listAll(): Promise<{ reputations: ReputationEntry[]; count?: number }> {
    return this.client.get('/api/v1/reputation/all');
  }
}
