/**
 * Learning Namespace API
 *
 * Provides access to learning/evolution metrics.
 */

export interface LearningEvolutionResponse {
  summary?: Record<string, unknown>;
  timeline?: Array<Record<string, unknown>>;
}

interface LearningClientInterface {
  get<T>(path: string): Promise<T>;
}

export class LearningAPI {
  constructor(private client: LearningClientInterface) {}

  async getEvolution(): Promise<LearningEvolutionResponse> {
    return this.client.get('/api/v1/learning/evolution');
  }
}
