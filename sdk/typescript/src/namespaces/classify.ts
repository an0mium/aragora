/**
 * Classify Namespace API
 *
 * Provides policy-based classification endpoints.
 */

export interface ClassifyRequest {
  content: string;
  context?: string;
  policies?: string[];
}

export interface ClassifyResponse {
  label: string;
  confidence?: number;
  policy?: string;
  metadata?: Record<string, unknown>;
}

interface ClassifyClientInterface {
  post<T>(path: string, body?: unknown): Promise<T>;
  get<T>(path: string): Promise<T>;
}

export class ClassifyAPI {
  constructor(private client: ClassifyClientInterface) {}

  async classify(body: ClassifyRequest): Promise<ClassifyResponse> {
    return this.client.post('/api/v1/classify', body);
  }

  async getPolicy(level: string): Promise<Record<string, unknown>> {
    return this.client.get(`/api/v1/classify/policy/${level}`);
  }
}
