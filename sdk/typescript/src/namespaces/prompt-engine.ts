/**
 * Prompt Engine Namespace API
 *
 * Provides methods for prompt-to-specification pipeline operations.
 */

interface PromptEngineClientInterface {
  request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
}

export interface PromptEngineRunListOptions extends Record<string, unknown> {
  status?: string;
  plan_id?: string;
  debate_id?: string;
  execution_id?: string;
  limit?: number;
  offset?: number;
}

export interface PromptEngineRunRequest extends Record<string, unknown> {
  prompt: string;
  context?: unknown;
  profile?: string;
  autonomy?: string;
  skip_research?: boolean;
  skip_interrogation?: boolean;
  decision_plan?: Record<string, unknown>;
}

export interface PromptEnginePromptRequest extends Record<string, unknown> {
  prompt: string;
  context?: unknown;
}

export interface PromptEngineIntentRequest extends Record<string, unknown> {
  intent: Record<string, unknown>;
  context?: unknown;
}

export interface PromptEngineInterrogateRequest extends PromptEngineIntentRequest {
  depth?: string;
}

export interface PromptEngineSpecifyRequest extends PromptEngineIntentRequest {
  questions?: Array<Record<string, unknown>>;
  research?: Record<string, unknown>;
}

export interface PromptEngineValidateRequest extends Record<string, unknown> {
  specification: Record<string, unknown>;
}

export class PromptEngineAPI {
  constructor(private client: PromptEngineClientInterface) {}

  /** List persisted prompt-engine runs. */
  async listRuns(params?: PromptEngineRunListOptions): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/prompt-engine/runs', { params });
  }

  /** Fetch a persisted prompt-engine run. */
  async getRun(runId: string): Promise<Record<string, unknown>> {
    return this.client.request('GET', `/api/prompt-engine/runs/${encodeURIComponent(runId)}`);
  }

  /** Run the full prompt-to-specification pipeline. */
  async run(body: PromptEngineRunRequest): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/prompt-engine/run', { body });
  }

  /** Decompose a prompt into structured intent. */
  async decompose(body: PromptEnginePromptRequest): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/prompt-engine/decompose', { body });
  }

  /** Generate clarifying questions for an intent. */
  async interrogate(body: PromptEngineInterrogateRequest): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/prompt-engine/interrogate', { body });
  }

  /** Research supporting context for an intent. */
  async research(body: PromptEngineIntentRequest): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/prompt-engine/research', { body });
  }

  /** Build a specification from intent, questions, research, and context. */
  async specify(body: PromptEngineSpecifyRequest): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/prompt-engine/specify', { body });
  }

  /** Validate a prompt-engine specification. */
  async validate(body: PromptEngineValidateRequest): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/prompt-engine/validate', { body });
  }
}
