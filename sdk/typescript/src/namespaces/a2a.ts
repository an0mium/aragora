/**
 * A2A (Agent-to-Agent) Namespace API
 *
 * Provides methods for the Agent-to-Agent protocol:
 * - Agent discovery and metadata
 * - Task submission and execution
 * - Streaming task responses
 */

/**
 * A2A Agent card (well-known metadata)
 */
export interface AgentCard {
  name: string;
  description: string;
  version: string;
  capabilities: string[];
  supported_tasks: string[];
  endpoints: {
    tasks: string;
    stream: string;
  };
  metadata?: Record<string, unknown>;
}

/**
 * A2A Agent
 */
export interface A2AAgent {
  name: string;
  description: string;
  version: string;
  status: 'available' | 'busy' | 'offline';
  capabilities: string[];
  supported_tasks: string[];
  avg_response_time_ms: number;
  success_rate: number;
  metadata?: Record<string, unknown>;
}

/**
 * A2A Task
 */
export interface A2ATask {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  agent?: string;
  task: string;
  context?: Record<string, unknown>;
  result?: unknown;
  error?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  progress?: number;
}

/**
 * Submit task request
 */
export interface SubmitTaskRequest {
  task: string;
  agent?: string;
  context?: Record<string, unknown>;
  timeout?: number;
  priority?: 'low' | 'normal' | 'high';
}

/**
 * Stream task request
 */
export interface StreamTaskRequest {
  from_sequence?: number;
}

/**
 * Stream chunk
 */
export interface StreamChunk {
  sequence: number;
  type: 'progress' | 'output' | 'error' | 'complete';
  data: unknown;
  timestamp: string;
}

/**
 * Interface for the internal client used by A2AAPI.
 */
interface A2AClientInterface {
  get<T>(path: string): Promise<T>;
  post<T>(path: string, body?: unknown): Promise<T>;
}

/**
 * A2A (Agent-to-Agent) API namespace.
 *
 * Implements the Agent-to-Agent protocol for inter-agent communication
 * and task delegation.
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // List available agents
 * const { agents } = await client.a2a.listAgents();
 *
 * // Submit a task
 * const task = await client.a2a.submitTask({
 *   task: 'Analyze this document and summarize key points',
 *   agent: 'claude',
 *   context: { document_url: 'https://...' }
 * });
 *
 * // Check task status
 * const status = await client.a2a.getTask(task.task_id);
 *
 * // Get agent card
 * const card = await client.a2a.getAgentCard();
 * ```
 */
export class A2AAPI {
  constructor(private client: A2AClientInterface) {}

  // ===========================================================================
  // Agent Discovery
  // ===========================================================================

  /**
   * Get the agent card (well-known metadata).
   */
  async getAgentCard(): Promise<AgentCard> {
    return this.client.get('/api/v1/a2a/.well-known/agent.json');
  }

  /**
   * List available A2A agents.
   */
  async listAgents(): Promise<{ agents: A2AAgent[] }> {
    return this.client.get('/api/v1/a2a/agents');
  }

  /**
   * Get details for a specific A2A agent.
   */
  async getAgent(name: string): Promise<A2AAgent> {
    return this.client.get(`/api/v1/a2a/agents/${name}`);
  }

  /**
   * Get the A2A OpenAPI specification.
   */
  async getOpenAPISpec(): Promise<Record<string, unknown>> {
    return this.client.get('/api/v1/a2a/openapi.json');
  }

  // ===========================================================================
  // Task Management
  // ===========================================================================

  /**
   * Submit a task to an A2A agent.
   */
  async submitTask(body: SubmitTaskRequest): Promise<A2ATask> {
    return this.client.post('/api/v1/a2a/tasks', body);
  }

  /**
   * Get the status and result of a task.
   */
  async getTask(taskId: string): Promise<A2ATask> {
    return this.client.get(`/api/v1/a2a/tasks/${taskId}`);
  }

  /**
   * Stream task output (for long-running tasks).
   */
  async streamTask(taskId: string, body?: StreamTaskRequest): Promise<{ stream_url: string; chunks: StreamChunk[] }> {
    return this.client.post(`/api/v1/a2a/tasks/${taskId}/stream`, body);
  }
}
