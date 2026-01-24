/**
 * Main client for the Aragora TypeScript SDK.
 *
 * @example
 * ```typescript
 * import { AragoraClient } from '@aragora/client';
 *
 * const client = new AragoraClient('http://localhost:8080');
 *
 * // Run a debate
 * const debate = await client.debates.run('Should we use microservices?');
 * console.log(debate.consensus?.conclusion);
 *
 * // Use control plane
 * const status = await client.controlPlane.getStatus();
 * console.log(status);
 * ```
 */

import { ControlPlaneAPI } from './control-plane';
import type {
  Debate,
  HealthStatus,
  AgentProfile,
  GraphDebate,
  GraphBranch,
  MatrixDebate,
  MatrixConclusion,
  VerificationResult,
  GauntletReceipt,
  TeamSelection,
  SelectionPlugins,
} from './types';

export interface AragoraClientOptions {
  apiKey?: string;
  timeout?: number;
  headers?: Record<string, string>;
}

export class AragoraError extends Error {
  code?: string;
  status?: number;
  details?: Record<string, unknown>;

  constructor(
    message: string,
    options: { code?: string; status?: number; details?: Record<string, unknown> } = {}
  ) {
    super(message);
    this.name = 'AragoraError';
    this.code = options.code;
    this.status = options.status;
    this.details = options.details;
  }
}

export class DebatesAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  /**
   * Create a new debate.
   */
  async create(options: {
    task: string;
    agents?: string[];
    maxRounds?: number;
    consensusThreshold?: number;
    metadata?: Record<string, unknown>;
  }): Promise<{ id: string }> {
    return this.client.post<{ id: string }>('/api/v1/debates', {
      task: options.task,
      agents: options.agents,
      max_rounds: options.maxRounds ?? 5,
      consensus_threshold: options.consensusThreshold ?? 0.8,
      metadata: options.metadata ?? {},
    });
  }

  /**
   * Get a debate by ID.
   */
  async get(debateId: string): Promise<Debate> {
    return this.client.get<Debate>(`/api/v1/debates/${debateId}`);
  }

  /**
   * List debates.
   */
  async list(options: {
    limit?: number;
    offset?: number;
    status?: string;
  } = {}): Promise<Debate[]> {
    const data = await this.client.get<{ debates: Debate[] }>('/api/v1/debates', {
      limit: options.limit ?? 50,
      offset: options.offset ?? 0,
      ...(options.status && { status: options.status }),
    });
    return data.debates ?? [];
  }

  /**
   * Run a debate and wait for completion.
   */
  async run(
    task: string,
    options: {
      agents?: string[];
      maxRounds?: number;
      consensusThreshold?: number;
      pollInterval?: number;
      timeout?: number;
    } = {}
  ): Promise<Debate> {
    const response = await this.create({
      task,
      agents: options.agents,
      maxRounds: options.maxRounds ?? 5,
      consensusThreshold: options.consensusThreshold ?? 0.8,
    });

    const debateId = response.id;
    const pollInterval = options.pollInterval ?? 1000;
    const timeout = options.timeout ?? 300000;
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const debate = await this.get(debateId);
      if (['completed', 'failed', 'cancelled'].includes(debate.status)) {
        return debate;
      }
      await new Promise((resolve) => setTimeout(resolve, pollInterval));
    }

    throw new AragoraError(`Debate ${debateId} did not complete within ${timeout}ms`);
  }
}

export class AgentsAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  /**
   * List all available agents.
   */
  async list(): Promise<AgentProfile[]> {
    const data = await this.client.get<{ agents: AgentProfile[] }>('/api/v1/agents');
    return data.agents ?? [];
  }

  /**
   * Get an agent profile.
   */
  async get(agentId: string): Promise<AgentProfile> {
    return this.client.get<AgentProfile>(`/api/v1/agents/${agentId}`);
  }
}

export class GraphDebatesAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  /**
   * Create a new graph debate.
   */
  async create(options: {
    task: string;
    agents?: string[];
    maxRounds?: number;
    branchThreshold?: number;
    maxBranches?: number;
  }): Promise<{ id: string }> {
    return this.client.post<{ id: string }>('/api/v1/graph-debates', {
      task: options.task,
      agents: options.agents,
      max_rounds: options.maxRounds ?? 5,
      branch_threshold: options.branchThreshold ?? 0.5,
      max_branches: options.maxBranches ?? 10,
    });
  }

  /**
   * Get a graph debate by ID.
   */
  async get(debateId: string): Promise<GraphDebate> {
    return this.client.get<GraphDebate>(`/api/v1/graph-debates/${debateId}`);
  }

  /**
   * Get branches for a graph debate.
   */
  async getBranches(debateId: string): Promise<GraphBranch[]> {
    const data = await this.client.get<{ branches: GraphBranch[] }>(
      `/api/v1/graph-debates/${debateId}/branches`
    );
    return data.branches ?? [];
  }
}

export class MatrixDebatesAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  /**
   * Create a new matrix debate.
   */
  async create(options: {
    task: string;
    scenarios: Array<{ name: string; parameters: Record<string, unknown>; weight?: number }>;
    agents?: string[];
    maxRounds?: number;
  }): Promise<{ id: string }> {
    return this.client.post<{ id: string }>('/api/v1/matrix-debates', {
      task: options.task,
      scenarios: options.scenarios,
      agents: options.agents,
      max_rounds: options.maxRounds ?? 3,
    });
  }

  /**
   * Get a matrix debate by ID.
   */
  async get(debateId: string): Promise<MatrixDebate> {
    return this.client.get<MatrixDebate>(`/api/v1/matrix-debates/${debateId}`);
  }

  /**
   * Get conclusions for a matrix debate.
   */
  async getConclusions(debateId: string): Promise<MatrixConclusion> {
    return this.client.get<MatrixConclusion>(
      `/api/v1/matrix-debates/${debateId}/conclusions`
    );
  }
}

export class VerificationAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  /**
   * Verify a claim using multi-agent deliberation.
   */
  async verifyClaim(options: {
    claim: string;
    context?: string;
    evidenceSources?: string[];
    minConfidence?: number;
  }): Promise<VerificationResult> {
    return this.client.post<VerificationResult>('/api/v1/verification/claim', {
      claim: options.claim,
      context: options.context,
      evidence_sources: options.evidenceSources,
      min_confidence: options.minConfidence ?? 0.7,
    });
  }

  /**
   * Get a verification result by claim ID.
   */
  async get(claimId: string): Promise<VerificationResult> {
    return this.client.get<VerificationResult>(`/api/v1/verification/${claimId}`);
  }
}

export class GauntletAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  /**
   * Run an agent through the gauntlet certification process.
   */
  async run(options: {
    agentId: string;
    challengeTypes?: string[];
    minPassScore?: number;
  }): Promise<GauntletReceipt> {
    return this.client.post<GauntletReceipt>('/api/v1/gauntlet/run', {
      agent_id: options.agentId,
      challenge_types: options.challengeTypes,
      min_pass_score: options.minPassScore ?? 0.7,
    });
  }

  /**
   * Get a gauntlet receipt by ID.
   */
  async get(receiptId: string): Promise<GauntletReceipt> {
    return this.client.get<GauntletReceipt>(`/api/v1/gauntlet/${receiptId}`);
  }
}

export class TeamSelectionAPI {
  private client: AragoraClient;

  constructor(client: AragoraClient) {
    this.client = client;
  }

  /**
   * Select an optimal team of agents for a task.
   */
  async selectTeam(options: {
    task: string;
    teamSize?: number;
    requiredCapabilities?: string[];
    excludeAgents?: string[];
    plugins?: SelectionPlugins;
  }): Promise<TeamSelection> {
    return this.client.post<TeamSelection>('/api/v1/team-selection', {
      task: options.task,
      team_size: options.teamSize ?? 3,
      required_capabilities: options.requiredCapabilities,
      exclude_agents: options.excludeAgents,
      plugins: options.plugins,
    });
  }
}

export class AragoraClient {
  public readonly baseUrl: string;
  private apiKey?: string;
  private timeout: number;
  private headers: Record<string, string>;

  // API namespaces
  public readonly debates: DebatesAPI;
  public readonly graphDebates: GraphDebatesAPI;
  public readonly matrixDebates: MatrixDebatesAPI;
  public readonly agents: AgentsAPI;
  public readonly verification: VerificationAPI;
  public readonly gauntlet: GauntletAPI;
  public readonly teamSelection: TeamSelectionAPI;
  public readonly controlPlane: ControlPlaneAPI;

  constructor(baseUrl: string = 'http://localhost:8080', options: AragoraClientOptions = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = options.apiKey;
    this.timeout = options.timeout ?? 30000;
    this.headers = {
      'Content-Type': 'application/json',
      'User-Agent': 'aragora-client-js/2.1.13',
      ...(options.apiKey && { Authorization: `Bearer ${options.apiKey}` }),
      ...options.headers,
    };

    // Initialize API namespaces
    this.debates = new DebatesAPI(this);
    this.graphDebates = new GraphDebatesAPI(this);
    this.matrixDebates = new MatrixDebatesAPI(this);
    this.agents = new AgentsAPI(this);
    this.verification = new VerificationAPI(this);
    this.gauntlet = new GauntletAPI(this);
    this.teamSelection = new TeamSelectionAPI(this);
    this.controlPlane = new ControlPlaneAPI(this);
  }

  /**
   * Get server health status.
   */
  async health(): Promise<HealthStatus> {
    return this.get<HealthStatus>('/api/v1/health');
  }

  /**
   * Make a GET request.
   */
  async get<T>(
    path: string,
    params?: Record<string, string | number | boolean>
  ): Promise<T> {
    const url = new URL(path, this.baseUrl);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        url.searchParams.set(key, String(value));
      });
    }

    const response = await fetch(url.toString(), {
      method: 'GET',
      headers: this.headers,
      signal: AbortSignal.timeout(this.timeout),
    });

    return this.handleResponse<T>(response);
  }

  /**
   * Make a POST request.
   */
  async post<T>(
    path: string,
    data: Record<string, unknown>
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(data),
      signal: AbortSignal.timeout(this.timeout),
    });

    return this.handleResponse<T>(response);
  }

  /**
   * Make a DELETE request.
   */
  async delete(path: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: 'DELETE',
      headers: this.headers,
      signal: AbortSignal.timeout(this.timeout),
    });

    if (!response.ok) {
      await this.handleError(response);
    }
  }

  private async handleResponse<T>(response: Response): Promise<T> {
    if (!response.ok) {
      await this.handleError(response);
    }

    const data = await response.json();
    return data as T;
  }

  private async handleError(response: Response): Promise<never> {
    let errorData: Record<string, unknown> = {};
    try {
      errorData = (await response.json()) as Record<string, unknown>;
    } catch {
      // Ignore JSON parse errors
    }

    if (response.status === 401) {
      throw new AragoraError('Authentication failed', { status: 401 });
    }
    if (response.status === 404) {
      throw new AragoraError('Resource not found', { status: 404 });
    }
    if (response.status === 400) {
      throw new AragoraError(
        (errorData.error as string) ?? 'Validation error',
        { status: 400, details: errorData.details as Record<string, unknown> }
      );
    }

    throw new AragoraError(
      (errorData.error as string) ?? `Request failed with status ${response.status}`,
      {
        status: response.status,
        code: errorData.code as string,
        details: errorData.details as Record<string, unknown>,
      }
    );
  }
}
