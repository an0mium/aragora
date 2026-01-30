/**
 * Graph Debates API
 *
 * Handles graph-structured debates with automatic branching,
 * merging, and visualization capabilities.
 */

import { BaseAPI, HttpClient } from './base';

// =============================================================================
// Types
// =============================================================================

export type MergeStrategy = 'synthesis' | 'vote' | 'best';

export interface BranchPolicy {
  min_disagreement?: number;
  max_branches?: number;
  merge_strategy?: MergeStrategy;
  auto_merge?: boolean;
}

export interface GraphDebateRequest {
  task: string;
  agents: string[];
  max_rounds?: number;
  branch_policy?: BranchPolicy;
}

export interface GraphNode {
  id: string;
  type: 'proposal' | 'critique' | 'revision' | 'branch_point' | 'merge_point';
  content: string;
  agent: string;
  parent_id?: string;
  branch_id?: string;
  round: number;
  metadata?: Record<string, unknown>;
}

export interface Branch {
  id: string;
  root_node_id: string;
  current_node_id: string;
  status: 'active' | 'merged' | 'abandoned';
  reason?: string;
  node_count: number;
  created_at: string;
}

export interface MergeResult {
  merged_node_id: string;
  source_branch_ids: string[];
  strategy: MergeStrategy;
  conflicts_resolved: number;
  insights_preserved: number;
}

export interface DebateGraph {
  nodes: Record<string, GraphNode>;
  edges: Array<{ source: string; target: string; type: string }>;
  root_id: string;
  metadata?: Record<string, unknown>;
}

export interface GraphDebateResponse {
  debate_id: string;
  task: string;
  graph: DebateGraph;
  branches: Branch[];
  merge_results: MergeResult[];
  node_count: number;
  branch_count: number;
}

export interface GraphDebate {
  debate_id: string;
  task: string;
  status: 'running' | 'completed' | 'failed';
  graph?: DebateGraph;
  branches?: Branch[];
  merge_results?: MergeResult[];
  created_at: string;
  completed_at?: string;
}

export interface BranchesResponse {
  debate_id: string;
  branches: Branch[];
}

export interface NodesResponse {
  debate_id: string;
  nodes: GraphNode[];
}

// =============================================================================
// Graph Debates API Class
// =============================================================================

export class GraphDebatesAPI extends BaseAPI {
  constructor(http: HttpClient) {
    super(http);
  }

  // ===========================================================================
  // Graph Debate Operations
  // ===========================================================================

  /**
   * Run a new graph-structured debate with automatic branching
   *
   * Graph debates automatically create branches when agents disagree
   * significantly, allowing parallel exploration of different approaches.
   * Branches can be merged back together using various strategies.
   */
  async run(request: GraphDebateRequest): Promise<GraphDebateResponse> {
    return this.http.post('/api/v1/debates/graph', request);
  }

  /**
   * Get a graph debate by ID
   */
  async get(debateId: string): Promise<GraphDebate> {
    return this.http.get(`/api/v1/debates/graph/${debateId}`);
  }

  /**
   * Get all branches for a graph debate
   */
  async getBranches(debateId: string): Promise<BranchesResponse> {
    return this.http.get(`/api/v1/debates/graph/${debateId}/branches`);
  }

  /**
   * Get all nodes in a graph debate
   */
  async getNodes(debateId: string): Promise<NodesResponse> {
    return this.http.get(`/api/v1/debates/graph/${debateId}/nodes`);
  }

  // ===========================================================================
  // Convenience Methods
  // ===========================================================================

  /**
   * Run a simple graph debate with default settings
   */
  async runSimple(task: string, agents: string[]): Promise<GraphDebateResponse> {
    return this.run({ task, agents });
  }

  /**
   * Run a graph debate optimized for exploration (more branching)
   */
  async runExplorative(task: string, agents: string[]): Promise<GraphDebateResponse> {
    return this.run({
      task,
      agents,
      max_rounds: 10,
      branch_policy: {
        min_disagreement: 0.5,
        max_branches: 5,
        merge_strategy: 'synthesis',
        auto_merge: true,
      },
    });
  }

  /**
   * Run a graph debate optimized for convergence (less branching)
   */
  async runFocused(task: string, agents: string[]): Promise<GraphDebateResponse> {
    return this.run({
      task,
      agents,
      max_rounds: 5,
      branch_policy: {
        min_disagreement: 0.8,
        max_branches: 2,
        merge_strategy: 'vote',
        auto_merge: true,
      },
    });
  }
}
