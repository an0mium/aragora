/**
 * Type definitions for Graph Debate components.
 * These types mirror the backend graph.py structures.
 */

import * as d3Force from 'd3-force';

export interface DebateNode {
  id: string;
  node_type: string;
  agent_id: string;
  content: string;
  timestamp: string;
  parent_ids: string[];
  child_ids: string[];
  branch_id: string | null;
  confidence: number;
  agreement_scores: Record<string, number>;
  claims: string[];
  evidence: string[];
  metadata: Record<string, unknown>;
  hash: string;
}

export interface Branch {
  id: string;
  name: string;
  reason: string;
  start_node_id: string;
  end_node_id: string | null;
  hypothesis: string;
  confidence: number;
  is_active: boolean;
  is_merged: boolean;
  merged_into: string | null;
  node_count: number;
  total_agreement: number;
}

export interface MergeResult {
  merged_node_id: string;
  source_branch_ids: string[];
  strategy: string;
  synthesis: string;
  confidence: number;
  insights_preserved: string[];
  conflicts_resolved: string[];
}

export interface GraphDebate {
  debate_id: string;
  task: string;
  graph: {
    debate_id: string;
    root_id: string;
    main_branch_id: string;
    created_at: string;
    nodes: Record<string, DebateNode>;
    branches: Record<string, Branch>;
    merge_history: MergeResult[];
    policy: {
      disagreement_threshold: number;
      uncertainty_threshold: number;
      max_branches: number;
      max_depth: number;
    };
  };
  branches: Branch[];
  merge_results: MergeResult[];
  node_count: number;
  branch_count: number;
}

// D3 Force simulation types
export interface SimulationNode extends d3Force.SimulationNodeDatum {
  id: string;
  node: DebateNode;
  depth: number;
  fx?: number | null;
  fy?: number | null;
}

export interface SimulationLink extends d3Force.SimulationLinkDatum<SimulationNode> {
  source: SimulationNode | string;
  target: SimulationNode | string;
  branchId: string;
}

export interface NodePosition {
  x: number;
  y: number;
  node: DebateNode;
  vx?: number;
  vy?: number;
}
