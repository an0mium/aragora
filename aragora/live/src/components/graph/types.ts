/**
 * Type definitions for Graph Debate visualization.
 */

import type * as d3Force from 'd3-force';

// Types from the backend graph.py
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

// Branch color mapping
export const BRANCH_COLORS: Record<string, string> = {
  main: 'text-acid-green',
  'Branch-1': 'text-acid-cyan',
  'Branch-2': 'text-gold',
  'Branch-3': 'text-purple',
  'Branch-4': 'text-crimson',
};

export function getBranchColor(branchId: string): string {
  return BRANCH_COLORS[branchId] || 'text-text-muted';
}

export function getBranchBgColor(branchId: string): string {
  const colorMap: Record<string, string> = {
    main: 'bg-acid-green/20',
    'Branch-1': 'bg-acid-cyan/20',
    'Branch-2': 'bg-gold/20',
    'Branch-3': 'bg-purple/20',
    'Branch-4': 'bg-crimson/20',
  };
  return colorMap[branchId] || 'bg-surface';
}

export function getEdgeColor(branchId: string): string {
  const colorMap: Record<string, string> = {
    main: '#00ff00',
    'Branch-1': '#00ffff',
    'Branch-2': '#ffd700',
    'Branch-3': '#9966ff',
    'Branch-4': '#ff3333',
  };
  return colorMap[branchId] || '#666666';
}
