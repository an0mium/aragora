/**
 * Types for Evidence Visualizer components
 */

export interface DissentRecord {
  topic: string;
  majority_view: string;
  dissenting_view: string;
  dissenting_agent: string;
  confidence: number;
  reasoning?: string;
}

export interface ContrarianView {
  agent: string;
  position: string;
  confidence: number;
  reasoning?: string;
  debate_id?: string;
}

export interface RiskWarning {
  domain: string;
  risk_type: string;
  severity: string;
  description: string;
  mitigation?: string;
  detected_at: string;
}

export interface ConsensusStats {
  total_topics: number;
  high_confidence_count: number;
  avg_confidence: number;
  total_dissents: number;
  domains: string[];
  by_strength: Record<string, number>;
}

export interface EvidenceCitation {
  source: string;
  claim: string;
  agent: string;
  round: number;
  confidence?: number;
  source_type?: string;
  freshness?: number;
  authority?: number;
  reliability_score?: number;
  url?: string;
}

export interface GraphNode {
  id: string;
  agent: string;
  content: string;
  type: 'argument' | 'rebuttal' | 'synthesis' | 'evidence' | 'root';
  parent_id?: string;
  children?: string[];
  confidence?: number;
  branch_id?: string;
}

// Source type configuration with icons and colors
export const SOURCE_TYPE_CONFIG: Record<string, { icon: string; color: string; label: string }> = {
  agent_generated: { icon: '🤖', color: 'text-acid-cyan', label: 'Agent' },
  user_provided: { icon: '👤', color: 'text-acid-green', label: 'User' },
  external_api: { icon: '🔌', color: 'text-acid-yellow', label: 'API' },
  web_search: { icon: '🌐', color: 'text-blue-400', label: 'Web' },
  document: { icon: '📄', color: 'text-orange-400', label: 'Doc' },
  code_analysis: { icon: '💻', color: 'text-purple-400', label: 'Code' },
  database: { icon: '🗄️', color: 'text-emerald-400', label: 'DB' },
  computation: { icon: '⚙️', color: 'text-pink-400', label: 'Compute' },
  synthesis: { icon: '🔀', color: 'text-acid-cyan', label: 'Synthesis' },
  arxiv: { icon: '📚', color: 'text-red-400', label: 'ArXiv' },
  hackernews: { icon: '📰', color: 'text-orange-500', label: 'HN' },
  github: { icon: '🐙', color: 'text-gray-400', label: 'GitHub' },
  unknown: { icon: '❓', color: 'text-text-muted', label: 'Unknown' },
};

export const SEVERITY_COLORS: Record<string, { text: string; bg: string }> = {
  critical: { text: 'text-acid-red', bg: 'bg-acid-red/20' },
  high: { text: 'text-acid-yellow', bg: 'bg-acid-yellow/20' },
  medium: { text: 'text-acid-cyan', bg: 'bg-acid-cyan/20' },
  low: { text: 'text-acid-green', bg: 'bg-acid-green/20' },
};
