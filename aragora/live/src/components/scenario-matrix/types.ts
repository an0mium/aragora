/**
 * Types for Scenario Matrix components.
 */

export interface ScenarioResult {
  scenario_name: string;
  parameters: Record<string, unknown>;
  constraints: string[];
  is_baseline: boolean;
  winner: string | null;
  final_answer: string;
  confidence: number;
  consensus_reached: boolean;
  rounds_used: number;
}

export interface ConditionalConclusion {
  condition: string;
  parameters: Record<string, unknown>;
  conclusion: string;
  confidence: number;
}

export interface ComparisonMatrix {
  scenarios: string[];
  consensus_rate: number;
  avg_confidence: number;
  avg_rounds: number;
}

export interface MatrixDebateResult {
  matrix_id: string;
  task: string;
  scenario_count: number;
  results: ScenarioResult[];
  universal_conclusions: string[];
  conditional_conclusions: ConditionalConclusion[];
  comparison_matrix: ComparisonMatrix;
}

export interface ScenarioInput {
  name: string;
  parameters: Record<string, string>;
  constraints: string[];
  is_baseline: boolean;
}

export interface FilterState {
  consensusOnly: boolean;
  minConfidence: number;
  searchTerm: string;
}

export type ViewMode = 'grid' | 'list' | 'compare';
