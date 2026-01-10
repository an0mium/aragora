export interface EmergentTrait {
  agent: string;
  trait: string;
  domain: string;
  confidence: number;
  evidence: string[];
  detected_at: string;
}

export interface CrossPollination {
  source_agent: string;
  target_agent: string;
  trait: string;
  expected_improvement: number;
  rationale: string;
}

export interface GenesisStats {
  total_events: number;
  total_births: number;
  total_deaths: number;
  net_population_change: number;
  avg_fitness_change_recent: number;
  integrity_verified: boolean;
  event_counts: Record<string, number>;
}

export interface GenesisEvent {
  event_type: string;
  genome_id: string;
  parent_id?: string;
  fitness_change?: number;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface Genome {
  genome_id: string;
  agent_name: string;
  generation: number;
  fitness: number;
  parent_id?: string;
  prompt_hash?: string;
  created_at: string;
}

export interface CritiquePattern {
  pattern: string;
  issue_type: string;
  suggested_rebuttal: string;
  success_rate: number;
  usage_count: number;
}

export interface LaboratoryPanelProps {
  apiBase?: string;
}

export type LaboratoryTab = 'traits' | 'pollinations' | 'evolution' | 'patterns';
