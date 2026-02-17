/**
 * Pipeline Canvas Types
 *
 * TypeScript types mirroring aragora/canvas/stages.py for the
 * four-stage idea-to-execution pipeline.
 */

// =============================================================================
// Stage Types
// =============================================================================

export type PipelineStageType = 'ideas' | 'goals' | 'actions' | 'orchestration';

export type IdeaType = 'concept' | 'cluster' | 'question' | 'insight' | 'evidence' | 'assumption' | 'constraint';
export type GoalType = 'goal' | 'principle' | 'strategy' | 'milestone' | 'metric' | 'risk';
export type ActionType = 'task' | 'epic' | 'checkpoint' | 'deliverable' | 'dependency';
export type OrchType = 'agent_task' | 'debate' | 'human_gate' | 'parallel_fan' | 'merge' | 'verification';

// =============================================================================
// Node Data Interfaces
// =============================================================================

export interface IdeaNodeData {
  label: string;
  ideaType: IdeaType;
  agent?: string;
  fullContent?: string;
  contentHash: string;
}

export interface GoalNodeData {
  label: string;
  goalType: GoalType;
  description: string;
  priority: 'high' | 'medium' | 'low';
  measurable?: boolean;
}

export interface ActionNodeData {
  label: string;
  stepType: ActionType;
  description?: string;
  optional?: boolean;
  timeoutSeconds?: number;
}

export interface OrchestrationNodeData {
  label: string;
  orchType: OrchType;
  assignedAgent?: string;
  capabilities?: string[];
  agentType?: string;
}

// =============================================================================
// Stage Configuration
// =============================================================================

export interface StageConfig {
  label: string;
  primary: string;
  secondary: string;
  accent: string;
  icon: string;
}

export const PIPELINE_STAGE_CONFIG: Record<PipelineStageType, StageConfig> = {
  ideas: {
    label: 'Ideas',
    primary: '#818cf8',
    secondary: '#c7d2fe',
    accent: '#4f46e5',
    icon: 'lightbulb',
  },
  goals: {
    label: 'Goals',
    primary: '#34d399',
    secondary: '#a7f3d0',
    accent: '#059669',
    icon: 'target',
  },
  actions: {
    label: 'Actions',
    primary: '#fbbf24',
    secondary: '#fde68a',
    accent: '#d97706',
    icon: 'list',
  },
  orchestration: {
    label: 'Orchestration',
    primary: '#f472b6',
    secondary: '#fbcfe8',
    accent: '#db2777',
    icon: 'cpu',
  },
};

// =============================================================================
// API Response Types
// =============================================================================

export interface ReactFlowData {
  nodes: Array<{
    id: string;
    type: string;
    position: { x: number; y: number };
    data: Record<string, unknown>;
    style?: Record<string, string>;
  }>;
  edges: Array<{
    id: string;
    source: string;
    target: string;
    type: string;
    label?: string;
    animated?: boolean;
    style?: Record<string, string>;
  }>;
  metadata: Record<string, unknown>;
}

export interface PipelineResultResponse {
  pipeline_id: string;
  ideas: ReactFlowData | null;
  goals: Record<string, unknown> | null;
  actions: ReactFlowData | null;
  orchestration: ReactFlowData | null;
  transitions: Array<Record<string, unknown>>;
  provenance_count: number;
  stage_status: Record<PipelineStageType, string>;
  integrity_hash: string;
}
