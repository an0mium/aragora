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

export type PipelineNodeData =
  | IdeaNodeData
  | GoalNodeData
  | ActionNodeData
  | OrchestrationNodeData;

// =============================================================================
// Node Type Configurations (per-stage palette items)
// =============================================================================

export interface NodeTypeConfig {
  label: string;
  icon: string;
  description: string;
  color: string;
  borderColor: string;
}

export const PIPELINE_NODE_TYPE_CONFIGS: Record<PipelineStageType, Record<string, NodeTypeConfig>> = {
  ideas: {
    concept: { label: 'Concept', icon: '💡', description: 'A raw idea or concept', color: 'bg-indigo-500/20', borderColor: 'border-indigo-500' },
    cluster: { label: 'Cluster', icon: '🔗', description: 'Group of related ideas', color: 'bg-indigo-500/20', borderColor: 'border-indigo-400' },
    question: { label: 'Question', icon: '❓', description: 'Open question to resolve', color: 'bg-indigo-500/20', borderColor: 'border-indigo-300' },
    insight: { label: 'Insight', icon: '🔍', description: 'Key insight or finding', color: 'bg-indigo-500/20', borderColor: 'border-indigo-500' },
    evidence: { label: 'Evidence', icon: '📊', description: 'Supporting evidence', color: 'bg-indigo-500/20', borderColor: 'border-indigo-400' },
    assumption: { label: 'Assumption', icon: '⚠️', description: 'Assumption to validate', color: 'bg-indigo-500/20', borderColor: 'border-indigo-300' },
    constraint: { label: 'Constraint', icon: '🚧', description: 'Known constraint', color: 'bg-indigo-500/20', borderColor: 'border-indigo-400' },
  },
  goals: {
    goal: { label: 'Goal', icon: '🎯', description: 'Concrete goal to achieve', color: 'bg-emerald-500/20', borderColor: 'border-emerald-500' },
    principle: { label: 'Principle', icon: '📐', description: 'Guiding principle', color: 'bg-emerald-500/20', borderColor: 'border-emerald-400' },
    strategy: { label: 'Strategy', icon: '♟️', description: 'Strategic approach', color: 'bg-emerald-500/20', borderColor: 'border-emerald-500' },
    milestone: { label: 'Milestone', icon: '🏁', description: 'Key milestone', color: 'bg-emerald-500/20', borderColor: 'border-emerald-400' },
    metric: { label: 'Metric', icon: '📈', description: 'Measurable metric', color: 'bg-emerald-500/20', borderColor: 'border-emerald-300' },
    risk: { label: 'Risk', icon: '⚡', description: 'Identified risk', color: 'bg-emerald-500/20', borderColor: 'border-emerald-500' },
  },
  actions: {
    task: { label: 'Task', icon: '✅', description: 'Actionable task', color: 'bg-amber-500/20', borderColor: 'border-amber-500' },
    epic: { label: 'Epic', icon: '📋', description: 'Large body of work', color: 'bg-amber-500/20', borderColor: 'border-amber-400' },
    checkpoint: { label: 'Checkpoint', icon: '🔖', description: 'Verification checkpoint', color: 'bg-amber-500/20', borderColor: 'border-amber-500' },
    deliverable: { label: 'Deliverable', icon: '📦', description: 'Tangible deliverable', color: 'bg-amber-500/20', borderColor: 'border-amber-400' },
    dependency: { label: 'Dependency', icon: '🔄', description: 'External dependency', color: 'bg-amber-500/20', borderColor: 'border-amber-300' },
  },
  orchestration: {
    agent_task: { label: 'Agent Task', icon: '🤖', description: 'Task assigned to an agent', color: 'bg-pink-500/20', borderColor: 'border-pink-500' },
    debate: { label: 'Debate', icon: '💬', description: 'Multi-agent debate', color: 'bg-pink-500/20', borderColor: 'border-pink-400' },
    human_gate: { label: 'Human Gate', icon: '👤', description: 'Human approval required', color: 'bg-pink-500/20', borderColor: 'border-pink-500' },
    parallel_fan: { label: 'Parallel Fan', icon: '🔀', description: 'Parallel execution', color: 'bg-pink-500/20', borderColor: 'border-pink-400' },
    merge: { label: 'Merge', icon: '🔁', description: 'Merge parallel results', color: 'bg-pink-500/20', borderColor: 'border-pink-300' },
    verification: { label: 'Verification', icon: '🔬', description: 'Verify results', color: 'bg-pink-500/20', borderColor: 'border-pink-500' },
  },
};

export function getDefaultPipelineNodeData(stage: PipelineStageType, subtype: string): PipelineNodeData {
  switch (stage) {
    case 'ideas':
      return { label: PIPELINE_NODE_TYPE_CONFIGS.ideas[subtype]?.label || 'New Idea', ideaType: subtype as IdeaType, contentHash: '', fullContent: '' };
    case 'goals':
      return { label: PIPELINE_NODE_TYPE_CONFIGS.goals[subtype]?.label || 'New Goal', goalType: subtype as GoalType, description: '', priority: 'medium' };
    case 'actions':
      return { label: PIPELINE_NODE_TYPE_CONFIGS.actions[subtype]?.label || 'New Action', stepType: subtype as ActionType, description: '', optional: false };
    case 'orchestration':
      return { label: PIPELINE_NODE_TYPE_CONFIGS.orchestration[subtype]?.label || 'New Node', orchType: subtype as OrchType, assignedAgent: '', capabilities: [] };
  }
}

export function getNodeTypeForStage(stage: PipelineStageType): string {
  const map: Record<PipelineStageType, string> = { ideas: 'ideaNode', goals: 'goalNode', actions: 'actionNode', orchestration: 'orchestrationNode' };
  return map[stage];
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
