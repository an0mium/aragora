'use client';

import { memo, useCallback } from 'react';
import {
  PIPELINE_STAGE_CONFIG,
  type PipelineStageType,
  type IdeaType,
  type GoalType,
  type ActionType,
  type OrchType,
} from '../types';
import {
  InputField,
  SelectField,
  SliderField,
  CheckboxField,
} from './shared-fields';

/* -------------------------------------------------------------------------- */
/*  Props                                                                     */
/* -------------------------------------------------------------------------- */

interface PipelinePropertyEditorProps {
  node: Record<string, unknown> | null;
  stage: PipelineStageType;
  onUpdate: (updates: Record<string, unknown>) => void;
  onDelete: () => void;
  onShowProvenance?: () => void;
  readOnly?: boolean;
}

/* -------------------------------------------------------------------------- */
/*  Option lists                                                              */
/* -------------------------------------------------------------------------- */

const IDEA_TYPE_OPTIONS: Array<{ value: IdeaType; label: string }> = [
  { value: 'concept', label: 'Concept' },
  { value: 'cluster', label: 'Cluster' },
  { value: 'question', label: 'Question' },
  { value: 'insight', label: 'Insight' },
  { value: 'evidence', label: 'Evidence' },
  { value: 'assumption', label: 'Assumption' },
  { value: 'constraint', label: 'Constraint' },
];

const GOAL_TYPE_OPTIONS: Array<{ value: GoalType; label: string }> = [
  { value: 'goal', label: 'Goal' },
  { value: 'principle', label: 'Principle' },
  { value: 'strategy', label: 'Strategy' },
  { value: 'milestone', label: 'Milestone' },
  { value: 'metric', label: 'Metric' },
  { value: 'risk', label: 'Risk' },
];

const ACTION_TYPE_OPTIONS: Array<{ value: ActionType; label: string }> = [
  { value: 'task', label: 'Task' },
  { value: 'epic', label: 'Epic' },
  { value: 'checkpoint', label: 'Checkpoint' },
  { value: 'deliverable', label: 'Deliverable' },
  { value: 'dependency', label: 'Dependency' },
];

const ORCH_TYPE_OPTIONS: Array<{ value: OrchType; label: string }> = [
  { value: 'agent_task', label: 'Agent Task' },
  { value: 'debate', label: 'Debate' },
  { value: 'human_gate', label: 'Human Gate' },
  { value: 'parallel_fan', label: 'Parallel Fan' },
  { value: 'merge', label: 'Merge' },
  { value: 'verification', label: 'Verification' },
];

const PRIORITY_OPTIONS = [
  { value: 'high', label: 'High' },
  { value: 'medium', label: 'Medium' },
  { value: 'low', label: 'Low' },
];

/* -------------------------------------------------------------------------- */
/*  Stage sub-editors                                                         */
/* -------------------------------------------------------------------------- */

function IdeasEditor({
  node,
  onUpdate,
  readOnly,
}: {
  node: Record<string, unknown>;
  onUpdate: (updates: Record<string, unknown>) => void;
  readOnly?: boolean;
}) {
  if (readOnly) {
    return (
      <>
        <ReadOnlyField label="Idea Type" value={String(node.ideaType ?? '')} />
        <ReadOnlyField label="Full Content" value={String(node.fullContent ?? '')} />
        <ReadOnlyField label="Agent" value={String(node.agent ?? '')} />
      </>
    );
  }

  return (
    <>
      <SelectField
        label="Idea Type"
        value={String(node.ideaType ?? 'concept')}
        options={IDEA_TYPE_OPTIONS}
        onChange={(v) => onUpdate({ ideaType: v })}
      />
      <InputField
        label="Full Content"
        value={String(node.fullContent ?? '')}
        onChange={(v) => onUpdate({ fullContent: v })}
        placeholder="Describe this idea..."
        type="textarea"
      />
      <InputField
        label="Agent"
        value={String(node.agent ?? '')}
        onChange={(v) => onUpdate({ agent: v })}
        placeholder="Originating agent"
      />
    </>
  );
}

function GoalsEditor({
  node,
  onUpdate,
  readOnly,
}: {
  node: Record<string, unknown>;
  onUpdate: (updates: Record<string, unknown>) => void;
  readOnly?: boolean;
}) {
  const confidence = typeof node.confidence === 'number' ? node.confidence : 50;
  const tagsRaw = node.tags;
  const tagsStr = Array.isArray(tagsRaw)
    ? tagsRaw.join(', ')
    : typeof tagsRaw === 'string'
      ? tagsRaw
      : '';

  if (readOnly) {
    return (
      <>
        <ReadOnlyField label="Goal Type" value={String(node.goalType ?? '')} />
        <ReadOnlyField label="Description" value={String(node.description ?? '')} />
        <ReadOnlyField label="Priority" value={String(node.priority ?? '')} />
        <ReadOnlyField label="Confidence" value={`${confidence}%`} />
        <ReadOnlyField label="Tags" value={tagsStr} />
      </>
    );
  }

  return (
    <>
      <SelectField
        label="Goal Type"
        value={String(node.goalType ?? 'goal')}
        options={GOAL_TYPE_OPTIONS}
        onChange={(v) => onUpdate({ goalType: v })}
      />
      <InputField
        label="Description"
        value={String(node.description ?? '')}
        onChange={(v) => onUpdate({ description: v })}
        placeholder="Describe this goal..."
        type="textarea"
      />
      <SelectField
        label="Priority"
        value={String(node.priority ?? 'medium')}
        options={PRIORITY_OPTIONS}
        onChange={(v) => onUpdate({ priority: v })}
      />
      <SliderField
        label="Confidence"
        value={confidence}
        onChange={(v) => onUpdate({ confidence: v })}
        min={0}
        max={100}
        formatLabel={(v) => `${v}%`}
      />
      <InputField
        label="Tags (comma-separated)"
        value={tagsStr}
        onChange={(v) =>
          onUpdate({
            tags: v
              .split(',')
              .map((t) => t.trim())
              .filter(Boolean),
          })
        }
        placeholder="e.g., ux, backend, critical"
      />
    </>
  );
}

function ActionsEditor({
  node,
  onUpdate,
  readOnly,
}: {
  node: Record<string, unknown>;
  onUpdate: (updates: Record<string, unknown>) => void;
  readOnly?: boolean;
}) {
  if (readOnly) {
    return (
      <>
        <ReadOnlyField label="Step Type" value={String(node.stepType ?? '')} />
        <ReadOnlyField label="Description" value={String(node.description ?? '')} />
        <ReadOnlyField label="Optional" value={node.optional ? 'Yes' : 'No'} />
        <ReadOnlyField
          label="Timeout (seconds)"
          value={node.timeoutSeconds != null ? String(node.timeoutSeconds) : ''}
        />
      </>
    );
  }

  return (
    <>
      <SelectField
        label="Step Type"
        value={String(node.stepType ?? 'task')}
        options={ACTION_TYPE_OPTIONS}
        onChange={(v) => onUpdate({ stepType: v })}
      />
      <InputField
        label="Description"
        value={String(node.description ?? '')}
        onChange={(v) => onUpdate({ description: v })}
        placeholder="What does this step do?"
        type="textarea"
      />
      <CheckboxField
        label="Optional"
        checked={!!node.optional}
        onChange={(v) => onUpdate({ optional: v })}
        description="Can be skipped without blocking the pipeline"
      />
      <InputField
        label="Timeout (seconds)"
        value={node.timeoutSeconds != null ? String(node.timeoutSeconds) : ''}
        onChange={(v) => onUpdate({ timeoutSeconds: v ? parseInt(v, 10) || 0 : undefined })}
        placeholder="e.g., 300"
        type="number"
      />
    </>
  );
}

function OrchestrationEditor({
  node,
  onUpdate,
  readOnly,
}: {
  node: Record<string, unknown>;
  onUpdate: (updates: Record<string, unknown>) => void;
  readOnly?: boolean;
}) {
  const capsRaw = node.capabilities;
  const capsStr = Array.isArray(capsRaw)
    ? capsRaw.join(', ')
    : typeof capsRaw === 'string'
      ? capsRaw
      : '';

  if (readOnly) {
    return (
      <>
        <ReadOnlyField label="Orchestration Type" value={String(node.orchType ?? '')} />
        <ReadOnlyField label="Assigned Agent" value={String(node.assignedAgent ?? '')} />
        <ReadOnlyField label="Agent Type" value={String(node.agentType ?? '')} />
        <ReadOnlyField label="Capabilities" value={capsStr} />
      </>
    );
  }

  return (
    <>
      <SelectField
        label="Orchestration Type"
        value={String(node.orchType ?? 'agent_task')}
        options={ORCH_TYPE_OPTIONS}
        onChange={(v) => onUpdate({ orchType: v })}
      />
      <InputField
        label="Assigned Agent"
        value={String(node.assignedAgent ?? '')}
        onChange={(v) => onUpdate({ assignedAgent: v })}
        placeholder="e.g., claude, gpt-4"
      />
      <InputField
        label="Agent Type"
        value={String(node.agentType ?? '')}
        onChange={(v) => onUpdate({ agentType: v })}
        placeholder="e.g., reviewer, coder"
      />
      <InputField
        label="Capabilities (comma-separated)"
        value={capsStr}
        onChange={(v) =>
          onUpdate({
            capabilities: v
              .split(',')
              .map((c) => c.trim())
              .filter(Boolean),
          })
        }
        placeholder="e.g., code_review, testing"
      />
    </>
  );
}

/* -------------------------------------------------------------------------- */
/*  ReadOnlyField helper                                                      */
/* -------------------------------------------------------------------------- */

function ReadOnlyField({ label, value }: { label: string; value: string }) {
  return (
    <div className="mb-3">
      <label className="block text-xs text-text-muted mb-1">{label}</label>
      <p className="text-sm text-text font-mono bg-bg border border-border rounded px-2 py-1.5 opacity-70">
        {value || <span className="text-text-muted italic">--</span>}
      </p>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*  Main component                                                            */
/* -------------------------------------------------------------------------- */

export const PipelinePropertyEditor = memo(function PipelinePropertyEditor({
  node,
  stage,
  onUpdate,
  onDelete,
  onShowProvenance,
  readOnly,
}: PipelinePropertyEditorProps) {
  const handleLabelChange = useCallback(
    (label: string) => onUpdate({ label }),
    [onUpdate],
  );

  /* -- Empty state -------------------------------------------------------- */
  if (!node) {
    return (
      <div className="w-72 flex-shrink-0 bg-surface border-l border-border h-full overflow-y-auto p-4">
        <div className="text-center text-text-muted py-8">
          <p className="text-sm font-mono">Select a node to edit its properties.</p>
        </div>
      </div>
    );
  }

  const stageConfig = PIPELINE_STAGE_CONFIG[stage];

  return (
    <div className="w-72 flex-shrink-0 bg-surface border-l border-border h-full overflow-y-auto p-4">
      {/* Stage-colored header bar */}
      <div
        className="flex items-center gap-2 mb-4 pb-3 border-b border-border"
        style={{ borderBottomColor: stageConfig.primary }}
      >
        <div
          className="w-1 h-8 rounded-full"
          style={{ backgroundColor: stageConfig.primary }}
        />
        <div>
          <h3 className="text-sm font-mono font-bold text-text uppercase">
            {stageConfig.label} Properties
          </h3>
          <p className="text-xs text-text-muted font-mono">
            Stage: <span style={{ color: stageConfig.primary }}>{stage}</span>
          </p>
        </div>
      </div>

      {/* Common field: Label */}
      {readOnly ? (
        <ReadOnlyField label="Label" value={String(node.label ?? '')} />
      ) : (
        <InputField
          label="Label"
          value={String(node.label ?? '')}
          onChange={handleLabelChange}
          placeholder="Node label"
        />
      )}

      {/* Stage-specific fields */}
      <div className="mt-4 pt-4 border-t border-border">
        {stage === 'ideas' && (
          <IdeasEditor node={node} onUpdate={onUpdate} readOnly={readOnly} />
        )}
        {stage === 'goals' && (
          <GoalsEditor node={node} onUpdate={onUpdate} readOnly={readOnly} />
        )}
        {stage === 'actions' && (
          <ActionsEditor node={node} onUpdate={onUpdate} readOnly={readOnly} />
        )}
        {stage === 'orchestration' && (
          <OrchestrationEditor node={node} onUpdate={onUpdate} readOnly={readOnly} />
        )}
      </div>

      {/* Bottom actions */}
      <div className="mt-6 pt-4 border-t border-border space-y-2">
        {onShowProvenance && (
          <button
            onClick={onShowProvenance}
            className="w-full px-4 py-2 bg-surface border border-border text-text font-mono text-sm hover:bg-bg transition-colors rounded"
          >
            View Provenance
          </button>
        )}
        {!readOnly && (
          <button
            onClick={onDelete}
            className="w-full px-4 py-2 bg-red-500/20 border border-red-500/50 text-red-400 font-mono text-sm hover:bg-red-500/30 transition-colors rounded"
          >
            Delete Node
          </button>
        )}
      </div>
    </div>
  );
});

export default PipelinePropertyEditor;
