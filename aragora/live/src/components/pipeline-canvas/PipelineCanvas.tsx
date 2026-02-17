'use client';

import { useCallback, useState, useMemo } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  MiniMap,
  useNodesState,
  useEdgesState,
  type NodeTypes,
  Panel,
  BackgroundVariant,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { IdeaNode, GoalNode, ActionNode, OrchestrationNode } from './nodes';
import { StageNavigator } from './StageNavigator';
import {
  PIPELINE_STAGE_CONFIG,
  type PipelineStageType,
  type PipelineResultResponse,
  type ReactFlowData,
} from './types';

const nodeTypes: NodeTypes = {
  ideaNode: IdeaNode,
  goalNode: GoalNode,
  actionNode: ActionNode,
  orchestrationNode: OrchestrationNode,
};

interface PipelineCanvasProps {
  pipelineId?: string;
  initialData?: PipelineResultResponse;
  onStageAdvance?: (pipelineId: string, stage: PipelineStageType) => void;
  readOnly?: boolean;
}

function getStageData(
  data: PipelineResultResponse | undefined,
  stage: PipelineStageType
): ReactFlowData | null {
  if (!data) return null;
  switch (stage) {
    case 'ideas':
      return data.ideas;
    case 'actions':
      return data.actions;
    case 'orchestration':
      return data.orchestration;
    case 'goals':
      // Goals are structured data, not ReactFlow — show ideas as fallback
      return data.ideas;
    default:
      return null;
  }
}

export function PipelineCanvas({
  pipelineId,
  initialData,
  onStageAdvance,
  readOnly = false,
}: PipelineCanvasProps) {
  const [activeStage, setActiveStage] = useState<PipelineStageType>('ideas');
  const [data] = useState(initialData);

  const stageStatus = useMemo(
    () =>
      data?.stage_status ?? {
        ideas: 'pending',
        goals: 'pending',
        actions: 'pending',
        orchestration: 'pending',
      },
    [data]
  );

  const stageData = useMemo(() => getStageData(data, activeStage), [data, activeStage]);

  const [nodes, , onNodesChange] = useNodesState(stageData?.nodes ?? []);
  const [edges, , onEdgesChange] = useEdgesState(stageData?.edges ?? []);

  const handleStageSelect = useCallback((stage: PipelineStageType) => {
    setActiveStage(stage);
  }, []);

  const handleAdvance = useCallback(
    (stage: PipelineStageType) => {
      if (pipelineId && onStageAdvance) {
        onStageAdvance(pipelineId, stage);
      }
    },
    [pipelineId, onStageAdvance]
  );

  const miniMapNodeColor = useCallback((node: { type?: string }) => {
    switch (node.type) {
      case 'ideaNode':
        return PIPELINE_STAGE_CONFIG.ideas.primary;
      case 'goalNode':
        return PIPELINE_STAGE_CONFIG.goals.primary;
      case 'actionNode':
        return PIPELINE_STAGE_CONFIG.actions.primary;
      case 'orchestrationNode':
        return PIPELINE_STAGE_CONFIG.orchestration.primary;
      default:
        return '#6b7280';
    }
  }, []);

  const stageConfig = PIPELINE_STAGE_CONFIG[activeStage];

  return (
    <div className="flex flex-col h-full bg-bg">
      {/* Stage Navigator */}
      <div className="flex justify-center p-2">
        <StageNavigator
          stageStatus={stageStatus}
          activeStage={activeStage}
          onStageSelect={handleStageSelect}
          onAdvance={readOnly ? undefined : handleAdvance}
          readOnly={readOnly}
        />
      </div>

      {/* Canvas */}
      <div className="flex-1">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={readOnly ? undefined : onNodesChange}
          onEdgesChange={readOnly ? undefined : onEdgesChange}
          nodeTypes={nodeTypes}
          fitView
          snapToGrid
          snapGrid={[16, 16]}
          defaultEdgeOptions={{
            animated: true,
            style: { stroke: stageConfig.primary, strokeWidth: 2 },
          }}
          proOptions={{ hideAttribution: true }}
        >
          <Background
            variant={BackgroundVariant.Dots}
            gap={16}
            size={1}
            color="#333"
          />
          <Controls
            className="bg-surface border border-border rounded"
            showInteractive={!readOnly}
          />
          <MiniMap
            className="bg-surface border border-border rounded"
            nodeColor={miniMapNodeColor}
          />

          {/* Stats panel */}
          <Panel position="bottom-left" className="bg-surface/90 border border-border rounded p-2">
            <div className="text-xs font-mono text-text-muted">
              <span className="text-text">{nodes.length}</span> nodes |{' '}
              <span className="text-text">{edges.length}</span> edges |{' '}
              <span style={{ color: stageConfig.primary }} className="uppercase font-bold">
                {stageConfig.label}
              </span>
            </div>
          </Panel>

          {/* Pipeline ID */}
          {pipelineId && (
            <Panel position="top-right" className="bg-surface/90 border border-border rounded p-2">
              <div className="text-xs font-mono text-text-muted">
                Pipeline: <span className="text-text">{pipelineId}</span>
              </div>
            </Panel>
          )}
        </ReactFlow>
      </div>
    </div>
  );
}

export default PipelineCanvas;
