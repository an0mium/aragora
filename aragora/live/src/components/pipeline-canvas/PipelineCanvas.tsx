'use client';

import { useCallback, useState, useMemo, useEffect } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  MiniMap,
  useNodesState,
  useEdgesState,
  useReactFlow,
  ReactFlowProvider,
  type NodeTypes,
  type Node,
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

type ViewMode = PipelineStageType | 'all';

interface PipelineCanvasProps {
  pipelineId?: string;
  initialData?: PipelineResultResponse;
  onStageAdvance?: (pipelineId: string, stage: PipelineStageType) => void;
  readOnly?: boolean;
}

/** Convert goals structured data into ReactFlow nodes/edges for stage 2 */
function goalsToReactFlow(goalsData: Record<string, unknown> | null): ReactFlowData | null {
  if (!goalsData) return null;
  const goals = (goalsData.goals || []) as Array<Record<string, unknown>>;
  if (goals.length === 0) return null;

  const nodes = goals.map((goal, i) => ({
    id: goal.id as string || `goal-${i}`,
    type: 'goalNode',
    position: { x: (i % 3) * 300, y: Math.floor(i / 3) * 180 },
    data: {
      label: goal.title as string || 'Untitled',
      goalType: goal.type as string || 'goal',
      goal_type: goal.type as string || 'goal',
      description: goal.description as string || '',
      priority: goal.priority as string || 'medium',
      confidence: goal.confidence as number | undefined,
      stage: 'goals',
      rf_type: 'goalNode',
    },
  }));

  const edges: ReactFlowData['edges'] = [];
  goals.forEach((goal) => {
    const deps = (goal.dependencies || []) as string[];
    deps.forEach((depId) => {
      edges.push({
        id: `dep-${depId}-${goal.id}`,
        source: depId,
        target: goal.id as string,
        type: 'smoothstep',
        label: 'requires',
        animated: true,
      });
    });
  });

  return { nodes, edges, metadata: { stage: 'goals', canvas_name: 'Goal Map' } };
}

/** Collect all nodes/edges across all stages for "All" view */
function getAllStagesData(data: PipelineResultResponse): ReactFlowData {
  const allNodes: ReactFlowData['nodes'] = [];
  const allEdges: ReactFlowData['edges'] = [];

  // Stage offsets for layout
  const stageOffsetX: Record<string, number> = { ideas: 0, goals: 600, actions: 1200, orchestration: 1800 };

  const addStage = (rfData: ReactFlowData | null, stage: string) => {
    if (!rfData) return;
    const offsetX = stageOffsetX[stage] || 0;
    rfData.nodes.forEach((n) => {
      allNodes.push({ ...n, position: { x: n.position.x + offsetX, y: n.position.y } });
    });
    allEdges.push(...rfData.edges);
  };

  addStage(data.ideas, 'ideas');
  addStage(goalsToReactFlow(data.goals), 'goals');
  addStage(data.actions, 'actions');
  addStage(data.orchestration, 'orchestration');

  return { nodes: allNodes, edges: allEdges, metadata: { stage: 'all' } };
}

function getStageData(
  data: PipelineResultResponse | undefined,
  view: ViewMode,
): ReactFlowData | null {
  if (!data) return null;
  if (view === 'all') return getAllStagesData(data);

  switch (view) {
    case 'ideas':
      return data.ideas;
    case 'goals':
      return goalsToReactFlow(data.goals);
    case 'actions':
      return data.actions;
    case 'orchestration':
      return data.orchestration;
    default:
      return null;
  }
}

function PipelineCanvasInner({
  pipelineId,
  initialData,
  onStageAdvance,
  readOnly = false,
}: PipelineCanvasProps) {
  const [activeView, setActiveView] = useState<ViewMode>('all');
  const [data] = useState(initialData);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [showProvenance, setShowProvenance] = useState(false);
  const { fitView } = useReactFlow();

  const stageStatus = useMemo(
    () =>
      data?.stage_status ?? {
        ideas: 'pending',
        goals: 'pending',
        actions: 'pending',
        orchestration: 'pending',
      },
    [data],
  );

  const stageData = useMemo(() => getStageData(data, activeView), [data, activeView]);

  const [nodes, setNodes, onNodesChange] = useNodesState(stageData?.nodes ?? []);
  const [edges, setEdges, onEdgesChange] = useEdgesState(stageData?.edges ?? []);

  // Update nodes/edges when stage changes
  useEffect(() => {
    const sd = getStageData(data, activeView);
    setNodes(sd?.nodes ?? []);
    setEdges(sd?.edges ?? []);
    // Fit view after stage switch
    setTimeout(() => fitView({ padding: 0.2 }), 50);
  }, [activeView, data, setNodes, setEdges, fitView]);

  const handleStageSelect = useCallback((stage: PipelineStageType) => {
    setActiveView(stage);
    setSelectedNodeId(null);
    setShowProvenance(false);
  }, []);

  const handleViewAll = useCallback(() => {
    setActiveView('all');
    setSelectedNodeId(null);
  }, []);

  const handleAdvance = useCallback(
    (stage: PipelineStageType) => {
      if (pipelineId && onStageAdvance) {
        onStageAdvance(pipelineId, stage);
      }
    },
    [pipelineId, onStageAdvance],
  );

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedNodeId(node.id);
    setShowProvenance(true);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
    setShowProvenance(false);
  }, []);

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

  // Find pending transitions for gate UI
  const pendingTransitions = useMemo(
    () => (data?.transitions || []).filter((t) => t.status === 'pending'),
    [data],
  );

  // Provenance for selected node
  const selectedProvenance = useMemo(() => {
    if (!selectedNodeId || !data?.goals?.provenance) return [];
    const provLinks = (data.goals.provenance || []) as Array<{
      source_node_id: string;
      target_node_id: string;
      source_stage: string;
      target_stage: string;
      content_hash: string;
      method: string;
    }>;
    return provLinks.filter(
      (p) => p.source_node_id === selectedNodeId || p.target_node_id === selectedNodeId,
    );
  }, [selectedNodeId, data]);

  const selectedNodeLabel = useMemo(() => {
    if (!selectedNodeId) return '';
    const node = nodes.find((n) => n.id === selectedNodeId);
    return (node?.data as Record<string, unknown>)?.label as string || selectedNodeId;
  }, [selectedNodeId, nodes]);

  const stageConfig = activeView === 'all' ? null : PIPELINE_STAGE_CONFIG[activeView];
  const edgeColor = stageConfig?.primary || '#6b7280';

  return (
    <div className="flex h-full bg-bg">
      <div className="flex flex-col flex-1">
        {/* Stage Navigator + All toggle */}
        <div className="flex items-center justify-center gap-2 p-2">
          <button
            onClick={handleViewAll}
            className={`px-3 py-1.5 rounded font-mono text-xs font-bold uppercase tracking-wide transition-all duration-200 ${
              activeView === 'all'
                ? 'bg-surface ring-2 ring-acid-green ring-offset-1 ring-offset-bg text-text'
                : 'bg-transparent text-text-muted hover:text-text hover:bg-surface/50'
            }`}
          >
            All Stages
          </button>
          <StageNavigator
            stageStatus={stageStatus}
            activeStage={activeView === 'all' ? 'ideas' : activeView}
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
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            nodeTypes={nodeTypes}
            fitView
            snapToGrid
            snapGrid={[16, 16]}
            defaultEdgeOptions={{
              animated: true,
              style: { stroke: edgeColor, strokeWidth: 2 },
            }}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#333" />
            <Controls className="bg-surface border border-border rounded" showInteractive={!readOnly} />
            <MiniMap className="bg-surface border border-border rounded" nodeColor={miniMapNodeColor} />

            {/* Stats panel */}
            <Panel position="bottom-left" className="bg-surface/90 border border-border rounded p-2">
              <div className="text-xs font-mono text-text-muted">
                <span className="text-text">{nodes.length}</span> nodes |{' '}
                <span className="text-text">{edges.length}</span> edges
                {stageConfig && (
                  <>
                    {' | '}
                    <span style={{ color: stageConfig.primary }} className="uppercase font-bold">
                      {stageConfig.label}
                    </span>
                  </>
                )}
                {activeView === 'all' && (
                  <>
                    {' | '}
                    <span className="text-acid-green uppercase font-bold">ALL STAGES</span>
                  </>
                )}
              </div>
            </Panel>

            {/* Pipeline ID + integrity */}
            {pipelineId && (
              <Panel position="top-right" className="bg-surface/90 border border-border rounded p-2">
                <div className="text-xs font-mono text-text-muted">
                  Pipeline: <span className="text-text">{pipelineId}</span>
                  {data?.integrity_hash && (
                    <span className="ml-2 text-emerald-400">#{data.integrity_hash.slice(0, 8)}</span>
                  )}
                </div>
              </Panel>
            )}

            {/* Pending transition gates */}
            {pendingTransitions.length > 0 && !readOnly && (
              <Panel position="bottom-right" className="space-y-2">
                {pendingTransitions.map((transition) => (
                  <div
                    key={transition.id}
                    className="bg-surface border border-border rounded-lg p-3 max-w-xs"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
                      <span className="text-xs font-mono font-bold text-text uppercase">
                        {transition.from_stage} &rarr; {transition.to_stage}
                      </span>
                    </div>
                    <p className="text-xs text-text-muted mb-2">{transition.ai_rationale}</p>
                    <div className="flex items-center gap-1 mb-2">
                      <span className="text-xs text-text-muted font-mono">Confidence:</span>
                      <div className="w-16 h-1 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-emerald-400 rounded-full"
                          style={{ width: `${Math.round((transition.confidence || 0) * 100)}%` }}
                        />
                      </div>
                      <span className="text-xs text-text font-mono">
                        {Math.round((transition.confidence || 0) * 100)}%
                      </span>
                    </div>
                  </div>
                ))}
              </Panel>
            )}
          </ReactFlow>
        </div>
      </div>

      {/* Provenance sidebar */}
      {showProvenance && selectedNodeId && (
        <div className="w-72 flex-shrink-0 bg-surface border-l border-border h-full overflow-y-auto p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-mono font-bold text-text uppercase">Provenance</h3>
            <button
              onClick={() => { setShowProvenance(false); setSelectedNodeId(null); }}
              className="text-text-muted hover:text-text text-lg leading-none"
            >
              &times;
            </button>
          </div>

          <div className="mb-4">
            <p className="text-sm text-text truncate">{selectedNodeLabel}</p>
            <p className="text-xs text-text-muted font-mono">{selectedNodeId}</p>
          </div>

          {selectedProvenance.length > 0 ? (
            <div className="space-y-2">
              {selectedProvenance.map((link, i) => {
                const isSource = link.source_node_id === selectedNodeId;
                const stageColors: Record<string, string> = {
                  ideas: 'text-indigo-300 bg-indigo-500/20',
                  goals: 'text-emerald-300 bg-emerald-500/20',
                  actions: 'text-amber-300 bg-amber-500/20',
                  orchestration: 'text-pink-300 bg-pink-500/20',
                };
                return (
                  <div key={i} className="p-2 bg-bg rounded border border-border">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs text-text-muted font-mono">
                        {isSource ? 'Produces' : 'Derived from'}
                      </span>
                      <span className={`px-1.5 py-0.5 text-xs rounded font-mono ${stageColors[isSource ? link.target_stage : link.source_stage] || ''}`}>
                        {isSource ? link.target_stage : link.source_stage}
                      </span>
                    </div>
                    <p className="text-xs text-text font-mono truncate">
                      {isSource ? link.target_node_id : link.source_node_id}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-xs text-text-muted font-mono">
                        #{link.content_hash.slice(0, 8)}
                      </span>
                      <span className="text-xs text-text-muted font-mono">
                        {link.method}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-sm text-text-muted">No provenance links for this node.</p>
          )}

          {data?.provenance_count !== undefined && (
            <div className="mt-4 pt-4 border-t border-border">
              <p className="text-xs text-text-muted font-mono">
                Total provenance links: <span className="text-text">{data.provenance_count}</span>
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function PipelineCanvas(props: PipelineCanvasProps) {
  return (
    <ReactFlowProvider>
      <PipelineCanvasInner {...props} />
    </ReactFlowProvider>
  );
}

export default PipelineCanvas;
