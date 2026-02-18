'use client';

import { useCallback, useState, useMemo, useEffect, useRef } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  MiniMap,
  useReactFlow,
  ReactFlowProvider,
  type NodeTypes,
  type Node,
  type Edge,
  Panel,
  BackgroundVariant,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { IdeaNode, GoalNode, ActionNode, OrchestrationNode } from './nodes';
import { StageNavigator } from './StageNavigator';
import { PipelinePalette } from './PipelinePalette';
import { PipelineToolbar } from './PipelineToolbar';
import { PipelinePropertyEditor } from './editors/PipelinePropertyEditor';
import {
  PIPELINE_STAGE_CONFIG,
  type PipelineStageType,
  type PipelineResultResponse,
} from './types';
import { usePipelineCanvas } from '../../hooks/usePipelineCanvas';

// =============================================================================
// Constants
// =============================================================================

const nodeTypes: NodeTypes = {
  ideaNode: IdeaNode,
  goalNode: GoalNode,
  actionNode: ActionNode,
  orchestrationNode: OrchestrationNode,
};

type ViewMode = PipelineStageType | 'all';

const STAGE_KEYS: Record<string, ViewMode> = {
  '1': 'ideas',
  '2': 'goals',
  '3': 'actions',
  '4': 'orchestration',
  a: 'all',
};

const ALL_STAGES: PipelineStageType[] = ['ideas', 'goals', 'actions', 'orchestration'];

const STAGE_OFFSET_X: Record<string, number> = {
  ideas: 0,
  goals: 600,
  actions: 1200,
  orchestration: 1800,
};

// =============================================================================
// Props
// =============================================================================

interface PipelineCanvasProps {
  pipelineId?: string;
  initialData?: PipelineResultResponse;
  onStageAdvance?: (pipelineId: string, stage: PipelineStageType) => void;
  onTransitionApprove?: (pipelineId: string, transitionId: string) => void;
  onTransitionReject?: (pipelineId: string, transitionId: string) => void;
  readOnly?: boolean;
}

// =============================================================================
// Inner component (inside ReactFlowProvider)
// =============================================================================

function PipelineCanvasInner({
  pipelineId,
  initialData,
  onStageAdvance,
  onTransitionApprove,
  onTransitionReject,
  readOnly = false,
}: PipelineCanvasProps) {
  // -- Hook: central state management -----------------------------------------
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    selectedNodeId,
    setSelectedNodeId,
    selectedNodeData,
    updateSelectedNode,
    deleteSelectedNode,
    addNode,
    activeStage,
    setActiveStage,
    stageStatus,
    stageNodes,
    stageEdges,
    savePipeline,
    aiGenerate,
    clearStage,
    loading,
    error: _hookError,
    onDragOver: _hookDragOver,
  } = usePipelineCanvas(pipelineId ?? null, initialData);

  const { fitView, screenToFlowPosition } = useReactFlow();
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  // -- View mode: adds 'all' view on top of the hook's active stage -----------
  const [viewMode, setViewMode] = useState<ViewMode>('all');
  const [showProvenance, setShowProvenance] = useState(false);

  const isEditable = viewMode !== 'all' && !readOnly;

  // -- Stage switching --------------------------------------------------------
  const handleStageSelect = useCallback(
    (stage: PipelineStageType) => {
      setViewMode(stage);
      setActiveStage(stage);
      setSelectedNodeId(null);
      setShowProvenance(false);
    },
    [setActiveStage, setSelectedNodeId],
  );

  const handleViewAll = useCallback(() => {
    setViewMode('all');
    setSelectedNodeId(null);
    setShowProvenance(false);
  }, [setSelectedNodeId]);

  // -- Keyboard shortcuts: 1-4 for stages, A for all -------------------------
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      const view = STAGE_KEYS[e.key];
      if (view) {
        e.preventDefault();
        if (view === 'all') {
          handleViewAll();
        } else {
          handleStageSelect(view);
        }
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [handleStageSelect, handleViewAll]);

  // -- Combined "all stages" view from caches --------------------------------
  const allStagesData = useMemo(() => {
    if (viewMode !== 'all') return { nodes: [] as Node[], edges: [] as Edge[] };
    const allNodes: Node[] = [];
    const allEdges: Edge[] = [];
    for (const stage of ALL_STAGES) {
      const offsetX = STAGE_OFFSET_X[stage] || 0;
      for (const n of stageNodes[stage]) {
        allNodes.push({
          ...n,
          position: { x: n.position.x + offsetX, y: n.position.y },
        });
      }
      allEdges.push(...stageEdges[stage]);
    }
    return { nodes: allNodes, edges: allEdges };
  }, [viewMode, stageNodes, stageEdges]);

  const displayNodes = viewMode === 'all' ? allStagesData.nodes : nodes;
  const displayEdges = viewMode === 'all' ? allStagesData.edges : edges;

  // -- Fit view on stage switch -----------------------------------------------
  useEffect(() => {
    setTimeout(() => fitView({ padding: 0.2 }), 50);
  }, [viewMode, fitView]);

  // -- Node click -------------------------------------------------------------
  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setSelectedNodeId(node.id);
      if (readOnly || viewMode === 'all') {
        setShowProvenance(true);
      }
    },
    [setSelectedNodeId, readOnly, viewMode],
  );

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
    setShowProvenance(false);
  }, [setSelectedNodeId]);

  // -- DnD: zoom-aware drop coordinates ---------------------------------------
  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const raw = event.dataTransfer.getData('application/pipeline-node');
      if (!raw) return;

      let parsed: { stage: PipelineStageType; subtype: string };
      try {
        parsed = JSON.parse(raw);
      } catch {
        return;
      }

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });
      addNode(parsed.stage, parsed.subtype, position);
    },
    [screenToFlowPosition, addNode],
  );

  // -- Toolbar handlers -------------------------------------------------------
  const handleAdvance = useCallback(
    (stage: PipelineStageType) => {
      if (pipelineId && onStageAdvance) {
        onStageAdvance(pipelineId, stage);
      }
    },
    [pipelineId, onStageAdvance],
  );

  const handleToolbarAdvance = useCallback(() => {
    const idx = ALL_STAGES.indexOf(activeStage);
    if (idx >= 0 && idx < ALL_STAGES.length - 1) {
      const next = ALL_STAGES[idx + 1];
      handleAdvance(next);
      handleStageSelect(next);
    }
  }, [activeStage, handleAdvance, handleStageSelect]);

  const handleAIGenerate = useCallback(() => {
    aiGenerate(activeStage);
  }, [aiGenerate, activeStage]);

  const handleClear = useCallback(() => {
    clearStage();
  }, [clearStage]);

  const handleSave = useCallback(() => {
    savePipeline();
  }, [savePipeline]);

  // -- Can advance: current stage has nodes and next stage exists -------------
  const canAdvance = useMemo(() => {
    const idx = ALL_STAGES.indexOf(activeStage);
    return idx >= 0 && idx < ALL_STAGES.length - 1 && nodes.length > 0;
  }, [activeStage, nodes.length]);

  // -- MiniMap color ----------------------------------------------------------
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

  // -- Provenance data (from initialData) -------------------------------------
  const pendingTransitions = useMemo(
    () => (initialData?.transitions || []).filter((t) => t.status === 'pending'),
    [initialData],
  );

  const selectedProvenance = useMemo(() => {
    if (!selectedNodeId || !initialData?.goals?.provenance) return [];
    const provLinks = (initialData.goals.provenance || []) as Array<{
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
  }, [selectedNodeId, initialData]);

  const selectedNodeLabel = useMemo(() => {
    if (!selectedNodeId) return '';
    const node = displayNodes.find((n) => n.id === selectedNodeId);
    return (node?.data as Record<string, unknown>)?.label as string || selectedNodeId;
  }, [selectedNodeId, displayNodes]);

  // -- Visual config ----------------------------------------------------------
  const stageConfig = viewMode === 'all' ? null : PIPELINE_STAGE_CONFIG[viewMode];
  const edgeColor = stageConfig?.primary || '#6b7280';

  // -- Right panel logic ------------------------------------------------------
  const showPropertyEditor = !!selectedNodeId && !showProvenance && isEditable;
  const showProvenanceSidebar = !!selectedNodeId && (showProvenance || readOnly || viewMode === 'all') && !showPropertyEditor;

  return (
    <div className="flex h-full bg-bg">
      {/* Left: Node Palette */}
      {isEditable && (
        <div className="w-56 flex-shrink-0">
          <PipelinePalette stage={activeStage} />
        </div>
      )}

      {/* Center: Navigator + Canvas */}
      <div className="flex flex-col flex-1">
        {/* Stage Navigator */}
        <div className="flex items-center justify-center gap-2 p-2">
          <button
            onClick={handleViewAll}
            className={`px-3 py-1.5 rounded font-mono text-xs font-bold uppercase tracking-wide transition-all duration-200 ${
              viewMode === 'all'
                ? 'bg-surface ring-2 ring-acid-green ring-offset-1 ring-offset-bg text-text'
                : 'bg-transparent text-text-muted hover:text-text hover:bg-surface/50'
            }`}
          >
            All Stages
          </button>
          <StageNavigator
            stageStatus={stageStatus}
            activeStage={viewMode === 'all' ? 'ideas' : viewMode}
            onStageSelect={handleStageSelect}
            onAdvance={readOnly ? undefined : handleAdvance}
            readOnly={readOnly}
          />
        </div>

        {/* Canvas */}
        <div className="flex-1" ref={reactFlowWrapper}>
          <ReactFlow
            nodes={displayNodes}
            edges={displayEdges}
            onNodesChange={isEditable ? onNodesChange : undefined}
            onEdgesChange={isEditable ? onEdgesChange : undefined}
            onConnect={isEditable ? onConnect : undefined}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            onDragOver={isEditable ? handleDragOver : undefined}
            onDrop={isEditable ? handleDrop : undefined}
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
            <Controls className="bg-surface border border-border rounded" showInteractive={isEditable} />
            <MiniMap className="bg-surface border border-border rounded" nodeColor={miniMapNodeColor} />

            {/* Toolbar */}
            {isEditable && (
              <Panel position="top-center">
                <PipelineToolbar
                  stage={activeStage}
                  nodeCount={nodes.length}
                  edgeCount={edges.length}
                  readOnly={readOnly}
                  loading={loading}
                  onSave={handleSave}
                  onClear={handleClear}
                  onAIGenerate={handleAIGenerate}
                  canAdvance={canAdvance}
                  onAdvance={handleToolbarAdvance}
                />
              </Panel>
            )}

            {/* Stats panel */}
            <Panel position="bottom-left" className="bg-surface/90 border border-border rounded p-2">
              <div className="text-xs font-mono text-text-muted">
                <span className="text-text">{displayNodes.length}</span> nodes |{' '}
                <span className="text-text">{displayEdges.length}</span> edges
                {stageConfig && (
                  <>
                    {' | '}
                    <span style={{ color: stageConfig.primary }} className="uppercase font-bold">
                      {stageConfig.label}
                    </span>
                  </>
                )}
                {viewMode === 'all' && (
                  <>
                    {' | '}
                    <span className="text-acid-green uppercase font-bold">ALL STAGES</span>
                  </>
                )}
                <span className="ml-2 opacity-50">1-4: stages | A: all</span>
              </div>
            </Panel>

            {/* Pipeline ID + integrity */}
            {pipelineId && (
              <Panel position="top-right" className="bg-surface/90 border border-border rounded p-2">
                <div className="text-xs font-mono text-text-muted">
                  Pipeline: <span className="text-text">{pipelineId}</span>
                  {initialData?.integrity_hash && (
                    <span className="ml-2 text-emerald-400">#{initialData.integrity_hash.slice(0, 8)}</span>
                  )}
                </div>
              </Panel>
            )}

            {/* Pending transition gates */}
            {pendingTransitions.length > 0 && !readOnly && (
              <Panel position="bottom-right" className="space-y-2">
                {pendingTransitions.map((transition, idx) => (
                  <div
                    key={(transition.id as string) || idx}
                    className="bg-surface border border-border rounded-lg p-3 max-w-xs"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
                      <span className="text-xs font-mono font-bold text-text uppercase">
                        {transition.from_stage as string} &rarr; {transition.to_stage as string}
                      </span>
                    </div>
                    <p className="text-xs text-text-muted mb-2">{transition.ai_rationale as string}</p>
                    <div className="flex items-center gap-1 mb-2">
                      <span className="text-xs text-text-muted font-mono">Confidence:</span>
                      <div className="w-16 h-1 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-emerald-400 rounded-full"
                          style={{ width: `${Math.round(((transition.confidence as number) || 0) * 100)}%` }}
                        />
                      </div>
                      <span className="text-xs text-text font-mono">
                        {Math.round(((transition.confidence as number) || 0) * 100)}%
                      </span>
                    </div>
                    {(onTransitionApprove || onTransitionReject) && pipelineId && (
                      <div className="flex gap-2 mt-2">
                        {onTransitionApprove && (
                          <button
                            onClick={() => onTransitionApprove(pipelineId, transition.id as string)}
                            className="flex-1 px-2 py-1 bg-emerald-600 text-white text-xs font-mono rounded hover:bg-emerald-500 transition-colors"
                          >
                            Approve
                          </button>
                        )}
                        {onTransitionReject && (
                          <button
                            onClick={() => onTransitionReject(pipelineId, transition.id as string)}
                            className="flex-1 px-2 py-1 bg-red-600 text-white text-xs font-mono rounded hover:bg-red-500 transition-colors"
                          >
                            Reject
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </Panel>
            )}
          </ReactFlow>
        </div>
      </div>

      {/* Right: Property Editor */}
      {showPropertyEditor && (
        <PipelinePropertyEditor
          node={selectedNodeData}
          stage={activeStage}
          onUpdate={updateSelectedNode}
          onDelete={deleteSelectedNode}
          onShowProvenance={() => setShowProvenance(true)}
          readOnly={readOnly}
        />
      )}

      {/* Right: Provenance Sidebar */}
      {showProvenanceSidebar && (
        <div className="w-72 flex-shrink-0 bg-surface border-l border-border h-full overflow-y-auto p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-mono font-bold text-text uppercase">Provenance</h3>
            <button
              onClick={() => {
                setShowProvenance(false);
                setSelectedNodeId(null);
              }}
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
                      <span
                        className={`px-1.5 py-0.5 text-xs rounded font-mono ${stageColors[isSource ? link.target_stage : link.source_stage] || ''}`}
                      >
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
                      <span className="text-xs text-text-muted font-mono">{link.method}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-sm text-text-muted">No provenance links for this node.</p>
          )}

          {initialData?.provenance_count !== undefined && (
            <div className="mt-4 pt-4 border-t border-border">
              <p className="text-xs text-text-muted font-mono">
                Total provenance links: <span className="text-text">{initialData.provenance_count}</span>
              </p>
            </div>
          )}

          {/* Back to editor button (edit mode only) */}
          {isEditable && (
            <button
              onClick={() => setShowProvenance(false)}
              className="mt-4 w-full px-4 py-2 bg-surface border border-border text-text font-mono text-sm hover:bg-bg transition-colors rounded"
            >
              Back to Editor
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Exported wrapper with ReactFlowProvider
// =============================================================================

export function PipelineCanvas(props: PipelineCanvasProps) {
  return (
    <ReactFlowProvider>
      <PipelineCanvasInner {...props} />
    </ReactFlowProvider>
  );
}

export default PipelineCanvas;
