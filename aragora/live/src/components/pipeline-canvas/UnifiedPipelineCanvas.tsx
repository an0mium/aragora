'use client';

/**
 * UnifiedPipelineCanvas - All 4 pipeline stages on a single React Flow canvas
 * with semantic zoom: zoom level determines which stages show full detail vs collapsed.
 *
 * Stages:
 *   Idea (blue #3B82F6) -> Goal (green #10B981) -> Action (orange #F59E0B) -> Orchestration (purple #8B5CF6)
 *
 * Semantic zoom levels:
 *   > 1.5   : All stages with full detail
 *   0.8-1.5 : Ideas + Goals + Actions (Orchestration collapsed)
 *   < 0.8   : Ideas + Goals only (collapsed view)
 */

import { useCallback, useState, useMemo, useEffect } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  MiniMap,
  useReactFlow,
  ReactFlowProvider,
  Panel,
  BackgroundVariant,
  type NodeTypes,
  type Node,
  type Edge,
  type Viewport,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { IdeaNode, GoalNode, ActionNode, OrchestrationNode } from './nodes';
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

const ALL_STAGES: PipelineStageType[] = ['ideas', 'goals', 'actions', 'orchestration'];

const STAGE_OFFSET_X: Record<string, number> = {
  ideas: 0,
  goals: 600,
  actions: 1200,
  orchestration: 1800,
};

/** Stage colors from the spec */
const STAGE_COLORS: Record<PipelineStageType, string> = {
  ideas: '#3B82F6',
  goals: '#10B981',
  actions: '#F59E0B',
  orchestration: '#8B5CF6',
};

/** Edge type visual configs */
const EDGE_STYLES: Record<string, { stroke: string; strokeDasharray?: string; animated?: boolean }> = {
  inspires: { stroke: '#3B82F6', strokeDasharray: '5 5' },
  derives: { stroke: '#10B981' },
  decomposes: { stroke: '#F59E0B' },
  triggers: { stroke: '#8B5CF6' },
  depends_on: { stroke: '#6b7280', strokeDasharray: '2 2' },
};

/** Semantic zoom thresholds */
const ZOOM_FULL_DETAIL = 1.5;
const ZOOM_PARTIAL = 0.8;

// =============================================================================
// Helper: determine visible stages at a zoom level
// =============================================================================

function getVisibleStages(zoom: number): Set<PipelineStageType> {
  if (zoom > ZOOM_FULL_DETAIL) {
    return new Set(ALL_STAGES);
  }
  if (zoom >= ZOOM_PARTIAL) {
    return new Set<PipelineStageType>(['ideas', 'goals', 'actions']);
  }
  return new Set<PipelineStageType>(['ideas', 'goals']);
}

function getStageForNodeType(type: string): PipelineStageType | null {
  switch (type) {
    case 'ideaNode': return 'ideas';
    case 'goalNode': return 'goals';
    case 'actionNode': return 'actions';
    case 'orchestrationNode': return 'orchestration';
    default: return null;
  }
}

// =============================================================================
// Props
// =============================================================================

export interface UnifiedPipelineCanvasProps {
  pipelineId?: string;
  initialData?: PipelineResultResponse;
  readOnly?: boolean;
}

// =============================================================================
// Stage Filter Sidebar
// =============================================================================

interface StageFilterProps {
  enabledStages: Set<PipelineStageType>;
  onToggle: (stage: PipelineStageType) => void;
  onFocus: (stage: PipelineStageType) => void;
  nodeCounts: Record<PipelineStageType, number>;
}

function StageFilterSidebar({ enabledStages, onToggle, onFocus, nodeCounts }: StageFilterProps) {
  return (
    <div
      className="w-48 flex-shrink-0 bg-surface border-r border-border h-full overflow-y-auto p-4"
      data-testid="stage-filter-sidebar"
    >
      <h3 className="text-sm font-mono font-bold text-text-muted uppercase tracking-wide mb-4">
        Stages
      </h3>
      <div className="space-y-2">
        {ALL_STAGES.map((stage) => {
          const config = PIPELINE_STAGE_CONFIG[stage];
          const color = STAGE_COLORS[stage];
          const enabled = enabledStages.has(stage);
          const count = nodeCounts[stage];

          return (
            <div key={stage} className="space-y-1">
              <button
                onClick={() => onToggle(stage)}
                className={`
                  w-full flex items-center justify-between px-3 py-2 rounded font-mono text-xs
                  transition-all duration-200 border
                  ${enabled
                    ? 'border-current opacity-100'
                    : 'border-border opacity-40 hover:opacity-60'
                  }
                `}
                style={{ color, borderColor: enabled ? color : undefined }}
                data-testid={`stage-toggle-${stage}`}
              >
                <span className="font-bold uppercase">{config.label}</span>
                <span
                  className="px-1.5 py-0.5 rounded-full text-xs font-mono"
                  style={{
                    backgroundColor: enabled ? `${color}33` : 'transparent',
                  }}
                  data-testid={`stage-count-${stage}`}
                >
                  {count}
                </span>
              </button>
              <button
                onClick={() => onFocus(stage)}
                className="w-full text-center text-xs font-mono text-text-muted hover:text-text transition-colors"
                data-testid={`stage-focus-${stage}`}
              >
                Focus
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// =============================================================================
// AI Transition Toolbar
// =============================================================================

interface AITransitionToolbarProps {
  selectedStages: Set<PipelineStageType>;
  loading: boolean;
  onGenerateGoals: () => void;
  onGenerateTasks: () => void;
  onGenerateWorkflow: () => void;
}

function AITransitionToolbar({
  selectedStages,
  loading,
  onGenerateGoals,
  onGenerateTasks,
  onGenerateWorkflow,
}: AITransitionToolbarProps) {
  const hasIdeas = selectedStages.has('ideas');
  const hasGoals = selectedStages.has('goals');
  const hasActions = selectedStages.has('actions');

  return (
    <div className="flex items-center gap-2" data-testid="ai-transition-toolbar">
      <button
        onClick={onGenerateGoals}
        disabled={!hasIdeas || loading}
        className="px-3 py-1.5 bg-emerald-600 text-white font-mono text-xs font-bold rounded
                   hover:bg-emerald-500 transition-colors
                   disabled:opacity-40 disabled:cursor-not-allowed"
        data-testid="btn-generate-goals"
      >
        {loading ? 'Generating...' : 'Generate Goals'}
      </button>
      <button
        onClick={onGenerateTasks}
        disabled={!hasGoals || loading}
        className="px-3 py-1.5 bg-amber-600 text-white font-mono text-xs font-bold rounded
                   hover:bg-amber-500 transition-colors
                   disabled:opacity-40 disabled:cursor-not-allowed"
        data-testid="btn-generate-tasks"
      >
        {loading ? 'Generating...' : 'Generate Tasks'}
      </button>
      <button
        onClick={onGenerateWorkflow}
        disabled={!hasActions || loading}
        className="px-3 py-1.5 bg-purple-600 text-white font-mono text-xs font-bold rounded
                   hover:bg-purple-500 transition-colors
                   disabled:opacity-40 disabled:cursor-not-allowed"
        data-testid="btn-generate-workflow"
      >
        {loading ? 'Generating...' : 'Generate Workflow'}
      </button>
    </div>
  );
}

// =============================================================================
// Provenance Sidebar
// =============================================================================

interface ProvenanceSidebarProps {
  nodeId: string;
  nodeLabel: string;
  provenanceChain: Array<{
    stage: string;
    nodeId: string;
    label: string;
    hash: string;
  }>;
  onClose: () => void;
}

function ProvenanceSidebar({ nodeId, nodeLabel, provenanceChain, onClose }: ProvenanceSidebarProps) {
  return (
    <div
      className="w-72 flex-shrink-0 bg-surface border-l border-border h-full overflow-y-auto p-4"
      data-testid="provenance-sidebar"
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-mono font-bold text-text uppercase">Provenance</h3>
        <button
          onClick={onClose}
          className="text-text-muted hover:text-text text-lg leading-none"
          data-testid="provenance-close"
        >
          &times;
        </button>
      </div>

      <div className="mb-4">
        <p className="text-sm text-text truncate">{nodeLabel}</p>
        <p className="text-xs text-text-muted font-mono">{nodeId}</p>
      </div>

      {provenanceChain.length > 0 ? (
        <div className="space-y-2">
          <h4 className="text-xs font-mono font-bold text-text-muted uppercase mb-2">
            Derivation Chain
          </h4>
          {provenanceChain.map((entry, i) => {
            const stageColor = STAGE_COLORS[entry.stage as PipelineStageType] || '#6b7280';
            return (
              <div key={i} className="p-2 bg-bg rounded border border-border" data-testid="provenance-entry">
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className="w-2 h-2 rounded-full inline-block"
                    style={{ backgroundColor: stageColor }}
                  />
                  <span className="text-xs font-mono uppercase" style={{ color: stageColor }}>
                    {entry.stage}
                  </span>
                </div>
                <p className="text-xs text-text truncate mb-1">{entry.label}</p>
                <p className="text-xs text-text-muted font-mono">
                  SHA-256: {entry.hash.slice(0, 12)}...
                </p>
              </div>
            );
          })}
        </div>
      ) : (
        <p className="text-sm text-text-muted">No provenance chain for this node.</p>
      )}
    </div>
  );
}

// =============================================================================
// Inner component (inside ReactFlowProvider)
// =============================================================================

function UnifiedPipelineCanvasInner({
  pipelineId,
  initialData,
  readOnly = false,
}: UnifiedPipelineCanvasProps) {
  const {
    stageNodes,
    stageEdges,
    loading,
    aiGenerate,
  } = usePipelineCanvas(pipelineId ?? null, initialData);

  const { fitView } = useReactFlow();

  // -- Zoom tracking --------------------------------------------------------
  const [zoomLevel, setZoomLevel] = useState(1.0);

  const onViewportChange = useCallback((viewport: Viewport) => {
    setZoomLevel(viewport.zoom);
  }, []);

  // -- Stage filter state ---------------------------------------------------
  const [stageFilterOverrides, setStageFilterOverrides] = useState<Set<PipelineStageType>>(
    new Set(ALL_STAGES),
  );

  const toggleStage = useCallback((stage: PipelineStageType) => {
    setStageFilterOverrides((prev) => {
      const next = new Set(prev);
      if (next.has(stage)) {
        next.delete(stage);
      } else {
        next.add(stage);
      }
      return next;
    });
  }, []);

  const focusStage = useCallback(
    (stage: PipelineStageType) => {
      // Ensure the stage is enabled
      setStageFilterOverrides((prev) => {
        const next = new Set(prev);
        next.add(stage);
        return next;
      });
      // Fit view to nodes of that stage after a tick
      setTimeout(() => {
        const offsetX = STAGE_OFFSET_X[stage];
        fitView({
          padding: 0.3,
          nodes: stageNodes[stage].map((n) => ({
            id: n.id,
            position: { x: n.position.x + offsetX, y: n.position.y },
            measured: { width: 250, height: 120 },
          })),
        });
      }, 50);
    },
    [fitView, stageNodes],
  );

  // -- Node selection & provenance ------------------------------------------
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [showProvenance, setShowProvenance] = useState(false);

  // -- Selected nodes per stage (for AI transition buttons) -----------------
  const [selectedNodeIds, setSelectedNodeIds] = useState<Set<string>>(new Set());
  const selectedNodeStages = useMemo(() => {
    const stages = new Set<PipelineStageType>();
    for (const stage of ALL_STAGES) {
      for (const n of stageNodes[stage]) {
        if (selectedNodeIds.has(n.id)) {
          stages.add(stage);
        }
      }
    }
    return stages;
  }, [selectedNodeIds, stageNodes]);

  // -- Compute visible stages: intersection of semantic zoom + filter -------
  const semanticVisible = useMemo(() => getVisibleStages(zoomLevel), [zoomLevel]);

  const visibleStages = useMemo(() => {
    const result = new Set<PipelineStageType>();
    for (const stage of ALL_STAGES) {
      if (semanticVisible.has(stage) && stageFilterOverrides.has(stage)) {
        result.add(stage);
      }
    }
    return result;
  }, [semanticVisible, stageFilterOverrides]);

  // -- Assemble nodes/edges from all visible stages -------------------------
  const { displayNodes, displayEdges } = useMemo(() => {
    const allNodes: Node[] = [];
    const allEdges: Edge[] = [];
    for (const stage of ALL_STAGES) {
      if (!visibleStages.has(stage)) continue;
      const offsetX = STAGE_OFFSET_X[stage];
      for (const n of stageNodes[stage]) {
        allNodes.push({
          ...n,
          position: { x: n.position.x + offsetX, y: n.position.y },
        });
      }
      for (const e of stageEdges[stage]) {
        const edgeType = (e.data as Record<string, unknown>)?.edgeType as string | undefined;
        const styleOverride = edgeType ? EDGE_STYLES[edgeType] : undefined;
        allEdges.push({
          ...e,
          style: {
            stroke: STAGE_COLORS[stage],
            strokeWidth: 2,
            ...styleOverride,
            ...(e.style || {}),
          },
          animated: styleOverride?.animated ?? e.animated ?? true,
        });
      }
    }
    return { displayNodes: allNodes, displayEdges: allEdges };
  }, [visibleStages, stageNodes, stageEdges]);

  // -- Per-stage node counts ------------------------------------------------
  const nodeCounts = useMemo(() => {
    const counts: Record<PipelineStageType, number> = {
      ideas: 0,
      goals: 0,
      actions: 0,
      orchestration: 0,
    };
    for (const stage of ALL_STAGES) {
      counts[stage] = stageNodes[stage].length;
    }
    return counts;
  }, [stageNodes]);

  // -- Provenance chain for selected node -----------------------------------
  const provenanceChain = useMemo(() => {
    if (!selectedNodeId || !initialData) return [];

    const chain: Array<{ stage: string; nodeId: string; label: string; hash: string }> = [];
    const provLinks = (initialData.goals?.provenance ?? []) as Array<{
      source_node_id: string;
      target_node_id: string;
      source_stage: string;
      target_stage: string;
      content_hash: string;
    }>;

    // Walk the chain backward from the selected node
    let currentId = selectedNodeId;
    const visited = new Set<string>();
    while (currentId && !visited.has(currentId)) {
      visited.add(currentId);
      const link = provLinks.find((l) => l.target_node_id === currentId);
      if (!link) break;

      // Find the source node label
      const sourceStage = link.source_stage as PipelineStageType;
      const sourceNode = stageNodes[sourceStage]?.find((n) => n.id === link.source_node_id);
      chain.unshift({
        stage: link.source_stage,
        nodeId: link.source_node_id,
        label: (sourceNode?.data as Record<string, unknown>)?.label as string || link.source_node_id,
        hash: link.content_hash || '',
      });
      currentId = link.source_node_id;
    }

    return chain;
  }, [selectedNodeId, initialData, stageNodes]);

  const selectedNodeLabel = useMemo(() => {
    if (!selectedNodeId) return '';
    const node = displayNodes.find((n) => n.id === selectedNodeId);
    return (node?.data as Record<string, unknown>)?.label as string || selectedNodeId;
  }, [selectedNodeId, displayNodes]);

  // -- Node click -----------------------------------------------------------
  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setSelectedNodeId(node.id);
      setShowProvenance(true);

      // Track selected nodes by stage for AI transition buttons
      const stage = getStageForNodeType(node.type || '');
      if (stage) {
        setSelectedNodeIds((prev) => {
          const next = new Set(prev);
          next.add(node.id);
          return next;
        });
      }
    },
    [],
  );

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
    setShowProvenance(false);
    setSelectedNodeIds(new Set());
  }, []);

  // -- AI transition handlers -----------------------------------------------
  const handleGenerateGoals = useCallback(() => {
    aiGenerate('goals');
  }, [aiGenerate]);

  const handleGenerateTasks = useCallback(() => {
    aiGenerate('actions');
  }, [aiGenerate]);

  const handleGenerateWorkflow = useCallback(() => {
    aiGenerate('orchestration');
  }, [aiGenerate]);

  // -- MiniMap color --------------------------------------------------------
  const miniMapNodeColor = useCallback((node: { type?: string }) => {
    switch (node.type) {
      case 'ideaNode': return STAGE_COLORS.ideas;
      case 'goalNode': return STAGE_COLORS.goals;
      case 'actionNode': return STAGE_COLORS.actions;
      case 'orchestrationNode': return STAGE_COLORS.orchestration;
      default: return '#6b7280';
    }
  }, []);

  // -- Fit view on mount ----------------------------------------------------
  useEffect(() => {
    setTimeout(() => fitView({ padding: 0.2 }), 50);
  }, [fitView]);

  return (
    <div className="flex h-full bg-bg" data-testid="unified-pipeline-canvas">
      {/* Left: Stage Filter Sidebar */}
      <StageFilterSidebar
        enabledStages={stageFilterOverrides}
        onToggle={toggleStage}
        onFocus={focusStage}
        nodeCounts={nodeCounts}
      />

      {/* Center: Canvas */}
      <div className="flex flex-col flex-1">
        <div className="flex-1">
          <ReactFlow
            nodes={displayNodes}
            edges={displayEdges}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            onViewportChange={onViewportChange}
            nodeTypes={nodeTypes}
            fitView
            snapToGrid
            snapGrid={[16, 16]}
            defaultEdgeOptions={{
              animated: true,
              style: { stroke: '#6b7280', strokeWidth: 2 },
            }}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#333" />
            <Controls className="bg-surface border border-border rounded" showInteractive={!readOnly} />
            <MiniMap className="bg-surface border border-border rounded" nodeColor={miniMapNodeColor} />

            {/* AI Transition Toolbar */}
            {!readOnly && (
              <Panel position="top-center">
                <AITransitionToolbar
                  selectedStages={selectedNodeStages}
                  loading={loading}
                  onGenerateGoals={handleGenerateGoals}
                  onGenerateTasks={handleGenerateTasks}
                  onGenerateWorkflow={handleGenerateWorkflow}
                />
              </Panel>
            )}

            {/* Stats + zoom info panel */}
            <Panel position="bottom-left" className="bg-surface/90 border border-border rounded p-2">
              <div className="text-xs font-mono text-text-muted">
                <span className="text-text">{displayNodes.length}</span> nodes |{' '}
                <span className="text-text">{displayEdges.length}</span> edges |{' '}
                <span className="text-text">zoom: {zoomLevel.toFixed(2)}</span>
                <span className="ml-2 opacity-50" data-testid="zoom-indicator">
                  {zoomLevel > ZOOM_FULL_DETAIL
                    ? 'all stages'
                    : zoomLevel >= ZOOM_PARTIAL
                      ? 'ideas + goals + actions'
                      : 'ideas + goals'
                  }
                </span>
              </div>
            </Panel>

            {/* Pipeline ID + integrity */}
            {pipelineId && (
              <Panel position="top-right" className="bg-surface/90 border border-border rounded p-2">
                <div className="text-xs font-mono text-text-muted">
                  Pipeline: <span className="text-text">{pipelineId}</span>
                  {initialData?.integrity_hash && (
                    <span className="ml-2 text-emerald-400">
                      #{initialData.integrity_hash.slice(0, 8)}
                    </span>
                  )}
                </div>
              </Panel>
            )}

            {/* Stage lane labels */}
            {ALL_STAGES.filter((s) => visibleStages.has(s)).map((stage) => (
              <Panel key={stage} position="top-left" className="pointer-events-none">
                <div
                  className="font-mono text-xs font-bold uppercase tracking-wide opacity-30 ml-2 mt-1"
                  style={{
                    color: STAGE_COLORS[stage],
                    transform: `translateX(${STAGE_OFFSET_X[stage]}px)`,
                  }}
                >
                  {PIPELINE_STAGE_CONFIG[stage].label}
                </div>
              </Panel>
            ))}
          </ReactFlow>
        </div>
      </div>

      {/* Right: Provenance Sidebar */}
      {showProvenance && selectedNodeId && (
        <ProvenanceSidebar
          nodeId={selectedNodeId}
          nodeLabel={selectedNodeLabel}
          provenanceChain={provenanceChain}
          onClose={() => {
            setShowProvenance(false);
            setSelectedNodeId(null);
          }}
        />
      )}
    </div>
  );
}

// =============================================================================
// Exported wrapper with ReactFlowProvider
// =============================================================================

export function UnifiedPipelineCanvas(props: UnifiedPipelineCanvasProps) {
  return (
    <ReactFlowProvider>
      <UnifiedPipelineCanvasInner {...props} />
    </ReactFlowProvider>
  );
}

export default UnifiedPipelineCanvas;
