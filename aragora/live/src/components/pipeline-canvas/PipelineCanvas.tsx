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
import { ProvenanceTrail } from './ProvenanceTrail';
import { TemplateSelector } from './TemplateSelector';
import { ProgressIndicator } from './ProgressIndicator';
import {
  PIPELINE_STAGE_CONFIG,
  STAGE_COLOR_CLASSES,
  type PipelineStageType,
  type PipelineResultResponse,
  type ProvenanceLink,
} from './types';
import { usePipelineCanvas } from '../../hooks/usePipelineCanvas';
import { usePipelineWebSocket } from '../../hooks/usePipelineWebSocket';
import { StageTransitionGate } from '../pipeline/StageTransitionGate';

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
  initialStage?: PipelineStageType;
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
  initialStage,
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
    runPipeline,
    clearStage,
    approveTransition,
    rejectTransition,
    loading,
    error: _hookError,
    onDragOver: _hookDragOver,
  } = usePipelineCanvas(pipelineId ?? null, initialData);

  const { fitView, screenToFlowPosition } = useReactFlow();
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  // -- View mode: adds 'all' view on top of the hook's active stage -----------
  const [viewMode, setViewMode] = useState<ViewMode>(initialStage || 'all');
  const [showProvenance, setShowProvenance] = useState(false);
  const [showTemplates, setShowTemplates] = useState(!pipelineId && !initialData);

  // -- Run pipeline state ----------------------------------------------------
  const [runInputText, setRunInputText] = useState('');
  const [activePipelineRunId, setActivePipelineRunId] = useState<string | null>(null);
  const [runNotification, setRunNotification] = useState<string | null>(null);

  // -- WebSocket: real-time pipeline progress --------------------------------
  const {
    completedStages,
    pendingTransitions: wsPendingTransitions,
    isComplete: pipelineComplete,
    isConnected,
  } = usePipelineWebSocket({
    pipelineId: activePipelineRunId ?? undefined,
    enabled: !!activePipelineRunId,
    onStageCompleted: (event) => {
      setRunNotification(`Stage "${event.stage}" completed`);
      setTimeout(() => setRunNotification(null), 3000);
    },
    onCompleted: () => {
      setRunNotification('Pipeline completed successfully');
      setActivePipelineRunId(null);
      setTimeout(() => setRunNotification(null), 5000);
    },
    onFailed: (error) => {
      setRunNotification(`Pipeline failed: ${error}`);
      setActivePipelineRunId(null);
      setTimeout(() => setRunNotification(null), 5000);
    },
  });

  // Sync initialStage into the hook's activeStage on mount
  useEffect(() => {
    if (initialStage) {
      setActiveStage(initialStage);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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

  // -- Provenance data (from initialData) -------------------------------------
  const allProvenance: ProvenanceLink[] = useMemo(
    () => (initialData?.provenance ?? []) as ProvenanceLink[],
    [initialData],
  );

  // Build a node lookup from all display nodes for the provenance trail
  const nodeLookup = useMemo(() => {
    const lookup: Record<string, { label: string; stage: PipelineStageType }> = {};
    // Gather from all stages to build complete lookup
    for (const stage of ALL_STAGES) {
      for (const n of stageNodes[stage]) {
        const data = n.data as Record<string, unknown>;
        lookup[n.id] = {
          label: (data?.label as string) || n.id,
          stage,
        };
      }
    }
    return lookup;
  }, [stageNodes]);

  // Provenance links relevant to the selected node
  const selectedProvenance = useMemo(() => {
    if (!selectedNodeId) return [];
    return allProvenance.filter(
      (p) => p.source_node_id === selectedNodeId || p.target_node_id === selectedNodeId,
    );
  }, [selectedNodeId, allProvenance]);

  // Compute the set of provenance-connected node IDs for highlighting
  const provenanceHighlightIds = useMemo(() => {
    if (!selectedNodeId) return new Set<string>();
    const ids = new Set<string>();
    // Walk the full chain (upstream and downstream)
    function walk(nodeId: string, visited: Set<string>) {
      if (visited.has(nodeId)) return;
      visited.add(nodeId);
      ids.add(nodeId);
      for (const link of allProvenance) {
        if (link.target_node_id === nodeId) walk(link.source_node_id, visited);
        if (link.source_node_id === nodeId) walk(link.target_node_id, visited);
      }
    }
    walk(selectedNodeId, new Set());
    return ids;
  }, [selectedNodeId, allProvenance]);

  // Cross-stage edge highlighting: modify edge styles when a node is selected
  const baseEdges = viewMode === 'all' ? allStagesData.edges : edges;
  const displayEdges = useMemo(() => {
    if (!selectedNodeId || provenanceHighlightIds.size === 0) return baseEdges;

    return baseEdges.map((edge) => {
      const sourceInChain = provenanceHighlightIds.has(edge.source);
      const targetInChain = provenanceHighlightIds.has(edge.target);
      const isCrossStage = edge.data?.provenance === true;

      if (sourceInChain && targetInChain) {
        // This edge connects two provenance-linked nodes: highlight it
        return {
          ...edge,
          animated: true,
          style: {
            ...edge.style,
            stroke: isCrossStage ? '#34d399' : '#818cf8',
            strokeWidth: 3,
            opacity: 1,
          },
        };
      }

      // Dim non-provenance edges when a node is selected
      return {
        ...edge,
        style: {
          ...edge.style,
          opacity: 0.25,
        },
      };
    });
  }, [baseEdges, selectedNodeId, provenanceHighlightIds]);

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

  // -- Run Pipeline handler -------------------------------------------------
  const handleRunPipeline = useCallback(async () => {
    if (!runInputText.trim()) return;
    const newPipelineId = await runPipeline(runInputText.trim());
    if (newPipelineId) {
      setActivePipelineRunId(newPipelineId);
      setRunInputText('');
      setRunNotification(`Pipeline started: ${newPipelineId}`);
      setTimeout(() => setRunNotification(null), 3000);
    }
  }, [runInputText, runPipeline]);

  // -- Export Receipt handler --------------------------------------------------
  const handleExportReceipt = useCallback(async () => {
    if (!pipelineId) return;
    try {
      const response = await fetch(`/api/v1/canvas/pipeline/${encodeURIComponent(pipelineId)}/receipt`);
      if (!response.ok) {
        console.error('Failed to fetch receipt:', response.status);
        return;
      }
      const receipt = await response.json();
      const blob = new Blob([JSON.stringify(receipt, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `receipt-${pipelineId}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export receipt failed:', err);
    }
  }, [pipelineId]);

  // -- Provenance navigation handler ------------------------------------------
  const handleProvenanceNavigate = useCallback(
    (nodeId: string, stage: PipelineStageType) => {
      setSelectedNodeId(nodeId);
      setViewMode(stage);
      setActiveStage(stage);
      setTimeout(() => fitView({ padding: 0.2 }), 50);
    },
    [setSelectedNodeId, setActiveStage, fitView],
  );

  // -- Template selection handlers --------------------------------------------
  const handleSelectTemplate = useCallback(async (templateName: string) => {
    try {
      const res = await fetch('/api/v1/canvas/pipeline/from-template', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ template_name: templateName, auto_advance: false }),
      });
      if (res.ok) {
        setShowTemplates(false);
        setRunNotification(`Pipeline created from "${templateName}" template`);
        setTimeout(() => setRunNotification(null), 3000);
      }
    } catch {
      // Template creation failed silently, user can retry
    }
  }, []);

  const handleStartBlank = useCallback(() => {
    setShowTemplates(false);
  }, []);

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

  // -- Transition data ---------------------------------------------------------
  // Merge static transitions from initialData with live WebSocket transitions
  const pendingTransitions = useMemo(() => {
    const staticPending = (initialData?.transitions || []).filter((t) => t.status === 'pending');
    if (wsPendingTransitions.length === 0) return staticPending;
    // WebSocket transitions use a slightly different shape; normalize them
    const wsNormalized = wsPendingTransitions.map((t) => ({
      id: `transition-${t.from_stage}-${t.to_stage}`,
      from_stage: t.from_stage,
      to_stage: t.to_stage,
      ai_rationale: t.ai_rationale,
      confidence: t.confidence,
      status: 'pending' as const,
    }));
    // Deduplicate by from_stage+to_stage
    const seen = new Set(staticPending.map((t) => `${t.from_stage}-${t.to_stage}`));
    const merged = [...staticPending];
    for (const t of wsNormalized) {
      const key = `${t.from_stage}-${t.to_stage}`;
      if (!seen.has(key)) {
        merged.push(t as typeof staticPending[number]);
        seen.add(key);
      }
    }
    return merged;
  }, [initialData, wsPendingTransitions]);

  const selectedNodeLabel = useMemo(() => {
    if (!selectedNodeId) return '';
    const node = displayNodes.find((n) => n.id === selectedNodeId);
    return (node?.data as Record<string, unknown>)?.label as string || selectedNodeId;
  }, [selectedNodeId, displayNodes]);

  // Stage of the selected node
  const selectedNodeStage = useMemo<PipelineStageType>(() => {
    if (!selectedNodeId) return activeStage;
    return nodeLookup[selectedNodeId]?.stage ?? activeStage;
  }, [selectedNodeId, nodeLookup, activeStage]);

  // Transitions relevant to the selected node's stage
  const selectedTransitions = useMemo(
    () => (initialData?.transitions || []).filter(
      (t) => t.from_stage === selectedNodeStage || t.to_stage === selectedNodeStage,
    ),
    [initialData, selectedNodeStage],
  );

  // -- Visual config ----------------------------------------------------------
  const stageConfig = viewMode === 'all' ? null : PIPELINE_STAGE_CONFIG[viewMode];
  const edgeColor = stageConfig?.primary || '#6b7280';

  // -- Right panel logic ------------------------------------------------------
  const showPropertyEditor = !!selectedNodeId && !showProvenance && isEditable;
  const showProvenanceSidebar = !!selectedNodeId && (showProvenance || readOnly || viewMode === 'all') && !showPropertyEditor;

  // -- Template selector: shown when no pipeline is active -------------------
  if (showTemplates) {
    return (
      <div className="flex h-full bg-bg">
        <TemplateSelector
          onSelectTemplate={handleSelectTemplate}
          onStartBlank={handleStartBlank}
        />
      </div>
    );
  }

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
        {/* Run Pipeline bar */}
        {!readOnly && (
          <div className="flex items-center gap-2 px-4 py-2 bg-surface/80 border-b border-border">
            <input
              type="text"
              value={runInputText}
              onChange={(e) => setRunInputText(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleRunPipeline(); }}
              placeholder="Describe your idea or problem..."
              className="flex-1 px-3 py-2 text-sm font-mono rounded bg-bg border border-border text-text placeholder:text-text-muted focus:outline-none focus:border-acid-green transition-colors"
            />
            <button
              onClick={handleRunPipeline}
              disabled={loading || !runInputText.trim() || !!activePipelineRunId}
              className="px-6 py-2 text-sm font-mono font-bold rounded bg-acid-green text-bg hover:bg-acid-green/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 whitespace-nowrap"
            >
              {activePipelineRunId ? (
                <>
                  <span className="inline-block w-3 h-3 border-2 border-bg/30 border-t-bg rounded-full animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  Run Pipeline
                </>
              )}
            </button>

            {/* Progress indicators */}
            {activePipelineRunId && (
              <div className="flex items-center gap-1.5">
                {['ideas', 'goals', 'actions', 'orchestration'].map((stage) => (
                  <div
                    key={stage}
                    className={`w-2.5 h-2.5 rounded-full transition-colors ${
                      completedStages.includes(stage) ? 'bg-acid-green' : 'bg-border'
                    }`}
                    title={`${stage}: ${completedStages.includes(stage) ? 'complete' : 'pending'}`}
                  />
                ))}
                {isConnected && (
                  <span className="text-xs font-mono text-text-muted ml-1">LIVE</span>
                )}
              </div>
            )}

            {/* Notification toast */}
            {runNotification && (
              <span className="text-xs font-mono text-acid-green animate-pulse truncate max-w-xs">
                {runNotification}
              </span>
            )}
          </div>
        )}

        {/* Stage Navigator + Progress */}
        <div className="flex items-center justify-center gap-2 p-2">
          <ProgressIndicator stageStatus={stageStatus} activeStage={activeStage} />
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
                  onExportReceipt={handleExportReceipt}
                  pipelineId={pipelineId}
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
                  <StageTransitionGate
                    key={(transition.id as string) || idx}
                    transition={{
                      id: transition.id as string,
                      from_stage: transition.from_stage as string,
                      to_stage: transition.to_stage as string,
                      ai_rationale: transition.ai_rationale as string | undefined,
                      confidence: transition.confidence as number | undefined,
                      status: transition.status as string | undefined,
                    }}
                    pipelineId={pipelineId || ''}
                    onApprove={(pid, tid) => {
                      approveTransition(tid);
                      onTransitionApprove?.(pid, tid);
                    }}
                    onReject={(pid, tid) => {
                      rejectTransition(tid);
                      onTransitionReject?.(pid, tid);
                    }}
                  />
                ))}
              </Panel>
            )}

            {/* Pipeline completion indicator */}
            {pipelineComplete && (
              <Panel position="bottom-center">
                <div className="bg-emerald-900/80 text-emerald-200 text-xs font-mono px-3 py-1 rounded-full">
                  Pipeline complete
                </div>
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
          provenanceLinks={selectedProvenance}
          transitions={selectedTransitions}
        />
      )}

      {/* Right: Provenance Sidebar */}
      {showProvenanceSidebar && selectedNodeId && (
        <div className="w-80 flex-shrink-0 bg-surface border-l border-border h-full overflow-y-auto p-4">
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

          {/* Node info */}
          <div className="mb-4">
            <p className="text-sm text-text truncate">{selectedNodeLabel}</p>
            <p className="text-xs text-text-muted font-mono">{selectedNodeId}</p>
          </div>

          {/* Provenance Trail breadcrumbs */}
          <div className="mb-4 p-3 bg-bg rounded border border-border">
            <label className="block text-xs text-text-muted mb-2 uppercase font-bold font-mono">
              Provenance Chain
            </label>
            <ProvenanceTrail
              selectedNodeId={selectedNodeId}
              selectedStage={selectedNodeStage}
              selectedLabel={selectedNodeLabel}
              provenance={allProvenance}
              nodeLookup={nodeLookup}
              onNavigate={handleProvenanceNavigate}
            />
          </div>

          {/* Individual provenance links */}
          {selectedProvenance.length > 0 ? (
            <>
              <label className="block text-xs text-text-muted mb-2 uppercase font-bold font-mono">
                Direct Links ({selectedProvenance.length})
              </label>
              <div className="space-y-2">
                {selectedProvenance.map((link, i) => {
                  const isSource = link.source_node_id === selectedNodeId;
                  const relatedStage = isSource ? link.target_stage : link.source_stage;
                  const colors = STAGE_COLOR_CLASSES[relatedStage] ?? { text: '', bg: '' };
                  return (
                    <div key={i} className="p-2 bg-bg rounded border border-border">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs text-text-muted font-mono">
                          {isSource ? 'Produces' : 'Derived from'}
                        </span>
                        <span className={`px-1.5 py-0.5 text-xs rounded font-mono ${colors.bg} ${colors.text}`}>
                          {relatedStage}
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
            </>
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
