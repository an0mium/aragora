'use client';

/**
 * useMissionControl - State management for the 6-stage Mission Control canvas.
 *
 * Extends the usePipelineCanvas pattern to support 6 stages:
 *   Ideas -> Principles -> Goals -> Actions -> Orchestration -> Execution
 *
 * Manages per-stage node/edge arrays, selection, provenance chain walking,
 * and WebSocket integration for real-time updates.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Node, Edge } from '@xyflow/react';
import type {
  PipelineStageType,
  PipelineResultResponse,
  ReactFlowData,
  ProvenanceBreadcrumb,
} from '../components/pipeline-canvas/types';
import {
  getNodeTypeForStage,
  PIPELINE_STAGE_CONFIG,
} from '../components/pipeline-canvas/types';
import {
  usePipelineWebSocket,
  type PipelineStageEvent,
  type PipelineNodeEvent,
} from './usePipelineWebSocket';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const API_PREFIX = '/api/v1/canvas/pipeline';

const ALL_STAGES: PipelineStageType[] = [
  'ideas', 'principles', 'goals', 'actions', 'orchestration',
];

const EXECUTION_STAGE = 'execution';

type MissionStageType = PipelineStageType | typeof EXECUTION_STAGE;

const _ALL_MISSION_STAGES: MissionStageType[] = [
  ...ALL_STAGES,
  EXECUTION_STAGE,
];

const EMPTY_STAGE_NODES: Record<PipelineStageType, Node[]> = {
  ideas: [],
  principles: [],
  goals: [],
  actions: [],
  orchestration: [],
};

const EMPTY_STAGE_EDGES: Record<PipelineStageType, Edge[]> = {
  ideas: [],
  principles: [],
  goals: [],
  actions: [],
  orchestration: [],
};

const DEFAULT_STATUS: Record<PipelineStageType, string> = {
  ideas: 'pending',
  principles: 'pending',
  goals: 'pending',
  actions: 'pending',
  orchestration: 'pending',
};

export const STAGE_OFFSET_X: Record<string, number> = {
  ideas: 0,
  principles: 600,
  goals: 1200,
  actions: 1800,
  orchestration: 2400,
  execution: 3000,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseStageNodes(stage: PipelineStageType, data: ReactFlowData | Record<string, unknown> | null): Node[] {
  if (!data) return [];

  const rawNodes: Array<Record<string, unknown>> =
    (data as ReactFlowData).nodes ??
    ((data as Record<string, unknown>)[stage] as Array<Record<string, unknown>>) ??
    ((data as Record<string, unknown>).goals as Array<Record<string, unknown>>) ??
    [];

  const nodeType = getNodeTypeForStage(stage);

  return rawNodes.map((n) => ({
    id: (n.id as string) || `${stage}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    type: (n.type as string) || nodeType,
    position: (n.position as { x: number; y: number }) || { x: 0, y: 0 },
    data: {
      ...(n.data as Record<string, unknown> ?? {}),
      label: (n.data as Record<string, unknown>)?.label ?? (n as Record<string, unknown>).label ?? (n as Record<string, unknown>).title ?? '',
      stage,
    },
    style: (n.style as Record<string, string>) ?? {},
  }));
}

function parseStageEdges(stage: PipelineStageType, data: ReactFlowData | Record<string, unknown> | null): Edge[] {
  if (!data) return [];

  const rawEdges: Array<Record<string, unknown>> = (data as ReactFlowData).edges ?? [];
  const stageColor = PIPELINE_STAGE_CONFIG[stage].primary;

  return rawEdges.map((e) => ({
    id: (e.id as string) || `e-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    source: ((e.source || e.source_id) as string),
    target: ((e.target || e.target_id) as string),
    type: (e.type as string) || 'default',
    label: e.label as string | undefined,
    animated: e.animated !== undefined ? !!e.animated : true,
    style: { stroke: stageColor, ...(e.style as Record<string, string> ?? {}) },
  }));
}

// ---------------------------------------------------------------------------
// Return type
// ---------------------------------------------------------------------------

export interface UseMissionControlReturn {
  // All nodes/edges (positioned with stage offsets)
  nodes: Node[];
  edges: Edge[];

  // Stage state
  stageStatus: Record<PipelineStageType, string>;
  stageNodeCounts: Record<PipelineStageType, number>;

  // Pipeline
  pipelineId: string | null;
  isExecuting: boolean;

  // Actions
  loadPipeline: (id: string) => Promise<void>;
  startBrainDump: (text: string, automationLevel: string) => Promise<string | null>;
  advanceStage: (targetStage: PipelineStageType) => Promise<void>;

  // Selection
  selectedNodeId: string | null;
  selectedNodeData: Record<string, unknown> | null;
  onNodeSelect: (nodeId: string | null) => void;

  // Provenance
  provenanceChain: ProvenanceBreadcrumb[];

  // WebSocket
  wsStatus: string;
  completedStages: string[];
  streamedNodes: PipelineNodeEvent[];

  // Loading
  loading: boolean;
  error: string | null;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useMissionControl(
  initialPipelineId?: string | null,
): UseMissionControlReturn {
  const [pipelineId, setPipelineId] = useState<string | null>(initialPipelineId ?? null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);

  // Per-stage caches
  const stageNodesRef = useRef<Record<PipelineStageType, Node[]>>({ ...EMPTY_STAGE_NODES });
  const stageEdgesRef = useRef<Record<PipelineStageType, Edge[]>>({ ...EMPTY_STAGE_EDGES });
  const [stageNodes, setStageNodes] = useState<Record<PipelineStageType, Node[]>>({ ...EMPTY_STAGE_NODES });
  const [stageEdges, setStageEdges] = useState<Record<PipelineStageType, Edge[]>>({ ...EMPTY_STAGE_EDGES });
  const [stageStatus, setStageStatus] = useState<Record<PipelineStageType, string>>({ ...DEFAULT_STATUS });

  // Selection
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // Provenance links from API
  const provenanceLinksRef = useRef<Array<{
    source_node_id: string;
    target_node_id: string;
    source_stage: string;
    target_stage: string;
    content_hash: string;
    method: string;
  }>>([]);

  const syncCacheToState = useCallback(() => {
    setStageNodes({ ...stageNodesRef.current });
    setStageEdges({ ...stageEdgesRef.current });
  }, []);

  // -- Populate from API result -------------------------------------------
  const populateFromResult = useCallback(
    (result: PipelineResultResponse) => {
      if (result.stage_status) {
        setStageStatus(result.stage_status);
      }

      for (const stage of ALL_STAGES) {
        const stageData = (result as unknown as Record<string, unknown>)[stage] as ReactFlowData | Record<string, unknown> | null;
        stageNodesRef.current[stage] = parseStageNodes(stage, stageData);
        stageEdgesRef.current[stage] = parseStageEdges(stage, stageData);
      }

      if (result.provenance) {
        provenanceLinksRef.current = result.provenance as typeof provenanceLinksRef.current;
      }

      syncCacheToState();
    },
    [syncCacheToState],
  );

  // -- Load pipeline from API ---------------------------------------------
  const loadPipeline = useCallback(
    async (id: string) => {
      setLoading(true);
      setError(null);
      setPipelineId(id);
      try {
        const res = await fetch(`${API_PREFIX}/${id}`);
        if (!res.ok) {
          setError(`Failed to load pipeline: ${res.status}`);
          return;
        }
        const data: PipelineResultResponse = await res.json();
        populateFromResult(data);
      } catch {
        setError('Failed to load pipeline');
      } finally {
        setLoading(false);
      }
    },
    [populateFromResult],
  );

  // -- Start brain dump ---------------------------------------------------
  const startBrainDump = useCallback(
    async (text: string, automationLevel: string): Promise<string | null> => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_PREFIX}/from-ideas`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            ideas: text.split('\n').map((s) => s.trim()).filter(Boolean),
            auto_advance: automationLevel === 'full',
          }),
        });

        if (!res.ok) {
          setError(`Failed to start brain dump: ${res.status}`);
          return null;
        }

        const data = await res.json();
        const newId = data.pipeline_id as string;
        setPipelineId(newId);

        if (data.result) {
          populateFromResult(data.result as PipelineResultResponse);
        }

        return newId;
      } catch {
        setError('Failed to start brain dump');
        return null;
      } finally {
        setLoading(false);
      }
    },
    [populateFromResult],
  );

  // -- Advance stage ------------------------------------------------------
  const advanceStage = useCallback(
    async (targetStage: PipelineStageType) => {
      if (!pipelineId) return;
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_PREFIX}/advance`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            pipeline_id: pipelineId,
            target_stage: targetStage,
          }),
        });

        if (!res.ok) {
          setError(`Failed to advance to ${targetStage}: ${res.status}`);
          return;
        }

        const data = await res.json();
        if (data.result) {
          populateFromResult(data.result as PipelineResultResponse);
        }
      } catch {
        setError(`Failed to advance to ${targetStage}`);
      } finally {
        setLoading(false);
      }
    },
    [pipelineId, populateFromResult],
  );

  // -- Node selection -----------------------------------------------------
  const onNodeSelect = useCallback((nodeId: string | null) => {
    setSelectedNodeId(nodeId);
  }, []);

  const selectedNodeData = useMemo(() => {
    if (!selectedNodeId) return null;
    for (const stage of ALL_STAGES) {
      const node = stageNodes[stage].find((n) => n.id === selectedNodeId);
      if (node) return node.data as Record<string, unknown>;
    }
    return null;
  }, [selectedNodeId, stageNodes]);

  // -- Provenance chain ---------------------------------------------------
  const provenanceChain = useMemo((): ProvenanceBreadcrumb[] => {
    if (!selectedNodeId) return [];

    const chain: ProvenanceBreadcrumb[] = [];
    let currentId = selectedNodeId;
    const visited = new Set<string>();

    while (currentId && !visited.has(currentId)) {
      visited.add(currentId);
      const link = provenanceLinksRef.current.find((l) => l.target_node_id === currentId);
      if (!link) break;

      const sourceStage = link.source_stage as PipelineStageType;
      const sourceNode = stageNodes[sourceStage]?.find((n) => n.id === link.source_node_id);

      chain.unshift({
        nodeId: link.source_node_id,
        nodeLabel: (sourceNode?.data as Record<string, unknown>)?.label as string || link.source_node_id,
        stage: sourceStage,
        contentHash: link.content_hash || '',
        method: link.method || '',
      });

      currentId = link.source_node_id;
    }

    return chain;
  }, [selectedNodeId, stageNodes]);

  // -- Stage node counts --------------------------------------------------
  const stageNodeCounts = useMemo(() => {
    const counts: Record<PipelineStageType, number> = {
      ideas: 0,
      principles: 0,
      goals: 0,
      actions: 0,
      orchestration: 0,
    };
    for (const stage of ALL_STAGES) {
      counts[stage] = stageNodes[stage].length;
    }
    return counts;
  }, [stageNodes]);

  // -- Assemble all nodes/edges with stage offsets ------------------------
  const { nodes, edges } = useMemo(() => {
    const allNodes: Node[] = [];
    const allEdges: Edge[] = [];

    for (const stage of ALL_STAGES) {
      const offsetX = STAGE_OFFSET_X[stage];
      for (const n of stageNodes[stage]) {
        allNodes.push({
          ...n,
          position: { x: n.position.x + offsetX, y: n.position.y },
        });
      }
      for (const e of stageEdges[stage]) {
        const stageColor = PIPELINE_STAGE_CONFIG[stage].primary;
        allEdges.push({
          ...e,
          style: {
            stroke: stageColor,
            strokeWidth: 2,
            ...(e.style || {}),
          },
          animated: e.animated ?? true,
        });
      }
    }

    return { nodes: allNodes, edges: allEdges };
  }, [stageNodes, stageEdges]);

  // -- WebSocket integration ----------------------------------------------
  const handleStageCompleted = useCallback(
    (event: PipelineStageEvent) => {
      const stage = event.stage as PipelineStageType;
      setStageStatus((prev) => ({ ...prev, [stage]: 'complete' }));

      // Reload stage data
      if (pipelineId) {
        fetch(`${API_PREFIX}/${pipelineId}/stage/${stage}`)
          .then((res) => res.ok ? res.json() : null)
          .then((data) => {
            if (!data) return;
            const stageData = data.data ?? data;
            stageNodesRef.current[stage] = parseStageNodes(stage, stageData);
            stageEdgesRef.current[stage] = parseStageEdges(stage, stageData);
            syncCacheToState();
          })
          .catch(() => { /* retain cache */ });
      }
    },
    [pipelineId, syncCacheToState],
  );

  const {
    status: wsStatus,
    completedStages,
    streamedNodes,
  } = usePipelineWebSocket({
    pipelineId: pipelineId ?? undefined,
    enabled: !!pipelineId,
    onStageCompleted: handleStageCompleted,
    onCompleted: () => setIsExecuting(false),
    onFailed: () => setIsExecuting(false),
  });

  // -- Initial load -------------------------------------------------------
  useEffect(() => {
    if (initialPipelineId) {
      loadPipeline(initialPipelineId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialPipelineId]);

  return {
    nodes,
    edges,
    stageStatus,
    stageNodeCounts,
    pipelineId,
    isExecuting,
    loadPipeline,
    startBrainDump,
    advanceStage,
    selectedNodeId,
    selectedNodeData,
    onNodeSelect,
    provenanceChain,
    wsStatus: wsStatus as string,
    completedStages,
    streamedNodes,
    loading,
    error,
  };
}

export default useMissionControl;
