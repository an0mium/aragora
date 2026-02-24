'use client';

/**
 * useUnifiedDAG - Full-lifecycle hook for the Unified DAG Canvas.
 *
 * Manages a server-side UniversalGraph and projects it to React Flow
 * Node[]/Edge[] for the canvas.  Exposes AI operations (debate, decompose,
 * prioritize, assign, execute, find-precedents) plus bulk ops (cluster,
 * auto-flow) and undo/redo.
 */

import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import type { Node, Edge } from '@xyflow/react';
import { useSWRFetch } from './useSWRFetch';
import { apiFetch } from '@/lib/api';
import type { ExecutionHistoryEntry } from '@/components/unified-dag/ExecutionSidebar';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type DAGStage = 'ideas' | 'goals' | 'actions' | 'orchestration';

export interface DAGNodeData {
  label: string;
  description: string;
  stage: DAGStage;
  subtype: string;
  status: string;
  priority: number;
  metadata: Record<string, unknown>;
  [key: string]: unknown;
}

export interface DAGOperationResult {
  success: boolean;
  message: string;
  created_nodes: string[];
  metadata: Record<string, unknown>;
}

interface GraphSnapshot {
  nodes: Node<DAGNodeData>[];
  edges: Edge[];
}

// Stage → swim-lane x position
const STAGE_X: Record<DAGStage, number> = {
  ideas: 0,
  goals: 300,
  actions: 600,
  orchestration: 900,
};

// Stage → color hint
export const STAGE_COLORS: Record<DAGStage, string> = {
  ideas: '#6366f1',       // indigo
  goals: '#10b981',       // emerald
  actions: '#f59e0b',     // amber
  orchestration: '#ec4899', // pink
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function serverNodeToReactFlow(n: Record<string, unknown>, yIndex: number): Node<DAGNodeData> {
  const stage = (n.stage as DAGStage) || 'ideas';
  return {
    id: n.id as string,
    type: `${stage}Node`,
    position: { x: STAGE_X[stage] + Math.random() * 60, y: 80 + yIndex * 120 },
    data: {
      label: (n.label as string) || '',
      description: (n.description as string) || '',
      stage,
      subtype: (n.subtype as string) || '',
      status: (n.status as string) || 'pending',
      priority: (n.priority as number) || 0,
      metadata: (n.metadata as Record<string, unknown>) || {},
    },
  };
}

function serverEdgeToReactFlow(e: Record<string, unknown>): Edge {
  return {
    id: (e.id as string) || `${e.source}-${e.target}`,
    source: e.source as string,
    target: e.target as string,
    label: (e.label as string) || (e.edge_type as string) || undefined,
    animated: (e.edge_type as string) === 'SIMILARITY',
    style: { stroke: '#6366f1' },
  };
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

const API_PREFIX = '/api/v1/pipeline/dag';

export function useUnifiedDAG(graphId: string | null) {
  // React Flow state
  const [nodes, setNodes] = useState<Node<DAGNodeData>[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [operationLoading, setOperationLoading] = useState(false);
  const [operationError, setOperationError] = useState<string | null>(null);

  // Undo/redo
  const undoStack = useRef<GraphSnapshot[]>([]);
  const redoStack = useRef<GraphSnapshot[]>([]);

  // Fetch initial graph
  const { data: graphData, mutate: mutateGraph } = useSWRFetch<{ data: Record<string, unknown> }>(
    graphId ? `${API_PREFIX}/${graphId}` : null
  );

  // Sync server graph → React Flow
  useEffect(() => {
    if (!graphData?.data) return;
    const g = graphData.data;
    const serverNodes = (g.nodes as Record<string, unknown>[]) || [];
    const serverEdges = (g.edges as Record<string, unknown>[]) || [];
    setNodes(serverNodes.map((n, i) => serverNodeToReactFlow(n, i)));
    setEdges(serverEdges.map(serverEdgeToReactFlow));
  }, [graphData]);

  // -------------------------------------------------------------------------
  // Snapshot helpers
  // -------------------------------------------------------------------------

  const pushUndo = useCallback(() => {
    undoStack.current.push({ nodes: [...nodes], edges: [...edges] });
    redoStack.current = [];
  }, [nodes, edges]);

  const undo = useCallback(() => {
    const snap = undoStack.current.pop();
    if (!snap) return;
    redoStack.current.push({ nodes, edges });
    setNodes(snap.nodes);
    setEdges(snap.edges);
  }, [nodes, edges]);

  const redo = useCallback(() => {
    const snap = redoStack.current.pop();
    if (!snap) return;
    undoStack.current.push({ nodes, edges });
    setNodes(snap.nodes);
    setEdges(snap.edges);
  }, [nodes, edges]);

  // -------------------------------------------------------------------------
  // Graph CRUD
  // -------------------------------------------------------------------------

  const addNode = useCallback((node: Node<DAGNodeData>) => {
    pushUndo();
    setNodes((prev) => [...prev, node]);
  }, [pushUndo]);

  const updateNode = useCallback((id: string, data: Partial<DAGNodeData>) => {
    pushUndo();
    setNodes((prev) =>
      prev.map((n) => (n.id === id ? { ...n, data: { ...n.data, ...data } } : n))
    );
  }, [pushUndo]);

  const deleteNode = useCallback((id: string) => {
    pushUndo();
    setNodes((prev) => prev.filter((n) => n.id !== id));
    setEdges((prev) => prev.filter((e) => e.source !== id && e.target !== id));
  }, [pushUndo]);

  const addEdge = useCallback((edge: Edge) => {
    pushUndo();
    setEdges((prev) => [...prev, edge]);
  }, [pushUndo]);

  const deleteEdge = useCallback((id: string) => {
    pushUndo();
    setEdges((prev) => prev.filter((e) => e.id !== id));
  }, [pushUndo]);

  // -------------------------------------------------------------------------
  // AI Operations
  // -------------------------------------------------------------------------

  const runOperation = useCallback(async (
    nodeId: string,
    operation: string,
    body?: Record<string, unknown>,
  ): Promise<DAGOperationResult | null> => {
    if (!graphId) return null;
    setOperationLoading(true);
    setOperationError(null);
    try {
      const result = await apiFetch<{ data: DAGOperationResult }>(
        `${API_PREFIX}/${graphId}/nodes/${nodeId}/${operation}`,
        { method: 'POST', body: JSON.stringify(body || {}) },
      );
      pushUndo();
      await mutateGraph();
      return result.data ?? null;
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Operation failed';
      setOperationError(msg);
      return null;
    } finally {
      setOperationLoading(false);
    }
  }, [graphId, pushUndo, mutateGraph]);

  const debateNode = useCallback((nodeId: string, agents?: string[], rounds?: number) =>
    runOperation(nodeId, 'debate', { agents, rounds }), [runOperation]);

  const decomposeNode = useCallback((nodeId: string) =>
    runOperation(nodeId, 'decompose'), [runOperation]);

  const prioritizeChildren = useCallback((nodeId: string) =>
    runOperation(nodeId, 'prioritize'), [runOperation]);

  const assignAgents = useCallback((nodeId: string) =>
    runOperation(nodeId, 'assign-agents'), [runOperation]);

  const executeNode = useCallback((nodeId: string) =>
    runOperation(nodeId, 'execute'), [runOperation]);

  const findPrecedents = useCallback((nodeId: string, maxResults?: number) =>
    runOperation(nodeId, 'find-precedents', { max_results: maxResults }), [runOperation]);

  // -------------------------------------------------------------------------
  // Bulk Operations
  // -------------------------------------------------------------------------

  const clusterIdeas = useCallback(async (ideas: string[], threshold?: number): Promise<DAGOperationResult | null> => {
    if (!graphId) return null;
    setOperationLoading(true);
    setOperationError(null);
    try {
      const result = await apiFetch<{ data: DAGOperationResult }>(
        `${API_PREFIX}/${graphId}/cluster-ideas`,
        { method: 'POST', body: JSON.stringify({ ideas, threshold }) },
      );
      pushUndo();
      await mutateGraph();
      return result.data ?? null;
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Clustering failed';
      setOperationError(msg);
      return null;
    } finally {
      setOperationLoading(false);
    }
  }, [graphId, pushUndo, mutateGraph]);

  const autoFlow = useCallback(async (ideas: string[], config?: Record<string, unknown>): Promise<DAGOperationResult | null> => {
    if (!graphId) return null;
    setOperationLoading(true);
    setOperationError(null);
    try {
      const result = await apiFetch<{ data: DAGOperationResult }>(
        `${API_PREFIX}/${graphId}/auto-flow`,
        { method: 'POST', body: JSON.stringify({ ideas, config }) },
      );
      pushUndo();
      await mutateGraph();
      return result.data ?? null;
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Auto-flow failed';
      setOperationError(msg);
      return null;
    } finally {
      setOperationLoading(false);
    }
  }, [graphId, pushUndo, mutateGraph]);

  // -------------------------------------------------------------------------
  // Validation
  // -------------------------------------------------------------------------

  const validateGraph = useCallback((): string[] => {
    const errors: string[] = [];
    if (nodes.length === 0) {
      errors.push('Graph is empty — add at least one idea node');
      return errors;
    }

    const stages: DAGStage[] = ['ideas', 'goals', 'actions', 'orchestration'];
    const byStage: Record<DAGStage, Node<DAGNodeData>[]> = {
      ideas: [], goals: [], actions: [], orchestration: [],
    };
    for (const n of nodes) {
      const s = (n.data as unknown as DAGNodeData).stage;
      if (byStage[s]) byStage[s].push(n);
    }

    if (byStage.ideas.length === 0) {
      errors.push('No idea nodes — ideas are required to start the pipeline');
    }

    // Check that each non-first stage has at least one incoming edge from a previous stage
    for (let i = 1; i < stages.length; i++) {
      const stage = stages[i];
      if (byStage[stage].length === 0) continue;
      const prevStageNodeIds = new Set(byStage[stages[i - 1]].map((n) => n.id));
      const hasIncoming = byStage[stage].some((n) =>
        edges.some((e) => e.target === n.id && prevStageNodeIds.has(e.source))
      );
      if (!hasIncoming) {
        errors.push(`${stage} nodes have no connections from ${stages[i - 1]} — add cross-stage edges`);
      }
    }

    // Check for orphan nodes with no edges
    const nodesWithEdges = new Set<string>();
    for (const e of edges) {
      nodesWithEdges.add(e.source);
      nodesWithEdges.add(e.target);
    }
    const orphans = nodes.filter((n) => !nodesWithEdges.has(n.id));
    if (orphans.length > 0 && nodes.length > 1) {
      errors.push(`${orphans.length} orphan node(s) with no connections`);
    }

    return errors;
  }, [nodes, edges]);

  // -------------------------------------------------------------------------
  // Batch Execution
  // -------------------------------------------------------------------------

  const [executionHistory, setExecutionHistory] = useState<ExecutionHistoryEntry[]>([]);
  const [batchExecuting, setBatchExecuting] = useState(false);

  const executeAllReady = useCallback(async (): Promise<void> => {
    if (!graphId) return;
    const readyNodes = nodes.filter(
      (n) => (n.data as unknown as DAGNodeData).status === 'ready'
    );
    if (readyNodes.length === 0) return;

    setBatchExecuting(true);
    pushUndo();

    // Mark all ready nodes as running
    setNodes((prev) =>
      prev.map((n) => {
        if ((n.data as unknown as DAGNodeData).status === 'ready') {
          return { ...n, data: { ...n.data, status: 'running' } as DAGNodeData };
        }
        return n;
      })
    );

    try {
      const result = await apiFetch<{ data: { results: Array<{ node_id: string; status: string; duration_ms: number }> } }>(
        `${API_PREFIX}/${graphId}/execute-batch`,
        { method: 'POST', body: JSON.stringify({ node_ids: readyNodes.map((n) => n.id) }) },
      );

      const batchResults = result?.data?.results || [];
      const newHistory: ExecutionHistoryEntry[] = batchResults.map((r) => {
        const node = readyNodes.find((n) => n.id === r.node_id);
        return {
          id: `${r.node_id}-${Date.now()}`,
          nodeId: r.node_id,
          nodeLabel: (node?.data as unknown as DAGNodeData)?.label || r.node_id,
          status: r.status === 'succeeded' ? 'succeeded' : 'failed',
          durationMs: r.duration_ms || 0,
          timestamp: Date.now(),
        };
      });
      setExecutionHistory((prev) => [...newHistory, ...prev]);

      // Update node statuses from batch results
      setNodes((prev) =>
        prev.map((n) => {
          const batchResult = batchResults.find((r) => r.node_id === n.id);
          if (batchResult) {
            return { ...n, data: { ...n.data, status: batchResult.status } as DAGNodeData };
          }
          return n;
        })
      );

      await mutateGraph();
    } catch (err) {
      // On failure, revert running nodes back to ready
      setNodes((prev) =>
        prev.map((n) => {
          if ((n.data as unknown as DAGNodeData).status === 'running') {
            return { ...n, data: { ...n.data, status: 'ready' } as DAGNodeData };
          }
          return n;
        })
      );
      setOperationError(err instanceof Error ? err.message : 'Batch execution failed');
    } finally {
      setBatchExecuting(false);
    }
  }, [graphId, nodes, pushUndo, mutateGraph]);

  const autoAdvanceAll = useCallback(async (): Promise<void> => {
    if (!graphId) return;
    setBatchExecuting(true);
    setOperationError(null);
    try {
      await apiFetch<{ data: DAGOperationResult }>(
        `${API_PREFIX}/${graphId}/auto-advance`,
        { method: 'POST', body: JSON.stringify({}) },
      );
      pushUndo();
      await mutateGraph();
    } catch (err) {
      setOperationError(err instanceof Error ? err.message : 'Auto-advance failed');
    } finally {
      setBatchExecuting(false);
    }
  }, [graphId, pushUndo, mutateGraph]);

  // Computed stats
  const graphStats = useMemo(() => {
    const total = nodes.length;
    const succeeded = nodes.filter((n) => (n.data as unknown as DAGNodeData).status === 'succeeded').length;
    const ready = nodes.filter((n) => (n.data as unknown as DAGNodeData).status === 'ready').length;
    const running = nodes.filter((n) => (n.data as unknown as DAGNodeData).status === 'running').length;
    const failed = nodes.filter((n) => (n.data as unknown as DAGNodeData).status === 'failed').length;
    return { total, succeeded, ready, running, failed, completionPct: total > 0 ? Math.round((succeeded / total) * 100) : 0 };
  }, [nodes]);

  return {
    // Graph state
    nodes,
    edges,
    setNodes,
    setEdges,

    // CRUD
    addNode,
    updateNode,
    deleteNode,
    addEdge,
    deleteEdge,

    // AI operations
    debateNode,
    decomposeNode,
    prioritizeChildren,
    assignAgents,
    executeNode,
    findPrecedents,

    // Bulk operations
    clusterIdeas,
    autoFlow,

    // Execution
    executeAllReady,
    autoAdvanceAll,
    validateGraph,
    executionHistory,
    batchExecuting,
    graphStats,

    // State
    operationLoading,
    operationError,

    // Undo/redo
    undo,
    redo,
    canUndo: undoStack.current.length > 0,
    canRedo: redoStack.current.length > 0,

    // Refresh
    refresh: mutateGraph,
  };
}
