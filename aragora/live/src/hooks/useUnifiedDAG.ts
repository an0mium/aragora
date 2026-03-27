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
import type { PipelineResultResponse } from '@/components/pipeline-canvas/types';

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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function normalizeStage(value: unknown): DAGStage {
  switch (String(value || '').toLowerCase()) {
    case 'goals':
      return 'goals';
    case 'actions':
      return 'actions';
    case 'orchestration':
      return 'orchestration';
    default:
      return 'ideas';
  }
}

function normalizeStatus(...candidates: unknown[]): string {
  for (const candidate of candidates) {
    const normalized = String(candidate || '').trim().toLowerCase();
    if (!normalized) continue;
    switch (normalized) {
      case 'succeeded':
      case 'complete':
      case 'completed':
      case 'approved':
        return 'succeeded';
      case 'in_progress':
      case 'running':
      case 'active':
        return 'running';
      case 'ready':
        return 'ready';
      case 'failed':
      case 'rejected':
        return 'failed';
      case 'partial':
      case 'awaiting_human':
      case 'archived':
      case 'blocked':
        return 'blocked';
      case 'pending':
        return 'pending';
      default:
        return normalized;
    }
  }
  return 'pending';
}

function priorityToRank(value: unknown): number {
  switch (String(value || '').toLowerCase()) {
    case 'critical':
      return 4;
    case 'high':
      return 3;
    case 'medium':
      return 2;
    case 'low':
      return 1;
    default:
      return 0;
  }
}

function getNumericPosition(value: unknown): { x: number; y: number } | null {
  if (!isRecord(value)) return null;
  const x = Number(value.x);
  const y = Number(value.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x, y };
}

function getLanePosition(stage: DAGStage, index: number, rawPosition?: unknown): { x: number; y: number } {
  const position = getNumericPosition(rawPosition);
  if (!position) {
    return { x: STAGE_X[stage] + 40, y: 80 + index * 120 };
  }
  const laneOffset = Math.max(24, Math.min(220, position.x));
  return {
    x: STAGE_X[stage] + laneOffset,
    y: Math.max(40, position.y),
  };
}

function ensureStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item)).filter(Boolean);
}

function createDAGEdge(edge: {
  id: string;
  source: string;
  target: string;
  label?: string;
  crossStage?: boolean;
  animated?: boolean;
  color?: string;
}): Edge {
  return {
    id: edge.id,
    source: edge.source,
    target: edge.target,
    type: edge.crossStage ? 'crossStage' : 'default',
    label: edge.label,
    animated: edge.animated ?? edge.crossStage ?? false,
    style: { stroke: edge.color ?? '#6366f1' },
    data: edge.crossStage ? { crossStage: true } : undefined,
  };
}

function serverNodeToReactFlow(n: Record<string, unknown>, yIndex: number): Node<DAGNodeData> {
  const data = isRecord(n.data) ? n.data : {};
  const metadata = isRecord(n.metadata) ? n.metadata : {};
  const stage = normalizeStage(n.stage ?? data.stage);
  const position = getLanePosition(stage, yIndex, {
    x: n.position_x ?? (isRecord(n.position) ? n.position.x : undefined),
    y: n.position_y ?? (isRecord(n.position) ? n.position.y : undefined),
  });
  const assignedAgent = data.assigned_agent ?? data.assignedAgent ?? metadata.assigned_agent;
  const capabilities = ensureStringArray(data.capabilities ?? metadata.capabilities);
  return {
    id: n.id as string,
    type: `${stage}Node`,
    position,
    data: {
      label: String(n.label ?? data.label ?? ''),
      description: String(n.description ?? data.description ?? data.full_content ?? data.output_preview ?? ''),
      stage,
      subtype: String(n.node_subtype ?? n.subtype ?? data.node_subtype ?? data.subtype ?? ''),
      status: normalizeStatus(
        n.execution_status,
        data.execution_status,
        n.status,
        data.status,
        n.approval_status,
        data.approval_status,
      ),
      priority: priorityToRank(data.priority ?? metadata.priority),
      metadata: {
        ...metadata,
        ...(isRecord(data.metadata) ? data.metadata : {}),
        ...(assignedAgent ? { agents: [String(assignedAgent)] } : {}),
        ...(capabilities.length > 0 ? { capabilities } : {}),
      },
      confidence: n.confidence ?? data.confidence,
      approvalStatus: n.approval_status ?? data.approval_status,
      executionStatus: n.execution_status ?? data.execution_status,
      assignedAgent,
      capabilities,
    },
  };
}

function serverEdgeToReactFlow(e: Record<string, unknown>): Edge {
  const source = String(e.source ?? e.source_id ?? '');
  const target = String(e.target ?? e.target_id ?? '');
  return createDAGEdge({
    id: String(e.id ?? `${source}-${target}`),
    source,
    target,
    label: String(e.label ?? e.edge_type ?? '').trim() || undefined,
    crossStage: Boolean(e.cross_stage ?? (isRecord(e.data) && e.data.crossStage)),
    animated: Boolean(e.cross_stage),
  });
}

function pipelineResultToReactFlowGraph(result: PipelineResultResponse): GraphSnapshot {
  const graphNodes: Node<DAGNodeData>[] = [];
  const graphEdges: Edge[] = [];
  const seenEdges = new Set<string>();
  const nodeIds = new Set<string>();

  const addEdge = (edge: Edge | null) => {
    if (!edge) return;
    if (!edge.source || !edge.target) return;
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) return;
    if (seenEdges.has(edge.id)) return;
    seenEdges.add(edge.id);
    graphEdges.push(edge);
  };

  const stageStatus = result.stage_status ?? {};

  const ideas = Array.isArray(result.ideas?.nodes) ? result.ideas.nodes : [];
  ideas.forEach((node, index) => {
    const data = isRecord(node.data) ? node.data : {};
    const dagNode: Node<DAGNodeData> = {
      id: String(node.id ?? `idea-${index}`),
      type: 'ideasNode',
      position: getLanePosition('ideas', index, node.position),
      data: {
        label: String(data.label ?? node.id ?? ''),
        description: String(data.full_content ?? data.fullContent ?? data.description ?? ''),
        stage: 'ideas',
        subtype: String(data.idea_type ?? data.ideaType ?? data.nodeType ?? 'concept'),
        status: normalizeStatus(stageStatus.ideas),
        priority: 0,
        metadata: {
          contentHash: data.content_hash ?? data.contentHash,
          agent: data.agent,
        },
      },
    };
    graphNodes.push(dagNode);
    nodeIds.add(dagNode.id);
  });

  const goalsData = isRecord(result.goals) ? result.goals : {};
  const goals = Array.isArray(goalsData.goals) ? goalsData.goals.filter(isRecord) : [];
  goals.forEach((goal, index) => {
    const dagNode: Node<DAGNodeData> = {
      id: String(goal.id ?? `goal-${index}`),
      type: 'goalsNode',
      position: getLanePosition('goals', index),
      data: {
        label: String(goal.title ?? goal.label ?? goal.id ?? ''),
        description: String(goal.description ?? ''),
        stage: 'goals',
        subtype: String(goal.type ?? 'goal'),
        status: normalizeStatus(goal.status, stageStatus.goals),
        priority: priorityToRank(goal.priority),
        metadata: {
          confidence: goal.confidence,
          sourceIdeaIds: ensureStringArray(goal.source_idea_ids),
          measurable: goal.measurable,
          priority: goal.priority,
        },
      },
    };
    graphNodes.push(dagNode);
    nodeIds.add(dagNode.id);
  });

  const actions = Array.isArray(result.actions?.nodes) ? result.actions.nodes : [];
  actions.forEach((node, index) => {
    const data = isRecord(node.data) ? node.data : {};
    const dagNode: Node<DAGNodeData> = {
      id: String(node.id ?? `action-${index}`),
      type: 'actionsNode',
      position: getLanePosition('actions', index, node.position),
      data: {
        label: String(data.label ?? node.id ?? ''),
        description: String(data.description ?? ''),
        stage: 'actions',
        subtype: String(data.step_type ?? data.stepType ?? 'task'),
        status: normalizeStatus(data.status, stageStatus.actions),
        priority: priorityToRank(data.priority),
        metadata: {
          optional: data.optional,
          assignee: data.assignee,
          sourceGoalId: data.source_goal_id ?? data.sourceGoalId,
          timeout: data.timeout ?? data.timeoutSeconds,
        },
      },
    };
    graphNodes.push(dagNode);
    nodeIds.add(dagNode.id);
  });

  const orchestration = Array.isArray(result.orchestration?.nodes) ? result.orchestration.nodes : [];
  orchestration.forEach((node, index) => {
    const data = isRecord(node.data) ? node.data : {};
    const assignedAgent = data.assigned_agent ?? data.assignedAgent;
    const capabilities = ensureStringArray(data.capabilities);
    const dagNode: Node<DAGNodeData> = {
      id: String(node.id ?? `orch-${index}`),
      type: 'orchestrationNode',
      position: getLanePosition('orchestration', index, node.position),
      data: {
        label: String(data.label ?? node.id ?? ''),
        description: String(
          data.description
            ?? data.output_preview
            ?? data.outputPreview
            ?? data.selection_rationale
            ?? data.selectionRationale
            ?? '',
        ),
        stage: 'orchestration',
        subtype: String(data.orch_type ?? data.orchType ?? 'agent_task'),
        status: normalizeStatus(data.execution_status, data.executionStatus, data.status, stageStatus.orchestration),
        priority: priorityToRank(data.priority),
        metadata: {
          agents: assignedAgent ? [String(assignedAgent)] : [],
          capabilities,
          agentType: data.agent_type ?? data.agentType,
          executionStatus: data.execution_status ?? data.executionStatus,
          approvalStatus: data.approval_status ?? data.approvalStatus,
          eloScore: data.elo_score ?? data.eloScore,
          outputPreview: data.output_preview ?? data.outputPreview,
        },
      },
    };
    graphNodes.push(dagNode);
    nodeIds.add(dagNode.id);
  });

  const stageEdges = [
    ...(Array.isArray(result.ideas?.edges) ? result.ideas.edges : []),
    ...(Array.isArray(result.actions?.edges) ? result.actions.edges : []),
    ...(Array.isArray(result.orchestration?.edges) ? result.orchestration.edges : []),
  ];

  stageEdges.forEach((edge) => {
    addEdge(
      createDAGEdge({
        id: String(edge.id ?? `${edge.source}-${edge.target}`),
        source: String(edge.source ?? ''),
        target: String(edge.target ?? ''),
        label: typeof edge.label === 'string' ? edge.label : undefined,
      }),
    );
  });

  goals.forEach((goal) => {
    const target = String(goal.id ?? '');
    ensureStringArray(goal.dependencies).forEach((dependencyId) => {
      addEdge(
        createDAGEdge({
          id: `goal-dep-${dependencyId}-${target}`,
          source: dependencyId,
          target,
          label: 'depends on',
          color: STAGE_COLORS.goals,
        }),
      );
    });
  });

  const provenance = Array.isArray(result.provenance) ? result.provenance.filter(isRecord) : [];
  provenance.forEach((link, index) => {
    addEdge(
      createDAGEdge({
        id: `prov-${String(link.source_node_id ?? index)}-${String(link.target_node_id ?? index)}-${index}`,
        source: String(link.source_node_id ?? ''),
        target: String(link.target_node_id ?? ''),
        label: String(link.method ?? 'derived from'),
        crossStage: true,
      }),
    );
  });

  return { nodes: graphNodes, edges: graphEdges };
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

const API_PREFIX = '/api/v1/pipeline/dag';

export function useUnifiedDAG(
  graphId: string | null,
  pipelineData?: PipelineResultResponse | null,
) {
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
    const embeddedGraph = isRecord(pipelineData?.universal_graph) ? pipelineData.universal_graph : null;
    const activeGraph = graphData?.data ?? embeddedGraph ?? null;

    if (isRecord(activeGraph) && Array.isArray(activeGraph.nodes) && Array.isArray(activeGraph.edges)) {
      const serverNodes = activeGraph.nodes.filter(isRecord);
      const serverEdges = activeGraph.edges.filter(isRecord);
      setNodes(serverNodes.map((node, index) => serverNodeToReactFlow(node, index)));
      setEdges(serverEdges.map(serverEdgeToReactFlow));
      return;
    }

    if (pipelineData) {
      const pipelineGraph = pipelineResultToReactFlowGraph(pipelineData);
      setNodes(pipelineGraph.nodes);
      setEdges(pipelineGraph.edges);
      return;
    }

    setNodes([]);
    setEdges([]);
  }, [graphData, pipelineData]);

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
