'use client';

/**
 * useUnifiedDAG - Full-lifecycle hook for the Unified DAG Canvas.
 *
 * Normalizes the pipeline DAG snapshot contract, projects it into React Flow,
 * and exposes DAG operations plus execution helpers for the canvas UI.
 */

import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import type { Edge, Node } from '@xyflow/react';
import { useSWRFetch } from './useSWRFetch';
import { apiFetch } from '@/lib/api';
import type { ExecutionHistoryEntry } from '@/components/unified-dag/ExecutionSidebar';

export type DAGStage = 'ideas' | 'principles' | 'goals' | 'actions' | 'orchestration';
export type DAGNodeStatus =
  | 'pending'
  | 'ready'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'blocked'
  | 'awaiting_human';
export type DAGStageStatus = 'pending' | 'in_progress' | 'complete' | 'partial' | 'failed' | 'awaiting_human';

export interface DAGNodeData {
  label: string;
  description: string;
  stage: DAGStage;
  subtype: string;
  status: DAGNodeStatus;
  runtimeStatus: string;
  priority: number;
  metadata: Record<string, unknown>;
  confidence: number;
  assignedAgent: string | null;
  dependencyCount: number;
  downstreamCount: number;
  blockedBy: string[];
  canExecute: boolean;
  outputPreview?: string;
  progress?: number;
  stageLabel?: string;
  stageStatus?: DAGStageStatus;
  [key: string]: unknown;
}

export interface DAGEdgeData {
  edgeType: string;
  sourceStage: DAGStage;
  targetStage: DAGStage;
  blocking: boolean;
  implicit: boolean;
  label: string;
  crossStage: boolean;
}

export interface DAGOperationResult {
  success: boolean;
  message: string;
  created_nodes: string[];
  metadata: Record<string, unknown>;
}

export interface DAGStageSummary {
  stage: DAGStage;
  label: string;
  order: number;
  status: DAGStageStatus;
  node_ids: string[];
  dependency_stage_ids: string[];
  node_count: number;
  status_counts: Record<string, number>;
  metadata: Record<string, unknown>;
}

interface NormalizedDAGNode {
  id: string;
  stage: DAGStage;
  subtype: string;
  label: string;
  description: string;
  status: DAGNodeStatus;
  runtimeStatus: string;
  priority: number;
  metadata: Record<string, unknown>;
  confidence: number;
  assignedAgent: string | null;
  dependencyCount: number;
  downstreamCount: number;
  blockedBy: string[];
  canExecute: boolean;
  outputPreview?: string;
  progress?: number;
  positionX?: number;
  positionY?: number;
  width?: number;
  height?: number;
}

interface DAGDependency {
  id: string;
  source_id: string;
  target_id: string;
  edge_type: string;
  source_stage: DAGStage;
  target_stage: DAGStage;
  label: string;
  cross_stage: boolean;
  blocking: boolean;
  implicit: boolean;
  metadata: Record<string, unknown>;
}

interface DAGSnapshot {
  graphId: string;
  name: string;
  nodes: NormalizedDAGNode[];
  dependencies: DAGDependency[];
  stages: DAGStageSummary[];
  stageStatus: Partial<Record<DAGStage, DAGStageStatus>>;
  dependencyMap: Record<string, string[]>;
  metadata: Record<string, unknown>;
  integrityHash: string;
  generatedAt: number;
}

interface GraphSnapshot {
  nodes: Node<DAGNodeData>[];
  edges: Edge<DAGEdgeData>[];
}

const API_PREFIX = '/api/v1/pipeline/dag';

export const DAG_STAGE_ORDER: DAGStage[] = [
  'ideas',
  'principles',
  'goals',
  'actions',
  'orchestration',
];

export const DAG_STAGE_LABELS: Record<DAGStage, string> = {
  ideas: 'Ideas',
  principles: 'Principles',
  goals: 'Goals',
  actions: 'Actions',
  orchestration: 'Orchestration',
};

export const STAGE_COLORS: Record<DAGStage, string> = {
  ideas: '#6366f1',
  principles: '#8b5cf6',
  goals: '#10b981',
  actions: '#f59e0b',
  orchestration: '#ec4899',
};

export const STAGE_LANE_WIDTH = 292;
export const STAGE_LANE_GAP = 28;
export const STAGE_LANE_PADDING_X = 28;
export const STAGE_LANE_PADDING_TOP = 92;
export const DAG_NODE_HEIGHT = 164;
export const DAG_NODE_ROW_GAP = 28;

const SUCCESS_STATUSES = new Set(['approved', 'complete', 'completed', 'succeeded']);
const FAILED_STATUSES = new Set(['error', 'failed', 'federation_error', 'rejected', 'timeout']);
const RUNNING_STATUSES = new Set(['active', 'in_progress', 'running']);
const WAITING_STATUSES = new Set(['awaiting_approval', 'awaiting_human', 'waiting_approval']);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined;
}

function asNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

function asRecordArray(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

function normalizeStage(value: unknown): DAGStage | null {
  if (typeof value !== 'string') {
    return null;
  }
  return DAG_STAGE_ORDER.includes(value as DAGStage) ? (value as DAGStage) : null;
}

function normalizeStageStatus(value: unknown): DAGStageStatus | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }
  const status = value.trim().toLowerCase();
  switch (status) {
    case 'complete':
    case 'completed':
      return 'complete';
    case 'running':
    case 'in_progress':
      return 'in_progress';
    case 'failed':
      return 'failed';
    case 'awaiting_approval':
    case 'awaiting_human':
    case 'waiting_approval':
      return 'awaiting_human';
    case 'partial':
      return 'partial';
    default:
      return 'pending';
  }
}

function normalizeNodeStatus(rawStatus: unknown, approvalStatus: unknown): DAGNodeStatus {
  const status = typeof rawStatus === 'string' ? rawStatus.trim().toLowerCase() : '';
  const approval = typeof approvalStatus === 'string' ? approvalStatus.trim().toLowerCase() : '';

  if (SUCCESS_STATUSES.has(status)) {
    return 'succeeded';
  }
  if (FAILED_STATUSES.has(status)) {
    return 'failed';
  }
  if (WAITING_STATUSES.has(status) || WAITING_STATUSES.has(approval)) {
    return 'awaiting_human';
  }
  if (RUNNING_STATUSES.has(status)) {
    return 'running';
  }
  if (status === 'ready') {
    return 'ready';
  }
  if (status === 'blocked') {
    return 'blocked';
  }
  return 'pending';
}

function aggregateStageStatus(
  statuses: DAGNodeStatus[],
  explicitStatus?: DAGStageStatus,
): DAGStageStatus {
  if (explicitStatus) {
    return explicitStatus;
  }
  if (statuses.length === 0) {
    return 'pending';
  }
  const statusSet = new Set(statuses);
  if (statusSet.size === 1 && statusSet.has('succeeded')) {
    return 'complete';
  }
  if (statusSet.has('failed')) {
    return 'failed';
  }
  if (statusSet.has('awaiting_human')) {
    return 'awaiting_human';
  }
  if (statusSet.has('running')) {
    return 'in_progress';
  }
  if (statusSet.has('succeeded')) {
    return 'partial';
  }
  return 'pending';
}

function buildDependencyMap(dependencies: DAGDependency[]): Record<string, string[]> {
  const dependencyMap = new Map<string, Set<string>>();
  for (const dependency of dependencies) {
    const current = dependencyMap.get(dependency.target_id) ?? new Set<string>();
    current.add(dependency.source_id);
    dependencyMap.set(dependency.target_id, current);
  }
  return Object.fromEntries(
    Array.from(dependencyMap.entries())
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([nodeId, deps]) => [nodeId, Array.from(deps).sort()]),
  );
}

function buildDownstreamMap(dependencies: DAGDependency[]): Record<string, string[]> {
  const downstreamMap = new Map<string, Set<string>>();
  for (const dependency of dependencies) {
    const current = downstreamMap.get(dependency.source_id) ?? new Set<string>();
    current.add(dependency.target_id);
    downstreamMap.set(dependency.source_id, current);
  }
  return Object.fromEntries(
    Array.from(downstreamMap.entries())
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([nodeId, deps]) => [nodeId, Array.from(deps).sort()]),
  );
}

function normalizeDependencies(
  payload: Record<string, unknown>,
  stageByNodeId: Record<string, DAGStage>,
): DAGDependency[] {
  const rawDependencies = asRecordArray(payload.dependencies);
  const sourceEdges = rawDependencies.length > 0 ? rawDependencies : asRecordArray(payload.edges);

  return sourceEdges
    .map((dependency, index) => {
      const sourceId = asString(dependency.source_id ?? dependency.source);
      const targetId = asString(dependency.target_id ?? dependency.target);
      const dependencyData = isRecord(dependency.data) ? dependency.data : {};
      if (!sourceId || !targetId) {
        return null;
      }

      const sourceStage =
        normalizeStage(dependency.source_stage ?? dependencyData.sourceStage) ?? stageByNodeId[sourceId];
      const targetStage =
        normalizeStage(dependency.target_stage ?? dependencyData.targetStage) ?? stageByNodeId[targetId];

      if (!sourceStage || !targetStage) {
        return null;
      }

      const edgeType =
        asString(dependency.edge_type ?? dependencyData.edgeType ?? dependency.label) ?? 'relates_to';
      return {
        id: asString(dependency.id) ?? `${sourceId}-${targetId}-${index}`,
        source_id: sourceId,
        target_id: targetId,
        edge_type: edgeType,
        source_stage: sourceStage,
        target_stage: targetStage,
        label: asString(dependency.label) ?? edgeType,
        cross_stage:
          typeof dependency.cross_stage === 'boolean'
            ? dependency.cross_stage
            : sourceStage !== targetStage,
        blocking:
          typeof dependency.blocking === 'boolean'
            ? dependency.blocking
            : Boolean(dependencyData.blocking),
        implicit:
          typeof dependency.implicit === 'boolean'
            ? dependency.implicit
            : Boolean(dependencyData.implicit),
        metadata: isRecord(dependency.metadata)
          ? dependency.metadata
          : isRecord(dependencyData)
            ? dependencyData
            : {},
      } satisfies DAGDependency;
    })
    .filter((dependency): dependency is DAGDependency => dependency !== null)
    .sort((left, right) => {
      const leftSourceIndex = DAG_STAGE_ORDER.indexOf(left.source_stage);
      const rightSourceIndex = DAG_STAGE_ORDER.indexOf(right.source_stage);
      if (leftSourceIndex !== rightSourceIndex) {
        return leftSourceIndex - rightSourceIndex;
      }
      const leftTargetIndex = DAG_STAGE_ORDER.indexOf(left.target_stage);
      const rightTargetIndex = DAG_STAGE_ORDER.indexOf(right.target_stage);
      if (leftTargetIndex !== rightTargetIndex) {
        return leftTargetIndex - rightTargetIndex;
      }
      return left.id.localeCompare(right.id);
    });
}

export function normalizeDagSnapshot(payload: Record<string, unknown>): DAGSnapshot {
  const rawNodes = asRecordArray(payload.nodes);
  const stageByNodeId = Object.fromEntries(
    rawNodes
      .map((node) => {
        const nodeId = asString(node.id);
        const nodeData = isRecord(node.data) ? node.data : {};
        const stage = normalizeStage(node.stage ?? nodeData.stage);
        return nodeId && stage ? [nodeId, stage] : null;
      })
      .filter((entry): entry is [string, DAGStage] => entry !== null),
  );

  const dependencies = normalizeDependencies(payload, stageByNodeId);
  const rawDependencyMap = isRecord(payload.dependency_map) ? payload.dependency_map : null;
  const dependencyMap =
    rawDependencyMap && Object.keys(rawDependencyMap).length > 0
      ? Object.fromEntries(
          Object.entries(rawDependencyMap).map(([nodeId, deps]) => [nodeId, asStringArray(deps)]),
        )
      : buildDependencyMap(dependencies);
  const downstreamMap = buildDownstreamMap(dependencies);
  const runtime = isRecord(payload.runtime) ? payload.runtime : {};

  const baseNodes = rawNodes
    .map((node) => {
      const nodeId = asString(node.id);
      const nodeData = isRecord(node.data) ? node.data : {};
      const stage = normalizeStage(node.stage ?? nodeData.stage);
      if (!nodeId || !stage) {
        return null;
      }

      const nodeMetadata = isRecord(node.metadata) ? node.metadata : {};
      const runtimeEntry = isRecord(runtime[nodeId]) ? runtime[nodeId] : {};
      const runtimeMetadata = isRecord(runtimeEntry.metadata) ? runtimeEntry.metadata : {};
      const metadata = { ...nodeMetadata, ...runtimeMetadata };

      const assignedAgent =
        asString(runtimeEntry.assigned_agent) ??
        asString(nodeData.assigned_agent) ??
        asString(metadata.assigned_agent) ??
        asString(nodeData.agent) ??
        asString(metadata.agent) ??
        null;
      const progress =
        asNumber(nodeData.progress) ??
        asNumber(metadata.progress) ??
        asNumber(runtimeEntry.progress);
      const confidence =
        asNumber(runtimeEntry.confidence) ??
        asNumber(node.confidence) ??
        asNumber(nodeData.confidence) ??
        0;
      const baseStatus = normalizeNodeStatus(
        runtimeEntry.execution_status ?? node.execution_status ?? nodeData.execution_status ?? nodeData.executionStatus,
        runtimeEntry.approval_status ?? node.approval_status ?? nodeData.approval_status ?? nodeData.approvalStatus,
      );

      return {
        id: nodeId,
        stage,
        subtype: asString(node.node_subtype ?? nodeData.subtype) ?? '',
        label: asString(node.label ?? nodeData.label) ?? nodeId,
        description: asString(node.description ?? nodeData.description) ?? '',
        status: baseStatus,
        runtimeStatus:
          asString(
            runtimeEntry.execution_status ??
            node.execution_status ??
            nodeData.execution_status ??
            nodeData.executionStatus,
          ) ??
          baseStatus,
        priority: asNumber(nodeData.priority ?? node.priority ?? metadata.priority) ?? 0,
        metadata,
        confidence,
        assignedAgent,
        dependencyCount: (dependencyMap[nodeId] ?? []).length,
        downstreamCount: (downstreamMap[nodeId] ?? []).length,
        blockedBy: [],
        canExecute: stage === 'orchestration',
        outputPreview:
          asString(metadata.output_preview) ??
          asString(nodeData.output_preview) ??
          asString(runtimeEntry.output_preview),
        progress,
        positionX: asNumber(node.position_x),
        positionY: asNumber(node.position_y),
        width: asNumber(node.width),
        height: asNumber(node.height),
      } satisfies NormalizedDAGNode;
    })
    .filter((node): node is NormalizedDAGNode => node !== null);

  const baseStatusById = Object.fromEntries(baseNodes.map((node) => [node.id, node.status]));
  const nodes = baseNodes.map((node) => {
    if (node.status !== 'pending') {
      return node;
    }

    const dependenciesForNode = dependencyMap[node.id] ?? [];
    if (dependenciesForNode.length === 0) {
      return node;
    }

    const unsatisfiedDependencies = dependenciesForNode.filter(
      (dependencyId) => baseStatusById[dependencyId] !== 'succeeded',
    );
    if (unsatisfiedDependencies.length > 0) {
      return {
        ...node,
        status: 'blocked',
        blockedBy: unsatisfiedDependencies,
      } satisfies NormalizedDAGNode;
    }

    return {
      ...node,
      status: 'ready',
    } satisfies NormalizedDAGNode;
  });

  const stageRecords = Object.fromEntries(
    asRecordArray(payload.stages)
      .map((stageRecord) => {
        const stage = normalizeStage(stageRecord.stage);
        return stage ? [stage, stageRecord] : null;
      })
      .filter((entry): entry is [DAGStage, Record<string, unknown>] => entry !== null),
  );

  const stages = DAG_STAGE_ORDER.map((stage, index) => {
    const nodesInStage = nodes.filter((node) => node.stage === stage);
    const stageRecord = stageRecords[stage];
    const rawStageStatus = isRecord(payload.stage_status) ? payload.stage_status : null;
    const statusCounts = nodesInStage.reduce<Record<string, number>>((counts, node) => {
      counts[node.status] = (counts[node.status] ?? 0) + 1;
      return counts;
    }, {});

    const explicitStatus =
      normalizeStageStatus(stageRecord?.status) ??
      normalizeStageStatus(rawStageStatus?.[stage]);
    const dependencyStageIds =
      asStringArray(stageRecord?.dependency_stage_ids).filter((dependencyStage): dependencyStage is DAGStage =>
        DAG_STAGE_ORDER.includes(dependencyStage as DAGStage),
      ) ??
      [];

    const inferredDependencyStages =
      dependencyStageIds.length > 0
        ? dependencyStageIds
        : Array.from(
            new Set(
              dependencies
                .filter((dependency) => dependency.target_stage === stage)
                .map((dependency) => dependency.source_stage),
            ),
          );

    return {
      stage,
      label: asString(stageRecord?.label) ?? DAG_STAGE_LABELS[stage],
      order: asNumber(stageRecord?.order) ?? index,
      status: aggregateStageStatus(
        nodesInStage.map((node) => node.status),
        explicitStatus,
      ),
      node_ids: nodesInStage.map((node) => node.id),
      dependency_stage_ids: inferredDependencyStages,
      node_count: nodesInStage.length,
      status_counts: statusCounts,
      metadata: isRecord(stageRecord?.metadata) ? stageRecord.metadata : {},
    } satisfies DAGStageSummary;
  }).sort((left, right) => left.order - right.order);

  const stageStatus = Object.fromEntries(stages.map((stage) => [stage.stage, stage.status]));

  return {
    graphId: asString(payload.graph_id ?? payload.id) ?? '',
    name: asString(payload.name) ?? 'Untitled Pipeline',
    nodes,
    dependencies,
    stages,
    stageStatus,
    dependencyMap,
    metadata: isRecord(payload.metadata) ? payload.metadata : {},
    integrityHash: asString(payload.integrity_hash) ?? '',
    generatedAt: asNumber(payload.generated_at) ?? 0,
  };
}

export function buildFlowFromSnapshot(snapshot: DAGSnapshot): GraphSnapshot {
  const stageOrder = snapshot.stages.length > 0
    ? snapshot.stages
    : DAG_STAGE_ORDER.map((stage, index) => ({
        stage,
        label: DAG_STAGE_LABELS[stage],
        order: index,
        status: 'pending' as DAGStageStatus,
        node_ids: [],
        dependency_stage_ids: [],
        node_count: 0,
        status_counts: {},
        metadata: {},
      }));
  const stageIndexById = Object.fromEntries(
    stageOrder.map((stage, index) => [stage.stage, index]),
  ) as Record<DAGStage, number>;
  const positionByNodeId = new Map<string, { x: number; y: number }>();

  for (const stage of stageOrder) {
    const nodesInStage = snapshot.nodes
      .filter((node) => node.stage === stage.stage)
      .map((node, index) => {
        const upstreamYs = (snapshot.dependencyMap[node.id] ?? [])
          .map((dependencyId) => positionByNodeId.get(dependencyId)?.y)
          .filter((value): value is number => typeof value === 'number');
        const preferredY =
          upstreamYs.length > 0
            ? upstreamYs.reduce((sum, value) => sum + value, 0) / upstreamYs.length
            : node.positionY ?? index * (DAG_NODE_HEIGHT + DAG_NODE_ROW_GAP);
        return { node, preferredY };
      })
      .sort((left, right) => {
        if (left.preferredY !== right.preferredY) {
          return left.preferredY - right.preferredY;
        }
        if (left.node.priority !== right.node.priority) {
          return right.node.priority - left.node.priority;
        }
        return left.node.label.localeCompare(right.node.label);
      });

    nodesInStage.forEach(({ node }, index) => {
      const stageIndex = stageIndexById[node.stage] ?? 0;
      positionByNodeId.set(node.id, {
        x: stageIndex * (STAGE_LANE_WIDTH + STAGE_LANE_GAP) + STAGE_LANE_PADDING_X,
        y: STAGE_LANE_PADDING_TOP + index * (DAG_NODE_HEIGHT + DAG_NODE_ROW_GAP),
      });
    });
  }

  const stageSummaryById = Object.fromEntries(
    snapshot.stages.map((stage) => [stage.stage, stage]),
  ) as Record<DAGStage, DAGStageSummary>;

  const nodes: Node<DAGNodeData>[] = snapshot.nodes.map((node) => ({
    id: node.id,
    type: `${node.stage}Node`,
    position: positionByNodeId.get(node.id) ?? { x: 0, y: 0 },
    data: {
      label: node.label,
      description: node.description,
      stage: node.stage,
      subtype: node.subtype,
      status: node.status,
      runtimeStatus: node.runtimeStatus,
      priority: node.priority,
      metadata: node.metadata,
      confidence: node.confidence,
      assignedAgent: node.assignedAgent,
      dependencyCount: node.dependencyCount,
      downstreamCount: node.downstreamCount,
      blockedBy: node.blockedBy,
      canExecute: node.canExecute,
      outputPreview: node.outputPreview,
      progress: node.progress,
      stageLabel: stageSummaryById[node.stage]?.label ?? DAG_STAGE_LABELS[node.stage],
      stageStatus: stageSummaryById[node.stage]?.status,
    },
  }));

  const edges: Edge<DAGEdgeData>[] = snapshot.dependencies.map((dependency) => ({
    id: dependency.id,
    source: dependency.source_id,
    target: dependency.target_id,
    type: 'crossStage',
    animated: dependency.cross_stage,
    data: {
      edgeType: dependency.edge_type,
      sourceStage: dependency.source_stage,
      targetStage: dependency.target_stage,
      blocking: dependency.blocking,
      implicit: dependency.implicit,
      label: dependency.label,
      crossStage: dependency.cross_stage,
    },
    style: {
      strokeWidth: dependency.blocking ? 2.5 : 1.75,
      strokeOpacity: dependency.implicit ? 0.55 : 0.95,
    },
  }));

  return { nodes, edges };
}

export function useUnifiedDAG(graphId: string | null) {
  const [nodes, setNodes] = useState<Node<DAGNodeData>[]>([]);
  const [edges, setEdges] = useState<Edge<DAGEdgeData>[]>([]);
  const [snapshot, setSnapshot] = useState<DAGSnapshot | null>(null);
  const [operationLoading, setOperationLoading] = useState(false);
  const [operationError, setOperationError] = useState<string | null>(null);

  const undoStack = useRef<GraphSnapshot[]>([]);
  const redoStack = useRef<GraphSnapshot[]>([]);

  const { data: graphData, mutate: mutateGraph, isLoading: isGraphLoading } = useSWRFetch<{
    data: Record<string, unknown>;
  }>(graphId ? `${API_PREFIX}/${graphId}` : null);

  useEffect(() => {
    if (!graphData?.data) {
      return;
    }
    const normalizedSnapshot = normalizeDagSnapshot(graphData.data);
    const flow = buildFlowFromSnapshot(normalizedSnapshot);
    setSnapshot(normalizedSnapshot);
    setNodes(flow.nodes);
    setEdges(flow.edges);
  }, [graphData]);

  const pushUndo = useCallback(() => {
    undoStack.current.push({ nodes: [...nodes], edges: [...edges] });
    redoStack.current = [];
  }, [nodes, edges]);

  const undo = useCallback(() => {
    const nextSnapshot = undoStack.current.pop();
    if (!nextSnapshot) {
      return;
    }
    redoStack.current.push({ nodes, edges });
    setNodes(nextSnapshot.nodes);
    setEdges(nextSnapshot.edges);
  }, [edges, nodes]);

  const redo = useCallback(() => {
    const nextSnapshot = redoStack.current.pop();
    if (!nextSnapshot) {
      return;
    }
    undoStack.current.push({ nodes, edges });
    setNodes(nextSnapshot.nodes);
    setEdges(nextSnapshot.edges);
  }, [edges, nodes]);

  const addNode = useCallback((node: Node<DAGNodeData>) => {
    pushUndo();
    setNodes((currentNodes) => [...currentNodes, node]);
  }, [pushUndo]);

  const updateNode = useCallback((id: string, data: Partial<DAGNodeData>) => {
    pushUndo();
    setNodes((currentNodes) =>
      currentNodes.map((node) => (node.id === id ? { ...node, data: { ...node.data, ...data } } : node)),
    );
  }, [pushUndo]);

  const deleteNode = useCallback((id: string) => {
    pushUndo();
    setNodes((currentNodes) => currentNodes.filter((node) => node.id !== id));
    setEdges((currentEdges) => currentEdges.filter((edge) => edge.source !== id && edge.target !== id));
  }, [pushUndo]);

  const addEdge = useCallback((edge: Edge<DAGEdgeData>) => {
    pushUndo();
    setEdges((currentEdges) => [...currentEdges, edge]);
  }, [pushUndo]);

  const deleteEdge = useCallback((id: string) => {
    pushUndo();
    setEdges((currentEdges) => currentEdges.filter((edge) => edge.id !== id));
  }, [pushUndo]);

  const runOperation = useCallback(async (
    nodeId: string,
    operation: string,
    body?: Record<string, unknown>,
  ): Promise<DAGOperationResult | null> => {
    if (!graphId) {
      return null;
    }

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
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Operation failed';
      setOperationError(message);
      return null;
    } finally {
      setOperationLoading(false);
    }
  }, [graphId, mutateGraph, pushUndo]);

  const debateNode = useCallback(
    (nodeId: string, agents?: string[], rounds?: number) => runOperation(nodeId, 'debate', { agents, rounds }),
    [runOperation],
  );
  const decomposeNode = useCallback((nodeId: string) => runOperation(nodeId, 'decompose'), [runOperation]);
  const prioritizeChildren = useCallback((nodeId: string) => runOperation(nodeId, 'prioritize'), [runOperation]);
  const assignAgents = useCallback((nodeId: string) => runOperation(nodeId, 'assign-agents'), [runOperation]);
  const executeNode = useCallback((nodeId: string) => runOperation(nodeId, 'execute'), [runOperation]);
  const findPrecedents = useCallback(
    (nodeId: string, maxResults?: number) => runOperation(nodeId, 'find-precedents', { max_results: maxResults }),
    [runOperation],
  );

  const clusterIdeas = useCallback(async (
    ideas: string[],
    threshold?: number,
  ): Promise<DAGOperationResult | null> => {
    if (!graphId) {
      return null;
    }

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
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Clustering failed';
      setOperationError(message);
      return null;
    } finally {
      setOperationLoading(false);
    }
  }, [graphId, mutateGraph, pushUndo]);

  const autoFlow = useCallback(async (
    ideas: string[],
    config?: Record<string, unknown>,
  ): Promise<DAGOperationResult | null> => {
    if (!graphId) {
      return null;
    }

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
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Auto-flow failed';
      setOperationError(message);
      return null;
    } finally {
      setOperationLoading(false);
    }
  }, [graphId, mutateGraph, pushUndo]);

  const validateGraph = useCallback((): string[] => {
    const errors: string[] = [];
    if (nodes.length === 0) {
      errors.push('Graph is empty - add at least one pipeline node');
      return errors;
    }

    const nodesByStage = Object.fromEntries(
      DAG_STAGE_ORDER.map((stage) => [stage, nodes.filter((node) => node.data.stage === stage)]),
    ) as Record<DAGStage, Node<DAGNodeData>[]>;

    if (nodesByStage.ideas.length === 0) {
      errors.push('No idea nodes - ideas are required to start the pipeline');
    }

    for (let index = 1; index < DAG_STAGE_ORDER.length; index += 1) {
      const stage = DAG_STAGE_ORDER[index];
      if (nodesByStage[stage].length === 0) {
        continue;
      }

      const upstreamNodeIds = new Set(
        DAG_STAGE_ORDER.slice(0, index).flatMap((upstreamStage) =>
          nodesByStage[upstreamStage].map((node) => node.id),
        ),
      );
      const hasUpstreamDependency = nodesByStage[stage].some((node) =>
        edges.some((edge) => edge.target === node.id && upstreamNodeIds.has(edge.source)),
      );
      if (!hasUpstreamDependency) {
        errors.push(`${DAG_STAGE_LABELS[stage]} nodes need dependencies from upstream stages`);
      }
    }

    const connectedNodeIds = new Set<string>();
    edges.forEach((edge) => {
      connectedNodeIds.add(edge.source);
      connectedNodeIds.add(edge.target);
    });

    const orphanedNodes = nodes.filter((node) => !connectedNodeIds.has(node.id));
    if (orphanedNodes.length > 0 && nodes.length > 1) {
      errors.push(`${orphanedNodes.length} orphan node(s) with no dependencies`);
    }

    return errors;
  }, [edges, nodes]);

  const [executionHistory, setExecutionHistory] = useState<ExecutionHistoryEntry[]>([]);
  const [batchExecuting, setBatchExecuting] = useState(false);

  const executeAllReady = useCallback(async (): Promise<void> => {
    if (!graphId) {
      return;
    }

    const readyNodes = nodes.filter(
      (node) => node.data.status === 'ready' && Boolean(node.data.canExecute),
    );
    if (readyNodes.length === 0) {
      return;
    }

    setBatchExecuting(true);
    pushUndo();
    setNodes((currentNodes) =>
      currentNodes.map((node) =>
        node.data.status === 'ready' && node.data.canExecute
          ? { ...node, data: { ...node.data, status: 'running', runtimeStatus: 'running' } }
          : node,
      ),
    );

    try {
      const result = await apiFetch<{
        data: { results: Array<{ node_id: string; status: string; duration_ms: number }> };
      }>(`${API_PREFIX}/${graphId}/execute-batch`, {
        method: 'POST',
        body: JSON.stringify({ node_ids: readyNodes.map((node) => node.id) }),
      });

      const batchResults = result?.data?.results ?? [];
      const nextHistory = batchResults.map((entry) => {
        const node = readyNodes.find((candidate) => candidate.id === entry.node_id);
        return {
          id: `${entry.node_id}-${Date.now()}`,
          nodeId: entry.node_id,
          nodeLabel: node?.data.label ?? entry.node_id,
          status: normalizeNodeStatus(entry.status, null) === 'succeeded' ? 'succeeded' : 'failed',
          durationMs: entry.duration_ms ?? 0,
          timestamp: Date.now(),
        } satisfies ExecutionHistoryEntry;
      });
      setExecutionHistory((currentHistory) => [...nextHistory, ...currentHistory]);

      setNodes((currentNodes) =>
        currentNodes.map((node) => {
          const batchResult = batchResults.find((entry) => entry.node_id === node.id);
          if (!batchResult) {
            return node;
          }
          const status = normalizeNodeStatus(batchResult.status, null);
          return {
            ...node,
            data: {
              ...node.data,
              status,
              runtimeStatus: batchResult.status,
            },
          };
        }),
      );
      await mutateGraph();
    } catch (error) {
      setNodes((currentNodes) =>
        currentNodes.map((node) =>
          node.data.status === 'running' && node.data.canExecute
            ? { ...node, data: { ...node.data, status: 'ready', runtimeStatus: 'ready' } }
            : node,
        ),
      );
      setOperationError(error instanceof Error ? error.message : 'Batch execution failed');
    } finally {
      setBatchExecuting(false);
    }
  }, [graphId, mutateGraph, nodes, pushUndo]);

  const autoAdvanceAll = useCallback(async (): Promise<void> => {
    if (!graphId) {
      return;
    }

    setBatchExecuting(true);
    setOperationError(null);
    try {
      await apiFetch<{ data: DAGOperationResult }>(
        `${API_PREFIX}/${graphId}/auto-advance`,
        { method: 'POST', body: JSON.stringify({}) },
      );
      pushUndo();
      await mutateGraph();
    } catch (error) {
      setOperationError(error instanceof Error ? error.message : 'Auto-advance failed');
    } finally {
      setBatchExecuting(false);
    }
  }, [graphId, mutateGraph, pushUndo]);

  const graphStats = useMemo(() => {
    const total = nodes.length;
    const succeeded = nodes.filter((node) => node.data.status === 'succeeded').length;
    const ready = nodes.filter((node) => node.data.status === 'ready').length;
    const executableReady = nodes.filter(
      (node) => node.data.status === 'ready' && Boolean(node.data.canExecute),
    ).length;
    const running = nodes.filter((node) => node.data.status === 'running').length;
    const failed = nodes.filter((node) => node.data.status === 'failed').length;
    const awaitingHuman = nodes.filter((node) => node.data.status === 'awaiting_human').length;
    return {
      total,
      succeeded,
      ready,
      executableReady,
      running,
      failed,
      awaitingHuman,
      completionPct: total > 0 ? Math.round((succeeded / total) * 100) : 0,
    };
  }, [nodes]);

  return {
    nodes,
    edges,
    stages: snapshot?.stages ?? [],
    stageStatus: snapshot?.stageStatus ?? {},
    snapshot,
    setNodes,
    setEdges,
    addNode,
    updateNode,
    deleteNode,
    addEdge,
    deleteEdge,
    debateNode,
    decomposeNode,
    prioritizeChildren,
    assignAgents,
    executeNode,
    findPrecedents,
    clusterIdeas,
    autoFlow,
    executeAllReady,
    autoAdvanceAll,
    validateGraph,
    executionHistory,
    batchExecuting,
    graphStats,
    operationLoading,
    operationError,
    isLoading: isGraphLoading,
    undo,
    redo,
    canUndo: undoStack.current.length > 0,
    canRedo: redoStack.current.length > 0,
    refresh: mutateGraph,
  };
}
