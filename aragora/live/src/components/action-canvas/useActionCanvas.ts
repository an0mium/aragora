'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type Connection,
  addEdge,
} from '@xyflow/react';
import type { ActionCanvasMeta, ActionNodeData, ActionNodeType, RemoteCursor } from './types';
import { ACTION_NODE_CONFIGS } from './types';

const API_BASE = '/api/v1/actions';

/**
 * Core state management hook for the Action Canvas.
 */
export function useActionCanvas(canvasId: string | null) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [canvasMeta, setCanvasMeta] = useState<ActionCanvasMeta | null>(null);
  const [loading, setLoading] = useState(false);
  const [cursors, setCursors] = useState<RemoteCursor[]>([]);
  const [onlineUsers, setOnlineUsers] = useState<string[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const cursorThrottleRef = useRef<number>(0);

  // ── Load canvas ──────────────────────────────────────────────
  const loadCanvas = useCallback(async () => {
    if (!canvasId) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/${canvasId}`);
      if (!res.ok) return;
      const data = await res.json();
      setCanvasMeta(data);

      const rfNodes: Node[] = (data.nodes || []).map((n: Record<string, unknown>) => ({
        id: n.id as string,
        type: 'actionNode',
        position: n.position as { x: number; y: number },
        data: {
          ...(n.data as Record<string, unknown>),
          label: n.label as string,
          actionType: ((n.data as Record<string, unknown>)?.action_type || (n.data as Record<string, unknown>)?.step_type || 'task') as ActionNodeType,
          description: (n.data as Record<string, unknown>)?.description || '',
          status: (n.data as Record<string, unknown>)?.status || 'pending',
          assignee: (n.data as Record<string, unknown>)?.assignee || '',
          optional: (n.data as Record<string, unknown>)?.optional || false,
          timeoutSeconds: (n.data as Record<string, unknown>)?.timeout_seconds || 0,
          tags: (n.data as Record<string, unknown>)?.tags || [],
          sourceGoalIds: (n.data as Record<string, unknown>)?.source_goal_ids || [],
          stage: 'actions' as const,
          rfType: 'actionNode' as const,
        },
      }));

      const rfEdges: Edge[] = (data.edges || []).map((e: Record<string, unknown>) => ({
        id: e.id as string,
        source: (e.source || e.source_id) as string,
        target: (e.target || e.target_id) as string,
        type: 'default',
        label: e.label as string,
        animated: !!e.animated,
      }));

      setNodes(rfNodes);
      setEdges(rfEdges);
    } finally {
      setLoading(false);
    }
  }, [canvasId, setNodes, setEdges]);

  useEffect(() => {
    loadCanvas();
  }, [loadCanvas]);

  // ── WebSocket ────────────────────────────────────────────────
  useEffect(() => {
    if (!canvasId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/canvas/${canvasId}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
          case 'actions:cursor:move':
            setCursors((prev) => {
              const filtered = prev.filter((c) => c.userId !== msg.user_id);
              return [...filtered, { userId: msg.user_id, position: msg.position, color: '' }];
            });
            break;
          case 'actions:presence:join':
          case 'actions:presence:leave':
            setOnlineUsers(msg.users || []);
            break;
          case 'canvas:node:create':
          case 'canvas:node:update':
          case 'canvas:node:delete':
          case 'canvas:edge:create':
          case 'canvas:edge:delete':
            loadCanvas();
            break;
        }
      } catch {
        // ignore parse errors
      }
    };

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'actions:presence:join' }));
    };

    return () => {
      ws.send(JSON.stringify({ type: 'actions:presence:leave' }));
      ws.close();
      wsRef.current = null;
    };
  }, [canvasId, loadCanvas]);

  // ── Cursor broadcasting ──────────────────────────────────────
  const sendCursorMove = useCallback(
    (position: { x: number; y: number }) => {
      const now = Date.now();
      if (now - cursorThrottleRef.current < 50) return;
      cursorThrottleRef.current = now;

      wsRef.current?.send(
        JSON.stringify({ type: 'actions:cursor:move', position })
      );
    },
    []
  );

  // ── Connection handler ───────────────────────────────────────
  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((eds) => addEdge({ ...connection, type: 'default' }, eds));
    },
    [setEdges]
  );

  // ── Drop handler ─────────────────────────────────────────────
  const onDrop = useCallback(
    (event: React.DragEvent, reactFlowBounds: DOMRect, screenToFlowPosition: (pos: { x: number; y: number }) => { x: number; y: number }) => {
      const actionType = event.dataTransfer.getData('application/action-node-type') as ActionNodeType;
      if (!actionType) return;

      const position = screenToFlowPosition({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      const config = ACTION_NODE_CONFIGS[actionType];
      const newNode: Node = {
        id: `action-${Date.now()}`,
        type: 'actionNode',
        position,
        data: {
          actionType,
          label: config.label,
          description: '',
          status: 'pending',
          assignee: '',
          optional: false,
          timeoutSeconds: 0,
          tags: [],
          stage: 'actions' as const,
          rfType: 'actionNode' as const,
        } satisfies ActionNodeData,
      };

      setNodes((nds) => [...nds, newNode]);
    },
    [setNodes]
  );

  // ── Node property updates ────────────────────────────────────
  const updateSelectedNode = useCallback(
    (updates: Partial<ActionNodeData>) => {
      if (!selectedNodeId) return;
      setNodes((nds) =>
        nds.map((n) =>
          n.id === selectedNodeId ? { ...n, data: { ...n.data, ...updates } } : n
        )
      );
    },
    [selectedNodeId, setNodes]
  );

  const deleteSelectedNode = useCallback(() => {
    if (!selectedNodeId) return;
    setNodes((nds) => nds.filter((n) => n.id !== selectedNodeId));
    setEdges((eds) =>
      eds.filter((e) => e.source !== selectedNodeId && e.target !== selectedNodeId)
    );
    setSelectedNodeId(null);
  }, [selectedNodeId, setNodes, setEdges]);

  // ── Save ─────────────────────────────────────────────────────
  const saveCanvas = useCallback(async () => {
    if (!canvasId) return;
    await fetch(`${API_BASE}/${canvasId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: canvasMeta?.name }),
    });
  }, [canvasId, canvasMeta]);

  // ── Selected node data ───────────────────────────────────────
  const selectedNode = nodes.find((n) => n.id === selectedNodeId);
  const selectedNodeData = selectedNode?.data as ActionNodeData | undefined;

  return {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    onDrop,
    selectedNodeId,
    setSelectedNodeId,
    selectedNodeData: selectedNodeData || null,
    updateSelectedNode,
    deleteSelectedNode,
    canvasMeta,
    loading,
    saveCanvas,
    cursors,
    onlineUsers,
    sendCursorMove,
  };
}

export default useActionCanvas;
