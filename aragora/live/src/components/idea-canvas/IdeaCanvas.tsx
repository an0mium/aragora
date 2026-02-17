'use client';

import { useCallback, useRef } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type NodeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { IdeaNode } from './IdeaNode';
import { IdeaPalette } from './IdeaPalette';
import { IdeaPropertyEditor } from './IdeaPropertyEditor';
import { CollaborationOverlay } from './CollaborationOverlay';
import { useIdeaCanvas } from './useIdeaCanvas';
import { IDEA_NODE_CONFIGS, type IdeaNodeType } from './types';

const nodeTypes: NodeTypes = {
  ideaNode: IdeaNode as unknown as NodeTypes[string],
};

interface IdeaCanvasProps {
  canvasId: string;
}

/**
 * Main Idea Canvas component with React Flow, palette, and property editor.
 */
export function IdeaCanvas({ canvasId }: IdeaCanvasProps) {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    onDrop,
    selectedNodeId,
    setSelectedNodeId,
    selectedNodeData,
    updateSelectedNode,
    deleteSelectedNode,
    saveCanvas,
    cursors,
    onlineUsers,
    sendCursorMove,
  } = useIdeaCanvas(canvasId);

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: { id: string }) => {
      setSelectedNodeId(node.id);
    },
    [setSelectedNodeId]
  );

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
  }, [setSelectedNodeId]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const bounds = reactFlowWrapper.current?.getBoundingClientRect();
      if (!bounds) return;
      // screenToFlowPosition is provided by ReactFlow via the hook
      // For drop we use a simple offset calculation
      onDrop(e, bounds, (pos) => pos);
    },
    [onDrop]
  );

  const handlePromote = useCallback(async () => {
    if (!selectedNodeId) return;
    try {
      const res = await fetch(`/api/v1/ideas/${canvasId}/promote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ node_ids: [selectedNodeId] }),
      });
      if (res.ok) {
        updateSelectedNode({ promotedToGoalId: 'pending' });
      }
    } catch {
      // ignore
    }
  }, [canvasId, selectedNodeId, updateSelectedNode]);

  // MiniMap color mapping
  const miniMapNodeColor = useCallback((node: { data?: Record<string, unknown> }) => {
    const ideaType = (node.data?.ideaType || 'concept') as IdeaNodeType;
    const config = IDEA_NODE_CONFIGS[ideaType];
    // Extract color name from tailwind class (e.g., 'bg-indigo-500/20' -> '#818cf8')
    const colorMap: Record<IdeaNodeType, string> = {
      concept: '#818cf8',
      observation: '#34d399',
      question: '#a78bfa',
      hypothesis: '#c084fc',
      insight: '#8b5cf6',
      evidence: '#7c3aed',
      cluster: '#6366f1',
      assumption: '#c4b5fd',
      constraint: '#ddd6fe',
    };
    return colorMap[ideaType] || '#818cf8';
  }, []);

  return (
    <div className="flex h-full">
      {/* Left: Palette */}
      <IdeaPalette />

      {/* Center: Canvas */}
      <div
        ref={reactFlowWrapper}
        className="flex-1 relative"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        {/* Toolbar */}
        <div className="absolute top-2 left-2 z-20 flex gap-2">
          <button
            onClick={saveCanvas}
            className="px-3 py-1 text-xs font-mono rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--text)] hover:border-[var(--acid-green)] transition-colors"
          >
            Save
          </button>
        </div>

        <CollaborationOverlay cursors={cursors} onlineUsers={onlineUsers} />

        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          onPaneClick={onPaneClick}
          nodeTypes={nodeTypes}
          snapToGrid
          snapGrid={[16, 16]}
          fitView
          className="bg-[var(--bg)]"
        >
          <Background gap={16} size={1} />
          <Controls className="!bg-[var(--surface)] !border-[var(--border)]" />
          <MiniMap
            nodeColor={miniMapNodeColor}
            maskColor="rgba(0,0,0,0.6)"
            className="!bg-[var(--surface)] !border-[var(--border)]"
          />
        </ReactFlow>
      </div>

      {/* Right: Property Editor */}
      <IdeaPropertyEditor
        data={selectedNodeData}
        onChange={updateSelectedNode}
        onPromote={handlePromote}
        onDelete={deleteSelectedNode}
      />
    </div>
  );
}

export default IdeaCanvas;
