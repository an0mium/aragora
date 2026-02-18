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

import { GoalNode } from './GoalNode';
import { GoalPalette } from './GoalPalette';
import { GoalPropertyEditor } from './GoalPropertyEditor';
import { useGoalCanvas } from './useGoalCanvas';
import { GOAL_NODE_CONFIGS, type GoalNodeType } from './types';

const nodeTypes: NodeTypes = {
  goalNode: GoalNode as unknown as NodeTypes[string],
};

interface GoalCanvasProps {
  canvasId: string;
}

/**
 * Main Goal Canvas component with React Flow, palette, and property editor.
 */
export function GoalCanvas({ canvasId }: GoalCanvasProps) {
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
  } = useGoalCanvas(canvasId);

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
      onDrop(e, bounds, (pos) => pos);
    },
    [onDrop]
  );

  // MiniMap color mapping
  const miniMapNodeColor = useCallback((node: { data?: Record<string, unknown> }) => {
    const goalType = (node.data?.goalType || 'goal') as GoalNodeType;
    const colorMap: Record<GoalNodeType, string> = {
      goal: '#34d399',
      principle: '#059669',
      strategy: '#14b8a6',
      milestone: '#6ee7b7',
      metric: '#2dd4bf',
      risk: '#ef4444',
    };
    return colorMap[goalType] || '#34d399';
  }, []);

  return (
    <div className="flex h-full">
      {/* Left: Palette */}
      <GoalPalette />

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
      <GoalPropertyEditor
        data={selectedNodeData}
        onChange={updateSelectedNode}
        onDelete={deleteSelectedNode}
      />
    </div>
  );
}

export default GoalCanvas;
