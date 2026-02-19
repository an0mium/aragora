'use client';

import { useCallback, useRef } from 'react';
import { ReactFlow, Background, Controls, MiniMap, type NodeTypes } from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { OrchNode } from './OrchNode';
import { OrchPalette } from './OrchPalette';
import { OrchPropertyEditor } from './OrchPropertyEditor';
import { useOrchCanvas } from './useOrchCanvas';
import { type OrchNodeType } from './types';

const nodeTypes: NodeTypes = { orchestrationNode: OrchNode as unknown as NodeTypes[string] };

interface OrchestrationCanvasProps { canvasId: string; }

export function OrchestrationCanvas({ canvasId }: OrchestrationCanvasProps) {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const {
    nodes, edges, onNodesChange, onEdgesChange, onConnect, onDrop,
    selectedNodeId: _selectedNodeId, setSelectedNodeId, selectedNodeData,
    updateSelectedNode, deleteSelectedNode, saveCanvas, executePipeline,
  } = useOrchCanvas(canvasId);

  const onNodeClick = useCallback((_: React.MouseEvent, node: { id: string }) => { setSelectedNodeId(node.id); }, [setSelectedNodeId]);
  const onPaneClick = useCallback(() => { setSelectedNodeId(null); }, [setSelectedNodeId]);
  const handleDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; }, []);
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const bounds = reactFlowWrapper.current?.getBoundingClientRect();
    if (!bounds) return;
    onDrop(e, bounds, (pos) => pos);
  }, [onDrop]);

  const miniMapNodeColor = useCallback((node: { data?: Record<string, unknown> }) => {
    const orchType = (node.data?.orchType || node.data?.orch_type || 'agent_task') as OrchNodeType;
    const colorMap: Record<OrchNodeType, string> = { agent_task: '#f472b6', debate: '#db2777', human_gate: '#f472b6', parallel_fan: '#f9a8d4', merge: '#f9a8d4', verification: '#ec4899' };
    return colorMap[orchType] || '#f472b6';
  }, []);

  return (
    <div className="flex h-full">
      <OrchPalette />
      <div ref={reactFlowWrapper} className="flex-1 relative" onDragOver={handleDragOver} onDrop={handleDrop}>
        <div className="absolute top-2 left-2 z-20 flex gap-2">
          <button onClick={saveCanvas} className="px-3 py-1 text-xs font-mono rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--text)] hover:border-pink-500 transition-colors">Save</button>
        </div>
        <ReactFlow nodes={nodes} edges={edges} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect} onNodeClick={onNodeClick} onPaneClick={onPaneClick} nodeTypes={nodeTypes} snapToGrid snapGrid={[16, 16]} fitView className="bg-[var(--bg)]">
          <Background gap={16} size={1} />
          <Controls className="!bg-[var(--surface)] !border-[var(--border)]" />
          <MiniMap nodeColor={miniMapNodeColor} maskColor="rgba(0,0,0,0.6)" className="!bg-[var(--surface)] !border-[var(--border)]" />
        </ReactFlow>
      </div>
      <OrchPropertyEditor data={selectedNodeData} onChange={updateSelectedNode} onExecute={executePipeline} onDelete={deleteSelectedNode} />
    </div>
  );
}

export default OrchestrationCanvas;
