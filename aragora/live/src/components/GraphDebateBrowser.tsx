'use client';

import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { getAgentColors } from '@/utils/agentColors';
import type { StreamEvent } from '@/types/events';
import * as d3Force from 'd3-force';
import * as d3Selection from 'd3-selection';
import { Skeleton } from './Skeleton';
import { useGraphDebateWebSocket } from '@/hooks/useGraphDebateWebSocket';

// Types from the backend graph.py
interface DebateNode {
  id: string;
  node_type: string;
  agent_id: string;
  content: string;
  timestamp: string;
  parent_ids: string[];
  child_ids: string[];
  branch_id: string | null;
  confidence: number;
  agreement_scores: Record<string, number>;
  claims: string[];
  evidence: string[];
  metadata: Record<string, unknown>;
  hash: string;
}

interface Branch {
  id: string;
  name: string;
  reason: string;
  start_node_id: string;
  end_node_id: string | null;
  hypothesis: string;
  confidence: number;
  is_active: boolean;
  is_merged: boolean;
  merged_into: string | null;
  node_count: number;
  total_agreement: number;
}

interface MergeResult {
  merged_node_id: string;
  source_branch_ids: string[];
  strategy: string;
  synthesis: string;
  confidence: number;
  insights_preserved: string[];
  conflicts_resolved: string[];
}

interface GraphDebate {
  debate_id: string;
  task: string;
  graph: {
    debate_id: string;
    root_id: string;
    main_branch_id: string;
    created_at: string;
    nodes: Record<string, DebateNode>;
    branches: Record<string, Branch>;
    merge_history: MergeResult[];
    policy: {
      disagreement_threshold: number;
      uncertainty_threshold: number;
      max_branches: number;
      max_depth: number;
    };
  };
  branches: Branch[];
  merge_results: MergeResult[];
  node_count: number;
  branch_count: number;
}

// D3 Force simulation types
interface SimulationNode extends d3Force.SimulationNodeDatum {
  id: string;
  node: DebateNode;
  depth: number;
  fx?: number | null;
  fy?: number | null;
}

interface SimulationLink extends d3Force.SimulationLinkDatum<SimulationNode> {
  source: SimulationNode | string;
  target: SimulationNode | string;
  branchId: string;
}

interface NodePosition {
  x: number;
  y: number;
  node: DebateNode;
  vx?: number;
  vy?: number;
}

// Calculate node depths using BFS
function calculateNodeDepths(
  nodes: Record<string, DebateNode>,
  rootId: string | null
): Map<string, number> {
  const depths = new Map<string, number>();
  if (!rootId || !nodes[rootId]) return depths;

  const visited = new Set<string>();
  const queue: Array<{ id: string; depth: number }> = [{ id: rootId, depth: 0 }];

  while (queue.length > 0) {
    const { id, depth } = queue.shift()!;
    if (visited.has(id)) continue;
    visited.add(id);
    depths.set(id, depth);

    const node = nodes[id];
    if (node) {
      for (const childId of node.child_ids) {
        if (!visited.has(childId)) {
          queue.push({ id: childId, depth: depth + 1 });
        }
      }
    }
  }

  return depths;
}

// Create force simulation for graph layout
function createForceSimulation(
  nodes: Record<string, DebateNode>,
  rootId: string | null,
  width: number,
  height: number
): { nodes: SimulationNode[]; links: SimulationLink[]; simulation: d3Force.Simulation<SimulationNode, SimulationLink> } {
  const depths = calculateNodeDepths(nodes, rootId);
  const maxDepth = Math.max(...Array.from(depths.values()), 0);
  const levelHeight = height / (maxDepth + 2);

  // Create simulation nodes
  const simNodes: SimulationNode[] = Object.values(nodes).map((node) => ({
    id: node.id,
    node,
    depth: depths.get(node.id) || 0,
    x: width / 2 + (Math.random() - 0.5) * 100,
    y: (depths.get(node.id) || 0) * levelHeight + 60,
  }));

  // Create links from parent-child relationships
  const simLinks: SimulationLink[] = [];
  Object.values(nodes).forEach((node) => {
    node.parent_ids.forEach((parentId) => {
      if (nodes[parentId]) {
        simLinks.push({
          source: parentId,
          target: node.id,
          branchId: node.branch_id || 'main',
        });
      }
    });
  });

  // Create D3 force simulation
  const simulation = d3Force.forceSimulation<SimulationNode, SimulationLink>(simNodes)
    .force('link', d3Force.forceLink<SimulationNode, SimulationLink>(simLinks)
      .id((d) => d.id)
      .distance(100)
      .strength(0.8))
    .force('charge', d3Force.forceManyBody<SimulationNode>()
      .strength(-300)
      .distanceMax(400))
    .force('collide', d3Force.forceCollide<SimulationNode>()
      .radius(50)
      .strength(0.7))
    .force('x', d3Force.forceX<SimulationNode>(width / 2).strength(0.05))
    .force('y', d3Force.forceY<SimulationNode>((d) => d.depth * levelHeight + 60).strength(0.3))
    .alphaDecay(0.02)
    .velocityDecay(0.4);

  // Run simulation for initial layout
  simulation.tick(150);
  simulation.stop();

  return { nodes: simNodes, links: simLinks, simulation };
}

// Legacy layout calculation (fallback)
function calculateLayout(
  nodes: Record<string, DebateNode>,
  rootId: string | null,
  _branches: Record<string, Branch>
): NodePosition[] {
  if (!rootId || !nodes[rootId]) return [];

  const { nodes: simNodes } = createForceSimulation(nodes, rootId, 800, 600);

  return simNodes.map((simNode) => ({
    x: simNode.x || 400,
    y: simNode.y || 60,
    node: simNode.node,
  }));
}

// Branch color mapping
const BRANCH_COLORS: Record<string, string> = {
  main: 'text-acid-green',
  'Branch-1': 'text-acid-cyan',
  'Branch-2': 'text-gold',
  'Branch-3': 'text-purple',
  'Branch-4': 'text-crimson',
};

function getBranchColor(branchId: string): string {
  return BRANCH_COLORS[branchId] || 'text-text-muted';
}

function getBranchBgColor(branchId: string): string {
  const colorMap: Record<string, string> = {
    main: 'bg-acid-green/20',
    'Branch-1': 'bg-acid-cyan/20',
    'Branch-2': 'bg-gold/20',
    'Branch-3': 'bg-purple/20',
    'Branch-4': 'bg-crimson/20',
  };
  return colorMap[branchId] || 'bg-surface';
}

function NodeTypeIcon({ type }: { type: string }) {
  const icons: Record<string, string> = {
    root: 'O',
    proposal: 'P',
    critique: 'C',
    synthesis: 'S',
    branch_point: '/',
    merge_point: 'M',
    counterfactual: '?',
    conclusion: 'X',
  };
  return <span className="font-bold">{icons[type] || '.'}</span>;
}

interface GraphNodeProps {
  position: NodePosition;
  isSelected: boolean;
  onClick: () => void;
}

function GraphNode({ position, isSelected, onClick }: GraphNodeProps) {
  const { node, x, y } = position;
  const colors = getAgentColors(node.agent_id);
  const branchColor = getBranchColor(node.branch_id || 'main');

  return (
    <g
      transform={`translate(${x}, ${y})`}
      onClick={onClick}
      className="cursor-pointer"
    >
      {/* Node circle */}
      <circle
        r={isSelected ? 28 : 24}
        className={`${isSelected ? 'fill-acid-green/30' : 'fill-surface'} stroke-2 transition-all duration-200`}
        style={{
          stroke: isSelected ? '#00ff00' : colors.text.replace('text-', '#'),
          filter: isSelected ? 'drop-shadow(0 0 8px #00ff00)' : undefined,
        }}
      />

      {/* Node type icon */}
      <text
        textAnchor="middle"
        dominantBaseline="central"
        className={`text-xs font-mono ${branchColor}`}
        style={{ pointerEvents: 'none' }}
      >
        <NodeTypeIcon type={node.node_type} />
      </text>

      {/* Confidence indicator */}
      {node.confidence > 0 && (
        <text
          y={35}
          textAnchor="middle"
          className="text-[10px] font-mono fill-text-muted"
        >
          {(node.confidence * 100).toFixed(0)}%
        </text>
      )}

      {/* Agent label */}
      <text
        y={-35}
        textAnchor="middle"
        className={`text-[10px] font-mono ${colors.text}`}
      >
        {node.agent_id.slice(0, 8)}
      </text>
    </g>
  );
}

function NodeDetailPanel({ node, onClose }: { node: DebateNode; onClose: () => void }) {
  const colors = getAgentColors(node.agent_id);

  return (
    <div className="absolute top-4 right-4 w-96 bg-surface border border-acid-green/30 shadow-lg z-10">
      <div className="px-4 py-3 border-b border-acid-green/20 bg-bg/50 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 ${colors.bg} ${colors.text} text-xs font-mono`}>
            {node.agent_id}
          </span>
          <span className="text-xs font-mono text-text-muted uppercase">
            {node.node_type.replace('_', ' ')}
          </span>
        </div>
        <button
          onClick={onClose}
          className="text-text-muted hover:text-acid-green text-xs font-mono"
          aria-label="Close node details"
        >
          [X]
        </button>
      </div>

      <div className="p-4 space-y-3 max-h-[60vh] overflow-y-auto">
        {/* Content */}
        <div>
          <div className="text-xs font-mono text-text-muted mb-1">CONTENT</div>
          <div className="text-sm font-mono text-text whitespace-pre-wrap">
            {node.content.length > 500 ? node.content.slice(0, 500) + '...' : node.content}
          </div>
        </div>

        {/* Claims */}
        {node.claims.length > 0 && (
          <div>
            <div className="text-xs font-mono text-acid-cyan mb-1">CLAIMS ({node.claims.length})</div>
            <ul className="space-y-1">
              {node.claims.slice(0, 5).map((claim, i) => (
                <li key={i} className="text-xs font-mono text-text-muted pl-2 border-l border-acid-cyan/30">
                  {claim.slice(0, 100)}{claim.length > 100 ? '...' : ''}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Metadata */}
        <div className="grid grid-cols-2 gap-2 text-xs font-mono">
          <div>
            <span className="text-text-muted">Branch: </span>
            <span className={getBranchColor(node.branch_id || 'main')}>
              {node.branch_id || 'main'}
            </span>
          </div>
          <div>
            <span className="text-text-muted">Confidence: </span>
            <span className="text-acid-green">{(node.confidence * 100).toFixed(0)}%</span>
          </div>
          <div>
            <span className="text-text-muted">Parents: </span>
            <span className="text-text">{node.parent_ids.length}</span>
          </div>
          <div>
            <span className="text-text-muted">Children: </span>
            <span className="text-text">{node.child_ids.length}</span>
          </div>
        </div>

        {/* Hash */}
        <div className="text-[10px] font-mono text-text-muted/50 pt-2 border-t border-border">
          Hash: {node.hash}
        </div>
      </div>
    </div>
  );
}

interface GraphVisualizationProps {
  graph: GraphDebate['graph'];
  selectedNodeId: string | null;
  onNodeSelect: (nodeId: string | null) => void;
  highlightedBranch: string | null;
  onBranchHover: (branchId: string | null) => void;
}

function GraphVisualization({
  graph,
  selectedNodeId,
  onNodeSelect,
  highlightedBranch,
  onBranchHover,
}: GraphVisualizationProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [positions, setPositions] = useState<NodePosition[]>([]);
  const [isSimulating, setIsSimulating] = useState(false);
  const [draggedNode, setDraggedNode] = useState<string | null>(null);
  const simulationRef = useRef<d3Force.Simulation<SimulationNode, SimulationLink> | null>(null);
  const nodesRef = useRef<SimulationNode[]>([]);
  const linksRef = useRef<SimulationLink[]>([]);

  // Initialize force simulation
  useEffect(() => {
    if (!graph.root_id || Object.keys(graph.nodes).length === 0) {
      setPositions([]);
      return;
    }

    const width = 800;
    const height = 600;
    const { nodes: simNodes, links: simLinks, simulation } = createForceSimulation(
      graph.nodes,
      graph.root_id,
      width,
      height
    );

    nodesRef.current = simNodes;
    linksRef.current = simLinks;
    simulationRef.current = simulation;

    // Update positions from simulation
    const updatePositions = () => {
      setPositions(
        nodesRef.current.map((simNode) => ({
          x: simNode.x || 400,
          y: simNode.y || 60,
          node: simNode.node,
        }))
      );
    };

    // Initial positions
    updatePositions();

    // Re-run simulation with animation on mount
    setIsSimulating(true);
    simulation.alpha(0.3).restart();

    simulation.on('tick', () => {
      updatePositions();
    });

    simulation.on('end', () => {
      setIsSimulating(false);
    });

    return () => {
      simulation.stop();
    };
  }, [graph.nodes, graph.root_id]);

  // Calculate SVG dimensions
  const minX = positions.length > 0 ? Math.min(...positions.map((p) => p.x)) - 60 : 0;
  const maxX = positions.length > 0 ? Math.max(...positions.map((p) => p.x)) + 60 : 800;
  const maxY = positions.length > 0 ? Math.max(...positions.map((p) => p.y)) + 80 : 400;

  const baseWidth = Math.max(800, maxX - minX);
  const baseHeight = Math.max(400, maxY);

  // Generate edges
  const edges: Array<{ from: NodePosition; to: NodePosition; branchId: string }> = [];
  positions.forEach((toPos) => {
    toPos.node.parent_ids.forEach((parentId) => {
      const fromPos = positions.find((p) => p.node.id === parentId);
      if (fromPos) {
        edges.push({
          from: fromPos,
          to: toPos,
          branchId: toPos.node.branch_id || 'main',
        });
      }
    });
  });

  const getEdgeColor = (branchId: string): string => {
    const colorMap: Record<string, string> = {
      main: '#00ff00',
      'Branch-1': '#00ffff',
      'Branch-2': '#ffd700',
      'Branch-3': '#9966ff',
      'Branch-4': '#ff3333',
    };
    return colorMap[branchId] || '#666666';
  };

  // Zoom controls
  const handleZoomIn = () => setZoom((z) => Math.min(z * 1.2, 3));
  const handleZoomOut = () => setZoom((z) => Math.max(z / 1.2, 0.3));
  const handleResetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Reheat simulation
  const handleReheat = useCallback(() => {
    if (simulationRef.current) {
      setIsSimulating(true);
      // Release all fixed positions
      nodesRef.current.forEach((node) => {
        node.fx = null;
        node.fy = null;
      });
      simulationRef.current.alpha(0.5).restart();
    }
  }, []);

  // Pan handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0 && e.shiftKey) {
      setIsPanning(true);
      setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isPanning) {
      setPan({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  // Wheel zoom
  const handleWheel = (e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setZoom((z) => Math.max(0.3, Math.min(3, z * delta)));
    }
  };

  // Node drag handlers
  const handleNodeDragStart = useCallback((nodeId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setDraggedNode(nodeId);

    if (simulationRef.current) {
      simulationRef.current.alphaTarget(0.3).restart();
    }

    const simNode = nodesRef.current.find((n) => n.id === nodeId);
    if (simNode) {
      simNode.fx = simNode.x;
      simNode.fy = simNode.y;
    }
  }, []);

  const handleNodeDrag = useCallback((nodeId: string, e: React.MouseEvent) => {
    if (draggedNode !== nodeId) return;

    const svg = svgRef.current;
    if (!svg) return;

    const rect = svg.getBoundingClientRect();
    const svgX = (e.clientX - rect.left - pan.x) / zoom + minX;
    const svgY = (e.clientY - rect.top - pan.y) / zoom;

    const simNode = nodesRef.current.find((n) => n.id === nodeId);
    if (simNode) {
      simNode.fx = svgX;
      simNode.fy = svgY;
    }
  }, [draggedNode, pan.x, pan.y, zoom, minX]);

  const handleNodeDragEnd = useCallback(() => {
    setDraggedNode(null);

    if (simulationRef.current) {
      simulationRef.current.alphaTarget(0);
    }
  }, []);

  // Global mouse move/up for drag
  useEffect(() => {
    if (!draggedNode) return;

    const handleGlobalMouseMove = (e: MouseEvent) => {
      const svg = svgRef.current;
      if (!svg) return;

      const rect = svg.getBoundingClientRect();
      const svgX = (e.clientX - rect.left - pan.x) / zoom + minX;
      const svgY = (e.clientY - rect.top - pan.y) / zoom;

      const simNode = nodesRef.current.find((n) => n.id === draggedNode);
      if (simNode) {
        simNode.fx = svgX;
        simNode.fy = svgY;
      }
    };

    const handleGlobalMouseUp = () => {
      handleNodeDragEnd();
    };

    window.addEventListener('mousemove', handleGlobalMouseMove);
    window.addEventListener('mouseup', handleGlobalMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleGlobalMouseMove);
      window.removeEventListener('mouseup', handleGlobalMouseUp);
    };
  }, [draggedNode, pan.x, pan.y, zoom, minX, handleNodeDragEnd]);

  const isNodeInHighlightedBranch = (branchId: string | null) => {
    if (!highlightedBranch) return true;
    return branchId === highlightedBranch || branchId === null;
  };

  return (
    <div className="relative" ref={containerRef}>
      {/* Zoom controls */}
      <div className="absolute top-2 left-2 z-10 flex gap-1">
        <button
          onClick={handleZoomIn}
          className="w-8 h-8 bg-surface border border-acid-green/30 text-acid-green font-mono text-sm hover:bg-acid-green/20"
          title="Zoom in"
        >
          +
        </button>
        <button
          onClick={handleZoomOut}
          className="w-8 h-8 bg-surface border border-acid-green/30 text-acid-green font-mono text-sm hover:bg-acid-green/20"
          title="Zoom out"
        >
          -
        </button>
        <button
          onClick={handleResetView}
          className="px-2 h-8 bg-surface border border-acid-green/30 text-acid-green font-mono text-xs hover:bg-acid-green/20"
          title="Reset view"
        >
          RESET
        </button>
        <button
          onClick={handleReheat}
          className={`px-2 h-8 bg-surface border border-acid-cyan/30 text-acid-cyan font-mono text-xs hover:bg-acid-cyan/20 ${
            isSimulating ? 'animate-pulse' : ''
          }`}
          title="Re-run force simulation"
        >
          {isSimulating ? 'SIMULATING...' : 'RELAYOUT'}
        </button>
        <span className="h-8 flex items-center px-2 text-xs font-mono text-text-muted">
          {Math.round(zoom * 100)}%
        </span>
      </div>

      {/* Pan hint */}
      <div className="absolute top-2 right-2 z-10 text-xs font-mono text-text-muted/50">
        Drag nodes | Shift+drag to pan | Ctrl+scroll to zoom
      </div>

      <svg
        ref={svgRef}
        width="100%"
        height={baseHeight}
        viewBox={`${minX - pan.x / zoom} ${-pan.y / zoom} ${baseWidth / zoom} ${baseHeight / zoom}`}
        className={`bg-bg/50 ${isPanning ? 'cursor-grabbing' : draggedNode ? 'cursor-grabbing' : 'cursor-default'}`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#00ff00" fillOpacity="0.5" />
          </marker>
          {/* Highlighted arrowhead */}
          <marker
            id="arrowhead-highlighted"
            markerWidth="12"
            markerHeight="9"
            refX="11"
            refY="4.5"
            orient="auto"
          >
            <polygon points="0 0, 12 4.5, 0 9" fill="#ffffff" fillOpacity="0.8" />
          </marker>
        </defs>

        {/* Edges */}
        {edges.map((edge, i) => {
          const isHighlighted = highlightedBranch === edge.branchId;
          const isDimmed = highlightedBranch && !isHighlighted;

          return (
            <line
              key={i}
              x1={edge.from.x}
              y1={edge.from.y + 24}
              x2={edge.to.x}
              y2={edge.to.y - 24}
              stroke={getEdgeColor(edge.branchId)}
              strokeWidth={isHighlighted ? 3 : 2}
              strokeOpacity={isDimmed ? 0.2 : isHighlighted ? 1 : 0.6}
              markerEnd={isHighlighted ? 'url(#arrowhead-highlighted)' : 'url(#arrowhead)'}
              className="transition-all duration-100"
              onMouseEnter={() => onBranchHover(edge.branchId)}
              onMouseLeave={() => onBranchHover(null)}
            />
          );
        })}

        {/* Nodes */}
        {positions.map((pos) => {
          const nodeInBranch = isNodeInHighlightedBranch(pos.node.branch_id);
          const isDragging = draggedNode === pos.node.id;

          return (
            <g
              key={pos.node.id}
              style={{
                opacity: nodeInBranch ? 1 : 0.3,
                cursor: isDragging ? 'grabbing' : 'grab',
              }}
              className="transition-opacity duration-200"
              onMouseDown={(e) => handleNodeDragStart(pos.node.id, e)}
            >
              <GraphNode
                position={pos}
                isSelected={selectedNodeId === pos.node.id || isDragging}
                onClick={() => !isDragging && onNodeSelect(pos.node.id)}
              />
            </g>
          );
        })}
      </svg>
    </div>
  );
}

interface GraphDebateBrowserProps {
  events?: StreamEvent[];
  initialDebateId?: string | null;
}

export function GraphDebateBrowser({ events = [], initialDebateId }: GraphDebateBrowserProps) {
  const [debates, setDebates] = useState<GraphDebate[]>([]);
  const [selectedDebate, setSelectedDebate] = useState<GraphDebate | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [highlightedBranch, setHighlightedBranch] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [newDebateTask, setNewDebateTask] = useState('');
  const [creating, setCreating] = useState(false);

  // WebSocket connection for real-time updates
  const {
    isConnected: wsConnected,
    lastEvent: wsLastEvent,
    status: wsStatus,
    reconnect: wsReconnect,
  } = useGraphDebateWebSocket({
    debateId: selectedDebate?.debate_id,
    enabled: !!selectedDebate,
  });

  // Listen for graph debate events from props
  const latestGraphEvent = useMemo(() => {
    const relevant = events.filter(e =>
      e.type === 'debate_branch' ||
      e.type === 'debate_merge' ||
      e.type === 'graph_node_added'
    );
    return relevant[relevant.length - 1];
  }, [events]);

  // Refresh debate on WebSocket events
  useEffect(() => {
    if (!wsLastEvent || !selectedDebate) return;

    // Re-fetch the selected debate to get updated graph
    const refreshDebate = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';
        const response = await fetch(
          `${apiUrl}/api/debates/graph/${selectedDebate.debate_id}`
        );
        if (response.ok) {
          const data = await response.json();
          setSelectedDebate(data);
          // Also update in list
          setDebates(prev =>
            prev.map(d => d.debate_id === data.debate_id ? data : d)
          );
        }
      } catch (e) {
        // Ignore refresh errors
      }
    };
    refreshDebate();
  }, [wsLastEvent, selectedDebate]);

  // Refresh on graph events from props (fallback)
  useEffect(() => {
    if (latestGraphEvent && selectedDebate) {
      // Re-fetch the selected debate to get updated graph
      const refreshDebate = async () => {
        try {
          const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';
          const response = await fetch(
            `${apiUrl}/api/debates/graph/${selectedDebate.debate_id}`
          );
          if (response.ok) {
            const data = await response.json();
            setSelectedDebate(data);
            // Also update in list
            setDebates(prev =>
              prev.map(d => d.debate_id === data.debate_id ? data : d)
            );
          }
        } catch (e) {
          // Ignore refresh errors
        }
      };
      refreshDebate();
    }
  }, [latestGraphEvent, selectedDebate]);

  const fetchDebates = useCallback(async () => {
    try {
      setLoading(true);
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';
      // For now, we'll show a placeholder since the API stores in memory
      // In production, this would fetch from storage
      setDebates([]);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch graph debates');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDebates();
  }, [fetchDebates]);

  // Auto-fetch and select debate when initialDebateId is provided
  useEffect(() => {
    if (!initialDebateId) return;

    const fetchInitialDebate = async () => {
      try {
        setLoading(true);
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';
        const response = await fetch(`${apiUrl}/api/debates/graph/${initialDebateId}`);
        if (response.ok) {
          const data = await response.json();
          setSelectedDebate(data);
          // Add to debates list if not already present
          setDebates((prev) => {
            const exists = prev.some((d) => d.debate_id === data.debate_id);
            return exists ? prev : [data, ...prev];
          });
        } else {
          setError(`Failed to load debate: ${initialDebateId}`);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load debate');
      } finally {
        setLoading(false);
      }
    };

    fetchInitialDebate();
  }, [initialDebateId]);

  const handleCreateDebate = async () => {
    if (!newDebateTask.trim()) return;

    try {
      setCreating(true);
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';
      const response = await fetch(`${apiUrl}/api/debates/graph`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task: newDebateTask,
          agents: ['claude', 'gpt4'],
          max_rounds: 5,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to create graph debate');
      }

      const data = await response.json();
      setDebates((prev) => [data, ...prev]);
      setSelectedDebate(data);
      setNewDebateTask('');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create debate');
    } finally {
      setCreating(false);
    }
  };

  const selectedNode = selectedDebate && selectedNodeId
    ? selectedDebate.graph.nodes[selectedNodeId]
    : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-surface border border-acid-green/30 p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-mono text-acid-green">{'>'} GRAPH DEBATES</h2>
            {/* WebSocket Status Indicator */}
            {selectedDebate && (
              <div className="flex items-center gap-1.5">
                <span
                  className={`w-2 h-2 rounded-full ${
                    wsConnected
                      ? 'bg-acid-green animate-pulse'
                      : wsStatus === 'connecting'
                      ? 'bg-gold animate-pulse'
                      : 'bg-crimson'
                  }`}
                />
                <span className="text-[10px] font-mono text-text-muted">
                  {wsConnected ? 'LIVE' : wsStatus === 'connecting' ? 'CONNECTING' : 'OFFLINE'}
                </span>
                {!wsConnected && wsStatus !== 'connecting' && (
                  <button
                    onClick={wsReconnect}
                    className="text-[10px] font-mono text-acid-cyan hover:text-acid-green"
                  >
                    [RECONNECT]
                  </button>
                )}
              </div>
            )}
          </div>
          <span className="text-xs font-mono text-text-muted">
            Branching & counterfactual exploration
          </span>
        </div>

        {/* Create new debate */}
        <div className="flex gap-2">
          <input
            type="text"
            value={newDebateTask}
            onChange={(e) => setNewDebateTask(e.target.value)}
            placeholder="Enter a topic for graph debate..."
            className="flex-1 px-3 py-2 bg-bg border border-acid-green/30 text-text font-mono text-sm focus:outline-none focus:border-acid-green"
            onKeyDown={(e) => e.key === 'Enter' && handleCreateDebate()}
          />
          <button
            onClick={handleCreateDebate}
            disabled={creating || !newDebateTask.trim()}
            className="px-4 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm hover:bg-acid-green/30 transition-colors disabled:opacity-50"
          >
            {creating ? 'CREATING...' : 'CREATE'}
          </button>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="bg-surface border border-crimson/30 p-4">
          <div className="text-xs font-mono text-crimson">Error: {error}</div>
        </div>
      )}

      {/* Main content */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Debate list */}
        <div className="lg:col-span-1 bg-surface border border-acid-green/30">
          <div className="px-4 py-3 border-b border-acid-green/20 bg-bg/50">
            <span className="text-xs font-mono text-acid-green uppercase tracking-wider">
              {'>'} DEBATES ({debates.length})
            </span>
          </div>

          <div className="max-h-[400px] overflow-y-auto">
            {loading && (
              <div className="p-4 space-y-3">
                {[1, 2, 3].map((i) => (
                  <div key={i} className="p-3 border-b border-border">
                    <Skeleton width="70%" height={14} className="mb-2" />
                    <div className="flex items-center gap-2">
                      <Skeleton width={60} height={10} />
                      <Skeleton width={60} height={10} />
                    </div>
                  </div>
                ))}
              </div>
            )}

            {!loading && debates.length === 0 && (
              <div className="p-4 text-xs font-mono text-text-muted">
                No graph debates yet. Create one above!
              </div>
            )}

            {debates.map((debate) => (
              <div
                key={debate.debate_id}
                onClick={() => {
                  setSelectedDebate(debate);
                  setSelectedNodeId(null);
                }}
                className={`p-3 border-b border-border cursor-pointer transition-colors ${
                  selectedDebate?.debate_id === debate.debate_id
                    ? 'bg-acid-green/10 border-l-2 border-l-acid-green'
                    : 'hover:bg-bg'
                }`}
              >
                <div className="text-sm font-mono text-text mb-1 truncate">
                  {debate.task.slice(0, 50)}{debate.task.length > 50 ? '...' : ''}
                </div>
                <div className="flex items-center gap-2 text-xs font-mono text-text-muted">
                  <span className="text-acid-green">{debate.node_count} nodes</span>
                  <span>/</span>
                  <span className="text-acid-cyan">{debate.branch_count} branches</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Graph visualization */}
        <div className="lg:col-span-3 bg-surface border border-acid-green/30 relative min-h-[500px]">
          <div className="px-4 py-3 border-b border-acid-green/20 bg-bg/50">
            <span className="text-xs font-mono text-acid-green uppercase tracking-wider">
              {'>'} GRAPH VISUALIZATION
            </span>
            {selectedDebate && (
              <span className="ml-4 text-xs font-mono text-text-muted">
                {selectedDebate.task.slice(0, 60)}{selectedDebate.task.length > 60 ? '...' : ''}
              </span>
            )}
          </div>

          {selectedDebate ? (
            <>
              <div className="p-4 overflow-x-auto">
                <GraphVisualization
                  graph={selectedDebate.graph}
                  selectedNodeId={selectedNodeId}
                  onNodeSelect={setSelectedNodeId}
                  highlightedBranch={highlightedBranch}
                  onBranchHover={setHighlightedBranch}
                />
              </div>

              {/* Branch legend - interactive */}
              <div className="px-4 py-2 border-t border-acid-green/20 bg-bg/30">
                <div className="flex flex-wrap gap-3 text-xs font-mono">
                  {Object.entries(selectedDebate.graph.branches).map(([id, branch]) => (
                    <div
                      key={id}
                      className={`flex items-center gap-1 cursor-pointer px-2 py-1 rounded transition-all duration-200 ${
                        highlightedBranch === branch.name
                          ? 'bg-acid-green/20 scale-105'
                          : highlightedBranch && highlightedBranch !== branch.name
                          ? 'opacity-40'
                          : 'hover:bg-surface'
                      }`}
                      onMouseEnter={() => setHighlightedBranch(branch.name)}
                      onMouseLeave={() => setHighlightedBranch(null)}
                    >
                      <div
                        className={`w-3 h-3 rounded-full ${getBranchBgColor(branch.name)} ${
                          highlightedBranch === branch.name ? 'ring-2 ring-white/50' : ''
                        }`}
                      />
                      <span className={getBranchColor(branch.name)}>
                        {branch.name}
                      </span>
                      <span className="text-text-muted">
                        ({branch.node_count} nodes)
                      </span>
                      {branch.is_merged && (
                        <span className="text-gold">[merged]</span>
                      )}
                      {branch.is_active && (
                        <span className="text-acid-green animate-pulse">[active]</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Node detail panel */}
              {selectedNode && (
                <NodeDetailPanel
                  node={selectedNode}
                  onClose={() => setSelectedNodeId(null)}
                />
              )}
            </>
          ) : (
            <div className="flex items-center justify-center h-[400px]">
              <div className="text-center">
                <div className="text-4xl font-mono text-acid-green/30 mb-4">/\\</div>
                <div className="text-sm font-mono text-text-muted">
                  Select or create a graph debate to visualize
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Merge history */}
      {selectedDebate && selectedDebate.merge_results.length > 0 && (
        <div className="bg-surface border border-purple/30">
          <div className="px-4 py-3 border-b border-purple/20 bg-bg/50">
            <span className="text-xs font-mono text-purple uppercase tracking-wider">
              {'>'} MERGE HISTORY ({selectedDebate.merge_results.length})
            </span>
          </div>
          <div className="p-4 space-y-3">
            {selectedDebate.merge_results.map((merge, i) => (
              <div key={i} className="p-3 bg-bg/50 border border-purple/20">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-mono text-purple">
                    Merged: {merge.source_branch_ids.join(' + ')}
                  </span>
                  <span className="text-xs font-mono text-text-muted">
                    Strategy: {merge.strategy}
                  </span>
                  <span className="text-xs font-mono text-acid-green">
                    {(merge.confidence * 100).toFixed(0)}% confidence
                  </span>
                </div>
                <div className="text-xs font-mono text-text">
                  {merge.synthesis.slice(0, 200)}{merge.synthesis.length > 200 ? '...' : ''}
                </div>
                {merge.insights_preserved.length > 0 && (
                  <div className="mt-2 text-[10px] font-mono text-text-muted">
                    Preserved: {merge.insights_preserved.slice(0, 3).join(', ')}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
