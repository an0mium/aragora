'use client';

import { useEffect, useRef, useMemo, useState } from 'react';
import * as d3 from 'd3-force';
import * as d3Select from 'd3-selection';
import { zoom as d3Zoom, type ZoomBehavior } from 'd3-zoom';
import { drag as d3Drag, type DragBehavior, type D3DragEvent } from 'd3-drag';

interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  type: 'argument' | 'rebuttal' | 'synthesis' | 'evidence' | 'root';
  agent: string;
  content: string;
  parent_id?: string;
  children?: string[];
  branch_id?: string;
  confidence?: number;
}

interface GraphLink {
  source: string;
  target: string;
  type: 'supports' | 'opposes' | 'extends' | 'synthesizes';
}

interface ForceGraphProps {
  nodes: GraphNode[];
  width?: number;
  height?: number;
  onNodeClick?: (node: GraphNode) => void;
}

const NODE_COLORS: Record<string, string> = {
  root: '#00ff00',
  argument: '#39ff14',
  rebuttal: '#ff3939',
  synthesis: '#39ffff',
  evidence: '#ffff39',
};

const NODE_RADIUS = {
  root: 16,
  argument: 12,
  rebuttal: 10,
  synthesis: 14,
  evidence: 8,
};

export function ForceGraph({ nodes, width = 800, height = 500, onNodeClick }: ForceGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  // Build links from parent-child relationships
  const links = useMemo(() => {
    const linkArray: GraphLink[] = [];
    nodes.forEach(node => {
      if (node.parent_id) {
        linkArray.push({
          source: node.parent_id,
          target: node.id,
          type: node.type === 'rebuttal' ? 'opposes' :
                node.type === 'synthesis' ? 'synthesizes' :
                node.type === 'evidence' ? 'supports' : 'extends',
        });
      }
    });
    return linkArray;
  }, [nodes]);

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3Select.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create container group for zoom/pan
    const g = svg.append('g').attr('class', 'graph-container');

    // Create arrow markers for directed edges
    svg.append('defs').selectAll('marker')
      .data(['supports', 'opposes', 'extends', 'synthesizes'])
      .join('marker')
      .attr('id', d => `arrow-${d}`)
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('fill', d => d === 'opposes' ? '#ff3939' : d === 'synthesizes' ? '#39ffff' : '#666')
      .attr('d', 'M0,-5L10,0L0,5');

    // Create simulation
    const simulation = d3.forceSimulation<GraphNode>(nodes)
      .force('link', d3.forceLink(links)
        .id((d: any) => d.id)
        .distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    // Draw links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', d => d.type === 'opposes' ? '#ff393980' : '#66666680')
      .attr('stroke-width', 2)
      .attr('marker-end', d => `url(#arrow-${d.type})`);

    // Draw nodes
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('class', 'node')
      .attr('cursor', 'pointer')
      .call(drag(simulation) as any);

    // Node circles
    node.append('circle')
      .attr('r', d => NODE_RADIUS[d.type] || 10)
      .attr('fill', d => NODE_COLORS[d.type] || '#666')
      .attr('stroke', '#000')
      .attr('stroke-width', 2)
      .attr('opacity', 0.9);

    // Node labels
    node.append('text')
      .attr('dx', 15)
      .attr('dy', 4)
      .attr('font-size', '10px')
      .attr('fill', '#ccc')
      .attr('font-family', 'monospace')
      .text(d => d.agent);

    // Tooltip on hover
    node.append('title')
      .text(d => `${d.type}: ${d.content.slice(0, 100)}${d.content.length > 100 ? '...' : ''}`);

    // Click handler
    node.on('click', (event, d) => {
      event.stopPropagation();
      setSelectedNode(d);
      onNodeClick?.(d);
    });

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    // Zoom behavior
    const zoom = d3Zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Drag behavior helper
    function drag(simulation: d3.Simulation<GraphNode, undefined>) {
      type DragEvent = D3DragEvent<SVGGElement, GraphNode, GraphNode>;

      function dragstarted(event: DragEvent) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
      }

      function dragged(event: DragEvent) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
      }

      function dragended(event: DragEvent) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
      }

      return d3Drag<SVGGElement, GraphNode>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended);
    }

    return () => {
      simulation.stop();
    };
  }, [nodes, links, width, height, onNodeClick]);

  return (
    <div className="relative">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="bg-bg/50 rounded border border-acid-green/20"
        style={{ minHeight: '400px' }}
      />

      {/* Selected Node Details */}
      {selectedNode && (
        <div className="absolute top-2 right-2 w-72 p-3 bg-surface border border-acid-green/30 rounded shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <span className={`px-2 py-0.5 rounded text-xs font-mono ${
              selectedNode.type === 'argument' ? 'bg-acid-green/20 text-acid-green' :
              selectedNode.type === 'rebuttal' ? 'bg-acid-red/20 text-acid-red' :
              selectedNode.type === 'synthesis' ? 'bg-acid-cyan/20 text-acid-cyan' :
              'bg-surface text-text-muted'
            }`}>
              {selectedNode.type}
            </span>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-text-muted hover:text-text text-xs"
            >
              [X]
            </button>
          </div>
          <div className="font-mono text-xs text-acid-yellow mb-2">
            {selectedNode.agent}
          </div>
          <p className="font-mono text-xs text-text leading-relaxed">
            {selectedNode.content}
          </p>
          {selectedNode.confidence !== undefined && (
            <div className="mt-2 text-xs font-mono text-text-muted">
              Confidence: {Math.round(selectedNode.confidence * 100)}%
            </div>
          )}
        </div>
      )}

      {/* Controls */}
      <div className="absolute bottom-2 left-2 flex gap-2">
        <div className="px-2 py-1 bg-surface/80 rounded text-xs font-mono text-text-muted">
          Scroll to zoom | Drag nodes | Click for details
        </div>
      </div>
    </div>
  );
}
