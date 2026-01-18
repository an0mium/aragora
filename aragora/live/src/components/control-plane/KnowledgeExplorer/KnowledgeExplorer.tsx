'use client';

import { useState, useCallback } from 'react';
import { PanelTemplate } from '@/components/shared/PanelTemplate';
import { useKnowledgeQuery } from '@/hooks/useKnowledgeQuery';
import { useKnowledgeExplorerStore } from '@/store/knowledgeExplorerStore';
import { QueryInterface } from './QueryInterface';
import { NodeBrowser } from './NodeBrowser';
import { GraphViewer } from './GraphViewer';
import type { KnowledgeNode, GraphNode } from '@/store/knowledgeExplorerStore';

export type ExplorerTab = 'search' | 'browse' | 'graph';

export interface KnowledgeExplorerProps {
  /** Initial tab to show */
  initialTab?: ExplorerTab;
  /** Callback when a node is selected */
  onSelectNode?: (node: KnowledgeNode) => void;
  /** Height of the explorer */
  height?: number;
  /** Show statistics summary */
  showStats?: boolean;
  /** Custom CSS classes */
  className?: string;
}

/**
 * Knowledge Explorer component for browsing and querying the Knowledge Mound.
 * Provides search, browse, and graph visualization modes.
 */
export function KnowledgeExplorer({
  initialTab = 'search',
  onSelectNode,
  height = 500,
  showStats = true,
  className = '',
}: KnowledgeExplorerProps) {
  const [recentQueries, setRecentQueries] = useState<string[]>([]);

  // Store state
  const { activeTab, setActiveTab } = useKnowledgeExplorerStore();

  // Query hook
  const {
    queryText,
    setQueryText,
    executeQuery,
    isQueryExecuting,
    queryResults,
    queryError,
    browserNodes,
    browserLoading,
    totalNodes,
    loadNodes,
    graphNodes,
    graphEdges,
    graphLoading,
    loadGraph,
    clearGraph,
    stats,
    statsLoading,
    loadStats,
  } = useKnowledgeQuery({
    autoLoadStats: showStats,
  });

  // Handle search
  const handleSearch = useCallback(
    async (text: string) => {
      await executeQuery(text);

      // Add to recent queries
      setRecentQueries((prev) => {
        const filtered = prev.filter((q) => q !== text);
        return [text, ...filtered].slice(0, 5);
      });
    },
    [executeQuery]
  );

  // Handle node selection
  const handleSelectNode = useCallback(
    (node: KnowledgeNode) => {
      onSelectNode?.(node);
    },
    [onSelectNode]
  );

  // Handle view in graph
  const handleViewInGraph = useCallback(
    async (node: KnowledgeNode) => {
      setActiveTab('graph');
      await loadGraph(node.id, 2, 'both');
    },
    [setActiveTab, loadGraph]
  );

  // Handle graph node click
  const handleGraphNodeClick = useCallback(
    (node: GraphNode) => {
      onSelectNode?.(node);
    },
    [onSelectNode]
  );

  // Stats summary
  const statsSummary = stats ? (
    <div className="grid grid-cols-4 gap-4 mb-4 p-3 bg-surface rounded-lg">
      <div className="text-center">
        <div className="text-2xl font-mono text-acid-green">
          {stats.total_nodes.toLocaleString()}
        </div>
        <div className="text-xs text-text-muted">Total Nodes</div>
      </div>
      <div className="text-center">
        <div className="text-2xl font-mono text-acid-cyan">
          {stats.total_relationships.toLocaleString()}
        </div>
        <div className="text-xs text-text-muted">Relationships</div>
      </div>
      <div className="text-center">
        <div className="text-2xl font-mono text-green-400">
          {Math.round(stats.avg_confidence * 100)}%
        </div>
        <div className="text-xs text-text-muted">Avg Confidence</div>
      </div>
      <div className="text-center">
        <div className="text-2xl font-mono text-yellow-400">
          {stats.stale_nodes_count}
        </div>
        <div className="text-xs text-text-muted">Stale Nodes</div>
      </div>
    </div>
  ) : null;

  // Tab content
  const tabContent: Record<ExplorerTab, React.ReactNode> = {
    search: (
      <div className="space-y-4">
        <QueryInterface
          value={queryText}
          onChange={setQueryText}
          onSearch={handleSearch}
          loading={isQueryExecuting}
          recentQueries={recentQueries}
        />

        {queryError && (
          <div className="p-3 bg-red-900/20 border border-red-800/30 rounded text-red-400 text-sm">
            {queryError}
          </div>
        )}

        {queryResults.length > 0 && (
          <div>
            <div className="text-xs text-text-muted mb-2">
              Found {queryResults.length} results
            </div>
            <div style={{ maxHeight: height - 200, overflowY: 'auto' }}>
              <NodeBrowser
                nodes={queryResults}
                onSelectNode={handleSelectNode}
                onViewInGraph={handleViewInGraph}
              />
            </div>
          </div>
        )}
      </div>
    ),

    browse: (
      <div>
        <div className="flex items-center justify-between mb-4">
          <div className="text-sm text-text-muted">
            {totalNodes.toLocaleString()} nodes in knowledge base
          </div>
          <button
            onClick={() => loadNodes()}
            disabled={browserLoading}
            className="text-xs text-acid-green hover:underline disabled:opacity-50"
          >
            {browserLoading ? 'Loading...' : 'Refresh'}
          </button>
        </div>

        <div style={{ maxHeight: height - 100, overflowY: 'auto' }}>
          <NodeBrowser
            nodes={browserNodes}
            loading={browserLoading}
            onSelectNode={handleSelectNode}
            onViewInGraph={handleViewInGraph}
            emptyMessage="No nodes found. Try adjusting filters."
          />
        </div>
      </div>
    ),

    graph: (
      <div>
        <div className="flex items-center justify-between mb-4">
          <div className="text-sm text-text-muted">
            {graphNodes.length > 0
              ? `Showing ${graphNodes.length} nodes, ${graphEdges.length} relationships`
              : 'Search or browse to select a node'}
          </div>
          {graphNodes.length > 0 && (
            <button
              onClick={clearGraph}
              className="text-xs text-text-muted hover:text-acid-green transition-colors"
            >
              Clear graph
            </button>
          )}
        </div>

        <GraphViewer
          nodes={graphNodes}
          edges={graphEdges}
          loading={graphLoading}
          onNodeClick={handleGraphNodeClick}
          width={800}
          height={height - 80}
          showLabels={graphNodes.length < 50}
        />
      </div>
    ),
  };

  return (
    <PanelTemplate
      title="Knowledge Explorer"
      icon="🧠"
      loading={statsLoading && showStats}
      onRefresh={loadStats}
      className={className}
      tabs={[
        { id: 'search', label: 'Search', content: tabContent.search },
        { id: 'browse', label: 'Browse', badge: totalNodes, content: tabContent.browse },
        { id: 'graph', label: 'Graph', content: tabContent.graph },
      ]}
      activeTab={activeTab}
      onTabChange={(tab) => setActiveTab(tab as ExplorerTab)}
    >
      {showStats && !statsLoading && stats && statsSummary}
    </PanelTemplate>
  );
}

export default KnowledgeExplorer;
