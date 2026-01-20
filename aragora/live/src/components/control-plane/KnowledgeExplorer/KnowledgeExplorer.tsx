'use client';

import { useState, useCallback, useEffect } from 'react';
import { PanelTemplate } from '@/components/shared/PanelTemplate';
import { useKnowledgeQuery } from '@/hooks/useKnowledgeQuery';
import { useSharing } from '@/hooks/useSharing';
import { useFederation } from '@/hooks/useFederation';
import { useKnowledgeExplorerStore } from '@/store/knowledgeExplorerStore';
import { QueryInterface } from './QueryInterface';
import { NodeBrowser } from './NodeBrowser';
import { GraphViewer } from './GraphViewer';
import { StaleKnowledgeTab, type StaleNode } from './StaleKnowledgeTab';
import { SharedWithMeTab, type SharedItem } from './SharedWithMeTab';
import { FederationStatus, type FederatedRegion } from './FederationStatus';
import { ShareDialog, type ShareGrant } from './ShareDialog';
import { VisibilitySelector, type VisibilityLevel } from './VisibilitySelector';
import { AccessGrantsList, type AccessGrant } from './AccessGrantsList';
import type { KnowledgeNode, GraphNode } from '@/store/knowledgeExplorerStore';

export type ExplorerTab = 'search' | 'browse' | 'graph' | 'stale' | 'shared' | 'federation';

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
  /** Whether user is admin (shows federation tab) */
  isAdmin?: boolean;
  /** Current workspace ID */
  workspaceId?: string;
  /** Available workspaces for sharing */
  availableWorkspaces?: Array<{ id: string; name: string }>;
  /** Callback when item visibility changes */
  onVisibilityChange?: (nodeId: string, visibility: VisibilityLevel) => void;
  /** Callback when sharing an item */
  onShare?: (nodeId: string, grant: ShareGrant) => void;
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
  isAdmin = false,
  workspaceId,
  availableWorkspaces = [],
  onVisibilityChange,
  onShare,
}: KnowledgeExplorerProps) {
  const [recentQueries, setRecentQueries] = useState<string[]>([]);
  const [shareDialogOpen, setShareDialogOpen] = useState(false);
  const [selectedNodeForShare, setSelectedNodeForShare] = useState<KnowledgeNode | null>(null);

  // Use hooks for sharing and federation
  const {
    sharedItems,
    isLoading: sharedItemsLoading,
    loadSharedWithMe,
    acceptSharedItem,
    declineSharedItem,
  } = useSharing({ workspaceId });

  const {
    regions: federatedRegions,
    isLoading: federationLoading,
    loadRegions,
    syncPush,
    syncPull,
    toggleRegionEnabled,
  } = useFederation({ workspaceId });

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

  // Handle share button click
  const handleShareClick = useCallback((node: KnowledgeNode) => {
    setSelectedNodeForShare(node);
    setShareDialogOpen(true);
  }, []);

  // Handle share dialog submit
  const handleShare = useCallback(
    (grant: ShareGrant) => {
      if (selectedNodeForShare && onShare) {
        onShare(selectedNodeForShare.id, grant);
      }
      setShareDialogOpen(false);
      setSelectedNodeForShare(null);
    },
    [selectedNodeForShare, onShare]
  );

  // Load shared items and federation status on mount and tab change
  useEffect(() => {
    if (activeTab === 'shared') {
      loadSharedWithMe();
    } else if (activeTab === 'federation' && isAdmin) {
      loadRegions();
    }
  }, [activeTab, isAdmin, loadSharedWithMe, loadRegions]);

  // Handle federation sync
  const handleFederationSync = useCallback(
    async (regionId: string, direction: 'push' | 'pull') => {
      try {
        if (direction === 'push') {
          await syncPush(regionId);
        } else {
          await syncPull(regionId);
        }
      } catch (error) {
        console.error(`Failed to sync ${direction} with region ${regionId}:`, error);
      }
    },
    [syncPush, syncPull]
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

    stale: (
      <StaleKnowledgeTab
        nodes={(stats?.stale_nodes || []) as StaleNode[]}
        loading={statsLoading}
        onRevalidate={async (_nodeId: string) => {
          await loadStats();
        }}
        onScheduleRevalidation={async (_nodeIds: string[]) => {
          await loadStats();
        }}
        onRefresh={loadStats}
      />
    ),

    shared: (
      <SharedWithMeTab
        items={sharedItems as unknown as SharedItem[]}
        isLoading={sharedItemsLoading}
        onItemClick={(item) => {
          // Navigate to the shared item
          onSelectNode?.({
            id: item.id,
            content: item.content,
            node_type: 'shared',
            confidence: 1,
            created_at: item.sharedAt.toISOString(),
            tier: 'medium',
            workspace_id: workspaceId || 'default',
            topics: [],
            metadata: {},
          } as unknown as KnowledgeNode);
        }}
        onAccept={async (item) => {
          try {
            await acceptSharedItem(item.id);
          } catch (error) {
            console.error('Failed to accept shared item:', error);
          }
        }}
        onDecline={async (item) => {
          try {
            await declineSharedItem(item.id);
          } catch (error) {
            console.error('Failed to decline shared item:', error);
          }
        }}
      />
    ),

    federation: (
      <FederationStatus
        regions={federatedRegions}
        isLoading={federationLoading}
        isAdmin={isAdmin}
        onSync={handleFederationSync}
        onToggleEnabled={async (regionId, enabled) => {
          try {
            await toggleRegionEnabled(regionId, enabled);
          } catch (error) {
            console.error(`Failed to toggle region ${regionId}:`, error);
          }
        }}
        onAddRegion={() => {
          // Open add region dialog - handled by parent component
          // The admin page has RegionDialog for this
          console.log('Add region clicked - use RegionDialog in admin page');
        }}
        onEditRegion={(regionId) => {
          // Open edit region dialog - handled by parent component
          console.log('Edit region:', regionId);
        }}
      />
    ),
  };

  // Build tabs array based on permissions
  const tabs = [
    { id: 'search', label: 'Search', content: tabContent.search },
    { id: 'browse', label: 'Browse', badge: totalNodes, content: tabContent.browse },
    { id: 'graph', label: 'Graph', content: tabContent.graph },
    { id: 'stale', label: 'Stale', badge: stats?.stale_nodes_count, content: tabContent.stale },
    { id: 'shared', label: 'Shared', badge: sharedItems.length || undefined, content: tabContent.shared },
  ];

  // Add federation tab for admins
  if (isAdmin) {
    tabs.push({
      id: 'federation',
      label: 'Federation',
      badge: federatedRegions.filter((r) => r.health !== 'healthy').length || undefined,
      content: tabContent.federation,
    });
  }

  return (
    <>
      <PanelTemplate
        title="Knowledge Explorer"
        icon="🧠"
        loading={statsLoading && showStats}
        onRefresh={loadStats}
        className={className}
        tabs={tabs}
        activeTab={activeTab}
        onTabChange={(tab) => {
          setActiveTab(tab as ExplorerTab);
          // Load data for new tabs (also handled by useEffect but this ensures immediate load)
          if (tab === 'shared') {
            loadSharedWithMe();
          } else if (tab === 'federation' && isAdmin) {
            loadRegions();
          }
        }}
      >
        {showStats && !statsLoading && stats && statsSummary}
      </PanelTemplate>

      {/* Share Dialog */}
      <ShareDialog
        isOpen={shareDialogOpen}
        onClose={() => {
          setShareDialogOpen(false);
          setSelectedNodeForShare(null);
        }}
        onShare={handleShare}
        itemId={selectedNodeForShare?.id || ''}
        itemTitle={selectedNodeForShare?.content?.slice(0, 100)}
        availableWorkspaces={availableWorkspaces}
      />
    </>
  );
}

export default KnowledgeExplorer;
