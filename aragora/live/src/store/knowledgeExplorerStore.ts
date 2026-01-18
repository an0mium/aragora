'use client';

import { create } from 'zustand';
import { devtools, subscribeWithSelector } from 'zustand/middleware';

// ============================================================================
// Types
// ============================================================================

export type NodeType = 'fact' | 'claim' | 'memory' | 'evidence' | 'consensus' | 'entity';
export type RelationshipType = 'supports' | 'contradicts' | 'derived_from' | 'related_to' | 'supersedes';
export type MemoryTier = 'fast' | 'medium' | 'slow' | 'glacial';
export type SortField = 'created' | 'confidence' | 'accessed' | 'relevance';

export interface ProvenanceInfo {
  source_type: 'debate' | 'document' | 'user' | 'agent' | 'import';
  source_id?: string;
  debate_id?: string;
  document_id?: string;
  agent_name?: string;
  created_at: string;
}

export interface KnowledgeNode {
  id: string;
  node_type: NodeType;
  content: string;
  confidence: number;
  tier: MemoryTier;
  workspace_id: string;
  topics: string[];
  metadata: Record<string, unknown>;
  provenance?: ProvenanceInfo;
  created_at: string;
  accessed_at?: string;
  staleness_score?: number;
}

export interface KnowledgeRelationship {
  id: string;
  from_node_id: string;
  to_node_id: string;
  relationship_type: RelationshipType;
  strength: number;
  created_at: string;
  metadata?: Record<string, unknown>;
}

export interface GraphNode extends KnowledgeNode {
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
  depth?: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: RelationshipType;
  strength: number;
}

export interface NodeFilters {
  nodeTypes: NodeType[];
  minConfidence: number;
  tier: MemoryTier | null;
  workspace: string;
  topics: string[];
  dateFrom?: string;
  dateTo?: string;
}

export interface QueryResult {
  nodes: KnowledgeNode[];
  total: number;
  took_ms: number;
}

export interface GraphResult {
  nodes: GraphNode[];
  edges: GraphEdge[];
  root_id: string;
  depth: number;
}

export interface MoundStats {
  total_nodes: number;
  nodes_by_type: Record<NodeType, number>;
  nodes_by_tier: Record<MemoryTier, number>;
  total_relationships: number;
  avg_confidence: number;
  stale_nodes_count: number;
}

// ============================================================================
// Store State
// ============================================================================

interface KnowledgeExplorerState {
  // Query state
  query: {
    text: string;
    isExecuting: boolean;
    results: KnowledgeNode[];
    total: number;
    error: string | null;
  };

  // Node browser state
  browser: {
    nodes: KnowledgeNode[];
    totalNodes: number;
    selectedNodeId: string | null;
    filters: NodeFilters;
    sortBy: SortField;
    sortDirection: 'asc' | 'desc';
    page: number;
    pageSize: number;
    isLoading: boolean;
    error: string | null;
  };

  // Graph visualization state
  graph: {
    rootNodeId: string | null;
    depth: number;
    direction: 'outgoing' | 'incoming' | 'both';
    nodes: GraphNode[];
    edges: GraphEdge[];
    selectedNodeId: string | null;
    hoveredNodeId: string | null;
    isLoading: boolean;
    error: string | null;
    zoom: number;
    panX: number;
    panY: number;
  };

  // Node detail panel
  detailPanel: {
    isOpen: boolean;
    nodeId: string | null;
    node: KnowledgeNode | null;
    relationships: KnowledgeRelationship[];
    isLoading: boolean;
  };

  // Relationship editor
  relationshipEditor: {
    isOpen: boolean;
    fromNodeId: string | null;
    toNodeId: string | null;
    editingRelationship: KnowledgeRelationship | null;
  };

  // Node editor
  nodeEditor: {
    isOpen: boolean;
    editingNode: KnowledgeNode | null;
    isNew: boolean;
  };

  // Statistics
  stats: MoundStats | null;
  statsLoading: boolean;

  // Active tab
  activeTab: 'search' | 'browse' | 'graph';
}

interface KnowledgeExplorerActions {
  // Query operations
  setQueryText: (text: string) => void;
  setQueryExecuting: (executing: boolean) => void;
  setQueryResults: (results: KnowledgeNode[], total?: number) => void;
  setQueryError: (error: string | null) => void;
  clearQueryResults: () => void;

  // Browser operations
  setBrowserNodes: (nodes: KnowledgeNode[], total?: number) => void;
  setBrowserLoading: (loading: boolean) => void;
  setBrowserError: (error: string | null) => void;
  selectBrowserNode: (id: string | null) => void;
  setFilters: (filters: Partial<NodeFilters>) => void;
  resetFilters: () => void;
  setSortBy: (field: SortField) => void;
  toggleSortDirection: () => void;
  setPage: (page: number) => void;
  setPageSize: (size: number) => void;

  // Graph operations
  setGraphRoot: (nodeId: string | null) => void;
  setGraphDepth: (depth: number) => void;
  setGraphDirection: (direction: 'outgoing' | 'incoming' | 'both') => void;
  setGraphData: (nodes: GraphNode[], edges: GraphEdge[]) => void;
  setGraphLoading: (loading: boolean) => void;
  setGraphError: (error: string | null) => void;
  selectGraphNode: (id: string | null) => void;
  hoverGraphNode: (id: string | null) => void;
  setGraphZoom: (zoom: number) => void;
  setGraphPan: (x: number, y: number) => void;
  updateNodePosition: (id: string, x: number, y: number) => void;
  clearGraph: () => void;

  // Detail panel operations
  openDetailPanel: (nodeId: string) => void;
  closeDetailPanel: () => void;
  setDetailNode: (node: KnowledgeNode | null) => void;
  setDetailRelationships: (relationships: KnowledgeRelationship[]) => void;
  setDetailLoading: (loading: boolean) => void;

  // Relationship editor operations
  openRelationshipEditor: (fromId: string, toId?: string | null) => void;
  closeRelationshipEditor: () => void;
  setEditingRelationship: (relationship: KnowledgeRelationship | null) => void;

  // Node editor operations
  openNodeEditor: (node?: KnowledgeNode | null) => void;
  closeNodeEditor: () => void;

  // Statistics
  setStats: (stats: MoundStats | null) => void;
  setStatsLoading: (loading: boolean) => void;

  // Tab navigation
  setActiveTab: (tab: 'search' | 'browse' | 'graph') => void;

  // Reset
  resetExplorer: () => void;
}

type KnowledgeExplorerStore = KnowledgeExplorerState & KnowledgeExplorerActions;

// ============================================================================
// Constants
// ============================================================================

const defaultFilters: NodeFilters = {
  nodeTypes: [],
  minConfidence: 0,
  tier: null,
  workspace: 'default',
  topics: [],
};

const initialQueryState = {
  text: '',
  isExecuting: false,
  results: [],
  total: 0,
  error: null,
};

const initialBrowserState = {
  nodes: [],
  totalNodes: 0,
  selectedNodeId: null,
  filters: { ...defaultFilters },
  sortBy: 'created' as SortField,
  sortDirection: 'desc' as const,
  page: 0,
  pageSize: 20,
  isLoading: false,
  error: null,
};

const initialGraphState = {
  rootNodeId: null,
  depth: 2,
  direction: 'both' as const,
  nodes: [],
  edges: [],
  selectedNodeId: null,
  hoveredNodeId: null,
  isLoading: false,
  error: null,
  zoom: 1,
  panX: 0,
  panY: 0,
};

const initialDetailPanelState = {
  isOpen: false,
  nodeId: null,
  node: null,
  relationships: [],
  isLoading: false,
};

const initialRelationshipEditorState = {
  isOpen: false,
  fromNodeId: null,
  toNodeId: null,
  editingRelationship: null,
};

const initialNodeEditorState = {
  isOpen: false,
  editingNode: null,
  isNew: false,
};

// ============================================================================
// Store Implementation
// ============================================================================

export const useKnowledgeExplorerStore = create<KnowledgeExplorerStore>()(
  devtools(
    subscribeWithSelector((set, get) => ({
      // Initial state
      query: { ...initialQueryState },
      browser: { ...initialBrowserState },
      graph: { ...initialGraphState },
      detailPanel: { ...initialDetailPanelState },
      relationshipEditor: { ...initialRelationshipEditorState },
      nodeEditor: { ...initialNodeEditorState },
      stats: null,
      statsLoading: false,
      activeTab: 'search',

      // Query operations
      setQueryText: (text) => set(
        (state) => ({ query: { ...state.query, text } }),
        false,
        'setQueryText'
      ),

      setQueryExecuting: (executing) => set(
        (state) => ({ query: { ...state.query, isExecuting: executing, error: null } }),
        false,
        'setQueryExecuting'
      ),

      setQueryResults: (results, total) => set(
        (state) => ({
          query: {
            ...state.query,
            results,
            total: total ?? results.length,
            isExecuting: false,
            error: null,
          },
        }),
        false,
        'setQueryResults'
      ),

      setQueryError: (error) => set(
        (state) => ({
          query: { ...state.query, error, isExecuting: false },
        }),
        false,
        'setQueryError'
      ),

      clearQueryResults: () => set(
        (state) => ({
          query: { ...state.query, results: [], total: 0, error: null },
        }),
        false,
        'clearQueryResults'
      ),

      // Browser operations
      setBrowserNodes: (nodes, total) => set(
        (state) => ({
          browser: {
            ...state.browser,
            nodes,
            totalNodes: total ?? nodes.length,
            isLoading: false,
            error: null,
          },
        }),
        false,
        'setBrowserNodes'
      ),

      setBrowserLoading: (loading) => set(
        (state) => ({ browser: { ...state.browser, isLoading: loading } }),
        false,
        'setBrowserLoading'
      ),

      setBrowserError: (error) => set(
        (state) => ({
          browser: { ...state.browser, error, isLoading: false },
        }),
        false,
        'setBrowserError'
      ),

      selectBrowserNode: (id) => set(
        (state) => ({ browser: { ...state.browser, selectedNodeId: id } }),
        false,
        'selectBrowserNode'
      ),

      setFilters: (filters) => set(
        (state) => ({
          browser: {
            ...state.browser,
            filters: { ...state.browser.filters, ...filters },
            page: 0, // Reset to first page on filter change
          },
        }),
        false,
        'setFilters'
      ),

      resetFilters: () => set(
        (state) => ({
          browser: { ...state.browser, filters: { ...defaultFilters }, page: 0 },
        }),
        false,
        'resetFilters'
      ),

      setSortBy: (field) => set(
        (state) => ({ browser: { ...state.browser, sortBy: field, page: 0 } }),
        false,
        'setSortBy'
      ),

      toggleSortDirection: () => set(
        (state) => ({
          browser: {
            ...state.browser,
            sortDirection: state.browser.sortDirection === 'asc' ? 'desc' : 'asc',
            page: 0,
          },
        }),
        false,
        'toggleSortDirection'
      ),

      setPage: (page) => set(
        (state) => ({ browser: { ...state.browser, page } }),
        false,
        'setPage'
      ),

      setPageSize: (size) => set(
        (state) => ({ browser: { ...state.browser, pageSize: size, page: 0 } }),
        false,
        'setPageSize'
      ),

      // Graph operations
      setGraphRoot: (nodeId) => set(
        (state) => ({ graph: { ...state.graph, rootNodeId: nodeId } }),
        false,
        'setGraphRoot'
      ),

      setGraphDepth: (depth) => set(
        (state) => ({ graph: { ...state.graph, depth: Math.max(1, Math.min(5, depth)) } }),
        false,
        'setGraphDepth'
      ),

      setGraphDirection: (direction) => set(
        (state) => ({ graph: { ...state.graph, direction } }),
        false,
        'setGraphDirection'
      ),

      setGraphData: (nodes, edges) => set(
        (state) => ({
          graph: { ...state.graph, nodes, edges, isLoading: false, error: null },
        }),
        false,
        'setGraphData'
      ),

      setGraphLoading: (loading) => set(
        (state) => ({ graph: { ...state.graph, isLoading: loading } }),
        false,
        'setGraphLoading'
      ),

      setGraphError: (error) => set(
        (state) => ({ graph: { ...state.graph, error, isLoading: false } }),
        false,
        'setGraphError'
      ),

      selectGraphNode: (id) => set(
        (state) => ({ graph: { ...state.graph, selectedNodeId: id } }),
        false,
        'selectGraphNode'
      ),

      hoverGraphNode: (id) => set(
        (state) => ({ graph: { ...state.graph, hoveredNodeId: id } }),
        false,
        'hoverGraphNode'
      ),

      setGraphZoom: (zoom) => set(
        (state) => ({
          graph: { ...state.graph, zoom: Math.max(0.1, Math.min(4, zoom)) },
        }),
        false,
        'setGraphZoom'
      ),

      setGraphPan: (x, y) => set(
        (state) => ({ graph: { ...state.graph, panX: x, panY: y } }),
        false,
        'setGraphPan'
      ),

      updateNodePosition: (id, x, y) => set(
        (state) => ({
          graph: {
            ...state.graph,
            nodes: state.graph.nodes.map(node =>
              node.id === id ? { ...node, x, y, fx: x, fy: y } : node
            ),
          },
        }),
        false,
        'updateNodePosition'
      ),

      clearGraph: () => set(
        (state) => ({
          graph: {
            ...initialGraphState,
            depth: state.graph.depth,
            direction: state.graph.direction,
          },
        }),
        false,
        'clearGraph'
      ),

      // Detail panel operations
      openDetailPanel: (nodeId) => set(
        {
          detailPanel: {
            isOpen: true,
            nodeId,
            node: null,
            relationships: [],
            isLoading: true,
          },
        },
        false,
        'openDetailPanel'
      ),

      closeDetailPanel: () => set(
        { detailPanel: { ...initialDetailPanelState } },
        false,
        'closeDetailPanel'
      ),

      setDetailNode: (node) => set(
        (state) => ({
          detailPanel: { ...state.detailPanel, node, isLoading: false },
        }),
        false,
        'setDetailNode'
      ),

      setDetailRelationships: (relationships) => set(
        (state) => ({
          detailPanel: { ...state.detailPanel, relationships },
        }),
        false,
        'setDetailRelationships'
      ),

      setDetailLoading: (loading) => set(
        (state) => ({
          detailPanel: { ...state.detailPanel, isLoading: loading },
        }),
        false,
        'setDetailLoading'
      ),

      // Relationship editor operations
      openRelationshipEditor: (fromId, toId = null) => set(
        {
          relationshipEditor: {
            isOpen: true,
            fromNodeId: fromId,
            toNodeId: toId,
            editingRelationship: null,
          },
        },
        false,
        'openRelationshipEditor'
      ),

      closeRelationshipEditor: () => set(
        { relationshipEditor: { ...initialRelationshipEditorState } },
        false,
        'closeRelationshipEditor'
      ),

      setEditingRelationship: (relationship) => set(
        (state) => ({
          relationshipEditor: { ...state.relationshipEditor, editingRelationship: relationship },
        }),
        false,
        'setEditingRelationship'
      ),

      // Node editor operations
      openNodeEditor: (node = null) => set(
        {
          nodeEditor: {
            isOpen: true,
            editingNode: node,
            isNew: node === null,
          },
        },
        false,
        'openNodeEditor'
      ),

      closeNodeEditor: () => set(
        { nodeEditor: { ...initialNodeEditorState } },
        false,
        'closeNodeEditor'
      ),

      // Statistics
      setStats: (stats) => set(
        { stats, statsLoading: false },
        false,
        'setStats'
      ),

      setStatsLoading: (loading) => set(
        { statsLoading: loading },
        false,
        'setStatsLoading'
      ),

      // Tab navigation
      setActiveTab: (tab) => set({ activeTab: tab }, false, 'setActiveTab'),

      // Reset
      resetExplorer: () => set({
        query: { ...initialQueryState },
        browser: { ...initialBrowserState },
        graph: { ...initialGraphState },
        detailPanel: { ...initialDetailPanelState },
        relationshipEditor: { ...initialRelationshipEditorState },
        nodeEditor: { ...initialNodeEditorState },
        stats: null,
        statsLoading: false,
        activeTab: 'search',
      }, false, 'resetExplorer'),
    })),
    { name: 'knowledge-explorer-store' }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectQuery = (state: KnowledgeExplorerStore) => state.query;
export const selectBrowser = (state: KnowledgeExplorerStore) => state.browser;
export const selectGraph = (state: KnowledgeExplorerStore) => state.graph;
export const selectDetailPanel = (state: KnowledgeExplorerStore) => state.detailPanel;
export const selectRelationshipEditor = (state: KnowledgeExplorerStore) => state.relationshipEditor;
export const selectNodeEditor = (state: KnowledgeExplorerStore) => state.nodeEditor;
export const selectStats = (state: KnowledgeExplorerStore) => state.stats;
export const selectActiveTab = (state: KnowledgeExplorerStore) => state.activeTab;

// Computed selectors
export const selectBrowserFiltersActive = (state: KnowledgeExplorerStore) => {
  const { filters } = state.browser;
  return (
    filters.nodeTypes.length > 0 ||
    filters.minConfidence > 0 ||
    filters.tier !== null ||
    filters.topics.length > 0 ||
    filters.dateFrom !== undefined ||
    filters.dateTo !== undefined
  );
};

export const selectGraphHasData = (state: KnowledgeExplorerStore) => {
  return state.graph.nodes.length > 0;
};

export const selectIsAnyPanelOpen = (state: KnowledgeExplorerStore) => {
  return (
    state.detailPanel.isOpen ||
    state.relationshipEditor.isOpen ||
    state.nodeEditor.isOpen
  );
};
