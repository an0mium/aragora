/**
 * Constants and initial state for Knowledge Explorer Store
 */

import type {
  NodeFilters,
  QueryState,
  BrowserState,
  GraphState,
  DetailPanelState,
  RelationshipEditorState,
  NodeEditorState,
  SortField,
} from './types';

export const defaultFilters: NodeFilters = {
  nodeTypes: [],
  minConfidence: 0,
  tier: null,
  workspace: 'default',
  topics: [],
};

export const initialQueryState: QueryState = {
  text: '',
  isExecuting: false,
  results: [],
  total: 0,
  error: null,
};

export const initialBrowserState: BrowserState = {
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

export const initialGraphState: GraphState = {
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

export const initialDetailPanelState: DetailPanelState = {
  isOpen: false,
  nodeId: null,
  node: null,
  relationships: [],
  isLoading: false,
};

export const initialRelationshipEditorState: RelationshipEditorState = {
  isOpen: false,
  fromNodeId: null,
  toNodeId: null,
  editingRelationship: null,
};

export const initialNodeEditorState: NodeEditorState = {
  isOpen: false,
  editingNode: null,
  isNew: false,
};
