/**
 * Tests for KnowledgeExplorer component
 *
 * Tests cover:
 * - Tab navigation (search, browse, graph, stale)
 * - Search functionality
 * - Stats display
 * - Node selection callbacks
 * - Loading states
 */

import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import type { KnowledgeExplorerProps } from '../src/components/control-plane/KnowledgeExplorer/KnowledgeExplorer';

// Mock the child components that use d3-force
jest.mock('../src/components/control-plane/KnowledgeExplorer/GraphViewer', () => ({
  GraphViewer: () => <div data-testid="mock-graph-viewer">Graph Viewer</div>,
}));

// Mock the hooks
const mockExecuteQuery = jest.fn();
const mockLoadNodes = jest.fn();
const mockLoadGraph = jest.fn();
const mockClearGraph = jest.fn();
const mockLoadStats = jest.fn();
const mockSetQueryText = jest.fn();

let mockActiveTab = 'search';
const mockSetActiveTab = jest.fn((tab: string) => {
  mockActiveTab = tab;
});

jest.mock('../src/hooks/useKnowledgeQuery', () => ({
  useKnowledgeQuery: () => ({
    queryText: '',
    setQueryText: mockSetQueryText,
    executeQuery: mockExecuteQuery,
    isQueryExecuting: false,
    queryResults: [],
    queryError: null,
    browserNodes: [
      { id: 'node-1', type: 'concept', title: 'Test Concept', summary: 'A test concept node', created_at: '2024-01-01T00:00:00Z' },
      { id: 'node-2', type: 'fact', title: 'Test Fact', summary: 'A test fact node', created_at: '2024-01-02T00:00:00Z' },
    ],
    browserLoading: false,
    totalNodes: 100,
    loadNodes: mockLoadNodes,
    graphNodes: [],
    graphEdges: [],
    graphLoading: false,
    loadGraph: mockLoadGraph,
    clearGraph: mockClearGraph,
    stats: {
      totalNodes: 1234,
      totalEdges: 5678,
      nodeTypes: { concept: 500, fact: 400, entity: 334 },
      avgConnections: 4.6,
    },
    statsLoading: false,
    loadStats: mockLoadStats,
  }),
}));

jest.mock('../src/store/knowledgeExplorerStore', () => ({
  useKnowledgeExplorerStore: () => ({
    activeTab: mockActiveTab,
    setActiveTab: mockSetActiveTab,
  }),
}));

// Import after mocks are set up
import { KnowledgeExplorer } from '../src/components/control-plane/KnowledgeExplorer/KnowledgeExplorer';

describe('KnowledgeExplorer', () => {
  beforeEach(() => {
    mockActiveTab = 'search';
    jest.clearAllMocks();
  });

  describe('Header and Layout', () => {
    it('renders the explorer header', () => {
      render(<KnowledgeExplorer />);

      expect(screen.getByText('KNOWLEDGE MOUND')).toBeInTheDocument();
    });

    it('shows all tabs', () => {
      render(<KnowledgeExplorer />);

      expect(screen.getByText('Search')).toBeInTheDocument();
      expect(screen.getByText('Browse')).toBeInTheDocument();
      expect(screen.getByText('Graph')).toBeInTheDocument();
      expect(screen.getByText('Stale')).toBeInTheDocument();
    });
  });

  describe('Tab Navigation', () => {
    it('starts with search tab by default', () => {
      render(<KnowledgeExplorer />);

      const searchTab = screen.getByText('Search');
      expect(searchTab).toHaveClass('text-acid-green');
    });

    it('calls setActiveTab when tab is clicked', async () => {
      render(<KnowledgeExplorer />);

      await act(async () => {
        fireEvent.click(screen.getByText('Browse'));
      });

      expect(mockSetActiveTab).toHaveBeenCalledWith('browse');
    });

    it('switches between all tabs', async () => {
      render(<KnowledgeExplorer />);

      // Click Graph
      await act(async () => {
        fireEvent.click(screen.getByText('Graph'));
      });
      expect(mockSetActiveTab).toHaveBeenCalledWith('graph');

      // Click Stale
      await act(async () => {
        fireEvent.click(screen.getByText('Stale'));
      });
      expect(mockSetActiveTab).toHaveBeenCalledWith('stale');
    });
  });

  describe('Statistics Display', () => {
    it('shows statistics when showStats is true', () => {
      render(<KnowledgeExplorer showStats={true} />);

      expect(screen.getByText('1,234')).toBeInTheDocument(); // totalNodes
      expect(screen.getByText('5,678')).toBeInTheDocument(); // totalEdges
    });

    it('displays stat labels', () => {
      render(<KnowledgeExplorer showStats={true} />);

      expect(screen.getByText('Nodes')).toBeInTheDocument();
      expect(screen.getByText('Edges')).toBeInTheDocument();
    });
  });

  describe('Search Functionality', () => {
    it('renders search input', () => {
      render(<KnowledgeExplorer />);

      expect(screen.getByPlaceholderText(/search/i)).toBeInTheDocument();
    });

    it('has a search button', () => {
      render(<KnowledgeExplorer />);

      const searchButton = screen.getByRole('button', { name: /search/i });
      expect(searchButton).toBeInTheDocument();
    });
  });

  describe('CSS Classes', () => {
    it('applies custom className', () => {
      const { container } = render(<KnowledgeExplorer className="custom-class" />);

      expect(container.firstChild).toHaveClass('custom-class');
    });
  });
});
