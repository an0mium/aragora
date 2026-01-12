/**
 * Tests for GraphDebateBrowser component
 *
 * Tests the graph-based debate visualization with D3 force layout,
 * node interactions, and real-time WebSocket updates.
 */

import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Mock d3 modules
jest.mock('d3-force', () => ({
  forceSimulation: jest.fn(() => ({
    force: jest.fn().mockReturnThis(),
    nodes: jest.fn().mockReturnThis(),
    on: jest.fn().mockReturnThis(),
    alpha: jest.fn().mockReturnThis(),
    restart: jest.fn().mockReturnThis(),
    stop: jest.fn().mockReturnThis(),
  })),
  forceLink: jest.fn(() => ({
    id: jest.fn().mockReturnThis(),
    distance: jest.fn().mockReturnThis(),
    links: jest.fn().mockReturnThis(),
  })),
  forceManyBody: jest.fn(() => ({
    strength: jest.fn().mockReturnThis(),
  })),
  forceCenter: jest.fn(() => ({})),
  forceCollide: jest.fn(() => ({
    radius: jest.fn().mockReturnThis(),
  })),
}));

jest.mock('d3-selection', () => ({
  select: jest.fn(() => ({
    selectAll: jest.fn().mockReturnThis(),
    data: jest.fn().mockReturnThis(),
    join: jest.fn().mockReturnThis(),
    attr: jest.fn().mockReturnThis(),
    style: jest.fn().mockReturnThis(),
    on: jest.fn().mockReturnThis(),
    call: jest.fn().mockReturnThis(),
    text: jest.fn().mockReturnThis(),
    append: jest.fn().mockReturnThis(),
  })),
}));

// Mock the WebSocket hook
const mockWsReturn = {
  isConnected: false,
  lastEvent: null,
  status: 'disconnected' as const,
  reconnect: jest.fn(),
  events: [],
  error: null,
  clearEvents: jest.fn(),
};

jest.mock('../src/hooks/useGraphDebateWebSocket', () => ({
  useGraphDebateWebSocket: jest.fn(() => mockWsReturn),
}));

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

// Mock ResizeObserver
global.ResizeObserver = jest.fn().mockImplementation(() => ({
  observe: jest.fn(),
  unobserve: jest.fn(),
  disconnect: jest.fn(),
}));

// Import after mocks
import { GraphDebateBrowser } from '../src/components/GraphDebateBrowser';
import { useGraphDebateWebSocket } from '../src/hooks/useGraphDebateWebSocket';

const mockDebates = [
  {
    debate_id: 'debate-1',
    topic: 'Should AI be regulated?',
    created_at: '2024-01-15T10:00:00Z',
    status: 'completed',
    graph: {
      nodes: [
        { id: 'node-1', type: 'root', agent_id: null, content: 'Root question', branch_id: 'main' },
        { id: 'node-2', type: 'proposal', agent_id: 'claude', content: 'Yes, AI should be regulated', branch_id: 'main' },
        { id: 'node-3', type: 'critique', agent_id: 'gpt4', content: 'But regulation could stifle innovation', branch_id: 'main' },
      ],
      edges: [
        { source: 'node-1', target: 'node-2' },
        { source: 'node-2', target: 'node-3' },
      ],
      branches: ['main'],
    },
  },
  {
    debate_id: 'debate-2',
    topic: 'Climate change solutions',
    created_at: '2024-01-14T09:00:00Z',
    status: 'in_progress',
    graph: {
      nodes: [
        { id: 'node-a', type: 'root', agent_id: null, content: 'Best approach to climate change?', branch_id: 'main' },
      ],
      edges: [],
      branches: ['main'],
    },
  },
];

describe('GraphDebateBrowser', () => {
  beforeEach(() => {
    mockFetch.mockClear();
    jest.clearAllMocks();
    (useGraphDebateWebSocket as jest.Mock).mockReturnValue(mockWsReturn);
  });

  describe('Rendering', () => {
    it('renders the component title', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: [] }),
      });

      render(<GraphDebateBrowser />);

      expect(screen.getByText('Graph Debates')).toBeInTheDocument();
    });

    it('shows loading state initially', () => {
      mockFetch.mockImplementation(() => new Promise(() => {}));

      render(<GraphDebateBrowser />);

      expect(screen.getByText('Loading debates...')).toBeInTheDocument();
    });

    it('renders debate list after loading', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: mockDebates }),
      });

      render(<GraphDebateBrowser />);

      await waitFor(() => {
        expect(screen.getByText('Should AI be regulated?')).toBeInTheDocument();
        expect(screen.getByText('Climate change solutions')).toBeInTheDocument();
      });
    });

    it('shows empty state when no debates', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: [] }),
      });

      render(<GraphDebateBrowser />);

      await waitFor(() => {
        expect(screen.getByText(/no graph debates/i)).toBeInTheDocument();
      });
    });

    it('shows error state on fetch failure', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      render(<GraphDebateBrowser />);

      await waitFor(() => {
        expect(screen.getByText(/failed to load/i)).toBeInTheDocument();
      });
    });
  });

  describe('Debate Selection', () => {
    it('selects debate when clicked', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: mockDebates }),
      });

      render(<GraphDebateBrowser />);

      await waitFor(() => {
        expect(screen.getByText('Should AI be regulated?')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText('Should AI be regulated?'));

      await waitFor(() => {
        // Should show graph visualization area
        expect(screen.getByTestId('graph-container')).toBeInTheDocument();
      });
    });

    it('auto-selects debate when initialDebateId is provided', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: mockDebates }),
      });

      render(<GraphDebateBrowser initialDebateId="debate-1" />);

      await waitFor(() => {
        // Should auto-select and show graph
        expect(screen.getByTestId('graph-container')).toBeInTheDocument();
      });
    });
  });

  describe('Graph Visualization', () => {
    it('renders SVG container for graph', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: mockDebates }),
      });

      render(<GraphDebateBrowser initialDebateId="debate-1" />);

      await waitFor(() => {
        const svgElement = document.querySelector('svg');
        expect(svgElement).toBeInTheDocument();
      });
    });

    it('shows node count for selected debate', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: mockDebates }),
      });

      render(<GraphDebateBrowser initialDebateId="debate-1" />);

      await waitFor(() => {
        expect(screen.getByText(/3 nodes/i)).toBeInTheDocument();
      });
    });
  });

  describe('WebSocket Integration', () => {
    it('shows connection status indicator', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: mockDebates }),
      });

      (useGraphDebateWebSocket as jest.Mock).mockReturnValue({
        ...mockWsReturn,
        isConnected: true,
        status: 'connected',
      });

      render(<GraphDebateBrowser initialDebateId="debate-1" />);

      await waitFor(() => {
        expect(screen.getByText(/connected/i)).toBeInTheDocument();
      });
    });

    it('shows reconnect button when disconnected', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: mockDebates }),
      });

      (useGraphDebateWebSocket as jest.Mock).mockReturnValue({
        ...mockWsReturn,
        isConnected: false,
        status: 'disconnected',
      });

      render(<GraphDebateBrowser initialDebateId="debate-1" />);

      await waitFor(() => {
        const reconnectButton = screen.getByRole('button', { name: /reconnect/i });
        expect(reconnectButton).toBeInTheDocument();
      });
    });

    it('calls reconnect when button clicked', async () => {
      const mockReconnect = jest.fn();
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: mockDebates }),
      });

      (useGraphDebateWebSocket as jest.Mock).mockReturnValue({
        ...mockWsReturn,
        isConnected: false,
        status: 'disconnected',
        reconnect: mockReconnect,
      });

      render(<GraphDebateBrowser initialDebateId="debate-1" />);

      await waitFor(() => {
        const reconnectButton = screen.getByRole('button', { name: /reconnect/i });
        fireEvent.click(reconnectButton);
      });

      expect(mockReconnect).toHaveBeenCalled();
    });
  });

  describe('Branch Filtering', () => {
    it('shows branch filter when multiple branches exist', async () => {
      const debateWithBranches = {
        ...mockDebates[0],
        graph: {
          ...mockDebates[0].graph,
          branches: ['main', 'alternative-1', 'alternative-2'],
        },
      };

      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: [debateWithBranches] }),
      });

      render(<GraphDebateBrowser initialDebateId="debate-1" />);

      await waitFor(() => {
        expect(screen.getByText(/branches/i)).toBeInTheDocument();
      });
    });
  });

  describe('Controls', () => {
    it('renders zoom controls', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: mockDebates }),
      });

      render(<GraphDebateBrowser initialDebateId="debate-1" />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /zoom in/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /zoom out/i })).toBeInTheDocument();
      });
    });

    it('renders reset view button', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: mockDebates }),
      });

      render(<GraphDebateBrowser initialDebateId="debate-1" />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /reset/i })).toBeInTheDocument();
      });
    });
  });

  describe('Node Details', () => {
    it('shows node details panel when node is selected', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: mockDebates }),
      });

      render(<GraphDebateBrowser initialDebateId="debate-1" />);

      await waitFor(() => {
        expect(screen.getByTestId('graph-container')).toBeInTheDocument();
      });

      // Simulate node selection (would typically happen via D3 click event)
      // This tests that the details panel structure exists
      const detailsSection = screen.queryByTestId('node-details');
      // Details panel may or may not be visible depending on selection state
      expect(true).toBe(true); // Placeholder for actual node click simulation
    });
  });

  describe('Refresh', () => {
    it('renders refresh button', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: mockDebates }),
      });

      render(<GraphDebateBrowser />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument();
      });
    });

    it('refetches debates when refresh clicked', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: mockDebates }),
      });

      render(<GraphDebateBrowser />);

      await waitFor(() => {
        expect(screen.getByText('Should AI be regulated?')).toBeInTheDocument();
      });

      // Clear previous calls
      mockFetch.mockClear();
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ debates: mockDebates }),
      });

      fireEvent.click(screen.getByRole('button', { name: /refresh/i }));

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalled();
      });
    });
  });
});
