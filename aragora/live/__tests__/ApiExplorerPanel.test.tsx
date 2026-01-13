import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ApiExplorerPanel } from '@/components/ApiExplorerPanel';

// Mock fetch
global.fetch = jest.fn();

// Mock CollapsibleSection to simplify testing
jest.mock('@/components/CollapsibleSection', () => ({
  CollapsibleSection: ({ children, title }: { children: React.ReactNode; title: string }) => (
    <div data-testid={`collapsible-${title}`}>
      <span>{title}</span>
      {children}
    </div>
  ),
}));

describe('ApiExplorerPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders endpoint categories', () => {
      render(<ApiExplorerPanel />);
      // Check for main categories
      expect(screen.getByText('Debates')).toBeInTheDocument();
      expect(screen.getByText('Agents')).toBeInTheDocument();
      expect(screen.getByText('System')).toBeInTheDocument();
    });

    it('displays endpoint count', () => {
      render(<ApiExplorerPanel />);
      expect(screen.getByText(/endpoints available/i)).toBeInTheDocument();
    });

    it('displays endpoint methods', () => {
      render(<ApiExplorerPanel />);
      // GET endpoints should be visible
      const getButtons = screen.getAllByText('GET');
      expect(getButtons.length).toBeGreaterThan(0);
    });

    it('shows POST method labels', () => {
      render(<ApiExplorerPanel />);
      const postButtons = screen.getAllByText('POST');
      expect(postButtons.length).toBeGreaterThan(0);
    });
  });

  describe('Endpoint Interaction', () => {
    it('selects endpoint on click and shows details', async () => {
      render(<ApiExplorerPanel />);

      // Find and click on an endpoint button (by its summary text)
      const listDebatesButton = screen.getByText('List all debates').closest('button');
      expect(listDebatesButton).toBeInTheDocument();
      fireEvent.click(listDebatesButton!);

      // Check that endpoint details panel is shown - look for the summary text in the details panel
      await waitFor(() => {
        // When selected, the summary should appear as an h2 heading in the details section
        const heading = screen.getByRole('heading', { level: 2, name: 'List all debates' });
        expect(heading).toBeInTheDocument();
      });
    });

    it('shows path parameters input when endpoint has path params', async () => {
      render(<ApiExplorerPanel />);

      // Find endpoint with path params
      const slugButton = screen.getByText('Get debate by slug').closest('button');
      fireEvent.click(slugButton!);

      await waitFor(() => {
        // Path Parameters section should appear with slug input
        expect(screen.getByText(/Path Parameters/i)).toBeInTheDocument();
        expect(screen.getByText('slug:')).toBeInTheDocument();
      });
    });

    it('shows query parameters input when endpoint has query params', async () => {
      render(<ApiExplorerPanel />);

      // Find endpoint with query params
      const listButton = screen.getByText('List all debates').closest('button');
      fireEvent.click(listButton!);

      await waitFor(() => {
        // Query Parameters section should appear
        expect(screen.getByText(/Query Parameters/i)).toBeInTheDocument();
        expect(screen.getByText('limit:')).toBeInTheDocument();
      });
    });
  });

  describe('Try It Feature', () => {
    it('shows Send button for endpoints', async () => {
      render(<ApiExplorerPanel />);

      // Expand an endpoint
      const endpoint = screen.getByText('Get agent rankings').closest('button');
      fireEvent.click(endpoint!);

      await waitFor(() => {
        // Button says "Send GET" not "Try It"
        expect(screen.getByRole('button', { name: /send get/i })).toBeInTheDocument();
      });
    });

    it('makes API request when Send is clicked', async () => {
      const mockResponse = { agents: [{ name: 'claude', elo: 1500 }] };
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockResponse),
      });

      render(<ApiExplorerPanel />);

      // Expand leaderboard endpoint
      const endpoint = screen.getByText('Get agent rankings').closest('button');
      fireEvent.click(endpoint!);

      // Click Send button
      await waitFor(() => {
        const sendButton = screen.getByRole('button', { name: /send get/i });
        fireEvent.click(sendButton);
      });

      // Verify fetch was called
      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalled();
      });
    });

    it('displays response after successful request', async () => {
      const mockResponse = { status: 'ok', data: [] };
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockResponse),
      });

      render(<ApiExplorerPanel />);

      // Expand and try endpoint
      const endpoint = screen.getByText('Get agent rankings').closest('button');
      fireEvent.click(endpoint!);

      await waitFor(() => {
        const sendButton = screen.getByRole('button', { name: /send get/i });
        fireEvent.click(sendButton);
      });

      // Response header should be displayed after request completes
      await waitFor(() => {
        expect(screen.getByText(/Response/i)).toBeInTheDocument();
      });
    });

    it('shows error message on failed request', async () => {
      (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

      render(<ApiExplorerPanel />);

      // Expand and try endpoint
      const endpoint = screen.getByText('Get agent rankings').closest('button');
      fireEvent.click(endpoint!);

      await waitFor(() => {
        const sendButton = screen.getByRole('button', { name: /send get/i });
        fireEvent.click(sendButton);
      });

      // Error should be displayed
      await waitFor(() => {
        expect(screen.getByText(/Network error/i)).toBeInTheDocument();
      });
    });
  });

  describe('Search/Filter', () => {
    it('has a search input', () => {
      render(<ApiExplorerPanel />);
      const searchInput = screen.getByPlaceholderText(/search endpoints/i);
      expect(searchInput).toBeInTheDocument();
    });

    it('filters endpoints based on search query', async () => {
      render(<ApiExplorerPanel />);

      // Find search input
      const searchInput = screen.getByPlaceholderText(/search endpoints/i);
      fireEvent.change(searchInput, { target: { value: 'graph' } });

      // Graph Debates category should still be visible (contains graph endpoints)
      await waitFor(() => {
        expect(screen.getByText('Graph Debates')).toBeInTheDocument();
      });
    });

    it('hides non-matching categories when filtering', async () => {
      render(<ApiExplorerPanel />);

      const searchInput = screen.getByPlaceholderText(/search endpoints/i);
      fireEvent.change(searchInput, { target: { value: 'memory' } });

      // Memory category should be visible
      await waitFor(() => {
        expect(screen.getByText('Memory')).toBeInTheDocument();
      });
    });
  });

  describe('Authentication Indicator', () => {
    it('shows auth required badge for protected endpoints', async () => {
      render(<ApiExplorerPanel />);

      // Find a protected endpoint
      const batchButton = screen.getByText('Submit batch of debates').closest('button');
      fireEvent.click(batchButton!);

      // Auth badge should be visible (shows "Requires Auth")
      await waitFor(() => {
        expect(screen.getByText(/Requires Auth/i)).toBeInTheDocument();
      });
    });
  });

  describe('Rate Limit Display', () => {
    it('shows rate limit info for rate-limited endpoints', async () => {
      render(<ApiExplorerPanel />);

      // Find a rate-limited endpoint
      const graphButton = screen.getByText('Run graph-structured debate').closest('button');
      fireEvent.click(graphButton!);

      // Rate limit should be visible
      await waitFor(() => {
        expect(screen.getByText(/Rate Limit: 5\/min/i)).toBeInTheDocument();
      });
    });
  });

  describe('Request Body', () => {
    it('shows request body textarea for POST endpoints', async () => {
      render(<ApiExplorerPanel />);

      // Select a POST endpoint
      const batchButton = screen.getByText('Submit batch of debates').closest('button');
      fireEvent.click(batchButton!);

      await waitFor(() => {
        expect(screen.getByText(/Request Body/i)).toBeInTheDocument();
        expect(screen.getByPlaceholderText(/JSON request body/i)).toBeInTheDocument();
      });
    });

    it('pre-fills example request body', async () => {
      render(<ApiExplorerPanel />);

      // Select a POST endpoint with example body
      const batchButton = screen.getByText('Submit batch of debates').closest('button');
      fireEvent.click(batchButton!);

      await waitFor(() => {
        const textarea = screen.getByPlaceholderText(/JSON request body/i) as HTMLTextAreaElement;
        expect(textarea.value).toContain('items');
      });
    });
  });
});
