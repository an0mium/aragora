/**
 * Tests for A/B Testing Dashboard page
 *
 * Tests cover:
 * - Loading state
 * - Empty state
 * - Test list display
 * - Test filtering
 * - Test creation form
 * - Test detail view
 * - Test actions (conclude, cancel)
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ABTestingPage from '../src/app/ab-testing/page';

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

// Mock next/navigation
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
  }),
}));

// Mock AuthContext
jest.mock('../src/context/AuthContext', () => ({
  useAuth: () => ({
    tokens: { access_token: 'test-token' },
    isLoading: false,
    isAuthenticated: true,
    organization: { tier: 'professional' },
  }),
}));

// Mock ProtectedRoute
jest.mock('../src/components/auth/ProtectedRoute', () => ({
  ProtectedRoute: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

// Mock MatrixRain
jest.mock('../src/components/MatrixRain', () => ({
  Scanlines: () => null,
  CRTVignette: () => null,
}));

// Mock AsciiBanner
jest.mock('../src/components/AsciiBanner', () => ({
  AsciiBannerCompact: () => <div>ARAGORA</div>,
}));

const mockTests = [
  {
    id: 'test-1',
    agent: 'claude-3-opus',
    baseline_prompt_version: 1,
    evolved_prompt_version: 2,
    baseline_wins: 8,
    evolved_wins: 12,
    baseline_debates: 10,
    evolved_debates: 10,
    evolved_win_rate: 0.6,
    baseline_win_rate: 0.4,
    total_debates: 20,
    sample_size: 20,
    is_significant: false,
    started_at: '2026-01-10T10:00:00Z',
    concluded_at: null,
    status: 'active',
    metadata: {},
  },
  {
    id: 'test-2',
    agent: 'gpt-4o',
    baseline_prompt_version: 3,
    evolved_prompt_version: 4,
    baseline_wins: 10,
    evolved_wins: 25,
    baseline_debates: 20,
    evolved_debates: 20,
    evolved_win_rate: 0.714,
    baseline_win_rate: 0.286,
    total_debates: 40,
    sample_size: 35,
    is_significant: true,
    started_at: '2026-01-09T08:00:00Z',
    concluded_at: '2026-01-10T08:00:00Z',
    status: 'concluded',
    metadata: { description: 'Testing improved reasoning' },
  },
];

describe('ABTestingPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
  });

  describe('Loading State', () => {
    it('shows loading state initially', async () => {
      mockFetch.mockImplementation(() => new Promise(() => {}));

      render(<ABTestingPage />);

      expect(screen.getByText(/loading/i)).toBeInTheDocument();
    });
  });

  describe('Empty State', () => {
    it('shows empty message when no tests exist', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ tests: [], count: 0 }),
      });

      render(<ABTestingPage />);

      await waitFor(() => {
        expect(screen.getByText(/no a\/b tests found/i)).toBeInTheDocument();
      });
    });
  });

  describe('Test List', () => {
    it('displays list of tests', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ tests: mockTests, count: 2 }),
      });

      render(<ABTestingPage />);

      await waitFor(() => {
        expect(screen.getByText('claude-3-opus')).toBeInTheDocument();
        expect(screen.getByText('gpt-4o')).toBeInTheDocument();
      });
    });

    it('shows version comparison', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ tests: mockTests, count: 2 }),
      });

      render(<ABTestingPage />);

      await waitFor(() => {
        expect(screen.getByText(/v1 vs v2/)).toBeInTheDocument();
        expect(screen.getByText(/v3 vs v4/)).toBeInTheDocument();
      });
    });

    it('displays status badges', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ tests: mockTests, count: 2 }),
      });

      render(<ABTestingPage />);

      await waitFor(() => {
        expect(screen.getByText('ACTIVE')).toBeInTheDocument();
        expect(screen.getByText('CONCLUDED')).toBeInTheDocument();
      });
    });

    it('shows win rates', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ tests: mockTests, count: 2 }),
      });

      render(<ABTestingPage />);

      await waitFor(() => {
        expect(screen.getByText('60.0%')).toBeInTheDocument();
        expect(screen.getByText('71.4%')).toBeInTheDocument();
      });
    });

    it('shows significance indicator for significant tests', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ tests: mockTests, count: 2 }),
      });

      render(<ABTestingPage />);

      await waitFor(() => {
        // The significant test should have an asterisk
        const significanceMarkers = screen.getAllByText('*');
        expect(significanceMarkers.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Filtering', () => {
    it('filters by status', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ tests: mockTests, count: 2 }),
      });

      render(<ABTestingPage />);

      await waitFor(() => {
        expect(screen.getByText('claude-3-opus')).toBeInTheDocument();
      });

      // Change status filter
      const statusSelect = screen.getByDisplayValue('All Statuses');
      fireEvent.change(statusSelect, { target: { value: 'active' } });

      // Should trigger refetch with status param
      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('status=active'),
          expect.any(Object)
        );
      });
    });
  });

  describe('Create Test View', () => {
    it('switches to create view when button clicked', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ tests: [], count: 0 }),
      });

      render(<ABTestingPage />);

      await waitFor(() => {
        expect(screen.getByText(/new test/i)).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText(/new test/i));

      expect(screen.getByText(/create new a\/b test/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/agent name/i)).toBeInTheDocument();
    });

    it('has form fields for test creation', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ tests: [], count: 0 }),
      });

      render(<ABTestingPage />);

      await waitFor(() => {
        expect(screen.getByText(/new test/i)).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText(/new test/i));

      expect(screen.getByPlaceholderText(/e.g., claude-3-opus/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/baseline version/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/evolved version/i)).toBeInTheDocument();
    });

    it('submits create form', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ tests: [], count: 0 }),
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            message: 'A/B test created',
            test: mockTests[0],
          }),
        });

      render(<ABTestingPage />);

      await waitFor(() => {
        expect(screen.getByText(/new test/i)).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText(/new test/i));

      // Fill form
      fireEvent.change(screen.getByPlaceholderText(/e.g., claude-3-opus/i), {
        target: { value: 'test-agent' },
      });
      fireEvent.change(screen.getByLabelText(/baseline version/i), {
        target: { value: '1' },
      });
      fireEvent.change(screen.getByLabelText(/evolved version/i), {
        target: { value: '2' },
      });

      // Submit
      fireEvent.click(screen.getByText(/start a\/b test/i));

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('/api/evolution/ab-tests'),
          expect.objectContaining({
            method: 'POST',
            body: expect.stringContaining('test-agent'),
          })
        );
      });
    });
  });

  describe('Detail View', () => {
    it('shows detail view when test clicked', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ tests: mockTests, count: 2 }),
      });

      render(<ABTestingPage />);

      await waitFor(() => {
        expect(screen.getByText('claude-3-opus')).toBeInTheDocument();
      });

      // Click on the test row
      fireEvent.click(screen.getByText('claude-3-opus'));

      await waitFor(() => {
        // Should show detail view with baseline and evolved sections
        expect(screen.getByText(/baseline/i)).toBeInTheDocument();
        expect(screen.getByText(/evolved/i)).toBeInTheDocument();
      });
    });

    it('shows action buttons for active test', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ tests: [mockTests[0]], count: 1 }),
      });

      render(<ABTestingPage />);

      await waitFor(() => {
        expect(screen.getByText('claude-3-opus')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText('claude-3-opus'));

      await waitFor(() => {
        expect(screen.getByText(/conclude test/i)).toBeInTheDocument();
        expect(screen.getByText(/cancel/i)).toBeInTheDocument();
      });
    });
  });

  describe('Refresh', () => {
    it('refetches when refresh clicked', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ tests: mockTests, count: 2 }),
      });

      render(<ABTestingPage />);

      await waitFor(() => {
        expect(screen.getByText('claude-3-opus')).toBeInTheDocument();
      });

      expect(mockFetch).toHaveBeenCalledTimes(1);

      fireEvent.click(screen.getByText(/refresh/i));

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledTimes(2);
      });
    });
  });
});
