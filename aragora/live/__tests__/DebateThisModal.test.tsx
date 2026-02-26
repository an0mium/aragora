import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { DebateThisModal } from '../src/components/DebateThisModal';

const mockPush = jest.fn();

jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
    replace: jest.fn(),
    prefetch: jest.fn(),
    back: jest.fn(),
  }),
}));

const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('DebateThisModal', () => {
  const defaultProps = {
    question: 'Should we switch to Rust?',
    context: 'Current backend is Python',
    source: 'pulse',
    onClose: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
  });

  describe('rendering', () => {
    it('renders modal header', () => {
      render(<DebateThisModal {...defaultProps} />);
      expect(screen.getByText('DEBATE THIS')).toBeInTheDocument();
    });

    it('shows source label', () => {
      render(<DebateThisModal {...defaultProps} />);
      expect(screen.getByText('Trending Topic')).toBeInTheDocument();
    });

    it('shows source label for receipt', () => {
      render(<DebateThisModal {...defaultProps} source="receipt" />);
      expect(screen.getByText('Decision Receipt')).toBeInTheDocument();
    });

    it('shows raw source string for unknown source', () => {
      render(<DebateThisModal {...defaultProps} source="custom_source" />);
      expect(screen.getByText('custom_source')).toBeInTheDocument();
    });

    it('pre-fills the question textarea', () => {
      render(<DebateThisModal {...defaultProps} />);
      const textarea = screen.getByPlaceholderText('What should be debated?');
      expect(textarea).toHaveValue('Should we switch to Rust?');
    });

    it('renders format toggle with quick and thorough', () => {
      render(<DebateThisModal {...defaultProps} />);
      expect(screen.getByText('QUICK')).toBeInTheDocument();
      expect(screen.getByText('THOROUGH')).toBeInTheDocument();
      expect(screen.getByText('4 rounds')).toBeInTheDocument();
      expect(screen.getByText('9 rounds')).toBeInTheDocument();
    });

    it('renders Start Debate and Cancel buttons', () => {
      render(<DebateThisModal {...defaultProps} />);
      expect(screen.getByText('Start Debate')).toBeInTheDocument();
      expect(screen.getByText('Cancel')).toBeInTheDocument();
    });
  });

  describe('context section', () => {
    it('shows context toggle when context is provided', () => {
      render(<DebateThisModal {...defaultProps} />);
      expect(screen.getByText(/Context \(25 chars\)/)).toBeInTheDocument();
    });

    it('does not show context toggle when no context', () => {
      render(<DebateThisModal {...defaultProps} context={undefined} />);
      expect(screen.queryByText(/Context/)).not.toBeInTheDocument();
    });

    it('expands context on click', () => {
      render(<DebateThisModal {...defaultProps} />);
      expect(screen.queryByText('Current backend is Python')).not.toBeInTheDocument();
      fireEvent.click(screen.getByText(/Context \(25 chars\)/));
      expect(screen.getByText('Current backend is Python')).toBeInTheDocument();
    });
  });

  describe('close behavior', () => {
    it('calls onClose when Cancel is clicked', () => {
      render(<DebateThisModal {...defaultProps} />);
      fireEvent.click(screen.getByText('Cancel'));
      expect(defaultProps.onClose).toHaveBeenCalledTimes(1);
    });

    it('calls onClose when close button (x) is clicked', () => {
      render(<DebateThisModal {...defaultProps} />);
      fireEvent.click(screen.getByLabelText('Close'));
      expect(defaultProps.onClose).toHaveBeenCalledTimes(1);
    });

    it('calls onClose when backdrop is clicked', () => {
      render(<DebateThisModal {...defaultProps} />);
      const backdrop = document.querySelector('.backdrop-blur-sm');
      if (backdrop) fireEvent.click(backdrop);
      expect(defaultProps.onClose).toHaveBeenCalledTimes(1);
    });
  });

  describe('format selection', () => {
    it('defaults to quick format', () => {
      render(<DebateThisModal {...defaultProps} />);
      const quickBtn = screen.getByText('QUICK').closest('button');
      expect(quickBtn?.className).toContain('border-acid-green');
    });

    it('switches to thorough format on click', () => {
      render(<DebateThisModal {...defaultProps} />);
      fireEvent.click(screen.getByText('THOROUGH'));
      const thoroughBtn = screen.getByText('THOROUGH').closest('button');
      expect(thoroughBtn?.className).toContain('border-acid-cyan');
    });
  });

  describe('question editing', () => {
    it('allows editing the question', () => {
      render(<DebateThisModal {...defaultProps} />);
      const textarea = screen.getByPlaceholderText('What should be debated?');
      fireEvent.change(textarea, { target: { value: 'New question' } });
      expect(textarea).toHaveValue('New question');
    });

    it('disables Start Debate when question is empty', () => {
      render(<DebateThisModal {...defaultProps} />);
      const textarea = screen.getByPlaceholderText('What should be debated?');
      fireEvent.change(textarea, { target: { value: '   ' } });
      expect(screen.getByText('Start Debate')).toBeDisabled();
    });
  });

  describe('debate creation', () => {
    it('creates debate with correct payload (quick format)', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ debate_id: 'debate-123' }),
      });

      render(<DebateThisModal {...defaultProps} />);

      await act(async () => {
        fireEvent.click(screen.getByText('Start Debate'));
      });

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('/api/v1/debates'),
          expect.objectContaining({
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: expect.stringContaining('"rounds":4'),
          })
        );
      });
    });

    it('creates debate with 9 rounds in thorough format', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ debate_id: 'debate-456' }),
      });

      render(<DebateThisModal {...defaultProps} />);
      fireEvent.click(screen.getByText('THOROUGH'));

      await act(async () => {
        fireEvent.click(screen.getByText('Start Debate'));
      });

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          expect.anything(),
          expect.objectContaining({
            body: expect.stringContaining('"rounds":9'),
          })
        );
      });
    });

    it('navigates to debate page on success', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ debate_id: 'debate-789' }),
      });

      render(<DebateThisModal {...defaultProps} />);

      await act(async () => {
        fireEvent.click(screen.getByText('Start Debate'));
      });

      await waitFor(() => {
        expect(mockPush).toHaveBeenCalledWith('/debates/debate-789');
        expect(defaultProps.onClose).toHaveBeenCalled();
      });
    });

    it('shows loading state during submission', async () => {
      let resolvePromise: (value: unknown) => void;
      const fetchPromise = new Promise((resolve) => {
        resolvePromise = resolve;
      });
      mockFetch.mockReturnValueOnce(fetchPromise);

      render(<DebateThisModal {...defaultProps} />);

      await act(async () => {
        fireEvent.click(screen.getByText('Start Debate'));
      });

      expect(screen.getByText('Starting...')).toBeInTheDocument();

      await act(async () => {
        resolvePromise!({
          ok: true,
          json: async () => ({ debate_id: 'debate-loaded' }),
        });
      });
    });

    it('shows error on API failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({ error: 'Server error' }),
      });

      render(<DebateThisModal {...defaultProps} />);

      await act(async () => {
        fireEvent.click(screen.getByText('Start Debate'));
      });

      await waitFor(() => {
        expect(screen.getByText('Server error')).toBeInTheDocument();
      });
    });

    it('shows error on network failure', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network timeout'));

      render(<DebateThisModal {...defaultProps} />);

      await act(async () => {
        fireEvent.click(screen.getByText('Start Debate'));
      });

      await waitFor(() => {
        expect(screen.getByText('Network timeout')).toBeInTheDocument();
      });
    });

    it('shows fallback error when no debate_id returned', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({}),
      });

      render(<DebateThisModal {...defaultProps} />);

      await act(async () => {
        fireEvent.click(screen.getByText('Start Debate'));
      });

      await waitFor(() => {
        expect(screen.getByText('No debate ID returned')).toBeInTheDocument();
      });
    });
  });
});
