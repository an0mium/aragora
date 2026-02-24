/**
 * Tests for InterventionPanel component
 *
 * Tests cover:
 * - Inactive state rendering
 * - Tab navigation (inject, nudge, control, weights)
 * - Pause/resume toggle
 * - Argument injection
 * - Nudge direction input
 * - Challenge claim input
 * - Consensus threshold control
 * - Agent weight sliders
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { InterventionPanel } from '../InterventionPanel';

// Mock logger
jest.mock('@/utils/logger', () => ({
  logger: {
    error: jest.fn(),
    debug: jest.fn(),
  },
}));

// Mock config
jest.mock('@/config', () => ({
  API_BASE_URL: 'http://localhost:8080',
}));

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

const defaultProps = {
  debateId: 'test-debate-123',
  isActive: true,
  isPaused: false,
  currentRound: 2,
  totalRounds: 5,
  agents: ['claude', 'gpt-4'],
  consensusThreshold: 0.75,
};

describe('InterventionPanel', () => {
  beforeEach(() => {
    mockFetch.mockReset();
    mockFetch.mockResolvedValue({ ok: true });
  });

  describe('inactive state', () => {
    it('shows inactive message when not active', () => {
      render(<InterventionPanel {...defaultProps} isActive={false} />);

      expect(
        screen.getByText(/Intervention controls are only available during active debates/)
      ).toBeInTheDocument();
    });

    it('does not show tabs when inactive', () => {
      render(<InterventionPanel {...defaultProps} isActive={false} />);

      expect(screen.queryByText('Inject')).not.toBeInTheDocument();
      expect(screen.queryByText('Nudge')).not.toBeInTheDocument();
    });
  });

  describe('active state', () => {
    it('renders intervention controls header', () => {
      render(<InterventionPanel {...defaultProps} />);

      expect(screen.getByText('Intervention Controls')).toBeInTheDocument();
    });

    it('shows round progress', () => {
      render(<InterventionPanel {...defaultProps} />);

      expect(screen.getByText('Round 2/5')).toBeInTheDocument();
    });

    it('renders all four tabs', () => {
      render(<InterventionPanel {...defaultProps} />);

      expect(screen.getByText('Inject')).toBeInTheDocument();
      expect(screen.getByText('Nudge')).toBeInTheDocument();
      expect(screen.getByText('Control')).toBeInTheDocument();
      expect(screen.getByText('Weights')).toBeInTheDocument();
    });
  });

  describe('pause/resume', () => {
    it('shows PAUSE button when not paused', () => {
      render(<InterventionPanel {...defaultProps} isPaused={false} />);

      expect(screen.getByText(/PAUSE/)).toBeInTheDocument();
    });

    it('shows RESUME button when paused', () => {
      render(<InterventionPanel {...defaultProps} isPaused={true} />);

      expect(screen.getByText(/RESUME/)).toBeInTheDocument();
    });

    it('calls pause API on toggle', async () => {
      render(<InterventionPanel {...defaultProps} isPaused={false} />);

      fireEvent.click(screen.getByText(/PAUSE/));

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8080/api/debates/test-debate-123/intervention/pause',
          { method: 'POST' }
        );
      });
    });
  });

  describe('inject tab', () => {
    it('shows argument injection textarea', () => {
      render(<InterventionPanel {...defaultProps} />);

      expect(screen.getByPlaceholderText('Add your argument to the debate...')).toBeInTheDocument();
    });

    it('shows inject button', () => {
      render(<InterventionPanel {...defaultProps} />);

      expect(screen.getByRole('button', { name: /INJECT ARGUMENT/ })).toBeInTheDocument();
    });

    it('calls inject API with argument', async () => {
      render(<InterventionPanel {...defaultProps} />);

      const textarea = screen.getByPlaceholderText('Add your argument to the debate...');
      fireEvent.change(textarea, { target: { value: 'My argument' } });

      fireEvent.click(screen.getByRole('button', { name: /INJECT ARGUMENT/ }));

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8080/api/debates/test-debate-123/intervention/inject',
          expect.objectContaining({
            method: 'POST',
            body: expect.stringContaining('My argument'),
          })
        );
      });
    });

    it('disables inject button when textarea empty', () => {
      render(<InterventionPanel {...defaultProps} />);

      expect(screen.getByRole('button', { name: /INJECT ARGUMENT/ })).toBeDisabled();
    });
  });

  describe('nudge tab', () => {
    it('shows nudge direction input after switching tab', () => {
      render(<InterventionPanel {...defaultProps} />);

      fireEvent.click(screen.getByText('Nudge'));

      expect(
        screen.getByPlaceholderText('e.g., Consider the economic implications...')
      ).toBeInTheDocument();
    });

    it('shows challenge claim textarea', () => {
      render(<InterventionPanel {...defaultProps} />);

      fireEvent.click(screen.getByText('Nudge'));

      expect(
        screen.getByPlaceholderText('e.g., The claim that X is incorrect because...')
      ).toBeInTheDocument();
    });

    it('calls nudge API with direction prefix', async () => {
      render(<InterventionPanel {...defaultProps} />);

      fireEvent.click(screen.getByText('Nudge'));

      const input = screen.getByPlaceholderText('e.g., Consider the economic implications...');
      fireEvent.change(input, { target: { value: 'Focus on scalability' } });

      fireEvent.click(screen.getByRole('button', { name: /NUDGE DIRECTION/ }));

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8080/api/debates/test-debate-123/intervention/inject',
          expect.objectContaining({
            body: expect.stringContaining('[DIRECTION] Focus on scalability'),
          })
        );
      });
    });

    it('calls challenge API with challenge prefix', async () => {
      render(<InterventionPanel {...defaultProps} />);

      fireEvent.click(screen.getByText('Nudge'));

      const textarea = screen.getByPlaceholderText(
        'e.g., The claim that X is incorrect because...'
      );
      fireEvent.change(textarea, { target: { value: 'That claim lacks evidence' } });

      // Use getByRole to specifically target the button (not the label)
      const challengeButton = screen.getByRole('button', { name: /CHALLENGE CLAIM/ });
      fireEvent.click(challengeButton);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:8080/api/debates/test-debate-123/intervention/inject',
          expect.objectContaining({
            body: expect.stringContaining('[CHALLENGE] That claim lacks evidence'),
          })
        );
      });
    });
  });

  describe('control tab', () => {
    it('shows consensus threshold slider', () => {
      render(<InterventionPanel {...defaultProps} />);

      fireEvent.click(screen.getByText('Control'));

      expect(screen.getByText(/CONSENSUS THRESHOLD/)).toBeInTheDocument();
    });

    it('shows quick action buttons', () => {
      render(<InterventionPanel {...defaultProps} />);

      fireEvent.click(screen.getByText('Control'));

      expect(screen.getByText(/Skip Round/)).toBeInTheDocument();
      expect(screen.getByText(/Add Round/)).toBeInTheDocument();
      expect(screen.getByText(/Force Vote/)).toBeInTheDocument();
      expect(screen.getByText(/End Debate/)).toBeInTheDocument();
    });
  });

  describe('weights tab', () => {
    it('shows agent weight sliders', () => {
      render(<InterventionPanel {...defaultProps} />);

      fireEvent.click(screen.getByText('Weights'));

      expect(screen.getByText('claude')).toBeInTheDocument();
      expect(screen.getByText('gpt-4')).toBeInTheDocument();
    });

    it('shows default weight values', () => {
      render(<InterventionPanel {...defaultProps} />);

      fireEvent.click(screen.getByText('Weights'));

      // Default weights are 1.0x
      const weights = screen.getAllByText('1.0x');
      expect(weights.length).toBe(2);
    });
  });

  describe('footer', () => {
    it('shows audit trail message', () => {
      render(<InterventionPanel {...defaultProps} />);

      expect(screen.getByText('Interventions are logged in the audit trail')).toBeInTheDocument();
    });
  });
});
