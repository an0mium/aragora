import { render, screen, fireEvent } from '@testing-library/react';
import { DebateThisButton } from '../src/components/DebateThisButton';

jest.mock('../src/components/DebateThisModal', () => ({
  DebateThisModal: ({ question, onClose }: { question: string; onClose: () => void }) => (
    <div data-testid="debate-modal">
      <span>Modal: {question}</span>
      <button onClick={onClose}>close-modal</button>
    </div>
  ),
}));

describe('DebateThisButton', () => {
  const defaultProps = {
    question: 'Should we migrate to Kubernetes?',
    context: 'Our infra is on ECS',
    source: 'dashboard' as const,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('variant="button" (default)', () => {
    it('renders the button text', () => {
      render(<DebateThisButton {...defaultProps} />);
      expect(screen.getByText('DEBATE THIS')).toBeInTheDocument();
    });

    it('opens modal on click', () => {
      render(<DebateThisButton {...defaultProps} />);
      expect(screen.queryByTestId('debate-modal')).not.toBeInTheDocument();
      fireEvent.click(screen.getByText('DEBATE THIS'));
      expect(screen.getByTestId('debate-modal')).toBeInTheDocument();
      expect(screen.getByText('Modal: Should we migrate to Kubernetes?')).toBeInTheDocument();
    });

    it('closes modal when onClose is called', () => {
      render(<DebateThisButton {...defaultProps} />);
      fireEvent.click(screen.getByText('DEBATE THIS'));
      expect(screen.getByTestId('debate-modal')).toBeInTheDocument();
      fireEvent.click(screen.getByText('close-modal'));
      expect(screen.queryByTestId('debate-modal')).not.toBeInTheDocument();
    });

    it('stops event propagation on click', () => {
      const parentClick = jest.fn();
      render(
        <div onClick={parentClick}>
          <DebateThisButton {...defaultProps} />
        </div>
      );
      fireEvent.click(screen.getByText('DEBATE THIS'));
      expect(parentClick).not.toHaveBeenCalled();
    });
  });

  describe('variant="icon"', () => {
    it('renders with aria-label', () => {
      render(<DebateThisButton {...defaultProps} variant="icon" />);
      expect(screen.getByLabelText('Debate This')).toBeInTheDocument();
    });

    it('opens modal on click', () => {
      render(<DebateThisButton {...defaultProps} variant="icon" />);
      fireEvent.click(screen.getByLabelText('Debate This'));
      expect(screen.getByTestId('debate-modal')).toBeInTheDocument();
    });
  });

  describe('variant="inline"', () => {
    it('renders inline text link', () => {
      render(<DebateThisButton {...defaultProps} variant="inline" />);
      expect(screen.getByText('Debate This')).toBeInTheDocument();
    });

    it('opens modal on click', () => {
      render(<DebateThisButton {...defaultProps} variant="inline" />);
      fireEvent.click(screen.getByText('Debate This'));
      expect(screen.getByTestId('debate-modal')).toBeInTheDocument();
    });
  });

  describe('className prop', () => {
    it('applies custom className to button variant', () => {
      render(<DebateThisButton {...defaultProps} className="custom-class" />);
      expect(screen.getByText('DEBATE THIS').className).toContain('custom-class');
    });

    it('applies custom className to icon variant', () => {
      render(<DebateThisButton {...defaultProps} variant="icon" className="custom-class" />);
      expect(screen.getByLabelText('Debate This').className).toContain('custom-class');
    });
  });
});
