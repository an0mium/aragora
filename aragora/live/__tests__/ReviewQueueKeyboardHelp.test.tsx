/**
 * Tests for the KeyboardHelp overlay.
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { KeyboardHelp } from '../src/components/review-queue/KeyboardHelp';

describe('KeyboardHelp', () => {
  it('renders nothing when closed', () => {
    const { container } = render(<KeyboardHelp open={false} onClose={jest.fn()} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders shortcut table when open', () => {
    render(<KeyboardHelp open onClose={jest.fn()} />);
    expect(screen.getByTestId('review-queue-keyboard-help')).toBeInTheDocument();
    expect(screen.getByText(/j \/ k/)).toBeInTheDocument();
    expect(screen.getByText(/Approve the selected PR/)).toBeInTheDocument();
  });

  it('calls onClose on Escape', () => {
    const onClose = jest.fn();
    render(<KeyboardHelp open onClose={onClose} />);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalled();
  });

  it('calls onClose on backdrop click', () => {
    const onClose = jest.fn();
    render(<KeyboardHelp open onClose={onClose} />);
    fireEvent.click(screen.getByTestId('review-queue-keyboard-help'));
    expect(onClose).toHaveBeenCalled();
  });
});
