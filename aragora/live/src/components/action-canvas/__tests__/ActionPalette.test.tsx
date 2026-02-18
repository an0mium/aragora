/**
 * Tests for ActionPalette component.
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { ActionPalette } from '../ActionPalette';

describe('ActionPalette', () => {
  it('renders all action type labels', () => {
    render(<ActionPalette />);

    expect(screen.getByText('Task')).toBeInTheDocument();
    expect(screen.getByText('Epic')).toBeInTheDocument();
    expect(screen.getByText('Checkpoint')).toBeInTheDocument();
    expect(screen.getByText('Deliverable')).toBeInTheDocument();
    expect(screen.getByText('Dependency')).toBeInTheDocument();
  });

  it('renders group labels', () => {
    render(<ActionPalette />);

    expect(screen.getByText('Execution')).toBeInTheDocument();
    expect(screen.getByText('Verification')).toBeInTheDocument();
    expect(screen.getByText('Management')).toBeInTheDocument();
  });

  it('renders exactly 5 draggable items', () => {
    render(<ActionPalette />);

    const items = screen.getAllByText(/^(Task|Epic|Checkpoint|Deliverable|Dependency)$/);
    expect(items).toHaveLength(5);
  });

  it('sets drag data on dragStart', () => {
    render(<ActionPalette />);

    const taskItem = screen.getByText('Task').closest('[draggable]') as HTMLElement;
    expect(taskItem).toBeTruthy();

    const setData = jest.fn();
    const dataTransfer = { setData, effectAllowed: '' };

    fireEvent.dragStart(taskItem, { dataTransfer });

    expect(setData).toHaveBeenCalledWith('application/action-node-type', 'task');
  });

  it('sets drag data for epic', () => {
    render(<ActionPalette />);

    const epicItem = screen.getByText('Epic').closest('[draggable]') as HTMLElement;

    const setData = jest.fn();
    const dataTransfer = { setData, effectAllowed: '' };

    fireEvent.dragStart(epicItem, { dataTransfer });

    expect(setData).toHaveBeenCalledWith('application/action-node-type', 'epic');
  });

  it('shows header text', () => {
    render(<ActionPalette />);
    expect(screen.getByText('Action Types')).toBeInTheDocument();
  });
});
