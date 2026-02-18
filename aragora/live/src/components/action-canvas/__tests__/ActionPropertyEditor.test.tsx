/**
 * Tests for ActionPropertyEditor component.
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { ActionPropertyEditor } from '../ActionPropertyEditor';
import type { ActionNodeData } from '../types';

describe('ActionPropertyEditor', () => {
  const baseData: ActionNodeData = {
    actionType: 'task',
    label: 'Write tests',
    description: 'Add unit tests for module',
    status: 'pending',
    assignee: 'alice',
    optional: false,
    timeoutSeconds: 300,
    tags: ['testing', 'quality'],
    stage: 'actions',
    rfType: 'actionNode',
  };

  it('shows empty state when no data', () => {
    render(<ActionPropertyEditor data={null} onChange={jest.fn()} />);
    expect(screen.getByText(/Select an action node/)).toBeInTheDocument();
  });

  it('renders header', () => {
    render(<ActionPropertyEditor data={baseData} onChange={jest.fn()} />);
    expect(screen.getByText('Action Properties')).toBeInTheDocument();
  });

  it('shows title field with current value', () => {
    render(<ActionPropertyEditor data={baseData} onChange={jest.fn()} />);
    const input = screen.getByDisplayValue('Write tests');
    expect(input).toBeInTheDocument();
  });

  it('shows description field', () => {
    render(<ActionPropertyEditor data={baseData} onChange={jest.fn()} />);
    const textarea = screen.getByDisplayValue('Add unit tests for module');
    expect(textarea).toBeInTheDocument();
  });

  it('shows assignee field', () => {
    render(<ActionPropertyEditor data={baseData} onChange={jest.fn()} />);
    const input = screen.getByDisplayValue('alice');
    expect(input).toBeInTheDocument();
  });

  it('shows tags', () => {
    render(<ActionPropertyEditor data={baseData} onChange={jest.fn()} />);
    const input = screen.getByDisplayValue('testing, quality');
    expect(input).toBeInTheDocument();
  });

  it('calls onChange when title changes', () => {
    const onChange = jest.fn();
    render(<ActionPropertyEditor data={baseData} onChange={onChange} />);
    const input = screen.getByDisplayValue('Write tests');
    fireEvent.change(input, { target: { value: 'Write integration tests' } });
    expect(onChange).toHaveBeenCalledWith({ label: 'Write integration tests' });
  });

  it('calls onChange when status changes', () => {
    const onChange = jest.fn();
    render(<ActionPropertyEditor data={baseData} onChange={onChange} />);
    const select = screen.getAllByRole('combobox')[1]; // second select is status
    fireEvent.change(select, { target: { value: 'completed' } });
    expect(onChange).toHaveBeenCalledWith({ status: 'completed' });
  });

  it('renders advance button when onAdvance provided', () => {
    render(<ActionPropertyEditor data={baseData} onChange={jest.fn()} onAdvance={jest.fn()} />);
    expect(screen.getByText('Advance to Orchestration')).toBeInTheDocument();
  });

  it('does not render advance button when onAdvance not provided', () => {
    render(<ActionPropertyEditor data={baseData} onChange={jest.fn()} />);
    expect(screen.queryByText('Advance to Orchestration')).not.toBeInTheDocument();
  });

  it('calls onAdvance when advance button clicked', () => {
    const onAdvance = jest.fn();
    render(<ActionPropertyEditor data={baseData} onChange={jest.fn()} onAdvance={onAdvance} />);
    fireEvent.click(screen.getByText('Advance to Orchestration'));
    expect(onAdvance).toHaveBeenCalledTimes(1);
  });

  it('renders delete button when onDelete provided', () => {
    render(<ActionPropertyEditor data={baseData} onChange={jest.fn()} onDelete={jest.fn()} />);
    expect(screen.getByText('Delete Action')).toBeInTheDocument();
  });

  it('calls onDelete when delete button clicked', () => {
    const onDelete = jest.fn();
    render(<ActionPropertyEditor data={baseData} onChange={jest.fn()} onDelete={onDelete} />);
    fireEvent.click(screen.getByText('Delete Action'));
    expect(onDelete).toHaveBeenCalledTimes(1);
  });

  it('shows source goal count', () => {
    const data = { ...baseData, sourceGoalIds: ['g1', 'g2', 'g3'] };
    render(<ActionPropertyEditor data={data} onChange={jest.fn()} />);
    expect(screen.getByText('Derived from 3 goal(s)')).toBeInTheDocument();
  });

  it('toggles optional checkbox', () => {
    const onChange = jest.fn();
    render(<ActionPropertyEditor data={baseData} onChange={onChange} />);
    const checkbox = screen.getByRole('checkbox');
    fireEvent.click(checkbox);
    expect(onChange).toHaveBeenCalledWith({ optional: true });
  });
});
