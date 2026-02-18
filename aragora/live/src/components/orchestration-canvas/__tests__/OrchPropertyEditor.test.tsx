/**
 * Tests for OrchPropertyEditor component.
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { OrchPropertyEditor } from '../OrchPropertyEditor';
import type { OrchNodeData } from '../types';

describe('OrchPropertyEditor', () => {
  const baseData: OrchNodeData = {
    orchType: 'agent_task',
    label: 'Research Agent',
    description: 'Investigate competitors',
    assignedAgent: 'claude',
    agentType: 'researcher',
    capabilities: ['analysis', 'synthesis'],
    status: 'pending',
    stage: 'orchestration',
    rfType: 'orchestrationNode',
  };

  it('shows empty state when no data', () => {
    render(<OrchPropertyEditor data={null} onChange={jest.fn()} />);
    expect(screen.getByText(/Select an orchestration node/)).toBeInTheDocument();
  });

  it('renders header', () => {
    render(<OrchPropertyEditor data={baseData} onChange={jest.fn()} />);
    expect(screen.getByText('Orchestration Properties')).toBeInTheDocument();
  });

  it('shows title field with current value', () => {
    render(<OrchPropertyEditor data={baseData} onChange={jest.fn()} />);
    expect(screen.getByDisplayValue('Research Agent')).toBeInTheDocument();
  });

  it('shows description field', () => {
    render(<OrchPropertyEditor data={baseData} onChange={jest.fn()} />);
    expect(screen.getByDisplayValue('Investigate competitors')).toBeInTheDocument();
  });

  it('shows assigned agent field', () => {
    render(<OrchPropertyEditor data={baseData} onChange={jest.fn()} />);
    expect(screen.getByDisplayValue('claude')).toBeInTheDocument();
  });

  it('shows agent type field', () => {
    render(<OrchPropertyEditor data={baseData} onChange={jest.fn()} />);
    expect(screen.getByDisplayValue('researcher')).toBeInTheDocument();
  });

  it('shows capabilities', () => {
    render(<OrchPropertyEditor data={baseData} onChange={jest.fn()} />);
    expect(screen.getByDisplayValue('analysis, synthesis')).toBeInTheDocument();
  });

  it('calls onChange when title changes', () => {
    const onChange = jest.fn();
    render(<OrchPropertyEditor data={baseData} onChange={onChange} />);
    const input = screen.getByDisplayValue('Research Agent');
    fireEvent.change(input, { target: { value: 'Analysis Agent' } });
    expect(onChange).toHaveBeenCalledWith({ label: 'Analysis Agent' });
  });

  it('calls onChange when status changes', () => {
    const onChange = jest.fn();
    render(<OrchPropertyEditor data={baseData} onChange={onChange} />);
    const select = screen.getAllByRole('combobox')[1]; // second select is status
    fireEvent.change(select, { target: { value: 'running' } });
    expect(onChange).toHaveBeenCalledWith({ status: 'running' });
  });

  it('calls onChange when capabilities change', () => {
    const onChange = jest.fn();
    render(<OrchPropertyEditor data={baseData} onChange={onChange} />);
    const input = screen.getByDisplayValue('analysis, synthesis');
    fireEvent.change(input, { target: { value: 'reasoning, code' } });
    expect(onChange).toHaveBeenCalledWith({ capabilities: ['reasoning', 'code'] });
  });

  it('renders execute button when onExecute provided', () => {
    render(<OrchPropertyEditor data={baseData} onChange={jest.fn()} onExecute={jest.fn()} />);
    expect(screen.getByText('Execute Pipeline')).toBeInTheDocument();
  });

  it('does not render execute button when onExecute not provided', () => {
    render(<OrchPropertyEditor data={baseData} onChange={jest.fn()} />);
    expect(screen.queryByText('Execute Pipeline')).not.toBeInTheDocument();
  });

  it('calls onExecute when execute button clicked', () => {
    const onExecute = jest.fn();
    render(<OrchPropertyEditor data={baseData} onChange={jest.fn()} onExecute={onExecute} />);
    fireEvent.click(screen.getByText('Execute Pipeline'));
    expect(onExecute).toHaveBeenCalledTimes(1);
  });

  it('renders delete button when onDelete provided', () => {
    render(<OrchPropertyEditor data={baseData} onChange={jest.fn()} onDelete={jest.fn()} />);
    expect(screen.getByText('Delete Node')).toBeInTheDocument();
  });

  it('calls onDelete when delete button clicked', () => {
    const onDelete = jest.fn();
    render(<OrchPropertyEditor data={baseData} onChange={jest.fn()} onDelete={onDelete} />);
    fireEvent.click(screen.getByText('Delete Node'));
    expect(onDelete).toHaveBeenCalledTimes(1);
  });

  it('shows source action count', () => {
    const data = { ...baseData, sourceActionIds: ['a1', 'a2'] };
    render(<OrchPropertyEditor data={data} onChange={jest.fn()} />);
    expect(screen.getByText('Derived from 2 action(s)')).toBeInTheDocument();
  });
});
