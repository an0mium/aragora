/**
 * Tests for standalone ActionNode component.
 */

import { render, screen } from '@testing-library/react';
import { ReactFlowProvider } from '@xyflow/react';
import { ActionNode } from '../ActionNode';

const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <ReactFlowProvider>{children}</ReactFlowProvider>
);

describe('ActionNode', () => {
  const baseData = {
    label: 'Deploy monitoring',
    actionType: 'task',
    description: 'Set up Prometheus dashboards',
    status: 'pending',
  };

  it('renders label', () => {
    render(<ActionNode data={baseData} />, { wrapper: Wrapper });
    expect(screen.getByText('Deploy monitoring')).toBeInTheDocument();
  });

  it('renders action type badge', () => {
    render(<ActionNode data={baseData} />, { wrapper: Wrapper });
    expect(screen.getByText('Task')).toBeInTheDocument();
  });

  it('renders status badge', () => {
    render(<ActionNode data={baseData} />, { wrapper: Wrapper });
    expect(screen.getByText('pending')).toBeInTheDocument();
  });

  it('renders description', () => {
    render(<ActionNode data={baseData} />, { wrapper: Wrapper });
    expect(screen.getByText('Set up Prometheus dashboards')).toBeInTheDocument();
  });

  it('renders optional badge when optional', () => {
    render(<ActionNode data={{ ...baseData, optional: true }} />, { wrapper: Wrapper });
    expect(screen.getByText('optional')).toBeInTheDocument();
  });

  it('does not render optional badge when not optional', () => {
    render(<ActionNode data={baseData} />, { wrapper: Wrapper });
    expect(screen.queryByText('optional')).not.toBeInTheDocument();
  });

  it('renders timeout when provided', () => {
    render(<ActionNode data={{ ...baseData, timeoutSeconds: 3600 }} />, { wrapper: Wrapper });
    expect(screen.getByText('timeout: 3600s')).toBeInTheDocument();
  });

  it('renders assignee when provided', () => {
    render(<ActionNode data={{ ...baseData, assignee: 'alice' }} />, { wrapper: Wrapper });
    expect(screen.getByText('assigned: alice')).toBeInTheDocument();
  });

  it('applies selection ring when selected', () => {
    const { container } = render(<ActionNode data={baseData} selected />, { wrapper: Wrapper });
    const root = container.firstChild as HTMLElement;
    expect(root.className).toContain('ring-acid-green');
  });

  it('renders in_progress status correctly', () => {
    render(<ActionNode data={{ ...baseData, status: 'in_progress' }} />, { wrapper: Wrapper });
    expect(screen.getByText('in progress')).toBeInTheDocument();
  });

  it('renders locked indicator', () => {
    render(<ActionNode data={{ ...baseData, lockedBy: 'bob' }} />, { wrapper: Wrapper });
    expect(screen.getByText('Locked by bob')).toBeInTheDocument();
  });

  it('supports step_type field for backward compat', () => {
    render(<ActionNode data={{ label: 'Test', step_type: 'checkpoint' }} />, { wrapper: Wrapper });
    expect(screen.getByText('Checkpoint')).toBeInTheDocument();
  });
});
