import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { ProvenanceTrail } from '../ProvenanceTrail';
import type { ProvenanceLink, PipelineStageType } from '../types';

const mockLinks: ProvenanceLink[] = [
  {
    source_node_id: 'idea-1',
    source_stage: 'ideas',
    target_node_id: 'goal-1',
    target_stage: 'goals',
    content_hash: 'abc123def456',
    timestamp: 1700000000,
    method: 'goal_extraction',
  },
  {
    source_node_id: 'goal-1',
    source_stage: 'goals',
    target_node_id: 'action-1',
    target_stage: 'actions',
    content_hash: 'xyz789abc123',
    timestamp: 1700000100,
    method: 'action_decomposition',
  },
];

const mockLookup: Record<string, { label: string; stage: PipelineStageType }> = {
  'idea-1': { label: 'Core Idea', stage: 'ideas' },
  'goal-1': { label: 'Primary Goal', stage: 'goals' },
  'action-1': { label: 'First Action', stage: 'actions' },
};

describe('ProvenanceTrail', () => {
  it('renders breadcrumbs for a node with provenance', () => {
    render(
      <ProvenanceTrail
        selectedNodeId="action-1"
        selectedStage="actions"
        selectedLabel="First Action"
        provenance={mockLinks}
        nodeLookup={mockLookup}
      />,
    );

    expect(screen.getByTestId('provenance-trail')).toBeInTheDocument();
    expect(screen.getByTestId('provenance-crumb-ideas')).toBeInTheDocument();
    expect(screen.getByTestId('provenance-crumb-goals')).toBeInTheDocument();
    expect(screen.getByTestId('provenance-crumb-actions')).toBeInTheDocument();
  });

  it('renders a single node when no provenance exists', () => {
    render(
      <ProvenanceTrail
        selectedNodeId="idea-1"
        selectedStage="ideas"
        selectedLabel="Standalone Idea"
        provenance={[]}
        nodeLookup={mockLookup}
      />,
    );

    expect(screen.getByTestId('provenance-crumb-ideas')).toBeInTheDocument();
    // No goals or actions breadcrumbs when there's no provenance
    expect(screen.queryByTestId('provenance-crumb-goals')).not.toBeInTheDocument();
  });

  it('calls onNavigate when a breadcrumb is clicked', () => {
    const onNavigate = jest.fn();
    render(
      <ProvenanceTrail
        selectedNodeId="action-1"
        selectedStage="actions"
        selectedLabel="First Action"
        provenance={mockLinks}
        nodeLookup={mockLookup}
        onNavigate={onNavigate}
      />,
    );

    fireEvent.click(screen.getByTestId('provenance-crumb-ideas'));
    expect(onNavigate).toHaveBeenCalledWith('idea-1', 'ideas');
  });

  it('shows content hash metadata for multi-step trails', () => {
    render(
      <ProvenanceTrail
        selectedNodeId="action-1"
        selectedStage="actions"
        selectedLabel="First Action"
        provenance={mockLinks}
        nodeLookup={mockLookup}
      />,
    );

    // Should show truncated hash
    expect(screen.getByText(/xyz789ab/)).toBeInTheDocument();
    // Should show depth
    expect(screen.getByText('3')).toBeInTheDocument();
  });

  it('highlights the active breadcrumb with a ring', () => {
    render(
      <ProvenanceTrail
        selectedNodeId="goal-1"
        selectedStage="goals"
        selectedLabel="Primary Goal"
        provenance={mockLinks}
        nodeLookup={mockLookup}
      />,
    );

    const goalCrumb = screen.getByTestId('provenance-crumb-goals');
    expect(goalCrumb.className).toContain('ring-2');
  });

  it('sorts breadcrumbs in stage order (ideas → goals → actions → orch)', () => {
    render(
      <ProvenanceTrail
        selectedNodeId="action-1"
        selectedStage="actions"
        selectedLabel="First Action"
        provenance={mockLinks}
        nodeLookup={mockLookup}
      />,
    );

    const crumbs = screen.getAllByRole('button');
    expect(crumbs).toHaveLength(3);
    // Ideas first, then goals, then actions
    expect(crumbs[0]).toHaveAttribute('data-testid', 'provenance-crumb-ideas');
    expect(crumbs[1]).toHaveAttribute('data-testid', 'provenance-crumb-goals');
    expect(crumbs[2]).toHaveAttribute('data-testid', 'provenance-crumb-actions');
  });
});
