/**
 * Smoke tests for pipeline canvas node components.
 *
 * Verifies each node type renders without errors and displays
 * its core data fields.
 */

import { render, screen } from '@testing-library/react';
import { ReactFlowProvider } from '@xyflow/react';
import { IdeaNode } from '../nodes/IdeaNode';
import { GoalNode } from '../nodes/GoalNode';
import { ActionNode } from '../nodes/ActionNode';
import { OrchestrationNode } from '../nodes/OrchestrationNode';

// ReactFlow nodes need to be wrapped in ReactFlowProvider to access internals
const Wrapper = ({ children }: { children: React.ReactNode }) => (
  <ReactFlowProvider>{children}</ReactFlowProvider>
);

describe('IdeaNode', () => {
  const data = {
    label: 'Use microservices',
    idea_type: 'concept',
    agent: 'claude',
    content_hash: 'abc12345def67890',
  };

  it('renders label', () => {
    render(<IdeaNode data={data} />, { wrapper: Wrapper });
    expect(screen.getByText('Use microservices')).toBeInTheDocument();
  });

  it('renders idea type badge', () => {
    render(<IdeaNode data={data} />, { wrapper: Wrapper });
    expect(screen.getByText('Concept')).toBeInTheDocument();
  });

  it('renders agent name', () => {
    render(<IdeaNode data={data} />, { wrapper: Wrapper });
    expect(screen.getByText('claude')).toBeInTheDocument();
  });

  it('renders content hash prefix', () => {
    render(<IdeaNode data={data} />, { wrapper: Wrapper });
    expect(screen.getByText('#abc12345')).toBeInTheDocument();
  });

  it('applies selection ring when selected', () => {
    const { container } = render(<IdeaNode data={data} selected />, { wrapper: Wrapper });
    const root = container.firstChild as HTMLElement;
    expect(root.className).toContain('ring-acid-green');
  });
});

describe('GoalNode', () => {
  const data = {
    label: 'Improve API reliability',
    goal_type: 'goal',
    description: 'Maintain P99 < 200ms',
    priority: 'high',
  };

  it('renders label', () => {
    render(<GoalNode data={data} />, { wrapper: Wrapper });
    expect(screen.getByText('Improve API reliability')).toBeInTheDocument();
  });

  it('renders goal type badge', () => {
    render(<GoalNode data={data} />, { wrapper: Wrapper });
    expect(screen.getByText('Goal')).toBeInTheDocument();
  });

  it('renders priority badge', () => {
    render(<GoalNode data={data} />, { wrapper: Wrapper });
    expect(screen.getByText('high')).toBeInTheDocument();
  });

  it('renders description', () => {
    render(<GoalNode data={data} />, { wrapper: Wrapper });
    expect(screen.getByText('Maintain P99 < 200ms')).toBeInTheDocument();
  });

  it('renders confidence bar when provided', () => {
    render(<GoalNode data={{ ...data, confidence: 0.85 }} />, { wrapper: Wrapper });
    expect(screen.getByText('85%')).toBeInTheDocument();
  });
});

describe('ActionNode', () => {
  const data = {
    label: 'Deploy monitoring',
    step_type: 'checkpoint',
    description: 'Set up Prometheus dashboards',
    optional: true,
    timeout: 3600,
  };

  it('renders label', () => {
    render(<ActionNode data={data} />, { wrapper: Wrapper });
    expect(screen.getByText('Deploy monitoring')).toBeInTheDocument();
  });

  it('renders step type badge', () => {
    render(<ActionNode data={data} />, { wrapper: Wrapper });
    expect(screen.getByText('Checkpoint')).toBeInTheDocument();
  });

  it('renders optional badge', () => {
    render(<ActionNode data={data} />, { wrapper: Wrapper });
    expect(screen.getByText('optional')).toBeInTheDocument();
  });

  it('renders timeout', () => {
    render(<ActionNode data={data} />, { wrapper: Wrapper });
    expect(screen.getByText('timeout: 3600s')).toBeInTheDocument();
  });
});

describe('OrchestrationNode', () => {
  const data = {
    label: 'Analyst Agent',
    orch_type: 'agent_task',
    agent_type: 'claude',
    capabilities: ['research', 'analysis', 'synthesis'],
  };

  it('renders label', () => {
    render(<OrchestrationNode data={data} />, { wrapper: Wrapper });
    expect(screen.getByText('Analyst Agent')).toBeInTheDocument();
  });

  it('renders orch type badge', () => {
    render(<OrchestrationNode data={data} />, { wrapper: Wrapper });
    expect(screen.getByText('Agent Task')).toBeInTheDocument();
  });

  it('renders agent type', () => {
    render(<OrchestrationNode data={data} />, { wrapper: Wrapper });
    expect(screen.getByText('claude')).toBeInTheDocument();
  });

  it('renders capabilities (max 3)', () => {
    render(<OrchestrationNode data={data} />, { wrapper: Wrapper });
    expect(screen.getByText('research')).toBeInTheDocument();
    expect(screen.getByText('analysis')).toBeInTheDocument();
    expect(screen.getByText('synthesis')).toBeInTheDocument();
  });

  it('uses rounded-full for agent type', () => {
    const { container } = render(<OrchestrationNode data={data} />, { wrapper: Wrapper });
    const root = container.firstChild as HTMLElement;
    expect(root.className).toContain('rounded-full');
  });

  it('uses border-dashed for human gate', () => {
    const { container } = render(
      <OrchestrationNode data={{ ...data, orch_type: 'human_gate' }} />,
      { wrapper: Wrapper },
    );
    const root = container.firstChild as HTMLElement;
    expect(root.className).toContain('border-dashed');
  });
});
