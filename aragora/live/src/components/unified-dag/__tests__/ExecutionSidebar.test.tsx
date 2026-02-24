import { render, screen, fireEvent } from '@testing-library/react';
import { ExecutionSidebar } from '../ExecutionSidebar';
import type { Node } from '@xyflow/react';
import type { DAGNodeData } from '@/hooks/useUnifiedDAG';

function makeNode(
  id: string,
  stage: string,
  status: string,
  label = `Node ${id}`,
): Node<DAGNodeData> {
  return {
    id,
    type: `${stage}Node`,
    position: { x: 0, y: 0 },
    data: {
      label,
      description: '',
      stage: stage as DAGNodeData['stage'],
      subtype: '',
      status,
      priority: 0,
      metadata: {},
    },
  };
}

const defaultProps = {
  nodes: [
    makeNode('1', 'ideas', 'succeeded'),
    makeNode('2', 'goals', 'ready'),
    makeNode('3', 'actions', 'pending'),
    makeNode('4', 'orchestration', 'failed'),
  ] as Node<DAGNodeData>[],
  executing: false,
  onExecuteAll: jest.fn(),
  onAutoAdvance: jest.fn(),
  onValidate: jest.fn(),
  validationErrors: [],
  executionHistory: [],
  onClose: jest.fn(),
};

describe('ExecutionSidebar', () => {
  it('renders the sidebar heading', () => {
    render(<ExecutionSidebar {...defaultProps} />);
    expect(screen.getByText('Execution')).toBeInTheDocument();
  });

  it('shows overall progress percentage', () => {
    render(<ExecutionSidebar {...defaultProps} />);
    // 1 out of 4 succeeded = 25%
    expect(screen.getByText('25%')).toBeInTheDocument();
    expect(screen.getByText('1/4 nodes complete')).toBeInTheDocument();
  });

  it('shows ready count', () => {
    render(<ExecutionSidebar {...defaultProps} />);
    // "1 ready" appears in both overall progress and stage breakdown
    expect(screen.getAllByText('1 ready').length).toBeGreaterThan(0);
  });

  it('renders stage progress sections', () => {
    render(<ExecutionSidebar {...defaultProps} />);
    expect(screen.getByText('ideas')).toBeInTheDocument();
    expect(screen.getByText('goals')).toBeInTheDocument();
    expect(screen.getByText('actions')).toBeInTheDocument();
    expect(screen.getByText('orchestration')).toBeInTheDocument();
  });

  it('shows validation success when no errors', () => {
    render(<ExecutionSidebar {...defaultProps} />);
    expect(screen.getByText('Graph is valid and executable')).toBeInTheDocument();
  });

  it('shows validation errors when present', () => {
    render(
      <ExecutionSidebar
        {...defaultProps}
        validationErrors={['No idea nodes', 'Orphan node detected']}
      />,
    );
    expect(screen.getByText('No idea nodes')).toBeInTheDocument();
    expect(screen.getByText('Orphan node detected')).toBeInTheDocument();
  });

  it('calls onExecuteAll when button clicked', () => {
    const onExecuteAll = jest.fn();
    render(<ExecutionSidebar {...defaultProps} onExecuteAll={onExecuteAll} />);
    fireEvent.click(screen.getByTestId('execute-all-btn'));
    expect(onExecuteAll).toHaveBeenCalledTimes(1);
  });

  it('calls onAutoAdvance when button clicked', () => {
    const onAutoAdvance = jest.fn();
    render(<ExecutionSidebar {...defaultProps} onAutoAdvance={onAutoAdvance} />);
    fireEvent.click(screen.getByTestId('auto-advance-btn'));
    expect(onAutoAdvance).toHaveBeenCalledTimes(1);
  });

  it('disables execute button when no ready nodes', () => {
    const nodes = [
      makeNode('1', 'ideas', 'succeeded'),
      makeNode('2', 'goals', 'pending'),
    ] as Node<DAGNodeData>[];
    render(<ExecutionSidebar {...defaultProps} nodes={nodes} />);
    expect(screen.getByTestId('execute-all-btn')).toBeDisabled();
  });

  it('disables buttons when executing', () => {
    render(<ExecutionSidebar {...defaultProps} executing={true} />);
    expect(screen.getByTestId('execute-all-btn')).toBeDisabled();
    expect(screen.getByTestId('auto-advance-btn')).toBeDisabled();
  });

  it('calls onClose when close button clicked', () => {
    const onClose = jest.fn();
    render(<ExecutionSidebar {...defaultProps} onClose={onClose} />);
    fireEvent.click(screen.getByTitle('Close'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('renders execution history entries', () => {
    render(
      <ExecutionSidebar
        {...defaultProps}
        executionHistory={[
          {
            id: 'h1',
            nodeId: '1',
            nodeLabel: 'Build rate limiter',
            status: 'succeeded',
            durationMs: 1500,
            timestamp: Date.now(),
          },
          {
            id: 'h2',
            nodeId: '2',
            nodeLabel: 'Set up cache',
            status: 'failed',
            durationMs: 45000,
            timestamp: Date.now(),
          },
        ]}
      />,
    );
    expect(screen.getByText('Build rate limiter')).toBeInTheDocument();
    expect(screen.getByText('Set up cache')).toBeInTheDocument();
    expect(screen.getByText('1.5s')).toBeInTheDocument();
    expect(screen.getByText('45.0s')).toBeInTheDocument();
  });

  it('renders progress bar element', () => {
    render(<ExecutionSidebar {...defaultProps} />);
    const bar = screen.getByTestId('progress-bar');
    expect(bar).toBeInTheDocument();
    expect(bar.style.width).toBe('25%');
  });
});
