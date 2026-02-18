/**
 * Tests for UnifiedPipelineCanvas component.
 *
 * Covers: rendering, all 4 node types, semantic zoom, stage filter toggles,
 * AI transition buttons, provenance sidebar, and cross-stage edges.
 */

import { render, screen, fireEvent, act } from '@testing-library/react';
import type { PipelineStageType } from '../types';

// ---------------------------------------------------------------------------
// Track onViewportChange callback so tests can simulate zoom
// ---------------------------------------------------------------------------

let capturedOnViewportChange: ((viewport: { zoom: number; x: number; y: number }) => void) | null = null;
let capturedOnNodeClick: ((event: React.MouseEvent, node: any) => void) | null = null;
let capturedOnPaneClick: (() => void) | null = null;

jest.mock('@xyflow/react', () => ({
  ReactFlow: ({ children, onViewportChange, onNodeClick, onPaneClick, nodes, edges, ...rest }: any) => {
    capturedOnViewportChange = onViewportChange || null;
    capturedOnNodeClick = onNodeClick || null;
    capturedOnPaneClick = onPaneClick || null;
    return (
      <div data-testid="react-flow">
        {children}
        {nodes?.map((n: any) => (
          <div
            key={n.id}
            data-testid={`node-${n.id}`}
            data-node-type={n.type}
            onClick={(e) => onNodeClick?.(e, n)}
          >
            {(n.data as Record<string, unknown>)?.label as string}
          </div>
        ))}
        {edges?.map((e: any) => (
          <div key={e.id} data-testid={`edge-${e.id}`} data-edge-style={JSON.stringify(e.style)} />
        ))}
      </div>
    );
  },
  ReactFlowProvider: ({ children }: any) => <div>{children}</div>,
  Controls: () => <div data-testid="controls" />,
  Background: () => <div data-testid="background" />,
  MiniMap: () => <div data-testid="minimap" />,
  Panel: ({ children, position }: any) => <div data-testid={`panel-${position}`}>{children}</div>,
  BackgroundVariant: { Dots: 'dots' },
  useNodesState: (initial: any[]) => {
    const [nodes, setNodes] = require('react').useState(initial);
    return [nodes, setNodes, jest.fn()];
  },
  useEdgesState: (initial: any[]) => {
    const [edges, setEdges] = require('react').useState(initial);
    return [edges, setEdges, jest.fn()];
  },
  useReactFlow: () => ({
    fitView: jest.fn(),
    screenToFlowPosition: ({ x, y }: { x: number; y: number }) => ({ x, y }),
  }),
  addEdge: jest.fn((connection: any, edges: any[]) => [...edges, { id: 'new-edge', ...connection }]),
}));

// ---------------------------------------------------------------------------
// Mock node components
// ---------------------------------------------------------------------------

jest.mock('../nodes', () => ({
  IdeaNode: () => <div />,
  GoalNode: () => <div />,
  ActionNode: () => <div />,
  OrchestrationNode: () => <div />,
}));

// ---------------------------------------------------------------------------
// Mock the usePipelineCanvas hook
// ---------------------------------------------------------------------------

import { usePipelineCanvas } from '../../../hooks/usePipelineCanvas';
jest.mock('../../../hooks/usePipelineCanvas', () => ({
  usePipelineCanvas: jest.fn(),
}));

const mockedUsePipelineCanvas = usePipelineCanvas as jest.MockedFunction<typeof usePipelineCanvas>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeIdeaNode(id: string, label: string) {
  return {
    id,
    type: 'ideaNode',
    position: { x: 0, y: 0 },
    data: { label, ideaType: 'concept', contentHash: 'abc123def456' },
  };
}

function makeGoalNode(id: string, label: string) {
  return {
    id,
    type: 'goalNode',
    position: { x: 0, y: 50 },
    data: { label, goalType: 'goal', priority: 'high', description: 'A goal' },
  };
}

function makeActionNode(id: string, label: string) {
  return {
    id,
    type: 'actionNode',
    position: { x: 0, y: 100 },
    data: { label, stepType: 'task', status: 'pending' },
  };
}

function makeOrchNode(id: string, label: string) {
  return {
    id,
    type: 'orchestrationNode',
    position: { x: 0, y: 150 },
    data: { label, orchType: 'agent_task', status: 'pending' },
  };
}

function makeEdge(id: string, source: string, target: string) {
  return { id, source, target, type: 'default', animated: true, style: {} };
}

function makeMockCanvas(overrides: Record<string, unknown> = {}) {
  return {
    nodes: [] as any[],
    edges: [] as any[],
    onNodesChange: jest.fn(),
    onEdgesChange: jest.fn(),
    onConnect: jest.fn(),
    selectedNodeId: null as string | null,
    setSelectedNodeId: jest.fn(),
    selectedNodeData: null as Record<string, unknown> | null,
    updateSelectedNode: jest.fn(),
    deleteSelectedNode: jest.fn(),
    addNode: jest.fn(),
    activeStage: 'ideas' as PipelineStageType,
    setActiveStage: jest.fn(),
    stageStatus: {
      ideas: 'pending',
      goals: 'pending',
      actions: 'pending',
      orchestration: 'pending',
    },
    stageNodes: {
      ideas: [] as any[],
      goals: [] as any[],
      actions: [] as any[],
      orchestration: [] as any[],
    },
    stageEdges: {
      ideas: [] as any[],
      goals: [] as any[],
      actions: [] as any[],
      orchestration: [] as any[],
    },
    savePipeline: jest.fn(),
    aiGenerate: jest.fn(),
    clearStage: jest.fn(),
    populateFromResult: jest.fn(),
    loading: false,
    error: null,
    onDrop: jest.fn(),
    onDragOver: jest.fn(),
    ...overrides,
  } as any;
}

// ---------------------------------------------------------------------------
// Import after mocks
// ---------------------------------------------------------------------------

import { UnifiedPipelineCanvas } from '../UnifiedPipelineCanvas';

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('UnifiedPipelineCanvas', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    capturedOnViewportChange = null;
    capturedOnNodeClick = null;
    capturedOnPaneClick = null;
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('renders without crashing', () => {
    mockedUsePipelineCanvas.mockReturnValue(makeMockCanvas());
    render(<UnifiedPipelineCanvas />);
    expect(screen.getByTestId('unified-pipeline-canvas')).toBeInTheDocument();
  });

  it('shows all 4 node types when stages have nodes', () => {
    const ideaNode = makeIdeaNode('idea-1', 'My Idea');
    const goalNode = makeGoalNode('goal-1', 'My Goal');
    const actionNode = makeActionNode('action-1', 'My Action');
    const orchNode = makeOrchNode('orch-1', 'My Agent');

    mockedUsePipelineCanvas.mockReturnValue(
      makeMockCanvas({
        stageNodes: {
          ideas: [ideaNode],
          goals: [goalNode],
          actions: [actionNode],
          orchestration: [orchNode],
        },
      }),
    );

    render(<UnifiedPipelineCanvas />);

    // At default zoom (1.0), only ideas + goals are visible (zoom < 0.8 threshold is not met,
    // but zoom is between 0.8 and 1.5, so ideas + goals + actions visible)
    expect(screen.getByTestId('node-idea-1')).toBeInTheDocument();
    expect(screen.getByTestId('node-goal-1')).toBeInTheDocument();
    expect(screen.getByTestId('node-action-1')).toBeInTheDocument();
    // Orchestration is hidden at zoom 1.0 (need > 1.5)
    expect(screen.queryByTestId('node-orch-1')).not.toBeInTheDocument();
  });

  it('semantic zoom: shows all stages when zoom > 1.5', () => {
    const orchNode = makeOrchNode('orch-1', 'My Agent');

    mockedUsePipelineCanvas.mockReturnValue(
      makeMockCanvas({
        stageNodes: {
          ideas: [makeIdeaNode('idea-1', 'Idea')],
          goals: [makeGoalNode('goal-1', 'Goal')],
          actions: [makeActionNode('action-1', 'Action')],
          orchestration: [orchNode],
        },
      }),
    );

    render(<UnifiedPipelineCanvas />);

    // Initially zoom is 1.0, so orchestration is hidden
    expect(screen.queryByTestId('node-orch-1')).not.toBeInTheDocument();

    // Simulate zoom to 2.0
    act(() => {
      capturedOnViewportChange?.({ zoom: 2.0, x: 0, y: 0 });
    });

    // Now all 4 stages should be visible
    expect(screen.getByTestId('node-idea-1')).toBeInTheDocument();
    expect(screen.getByTestId('node-goal-1')).toBeInTheDocument();
    expect(screen.getByTestId('node-action-1')).toBeInTheDocument();
    expect(screen.getByTestId('node-orch-1')).toBeInTheDocument();
  });

  it('semantic zoom: shows only ideas + goals when zoom < 0.8', () => {
    mockedUsePipelineCanvas.mockReturnValue(
      makeMockCanvas({
        stageNodes: {
          ideas: [makeIdeaNode('idea-1', 'Idea')],
          goals: [makeGoalNode('goal-1', 'Goal')],
          actions: [makeActionNode('action-1', 'Action')],
          orchestration: [makeOrchNode('orch-1', 'Agent')],
        },
      }),
    );

    render(<UnifiedPipelineCanvas />);

    // Simulate zoom to 0.5
    act(() => {
      capturedOnViewportChange?.({ zoom: 0.5, x: 0, y: 0 });
    });

    // Only ideas + goals visible
    expect(screen.getByTestId('node-idea-1')).toBeInTheDocument();
    expect(screen.getByTestId('node-goal-1')).toBeInTheDocument();
    expect(screen.queryByTestId('node-action-1')).not.toBeInTheDocument();
    expect(screen.queryByTestId('node-orch-1')).not.toBeInTheDocument();
  });

  it('stage filter toggles work', () => {
    mockedUsePipelineCanvas.mockReturnValue(
      makeMockCanvas({
        stageNodes: {
          ideas: [makeIdeaNode('idea-1', 'Idea')],
          goals: [makeGoalNode('goal-1', 'Goal')],
          actions: [],
          orchestration: [],
        },
      }),
    );

    render(<UnifiedPipelineCanvas />);

    // Ideas node should be visible initially
    expect(screen.getByTestId('node-idea-1')).toBeInTheDocument();

    // Toggle ideas stage off
    fireEvent.click(screen.getByTestId('stage-toggle-ideas'));

    // Ideas node should now be hidden
    expect(screen.queryByTestId('node-idea-1')).not.toBeInTheDocument();

    // Toggle it back on
    fireEvent.click(screen.getByTestId('stage-toggle-ideas'));
    expect(screen.getByTestId('node-idea-1')).toBeInTheDocument();
  });

  it('AI transition buttons enable/disable based on selected nodes', () => {
    mockedUsePipelineCanvas.mockReturnValue(
      makeMockCanvas({
        stageNodes: {
          ideas: [makeIdeaNode('idea-1', 'Idea')],
          goals: [makeGoalNode('goal-1', 'Goal')],
          actions: [makeActionNode('action-1', 'Action')],
          orchestration: [],
        },
      }),
    );

    render(<UnifiedPipelineCanvas />);

    // Initially no nodes selected, all buttons disabled
    const goalsBtn = screen.getByTestId('btn-generate-goals');
    const tasksBtn = screen.getByTestId('btn-generate-tasks');
    const workflowBtn = screen.getByTestId('btn-generate-workflow');

    expect(goalsBtn).toBeDisabled();
    expect(tasksBtn).toBeDisabled();
    expect(workflowBtn).toBeDisabled();

    // Click an idea node to select it
    fireEvent.click(screen.getByTestId('node-idea-1'));

    // Now "Generate Goals" should be enabled (idea selected)
    expect(screen.getByTestId('btn-generate-goals')).not.toBeDisabled();
    // Others still disabled (no goal/action selected yet)
    expect(screen.getByTestId('btn-generate-tasks')).toBeDisabled();
    expect(screen.getByTestId('btn-generate-workflow')).toBeDisabled();
  });

  it('provenance sidebar opens on node click and closes', () => {
    mockedUsePipelineCanvas.mockReturnValue(
      makeMockCanvas({
        stageNodes: {
          ideas: [makeIdeaNode('idea-1', 'Test Idea')],
          goals: [],
          actions: [],
          orchestration: [],
        },
      }),
    );

    render(<UnifiedPipelineCanvas />);

    // Initially no provenance sidebar
    expect(screen.queryByTestId('provenance-sidebar')).not.toBeInTheDocument();

    // Click a node
    fireEvent.click(screen.getByTestId('node-idea-1'));

    // Provenance sidebar should appear
    expect(screen.getByTestId('provenance-sidebar')).toBeInTheDocument();
    expect(screen.getByText('Provenance')).toBeInTheDocument();

    // Close it
    fireEvent.click(screen.getByTestId('provenance-close'));
    expect(screen.queryByTestId('provenance-sidebar')).not.toBeInTheDocument();
  });

  it('cross-stage edges render correctly', () => {
    const edge = makeEdge('e1', 'idea-1', 'goal-1');

    mockedUsePipelineCanvas.mockReturnValue(
      makeMockCanvas({
        stageNodes: {
          ideas: [makeIdeaNode('idea-1', 'Idea')],
          goals: [makeGoalNode('goal-1', 'Goal')],
          actions: [],
          orchestration: [],
        },
        stageEdges: {
          ideas: [edge],
          goals: [],
          actions: [],
          orchestration: [],
        },
      }),
    );

    render(<UnifiedPipelineCanvas />);

    // Edge should be rendered
    expect(screen.getByTestId('edge-e1')).toBeInTheDocument();
  });

  it('stage filter sidebar shows correct node counts', () => {
    mockedUsePipelineCanvas.mockReturnValue(
      makeMockCanvas({
        stageNodes: {
          ideas: [makeIdeaNode('i1', 'A'), makeIdeaNode('i2', 'B')],
          goals: [makeGoalNode('g1', 'G')],
          actions: [],
          orchestration: [makeOrchNode('o1', 'O'), makeOrchNode('o2', 'O2'), makeOrchNode('o3', 'O3')],
        },
      }),
    );

    render(<UnifiedPipelineCanvas />);

    expect(screen.getByTestId('stage-count-ideas')).toHaveTextContent('2');
    expect(screen.getByTestId('stage-count-goals')).toHaveTextContent('1');
    expect(screen.getByTestId('stage-count-actions')).toHaveTextContent('0');
    expect(screen.getByTestId('stage-count-orchestration')).toHaveTextContent('3');
  });

  it('displays zoom indicator text', () => {
    mockedUsePipelineCanvas.mockReturnValue(makeMockCanvas());
    render(<UnifiedPipelineCanvas />);

    // Default zoom 1.0 is between 0.8 and 1.5
    expect(screen.getByTestId('zoom-indicator')).toHaveTextContent('ideas + goals + actions');

    // Change to high zoom
    act(() => {
      capturedOnViewportChange?.({ zoom: 2.0, x: 0, y: 0 });
    });
    expect(screen.getByTestId('zoom-indicator')).toHaveTextContent('all stages');

    // Change to low zoom
    act(() => {
      capturedOnViewportChange?.({ zoom: 0.5, x: 0, y: 0 });
    });
    expect(screen.getByTestId('zoom-indicator')).toHaveTextContent('ideas + goals');
  });

  it('hides AI transition toolbar in readOnly mode', () => {
    mockedUsePipelineCanvas.mockReturnValue(makeMockCanvas());
    render(<UnifiedPipelineCanvas readOnly />);

    expect(screen.queryByTestId('ai-transition-toolbar')).not.toBeInTheDocument();
  });

  it('AI generate buttons call aiGenerate with correct stage', () => {
    const aiGenerate = jest.fn();
    mockedUsePipelineCanvas.mockReturnValue(
      makeMockCanvas({
        aiGenerate,
        stageNodes: {
          ideas: [makeIdeaNode('idea-1', 'Idea')],
          goals: [makeGoalNode('goal-1', 'Goal')],
          actions: [makeActionNode('action-1', 'Action')],
          orchestration: [],
        },
      }),
    );

    render(<UnifiedPipelineCanvas />);

    // Select an idea node to enable "Generate Goals"
    fireEvent.click(screen.getByTestId('node-idea-1'));
    fireEvent.click(screen.getByTestId('btn-generate-goals'));
    expect(aiGenerate).toHaveBeenCalledWith('goals');

    // Select a goal node to enable "Generate Tasks"
    fireEvent.click(screen.getByTestId('node-goal-1'));
    fireEvent.click(screen.getByTestId('btn-generate-tasks'));
    expect(aiGenerate).toHaveBeenCalledWith('actions');

    // Select an action node to enable "Generate Workflow"
    fireEvent.click(screen.getByTestId('node-action-1'));
    fireEvent.click(screen.getByTestId('btn-generate-workflow'));
    expect(aiGenerate).toHaveBeenCalledWith('orchestration');
  });
});
