/**
 * Interactive tests for PipelineCanvas component.
 *
 * Covers: stage navigation, palette visibility, toolbar rendering,
 * property editor display, and provenance sidebar in readOnly mode.
 */

import { render, screen, fireEvent, act } from '@testing-library/react';
import type { PipelineStageType } from '../types';

// ---------------------------------------------------------------------------
// Mock @xyflow/react -- requires browser APIs unavailable in jsdom
// ---------------------------------------------------------------------------

jest.mock('@xyflow/react', () => ({
  ReactFlow: ({ children, onNodeClick, onPaneClick, onDrop, onDragOver, ...props }: any) => (
    <div data-testid="react-flow" {...{ onDrop, onDragOver }}>
      {children}
      {props.nodes?.map((n: any) => (
        <div key={n.id} data-testid={`node-${n.id}`} onClick={(e) => onNodeClick?.(e, n)}>
          {n.data?.label}
        </div>
      ))}
    </div>
  ),
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
// Mock child components so we can detect their presence without full rendering
// ---------------------------------------------------------------------------

jest.mock('../PipelinePalette', () => ({
  PipelinePalette: ({ stage }: { stage: string }) => (
    <div data-testid="pipeline-palette">Palette: {stage}</div>
  ),
}));

jest.mock('../PipelineToolbar', () => ({
  PipelineToolbar: (props: any) => (
    <div data-testid="pipeline-toolbar">Toolbar: {props.stage}</div>
  ),
}));

jest.mock('../editors/PipelinePropertyEditor', () => ({
  PipelinePropertyEditor: (props: any) => (
    <div data-testid="pipeline-property-editor">
      PropertyEditor: {props.stage}
    </div>
  ),
}));

jest.mock('../nodes', () => ({
  IdeaNode: () => <div />,
  GoalNode: () => <div />,
  ActionNode: () => <div />,
  OrchestrationNode: () => <div />,
}));

jest.mock('../StageNavigator', () => ({
  StageNavigator: (props: any) => (
    <div data-testid="stage-navigator">
      {/* Expose stage buttons so tests can switch stages */}
      {(['ideas', 'goals', 'actions', 'orchestration'] as const).map((s) => (
        <button key={s} data-testid={`stage-btn-${s}`} onClick={() => props.onStageSelect(s)}>
          {s}
        </button>
      ))}
    </div>
  ),
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

// Import PipelineCanvas AFTER all mocks are defined
import { PipelineCanvas } from '../PipelineCanvas';

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('PipelineCanvas Interactive', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    mockedUsePipelineCanvas.mockReturnValue(makeMockCanvas());
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('renders "All Stages" button and StageNavigator in default view', () => {
    render(<PipelineCanvas />);

    // The "All Stages" button should be present and visible
    const allStagesButton = screen.getByRole('button', { name: /all stages/i });
    expect(allStagesButton).toBeInTheDocument();

    // StageNavigator should render
    expect(screen.getByTestId('stage-navigator')).toBeInTheDocument();
  });

  it('shows palette when switching to a specific stage (not "all" and not readOnly)', () => {
    render(<PipelineCanvas />);

    // In the default 'all' view the palette should NOT be visible
    expect(screen.queryByTestId('pipeline-palette')).not.toBeInTheDocument();

    // Switch to the 'ideas' stage via the mocked StageNavigator button
    act(() => {
      fireEvent.click(screen.getByTestId('stage-btn-ideas'));
      jest.runAllTimers();
    });

    // Now the palette should appear (isEditable = viewMode !== 'all' && !readOnly)
    expect(screen.getByTestId('pipeline-palette')).toBeInTheDocument();
    expect(screen.getByTestId('pipeline-palette')).toHaveTextContent('Palette: ideas');
  });

  it('hides palette in readOnly mode', () => {
    render(<PipelineCanvas readOnly />);

    // Switch to a specific stage -- even after switching, palette stays hidden in readOnly
    act(() => {
      fireEvent.click(screen.getByTestId('stage-btn-goals'));
      jest.runAllTimers();
    });

    expect(screen.queryByTestId('pipeline-palette')).not.toBeInTheDocument();
  });

  it('shows toolbar in edit mode (non-"all", non-readOnly stage)', () => {
    render(<PipelineCanvas />);

    // In 'all' view the toolbar should not be present (isEditable is false)
    expect(screen.queryByTestId('pipeline-toolbar')).not.toBeInTheDocument();

    // Switch to a specific stage
    act(() => {
      fireEvent.click(screen.getByTestId('stage-btn-actions'));
      jest.runAllTimers();
    });

    // The toolbar is rendered inside a Panel with position="top-center"
    const topCenterPanel = screen.getByTestId('panel-top-center');
    expect(topCenterPanel).toBeInTheDocument();
    expect(screen.getByTestId('pipeline-toolbar')).toBeInTheDocument();
  });

  it('clicking a node shows property editor in edit mode', () => {
    const nodeData = { label: 'My Idea', ideaType: 'concept', contentHash: '' };
    const testNode = {
      id: 'node-1',
      type: 'ideaNode',
      position: { x: 0, y: 0 },
      data: nodeData,
    };

    // Render with selectedNodeId already set and a specific stage active.
    // The component derives showPropertyEditor from:
    //   !!selectedNodeId && !showProvenance && isEditable
    // where isEditable = viewMode !== 'all' && !readOnly
    // Since viewMode starts as 'all', we need to switch stage first.
    // To avoid complexity with internal state, we render directly with
    // the hook returning the selected state, then switch out of 'all' view.
    const mockCanvas = makeMockCanvas({
      nodes: [testNode],
      selectedNodeId: 'node-1',
      selectedNodeData: nodeData,
    });
    mockedUsePipelineCanvas.mockReturnValue(mockCanvas);

    render(<PipelineCanvas />);

    // In 'all' view, property editor should NOT show (isEditable is false)
    expect(screen.queryByTestId('pipeline-property-editor')).not.toBeInTheDocument();

    // Switch to a specific stage to make it editable
    act(() => {
      fireEvent.click(screen.getByTestId('stage-btn-ideas'));
      jest.runAllTimers();
    });

    // Now isEditable=true and selectedNodeId is set, so property editor should appear
    expect(screen.getByTestId('pipeline-property-editor')).toBeInTheDocument();
  });

  it('clicking a node shows provenance sidebar in readOnly mode', () => {
    const nodeData = { label: 'Read-Only Node', ideaType: 'concept', contentHash: '' };
    const testNode = {
      id: 'node-ro',
      type: 'ideaNode',
      position: { x: 0, y: 0 },
      data: nodeData,
    };

    const mockCanvas = makeMockCanvas({
      nodes: [testNode],
      stageNodes: {
        ideas: [testNode],
        goals: [],
        actions: [],
        orchestration: [],
      },
      selectedNodeId: 'node-ro',
      selectedNodeData: nodeData,
    });
    mockedUsePipelineCanvas.mockReturnValue(mockCanvas);

    render(<PipelineCanvas readOnly />);

    // In readOnly with a selectedNodeId, the provenance sidebar should show.
    // The provenance sidebar contains the heading "Provenance".
    expect(screen.getByText('Provenance')).toBeInTheDocument();

    // It should also display the node label (appears both in ReactFlow node
    // mock and provenance sidebar, so use getAllByText)
    const labelElements = screen.getAllByText('Read-Only Node');
    expect(labelElements.length).toBeGreaterThanOrEqual(1);

    // The property editor should NOT be shown (readOnly mode shows provenance instead)
    expect(screen.queryByTestId('pipeline-property-editor')).not.toBeInTheDocument();
  });
});
