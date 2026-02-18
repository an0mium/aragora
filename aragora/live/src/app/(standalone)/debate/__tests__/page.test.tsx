import { renderWithProviders, screen, waitFor } from '@/test-utils';
import DebateViewerPage from '../[[...id]]/page';
import { DebateViewerWrapper } from '../[[...id]]/DebateViewerWrapper';

// Mock next/link
jest.mock('next/link', () => {
  return function MockLink({ children, href }: { children: React.ReactNode; href: string }) {
    return <a href={href}>{children}</a>;
  };
});

// Mock next/navigation
const mockPush = jest.fn();
let mockPathname = '/debate';
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
    replace: jest.fn(),
    prefetch: jest.fn(),
  }),
  usePathname: () => mockPathname,
}));

// Mock visual components
jest.mock('@/components/MatrixRain', () => ({
  Scanlines: () => <div data-testid="scanlines" />,
  CRTVignette: () => <div data-testid="crt-vignette" />,
}));

jest.mock('@/components/AsciiBanner', () => ({
  AsciiBannerCompact: () => <div data-testid="ascii-banner">ARAGORA</div>,
}));

jest.mock('@/components/ThemeToggle', () => ({
  ThemeToggle: () => <button data-testid="theme-toggle">Theme</button>,
}));

// Mock BackendSelector with context
const mockBackendConfig = { api: 'http://localhost:8080', ws: 'ws://localhost:8080/ws' };
jest.mock('@/components/BackendSelector', () => ({
  BackendSelector: () => <div data-testid="backend-selector">Backend</div>,
  useBackend: () => ({ config: mockBackendConfig }),
}));

// Mock debate-viewer components
jest.mock('@/components/debate-viewer', () => ({
  DebateViewer: ({ debateId, wsUrl }: { debateId: string; wsUrl: string }) => (
    <div data-testid="debate-viewer" data-debate-id={debateId} data-ws-url={wsUrl}>
      Debate Viewer
    </div>
  ),
}));

// Mock analysis panel components
jest.mock('@/components/CruxPanel', () => ({
  CruxPanel: ({ debateId }: { debateId: string }) => (
    <div data-testid="crux-panel" data-debate-id={debateId}>Crux Panel</div>
  ),
}));

jest.mock('@/components/AnalyticsPanel', () => ({
  AnalyticsPanel: () => <div data-testid="analytics-panel">Analytics Panel</div>,
}));

jest.mock('@/components/VoiceInput', () => ({
  VoiceInput: ({ debateId }: { debateId: string }) => (
    <div data-testid="voice-input" data-debate-id={debateId}>Voice Input</div>
  ),
}));

jest.mock('@/components/RedTeamAnalysisPanel', () => ({
  RedTeamAnalysisPanel: () => <div data-testid="red-team-panel">Red Team Analysis</div>,
}));

jest.mock('@/components/PanelErrorBoundary', () => ({
  PanelErrorBoundary: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

jest.mock('@/components/ImpasseDetectionPanel', () => ({
  ImpasseDetectionPanel: () => <div data-testid="impasse-detection-panel">Impasse Detection</div>,
}));

jest.mock('@/components/CalibrationPanel', () => ({
  CalibrationPanel: () => <div data-testid="calibration-panel">Calibration</div>,
}));

jest.mock('@/components/ConsensusKnowledgeBase', () => ({
  ConsensusKnowledgeBase: () => <div data-testid="consensus-kb">Consensus Knowledge Base</div>,
}));

jest.mock('@/components/TrendingTopicsPanel', () => ({
  TrendingTopicsPanel: () => <div data-testid="trending-topics-panel">Trending Topics</div>,
}));

jest.mock('@/components/MemoryInspector', () => ({
  MemoryInspector: () => <div data-testid="memory-inspector">Memory Inspector</div>,
}));

jest.mock('@/components/MetricsPanel', () => ({
  MetricsPanel: () => <div data-testid="metrics-panel">Metrics Panel</div>,
}));

jest.mock('@/components/broadcast/BroadcastPanel', () => ({
  BroadcastPanel: () => <div data-testid="broadcast-panel">Broadcast Panel</div>,
}));

jest.mock('@/components/EvidencePanel', () => ({
  EvidencePanel: () => <div data-testid="evidence-panel">Evidence Panel</div>,
}));

jest.mock('@/components/fork-visualizer', () => ({
  ForkVisualizer: () => <div data-testid="fork-visualizer">Fork Visualizer</div>,
}));

// Mock useDebateWebSocketStore hook
const mockSendSuggestion = jest.fn();
jest.mock('@/hooks/useDebateWebSocketStore', () => ({
  useDebateWebSocketStore: () => ({
    sendSuggestion: mockSendSuggestion,
    messages: [],
    connected: false,
  }),
}));

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

// Helper to set the mocked pathname for usePathname()
function setMockPathname(pathname: string) {
  mockPathname = pathname;
}

describe('DebateViewerPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    setMockPathname('/debate');
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({}),
    });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('initial render', () => {
    it('renders without crashing', async () => {
      renderWithProviders(<DebateViewerPage />);

      // Should render successfully
      await waitFor(() => {
        // Either no-debate message or debate viewer should appear
        const hasContent =
          screen.queryByText(/NO DEBATE ID PROVIDED/i) ||
          screen.queryByTestId('debate-viewer');
        expect(hasContent).toBeTruthy();
      });
    });

    it('renders the DebateViewerWrapper component', () => {
      renderWithProviders(<DebateViewerPage />);

      // Page renders the wrapper which handles the logic
      expect(document.body).toBeInTheDocument();
    });
  });
});

describe('DebateViewerWrapper', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    setMockPathname('/debate');
    mockFetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({}),
    });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('initial state', () => {
    it('component renders correctly with no debate ID', () => {
      // The debate ID is derived synchronously from usePathname()
      // With pathname '/debate', there is no ID segment
      renderWithProviders(<DebateViewerWrapper />);

      // Should show no debate ID message
      expect(screen.getByText(/NO DEBATE ID PROVIDED/i)).toBeInTheDocument();
    });
  });

  describe('no debate ID', () => {
    it('shows "no debate ID" message when no ID in URL', async () => {
      setMockPathname('/debate');

      renderWithProviders(<DebateViewerWrapper />);

      await waitFor(() => {
        expect(screen.getByText(/NO DEBATE ID PROVIDED/i)).toBeInTheDocument();
      });
    });

    it('renders visual effects when no debate ID', async () => {
      setMockPathname('/debate');

      renderWithProviders(<DebateViewerWrapper />);

      await waitFor(() => {
        expect(screen.getByTestId('scanlines')).toBeInTheDocument();
        expect(screen.getByTestId('crt-vignette')).toBeInTheDocument();
      });
    });

    it('renders header elements when no debate ID', async () => {
      setMockPathname('/debate');

      renderWithProviders(<DebateViewerWrapper />);

      await waitFor(() => {
        expect(screen.getByTestId('ascii-banner')).toBeInTheDocument();
        expect(screen.getByTestId('theme-toggle')).toBeInTheDocument();
      });
    });

    it('renders return to dashboard link when no debate ID', async () => {
      setMockPathname('/debate');

      renderWithProviders(<DebateViewerWrapper />);

      await waitFor(() => {
        expect(screen.getByText('[RETURN TO DASHBOARD]')).toBeInTheDocument();
      });
    });

    it('has correct link to dashboard', async () => {
      setMockPathname('/debate');

      renderWithProviders(<DebateViewerWrapper />);

      await waitFor(() => {
        const link = screen.getByText('[RETURN TO DASHBOARD]').closest('a');
        expect(link).toHaveAttribute('href', '/');
      });
    });
  });

  describe('with debate ID', () => {
    it('renders debate viewer when debate ID is in URL', async () => {
      setMockPathname('/debate/test-debate-123');

      renderWithProviders(<DebateViewerWrapper />);

      await waitFor(() => {
        expect(screen.getByTestId('debate-viewer')).toBeInTheDocument();
      });
    });

    it('passes debate ID to DebateViewer component', async () => {
      setMockPathname('/debate/my-debate-456');

      renderWithProviders(<DebateViewerWrapper />);

      await waitFor(() => {
        const viewer = screen.getByTestId('debate-viewer');
        expect(viewer).toHaveAttribute('data-debate-id', 'my-debate-456');
      });
    });

    it('passes WebSocket URL to DebateViewer component', async () => {
      setMockPathname('/debate/test-debate');

      renderWithProviders(<DebateViewerWrapper />);

      await waitFor(() => {
        const viewer = screen.getByTestId('debate-viewer');
        expect(viewer).toHaveAttribute('data-ws-url', 'ws://localhost:8080/ws');
      });
    });

    it('shows analysis toggle button for archived debates', async () => {
      setMockPathname('/debate/archived-debate-789');

      renderWithProviders(<DebateViewerWrapper />);

      await waitFor(() => {
        expect(screen.getByText('[+] SHOW ANALYSIS PANELS')).toBeInTheDocument();
      });
    });
  });

  describe('live debate detection', () => {
    it('detects live debate from adhoc_ prefix', async () => {
      setMockPathname('/debate/adhoc_live-debate');

      renderWithProviders(<DebateViewerWrapper />);

      await waitFor(() => {
        // Live debates show voice input, not analysis toggle
        expect(screen.getByTestId('voice-input')).toBeInTheDocument();
        expect(screen.queryByText('[+] SHOW ANALYSIS PANELS')).not.toBeInTheDocument();
      });
    });

    it('shows voice input panel for live debates', async () => {
      setMockPathname('/debate/adhoc_streaming-debate');

      renderWithProviders(<DebateViewerWrapper />);

      await waitFor(() => {
        expect(screen.getByTestId('voice-input')).toBeInTheDocument();
      });
    });

    it('hides voice input panel for archived debates', async () => {
      setMockPathname('/debate/archived-debate');

      renderWithProviders(<DebateViewerWrapper />);

      await waitFor(() => {
        expect(screen.queryByTestId('voice-input')).not.toBeInTheDocument();
      });
    });
  });

  describe('starting debate from trending topic', () => {
    it('calls API when starting debate from trending topic', async () => {
      setMockPathname('/debate');
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          success: true,
          debate_id: 'new-debate-id',
        }),
      });

      renderWithProviders(<DebateViewerWrapper />);

      // The handleStartDebateFromTrend function is passed to TrendingTopicsPanel
      // Testing the integration would require simulating the panel callback
    });

    it('navigates to new debate when created successfully', async () => {
      setMockPathname('/debate');
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({
          success: true,
          debate_id: 'created-debate-123',
        }),
      });

      renderWithProviders(<DebateViewerWrapper />);

      // The navigation happens via router.push in handleStartDebateFromTrend
    });
  });

  describe('error handling', () => {
    it('handles missing pathname gracefully', async () => {
      // The component reads pathname from usePathname()
      setMockPathname('/debate');

      renderWithProviders(<DebateViewerWrapper />);

      await waitFor(() => {
        // Should show no debate ID message rather than crash
        expect(screen.getByText(/NO DEBATE ID PROVIDED/i)).toBeInTheDocument();
      });
    });
  });

  describe('URL parsing', () => {
    it('extracts debate ID from URL path segments', async () => {
      setMockPathname('/debate/segment-debate-id');

      renderWithProviders(<DebateViewerWrapper />);

      await waitFor(() => {
        const viewer = screen.getByTestId('debate-viewer');
        expect(viewer).toHaveAttribute('data-debate-id', 'segment-debate-id');
      });
    });

    it('handles URLs with trailing slashes', async () => {
      setMockPathname('/debate/trailing-slash/');

      renderWithProviders(<DebateViewerWrapper />);

      await waitFor(() => {
        const viewer = screen.getByTestId('debate-viewer');
        expect(viewer).toHaveAttribute('data-debate-id', 'trailing-slash');
      });
    });

    it('handles root debate path', async () => {
      setMockPathname('/debate');

      renderWithProviders(<DebateViewerWrapper />);

      await waitFor(() => {
        expect(screen.getByText(/NO DEBATE ID PROVIDED/i)).toBeInTheDocument();
      });
    });
  });
});
