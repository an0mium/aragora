import { render, screen, waitFor } from '@testing-library/react';
import MemoryPage from '../page';

jest.mock('next/dynamic', () => {
  return (_loader: unknown, options?: { loading?: () => JSX.Element }) => {
    const Loading = options?.loading;
    return function MockDynamicComponent() {
      return Loading ? <Loading /> : <div data-testid="dynamic-stub" />;
    };
  };
});

jest.mock('@/components/MatrixRain', () => ({
  Scanlines: () => <div data-testid="scanlines" />,
  CRTVignette: () => <div data-testid="crt-vignette" />,
}));

jest.mock('@/components/BackendSelector', () => ({
  useBackend: () => ({ config: { api: 'http://localhost:8080', ws: 'ws://localhost:8080' } }),
}));

jest.mock('@/components/PanelErrorBoundary', () => ({
  PanelErrorBoundary: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

jest.mock('@/components/memory/UnifiedMemorySearch', () => ({
  UnifiedMemorySearch: () => <div data-testid="unified-memory-search" />,
}));

jest.mock('@/components/memory/RetentionDecisions', () => ({
  RetentionDecisions: () => <div data-testid="retention-decisions" />,
}));

jest.mock('@/components/memory/DedupClusters', () => ({
  DedupClusters: () => <div data-testid="dedup-clusters" />,
}));

jest.mock('@/components/memory/CrossDebateLearning', () => ({
  CrossDebateLearning: () => <div data-testid="cross-debate-learning" />,
}));

jest.mock('@/components/memory/MemoryTiersPanel', () => ({
  MemoryTiersPanel: () => <div data-testid="memory-tiers-panel" />,
}));

const mockFetch = jest.fn();
global.fetch = mockFetch as unknown as typeof fetch;

describe('MemoryPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        pressure: 0.45,
        status: 'normal',
        tier_utilization: {
          FAST: { count: 15, limit: 100, utilization: 0.15 },
          MEDIUM: { count: 42, limit: 500, utilization: 0.084 },
          SLOW: { count: 128, limit: 1000, utilization: 0.128 },
          GLACIAL: { count: 87, limit: 5000, utilization: 0.017 },
        },
        total_memories: 272,
        cleanup_recommended: false,
      }),
    });
  });

  it('renders memory pressure from tier_utilization payloads without crashing', async () => {
    render(<MemoryPage />);

    expect(await screen.findByRole('heading', { name: /continuum memory/i })).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Memory Pressure')).toBeInTheDocument();
      expect(screen.getByText('15%')).toBeInTheDocument();
      expect(screen.getByText('8%')).toBeInTheDocument();
      expect(screen.getByText('13%')).toBeInTheDocument();
      expect(screen.getByText('2%')).toBeInTheDocument();
    });
  });
});
