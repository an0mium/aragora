import { render, screen, waitFor } from '@testing-library/react';
import SettlementsPage from '../page';
import { apiFetch } from '@/lib/api';

jest.mock('@/lib/api', () => ({
  apiFetch: jest.fn(),
}));

const mockApiFetch = apiFetch as jest.MockedFunction<typeof apiFetch>;

describe('SettlementsPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders settlement and oracle telemetry strip when observability data is available', async () => {
    mockApiFetch.mockImplementation(async (path: string) => {
      if (path === '/api/v1/observability/dashboard') {
        return {
          settlement_review: {
            running: true,
            interval_hours: 6,
            available: true,
            stats: {
              success_rate: 0.875,
              total_receipts_updated: 11,
              last_result: { unresolved_due: 2 },
            },
          },
          oracle_stream: {
            active_sessions: 3,
            stalls_total: 1,
            ttft_avg_ms: 112.4,
            available: true,
          },
        };
      }
      if (path === '/api/v1/settlements') {
        return { settlements: [] };
      }
      throw new Error(`Unexpected path: ${path}`);
    });

    render(<SettlementsPage />);

    await waitFor(() => {
      expect(screen.getByText('Scheduler: RUNNING')).toBeInTheDocument();
    });

    expect(screen.getByText('Interval: 6h')).toBeInTheDocument();
    expect(screen.getByText('Success: 87.5%')).toBeInTheDocument();
    expect(screen.getByText('Updated: 11 | Unresolved: 2')).toBeInTheDocument();
    expect(screen.getByText('Active: 3 | Stalls: 1')).toBeInTheDocument();
    expect(screen.getByText('TTFT: 112ms')).toBeInTheDocument();
  });

  it('shows fallback telemetry messages when observability endpoint fails', async () => {
    mockApiFetch.mockImplementation(async (path: string) => {
      if (path === '/api/v1/observability/dashboard') {
        throw new Error('observability unavailable');
      }
      if (path === '/api/v1/settlements') {
        return { settlements: [] };
      }
      throw new Error(`Unexpected path: ${path}`);
    });

    render(<SettlementsPage />);

    await waitFor(() => {
      expect(screen.getByText('Ops telemetry unavailable')).toBeInTheDocument();
    });

    expect(screen.getByText('No settlement rollup')).toBeInTheDocument();
    expect(screen.getByText('No oracle stream telemetry')).toBeInTheDocument();
  });
});

