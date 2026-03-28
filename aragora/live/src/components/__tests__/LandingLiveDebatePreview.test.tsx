import { render, screen, waitFor } from '@testing-library/react';
import { LandingLiveDebatePreview } from '../LandingLiveDebatePreview';

jest.mock('next/link', () => {
  return function MockLink({
    children,
    href,
    className,
    ...rest
  }: {
    children: React.ReactNode;
    href: string;
    className?: string;
  }) {
    return (
      <a href={href} className={className} {...rest}>
        {children}
      </a>
    );
  };
});

const mockFetch = jest.fn();
global.fetch = mockFetch as typeof fetch;

function jsonResponse(data: unknown) {
  return Promise.resolve({
    ok: true,
    json: async () => data,
  });
}

describe('LandingLiveDebatePreview', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders a live debate preview when the public spectate feed exposes one', async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/api/v1/spectate/recent')) {
        return jsonResponse({
          events: [
            {
              event_type: 'proposal',
              timestamp: '2026-03-28T15:00:00Z',
              data: {
                content: 'Landing page visitors should see the live debate first.',
              },
              debate_id: 'debate-live-123456789',
              pipeline_id: null,
              agent_name: 'codex',
              round_number: 1,
            },
            {
              event_type: 'critique',
              timestamp: '2026-03-28T15:00:02Z',
              data: {
                critique: 'Show the back-and-forth instead of a static status card.',
              },
              debate_id: 'debate-live-123456789',
              pipeline_id: null,
              agent_name: 'claude',
              round_number: 1,
            },
          ],
        });
      }

      if (url.includes('/api/v1/spectate/status')) {
        return jsonResponse({
          active: true,
          subscribers: 4,
          buffer_size: 12,
          bridge_state: 'live_debates_available',
          last_event_at: '2026-03-28T15:00:02Z',
          activity_age_seconds: 0,
          recent_activity_window_seconds: 120,
          recent_event_count: 2,
          live_debate_count: 1,
          live_debate_ids: ['debate-live-123456789'],
          live_debates: [
            {
              debate_id: 'debate-live-123456789',
              recent_event_count: 2,
              last_event_at: '2026-03-28T15:00:02Z',
              event_types: ['critique', 'proposal'],
            },
          ],
          unattributed_recent_event_count: 0,
        });
      }

      return jsonResponse({});
    });

    render(<LandingLiveDebatePreview apiBase="http://localhost:8080" />);

    await waitFor(() => {
      expect(screen.getByText('Debate debate-live-...')).toBeInTheDocument();
    });

    expect(screen.getByTestId('landing-live-debate-status')).toHaveTextContent('LIVE');
    expect(screen.getByText('Show the back-and-forth instead of a static status card.')).toBeInTheDocument();
    expect(screen.getByText('claude')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open full spectate view' })).toHaveAttribute(
      'href',
      '/spectate/debate-live-123456789',
    );
  });

  it('renders an idle empty state when no live debate is discoverable', async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/api/v1/spectate/recent')) {
        return jsonResponse({ events: [] });
      }

      if (url.includes('/api/v1/spectate/status')) {
        return jsonResponse({
          active: true,
          subscribers: 0,
          buffer_size: 0,
          bridge_state: 'idle',
          last_event_at: null,
          activity_age_seconds: null,
          recent_activity_window_seconds: 120,
          recent_event_count: 0,
          live_debate_count: 0,
          live_debate_ids: [],
          live_debates: [],
          unattributed_recent_event_count: 0,
        });
      }

      return jsonResponse({});
    });

    render(<LandingLiveDebatePreview apiBase="http://localhost:8080" />);

    await waitFor(() => {
      expect(screen.getByTestId('landing-live-debate-empty')).toBeInTheDocument();
    });

    expect(screen.getByText('No public debate is live right now.')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Open spectate archive' })).toHaveAttribute(
      'href',
      '/spectate',
    );
  });
});
