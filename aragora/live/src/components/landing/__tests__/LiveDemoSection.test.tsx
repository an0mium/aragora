import type { ReactNode } from 'react';
import { render, screen } from '@testing-library/react';
import { LiveDemoSection } from '../LiveDemoSection';

const mockUseSpectate = jest.fn();

jest.mock('next/link', () => ({
  __esModule: true,
  default: ({
    href,
    children,
    ...props
  }: {
    href: string;
    children: ReactNode;
  }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

jest.mock('@/context/ThemeContext', () => ({
  useTheme: () => ({ theme: 'dark' }),
}));

jest.mock('@/hooks/useSpectate', () => ({
  useSpectate: () => mockUseSpectate(),
}));

describe('LiveDemoSection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders a live debate feed from spectate events', () => {
    mockUseSpectate.mockReturnValue({
      connected: true,
      loaded: true,
      refresh: jest.fn(),
      status: {
        active: true,
        subscribers: 2,
        buffer_size: 12,
        bridge_state: 'live_debates_available',
        last_event_at: '2026-03-28T10:00:04Z',
        activity_age_seconds: 1,
        recent_activity_window_seconds: 120,
        recent_event_count: 3,
        live_debate_count: 1,
        live_debate_ids: ['debate-live'],
        live_debates: [
          {
            debate_id: 'debate-live',
            recent_event_count: 2,
            last_event_at: '2026-03-28T10:00:04Z',
            event_types: ['proposal', 'critique'],
          },
        ],
        unattributed_recent_event_count: 0,
      },
      events: [
        {
          event_type: 'proposal',
          timestamp: '2026-03-28T10:00:01Z',
          data: { details: 'Extract payments first so the hot path can scale independently.' },
          debate_id: 'debate-live',
          pipeline_id: null,
          agent_name: 'Claude',
          round_number: 1,
        },
        {
          event_type: 'critique',
          timestamp: '2026-03-28T10:00:04Z',
          data: { details: 'That migration adds operational risk before the platform team exists.' },
          debate_id: 'debate-live',
          pipeline_id: null,
          agent_name: 'Gemini',
          round_number: 1,
        },
        {
          event_type: 'proposal',
          timestamp: '2026-03-28T10:00:03Z',
          data: { details: 'Ignore this unrelated debate.' },
          debate_id: 'debate-other',
          pipeline_id: null,
          agent_name: 'Mistral',
          round_number: 2,
        },
      ],
    });

    render(<LiveDemoSection />);

    expect(screen.getByText('Live stream active')).toBeInTheDocument();
    expect(screen.getByText('Debate debate-live')).toBeInTheDocument();
    expect(screen.getByText('Claude')).toBeInTheDocument();
    expect(screen.getByText('Gemini')).toBeInTheDocument();
    expect(
      screen.getByText('Extract payments first so the hot path can scale independently.')
    ).toBeInTheDocument();
    expect(
      screen.getByText('That migration adds operational risk before the platform team exists.')
    ).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /open the live debate/i })).toHaveAttribute(
      'href',
      '/debate/debate-live'
    );
    expect(screen.queryByText('Ignore this unrelated debate.')).not.toBeInTheDocument();
  });

  it('renders a truthful waiting state when no live debate is available', () => {
    const refresh = jest.fn();
    mockUseSpectate.mockReturnValue({
      connected: true,
      loaded: true,
      refresh,
      status: {
        active: true,
        subscribers: 1,
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
      },
      events: [],
    });

    render(<LiveDemoSection />);

    expect(screen.getByText('Awaiting live debate')).toBeInTheDocument();
    expect(screen.getByText('Waiting for the next live debate')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /refresh live feed/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /check for a live debate/i })).toBeInTheDocument();
  });
});
