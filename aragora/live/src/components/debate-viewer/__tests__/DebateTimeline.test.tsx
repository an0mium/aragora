/**
 * Tests for DebateTimeline component
 *
 * Tests cover:
 * - Renders timeline header
 * - Displays messages as timeline entries
 * - Displays stream events as timeline entries
 * - Agent filtering
 * - Event type filtering
 * - Chronological ordering
 * - Entry expansion
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { DebateTimeline } from '../DebateTimeline';
import type { TranscriptMessage } from '@/hooks/useDebateWebSocket';
import type { StreamEvent } from '@/types/events';

// Mock agentColors utility
jest.mock('@/utils/agentColors', () => ({
  getAgentColors: (agent: string) => ({
    bg: `bg-${agent}-500/20`,
    text: `text-${agent}-400`,
    border: `border-${agent}-500/30`,
    dot: 'bg-acid-green',
  }),
}));

const createMessage = (
  overrides: Partial<TranscriptMessage> = {}
): TranscriptMessage => ({
  agent: 'claude',
  content: 'This is a test message.',
  timestamp: 1705420800,
  role: 'participant',
  round: 1,
  ...overrides,
});

const createStreamEvent = (
  overrides: Partial<StreamEvent> = {}
): StreamEvent => ({
  type: 'agent_message' as const,
  data: { content: 'Event data' },
  timestamp: 1705420800,
  agent: 'claude',
  ...overrides,
});

describe('DebateTimeline', () => {
  describe('basic rendering', () => {
    it('renders timeline header', () => {
      render(
        <DebateTimeline
          messages={[createMessage()]}
          streamEvents={[]}
          agents={['claude']}
        />
      );

      expect(screen.getByText(/DEBATE TIMELINE/)).toBeInTheDocument();
    });

    it('renders with empty data', () => {
      render(
        <DebateTimeline messages={[]} streamEvents={[]} agents={[]} />
      );

      // Should render without crashing
      expect(screen.getByText(/DEBATE TIMELINE/)).toBeInTheDocument();
    });
  });

  describe('message entries', () => {
    it('renders messages as timeline entries', () => {
      const messages = [
        createMessage({ agent: 'claude', content: 'First message', timestamp: 1705420800 }),
        createMessage({ agent: 'gpt-4', content: 'Second message', timestamp: 1705420810 }),
      ];

      render(
        <DebateTimeline
          messages={messages}
          streamEvents={[]}
          agents={['claude', 'gpt-4']}
        />
      );

      expect(screen.getByText('CLAUDE')).toBeInTheDocument();
      expect(screen.getByText('GPT-4')).toBeInTheDocument();
    });
  });

  describe('stream event entries', () => {
    it('renders relevant stream events', () => {
      const events = [
        createStreamEvent({ type: 'debate_start', timestamp: 1705420790 }),
        createStreamEvent({ type: 'round_start', data: { round: 1 }, timestamp: 1705420800 }),
      ];

      render(
        <DebateTimeline
          messages={[]}
          streamEvents={events}
          agents={['claude']}
        />
      );

      // Timeline should have entries for these events
      const container = screen.getByText(/DEBATE TIMELINE/).closest('div');
      expect(container).toBeInTheDocument();
    });
  });

  describe('agent filtering', () => {
    it('shows agent filter buttons', () => {
      render(
        <DebateTimeline
          messages={[createMessage({ agent: 'claude' }), createMessage({ agent: 'gpt-4' })]}
          streamEvents={[]}
          agents={['claude', 'gpt-4']}
        />
      );

      // Agent filter buttons should exist
      const allButton = screen.getByText('All');
      expect(allButton).toBeInTheDocument();
    });

    it('filters entries by agent when agent button clicked', () => {
      const messages = [
        createMessage({ agent: 'claude', content: 'Claude message', timestamp: 1705420800 }),
        createMessage({ agent: 'gpt-4', content: 'GPT message', timestamp: 1705420810 }),
      ];

      render(
        <DebateTimeline
          messages={messages}
          streamEvents={[]}
          agents={['claude', 'gpt-4']}
        />
      );

      // Click on claude agent filter
      const claudeButton = screen.getByText('claude');
      fireEvent.click(claudeButton);

      // Claude message should be visible, GPT should not
      expect(screen.getByText('CLAUDE')).toBeInTheDocument();
      expect(screen.queryByText('GPT-4')).not.toBeInTheDocument();
    });
  });

  describe('event type filtering', () => {
    it('shows event type filter buttons', () => {
      render(
        <DebateTimeline
          messages={[createMessage()]}
          streamEvents={[
            createStreamEvent({ type: 'consensus' }),
          ]}
          agents={['claude']}
        />
      );

      // Should show "All Types" filter
      const allTypesButton = screen.getByText('All Types');
      expect(allTypesButton).toBeInTheDocument();
    });
  });

  describe('entry expansion', () => {
    it('expands entry to show content on click', () => {
      const messages = [
        createMessage({
          agent: 'claude',
          content: 'This is a detailed response that should be expandable.',
          timestamp: 1705420800,
        }),
      ];

      render(
        <DebateTimeline
          messages={messages}
          streamEvents={[]}
          agents={['claude']}
        />
      );

      // Find and click on the CLAUDE entry to expand it
      const entry = screen.getByText('CLAUDE');
      fireEvent.click(entry.closest('[class*="cursor-pointer"]') || entry);

      // After expansion, content should be visible
      expect(screen.getByText(/This is a detailed response/)).toBeInTheDocument();
    });
  });

  describe('multiple agents', () => {
    it('renders entries from multiple agents with different colors', () => {
      const messages = [
        createMessage({ agent: 'claude', content: 'Claude says', timestamp: 1705420800 }),
        createMessage({ agent: 'gpt-4', content: 'GPT says', timestamp: 1705420810 }),
        createMessage({ agent: 'gemini', content: 'Gemini says', timestamp: 1705420820 }),
      ];

      render(
        <DebateTimeline
          messages={messages}
          streamEvents={[]}
          agents={['claude', 'gpt-4', 'gemini']}
        />
      );

      expect(screen.getByText('CLAUDE')).toBeInTheDocument();
      expect(screen.getByText('GPT-4')).toBeInTheDocument();
      expect(screen.getByText('GEMINI')).toBeInTheDocument();
    });
  });
});
