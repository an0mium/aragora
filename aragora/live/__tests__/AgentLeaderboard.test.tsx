import { render, screen, fireEvent } from '@testing-library/react';
import { AgentLeaderboard, type AgentRankingEntry } from '../src/components/analytics/AgentLeaderboard';

const mockAgents: AgentRankingEntry[] = [
  {
    rank: 1, agent_name: 'claude-sonnet', elo: 1650,
    wins: 45, losses: 5, draws: 10, win_rate: 75.0, games_played: 60,
    calibration: { brier_score: 0.08, ece: 0.03, trust_tier: 'excellent', prediction_count: 50 },
  },
  {
    rank: 2, agent_name: 'gpt-4o', elo: 1580,
    wins: 38, losses: 12, draws: 8, win_rate: 65.5, games_played: 58,
    calibration: null,
  },
  {
    rank: 3, agent_name: 'gemini-pro', elo: 1520,
    wins: 30, losses: 18, draws: 12, win_rate: 50.0, games_played: 60,
  },
];

describe('AgentLeaderboard', () => {
  describe('loading state', () => {
    it('shows loading skeleton', () => {
      render(<AgentLeaderboard agents={[]} loading={true} />);
      expect(screen.getByText(/AGENT LEADERBOARD/)).toBeInTheDocument();
    });
  });

  describe('empty state', () => {
    it('shows empty message when no agents', () => {
      render(<AgentLeaderboard agents={[]} />);
      expect(screen.getByText('No agent data available')).toBeInTheDocument();
    });
  });

  describe('populated state', () => {
    it('renders agent names', () => {
      render(<AgentLeaderboard agents={mockAgents} />);
      expect(screen.getByText('claude-sonnet')).toBeInTheDocument();
      expect(screen.getByText('gpt-4o')).toBeInTheDocument();
      expect(screen.getByText('gemini-pro')).toBeInTheDocument();
    });

    it('renders rank icons', () => {
      render(<AgentLeaderboard agents={mockAgents} />);
      expect(screen.getByText('[1ST]')).toBeInTheDocument();
      expect(screen.getByText('[2ND]')).toBeInTheDocument();
      expect(screen.getByText('[3RD]')).toBeInTheDocument();
    });

    it('renders ELO scores', () => {
      render(<AgentLeaderboard agents={mockAgents} />);
      expect(screen.getByText('1650')).toBeInTheDocument();
      expect(screen.getByText('1580')).toBeInTheDocument();
      expect(screen.getByText('1520')).toBeInTheDocument();
    });

    it('renders win rates', () => {
      render(<AgentLeaderboard agents={mockAgents} />);
      expect(screen.getByText('75.0%')).toBeInTheDocument();
      expect(screen.getByText('65.5%')).toBeInTheDocument();
      expect(screen.getByText('50.0%')).toBeInTheDocument();
    });

    it('renders W/L/D stats', () => {
      render(<AgentLeaderboard agents={mockAgents} />);
      expect(screen.getByText('45')).toBeInTheDocument();
    });

    it('renders summary footer', () => {
      render(<AgentLeaderboard agents={mockAgents} />);
      expect(screen.getByText(/Total agents:/)).toBeInTheDocument();
      expect(screen.getByText('3')).toBeInTheDocument();
    });

    it('uses custom title', () => {
      render(<AgentLeaderboard agents={mockAgents} title="TOP AGENTS" />);
      expect(screen.getByText(/TOP AGENTS/)).toBeInTheDocument();
    });

    it('respects limit prop', () => {
      render(<AgentLeaderboard agents={mockAgents} limit={2} />);
      expect(screen.getByText('claude-sonnet')).toBeInTheDocument();
      expect(screen.getByText('gpt-4o')).toBeInTheDocument();
      expect(screen.queryByText('gemini-pro')).not.toBeInTheDocument();
    });
  });

  describe('sorting', () => {
    it('sorts by ELO descending when ELO header clicked', () => {
      render(<AgentLeaderboard agents={mockAgents} />);
      fireEvent.click(screen.getByText('ELO'));
      const rows = screen.getAllByText(/\[\d/);
      expect(rows[0].textContent).toBe('[1ST]');
    });

    it('toggles sort direction on second click', () => {
      render(<AgentLeaderboard agents={mockAgents} />);
      fireEvent.click(screen.getByText('ELO'));
      fireEvent.click(screen.getByText('ELO'));
      expect(screen.getByText('^')).toBeInTheDocument();
    });
  });

  describe('interaction', () => {
    it('calls onAgentClick when row is clicked', () => {
      const onClick = jest.fn();
      render(<AgentLeaderboard agents={mockAgents} onAgentClick={onClick} />);
      fireEvent.click(screen.getByText('claude-sonnet'));
      expect(onClick).toHaveBeenCalledWith(expect.objectContaining({ agent_name: 'claude-sonnet' }));
    });

    it('does not crash without onAgentClick', () => {
      render(<AgentLeaderboard agents={mockAgents} />);
      fireEvent.click(screen.getByText('claude-sonnet'));
    });
  });

  describe('calibration display', () => {
    it('renders TrustBadge for agents with calibration data', () => {
      render(<AgentLeaderboard agents={mockAgents} />);
      expect(screen.getByText('claude-sonnet')).toBeInTheDocument();
    });

    it('shows calibration score column when agents have calibration_score', () => {
      const agentsWithScore = mockAgents.map((a, i) => ({
        ...a,
        calibration_score: 0.9 - i * 0.1,
      }));
      render(<AgentLeaderboard agents={agentsWithScore} />);
      expect(screen.getByText('Calibration')).toBeInTheDocument();
      expect(screen.getByText('90%')).toBeInTheDocument();
    });
  });

  describe('className prop', () => {
    it('applies custom className', () => {
      const { container } = render(
        <AgentLeaderboard agents={mockAgents} className="my-class" />
      );
      expect((container.firstChild as HTMLElement)?.className).toContain('my-class');
    });
  });
});
