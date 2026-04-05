import { render, screen } from '@testing-library/react';

import { DecisionPackageView } from '../DecisionPackageView';

describe('DecisionPackageView', () => {
  it('renders provider routing details when present', () => {
    render(
      <DecisionPackageView
        pkg={{
          explanation: 'Routing explanation',
          agents: ['claude', 'gpt'],
          rounds: 3,
          consensus_reached: true,
          confidence: 0.92,
          total_cost: 0.0312,
          cost_breakdown: [],
          next_steps: [],
          provider_names: ['anthropic', 'openai'],
          provider_hints: ['claude-sonnet-4', 'gpt-4o'],
          provider_routing: {
            routing_applied: true,
            routing_strategy: 'provider_router_selection',
            routed_agent_names: ['claude', 'gpt'],
            provider_matches: {
              claude: 'anthropic',
              gpt: 'openai',
            },
            provider_hint_scores: {
              anthropic: 0.91,
              openai: 0.73,
            },
          },
          duration_seconds: 42,
        }}
      />
    );

    expect(screen.getByText(/provider routing/i)).toBeInTheDocument();
    expect(screen.getAllByText('anthropic').length).toBeGreaterThan(0);
    expect(screen.getAllByText('openai').length).toBeGreaterThan(0);
    expect(screen.getByText('provider_router_selection')).toBeInTheDocument();
    expect(screen.getByText('claude-sonnet-4')).toBeInTheDocument();
    expect(screen.getByText('gpt-4o')).toBeInTheDocument();
  });
});
