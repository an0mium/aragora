import { fireEvent, render, screen, within } from '@testing-library/react';
import { CostBreakdown, type CostBreakdownData } from '../src/components/receipts/CostBreakdown';

function makeCostData(overrides: Partial<CostBreakdownData> = {}): CostBreakdownData {
  return {
    total_cost_usd: 0.42,
    total_tokens_in: 12500,
    total_tokens_out: 3200,
    total_calls: 8,
    per_agent: {
      'claude-critic': {
        cost_usd: 0.25,
        tokens_in: 8000,
        tokens_out: 2000,
        call_count: 4,
        model: 'claude-opus-4-6',
        provider: 'anthropic',
        latency_ms: 1250,
      },
      'gpt-advocate': {
        cost_usd: 0.17,
        tokens_in: 4500,
        tokens_out: 1200,
        call_count: 4,
        model: 'gpt-4o',
        provider: 'openai',
        latency_ms: 890,
      },
    },
    model_usage: {
      'claude-opus-4-6': {
        call_count: 4,
        tokens_in: 8000,
        tokens_out: 2000,
        cost_usd: 0.25,
        avg_latency_ms: 1250,
        provider: 'anthropic',
      },
      'gpt-4o': {
        call_count: 4,
        tokens_in: 4500,
        tokens_out: 1200,
        cost_usd: 0.17,
        avg_latency_ms: 890,
        provider: 'openai',
      },
    },
    ...overrides,
  };
}

/** Click a tab button by its data-testid. */
function clickTab(tabKey: string) {
  fireEvent.click(screen.getByTestId(`tab-${tabKey}`));
}

describe('CostBreakdown', () => {
  it('renders summary header with total cost, tokens, and calls', () => {
    render(<CostBreakdown data={makeCostData()} />);

    expect(screen.getByTestId('total-cost')).toHaveTextContent('$0.42');
    expect(screen.getByTestId('total-tokens')).toHaveTextContent('15.7K');
    expect(screen.getByTestId('total-calls')).toHaveTextContent('8');
  });

  it('displays duration when provided', () => {
    render(<CostBreakdown data={makeCostData({ duration_seconds: 125 })} />);

    expect(screen.getByTestId('duration')).toHaveTextContent('2m 5s');
  });

  it('shows efficiency metrics on the summary tab', () => {
    render(<CostBreakdown data={makeCostData()} />);

    expect(screen.getByText('Agents used:')).toBeInTheDocument();
    expect(screen.getByText('Models used:')).toBeInTheDocument();
    expect(screen.getByText('Cost / call:')).toBeInTheDocument();
  });

  it('shows top agent by cost on summary tab', () => {
    render(<CostBreakdown data={makeCostData()} />);

    expect(screen.getByText('claude-critic')).toBeInTheDocument();
    expect(screen.getByText('$0.25')).toBeInTheDocument();
  });

  it('switches to agents tab and shows per-agent table', () => {
    render(<CostBreakdown data={makeCostData()} />);

    clickTab('agents');

    expect(screen.getByText('claude-critic')).toBeInTheDocument();
    expect(screen.getByText('gpt-advocate')).toBeInTheDocument();
    expect(screen.getByText('anthropic')).toBeInTheDocument();
    expect(screen.getByText('openai')).toBeInTheDocument();
  });

  it('switches to models tab and shows model cards', () => {
    render(<CostBreakdown data={makeCostData()} />);

    clickTab('models');

    expect(screen.getByText('claude-opus-4-6')).toBeInTheDocument();
    expect(screen.getByText('gpt-4o')).toBeInTheDocument();
  });

  it('displays latency in model cards when available', () => {
    render(<CostBreakdown data={makeCostData()} />);

    clickTab('models');

    expect(screen.getByText('avg 1.3s')).toBeInTheDocument();
    expect(screen.getByText('avg 890ms')).toBeInTheDocument();
  });

  it('shows rounds tab when per_round data is present', () => {
    const data = makeCostData({
      per_round: [
        { round: 1, cost_usd: 0.20, tokens_in: 6000, tokens_out: 1500, call_count: 4, duration_ms: 3200 },
        { round: 2, cost_usd: 0.22, tokens_in: 6500, tokens_out: 1700, call_count: 4, duration_ms: 2800 },
      ],
    });

    render(<CostBreakdown data={data} />);

    clickTab('rounds');

    expect(screen.getByText('Round 1')).toBeInTheDocument();
    expect(screen.getByText('Round 2')).toBeInTheDocument();
    expect(screen.getByText('3.2s')).toBeInTheDocument();
  });

  it('hides rounds tab when no per_round data', () => {
    render(<CostBreakdown data={makeCostData()} />);

    expect(screen.queryByTestId('tab-rounds')).not.toBeInTheDocument();
  });

  it('handles empty per_agent gracefully on agents tab', () => {
    render(<CostBreakdown data={makeCostData({ per_agent: {} })} />);

    clickTab('agents');

    expect(screen.getByText('No per-agent cost data available')).toBeInTheDocument();
  });

  it('handles empty model_usage gracefully on models tab', () => {
    render(<CostBreakdown data={makeCostData({ model_usage: {} })} />);

    clickTab('models');

    expect(screen.getByText('No model usage data available')).toBeInTheDocument();
  });

  it('formats very small costs with 4 decimal places', () => {
    render(
      <CostBreakdown
        data={makeCostData({ total_cost_usd: 0.0023 })}
      />
    );

    expect(screen.getByTestId('total-cost')).toHaveTextContent('$0.0023');
  });

  it('formats large token counts with M suffix', () => {
    render(
      <CostBreakdown
        data={makeCostData({ total_tokens_in: 1_500_000, total_tokens_out: 500_000 })}
      />
    );

    expect(screen.getByTestId('total-tokens')).toHaveTextContent('2.0M');
  });

  it('renders cost bars in agent table', () => {
    render(<CostBreakdown data={makeCostData()} />);

    clickTab('agents');

    const bars = screen.getAllByTestId('cost-bar');
    expect(bars.length).toBeGreaterThan(0);
  });

  it('applies custom className', () => {
    const { container } = render(
      <CostBreakdown data={makeCostData()} className="my-custom-class" />
    );

    expect(container.firstChild).toHaveClass('my-custom-class');
  });
});
