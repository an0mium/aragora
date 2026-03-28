import { render, screen, within } from '@testing-library/react';

import {
  ReceiptCostBreakdown,
  type AgentLineItem,
} from '../ReceiptCostBreakdown';

const SAMPLE_ITEMS: AgentLineItem[] = [
  {
    agent: 'claude-advocate',
    provider: 'Anthropic',
    model: 'claude-sonnet-4-5-20250514',
    tokens_in: 12500,
    tokens_out: 3200,
    cost_usd: 0.0423,
    latency_ms: 4200,
    api_calls: 3,
    role: 'proposer',
    rounds: 3,
  },
  {
    agent: 'gpt4-critic',
    provider: 'OpenAI',
    model: 'gpt-4o',
    tokens_in: 9800,
    tokens_out: 2100,
    cost_usd: 0.0312,
    latency_ms: 3100,
    api_calls: 3,
    role: 'critic',
    rounds: 3,
  },
  {
    agent: 'gemini-judge',
    provider: 'Google',
    model: 'gemini-2.5-pro',
    tokens_in: 15000,
    tokens_out: 1800,
    cost_usd: 0.0189,
    latency_ms: 2800,
    api_calls: 1,
    role: 'judge',
    rounds: 1,
  },
];

describe('ReceiptCostBreakdown', () => {
  it('renders the component with line items', () => {
    render(
      <ReceiptCostBreakdown
        lineItems={SAMPLE_ITEMS}
        totalCostUsd={0.0924}
        totalDurationMs={8500}
      />
    );

    expect(screen.getByTestId('receipt-cost-breakdown')).toBeInTheDocument();
    expect(screen.getByText(/RECEIPT COST BREAKDOWN/)).toBeInTheDocument();
  });

  it('shows all agent names', () => {
    render(
      <ReceiptCostBreakdown
        lineItems={SAMPLE_ITEMS}
        totalCostUsd={0.0924}
        totalDurationMs={8500}
      />
    );

    expect(screen.getByText('claude-advocate')).toBeInTheDocument();
    expect(screen.getByText('gpt4-critic')).toBeInTheDocument();
    expect(screen.getByText('gemini-judge')).toBeInTheDocument();
  });

  it('shows provider and model info for each agent', () => {
    render(
      <ReceiptCostBreakdown
        lineItems={SAMPLE_ITEMS}
        totalCostUsd={0.0924}
        totalDurationMs={8500}
      />
    );

    // Providers may appear in both agent rows and model summary table
    expect(screen.getAllByText('Anthropic').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('claude-sonnet-4-5-20250514').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('OpenAI').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('gpt-4o').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Google').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('gemini-2.5-pro').length).toBeGreaterThanOrEqual(1);
  });

  it('displays role badges', () => {
    render(
      <ReceiptCostBreakdown
        lineItems={SAMPLE_ITEMS}
        totalCostUsd={0.0924}
        totalDurationMs={8500}
      />
    );

    expect(screen.getByText('PRO')).toBeInTheDocument();
    expect(screen.getByText('CRT')).toBeInTheDocument();
    expect(screen.getByText('JDG')).toBeInTheDocument();
  });

  it('shows summary stats', () => {
    render(
      <ReceiptCostBreakdown
        lineItems={SAMPLE_ITEMS}
        totalCostUsd={0.0924}
        totalDurationMs={8500}
        roundsCompleted={5}
      />
    );

    // Total cost (formatCost: 0.01 <= 0.0924 < 1 => toFixed(3))
    expect(screen.getByText('$0.092')).toBeInTheDocument();
    // Duration
    expect(screen.getByText('8.5s')).toBeInTheDocument();
    // Rounds (unique value to avoid collision with agent count)
    expect(screen.getByText('5')).toBeInTheDocument();
  });

  it('shows empty state when no line items', () => {
    render(
      <ReceiptCostBreakdown lineItems={[]} totalCostUsd={0} totalDurationMs={0} />
    );

    expect(screen.getByText(/No cost data available/)).toBeInTheDocument();
  });

  it('shows loading state', () => {
    render(
      <ReceiptCostBreakdown
        lineItems={[]}
        totalCostUsd={0}
        totalDurationMs={0}
        loading
      />
    );

    // Loading state should not show "No cost data"
    expect(screen.queryByText(/No cost data available/)).not.toBeInTheDocument();
    // But should show the header
    expect(screen.getByText(/RECEIPT COST BREAKDOWN/)).toBeInTheDocument();
  });

  it('sorts agents by cost descending', () => {
    render(
      <ReceiptCostBreakdown
        lineItems={SAMPLE_ITEMS}
        totalCostUsd={0.0924}
        totalDurationMs={8500}
      />
    );

    const agentNames = screen
      .getAllByText(/claude-advocate|gpt4-critic|gemini-judge/)
      .map((el) => el.textContent);

    // claude-advocate has highest cost, then gpt4-critic, then gemini-judge
    expect(agentNames[0]).toBe('claude-advocate');
    expect(agentNames[1]).toBe('gpt4-critic');
    expect(agentNames[2]).toBe('gemini-judge');
  });

  it('renders model summary table when enabled', () => {
    render(
      <ReceiptCostBreakdown
        lineItems={SAMPLE_ITEMS}
        totalCostUsd={0.0924}
        totalDurationMs={8500}
        showModelSummary
      />
    );

    expect(screen.getByText(/MODEL USAGE SUMMARY/)).toBeInTheDocument();
    // Table headers
    expect(screen.getByText('Model')).toBeInTheDocument();
    expect(screen.getByText('Provider')).toBeInTheDocument();
  });

  it('hides model summary when disabled', () => {
    render(
      <ReceiptCostBreakdown
        lineItems={SAMPLE_ITEMS}
        totalCostUsd={0.0924}
        totalDurationMs={8500}
        showModelSummary={false}
      />
    );

    expect(screen.queryByText(/MODEL USAGE SUMMARY/)).not.toBeInTheDocument();
  });

  it('shows debate ID when provided', () => {
    render(
      <ReceiptCostBreakdown
        lineItems={SAMPLE_ITEMS}
        totalCostUsd={0.0924}
        totalDurationMs={8500}
        debateId="abc123def456xyz789"
      />
    );

    expect(screen.getByText('abc123def456...')).toBeInTheDocument();
  });

  it('renders cost bars with correct proportions', () => {
    render(
      <ReceiptCostBreakdown
        lineItems={SAMPLE_ITEMS}
        totalCostUsd={0.0924}
        totalDurationMs={8500}
      />
    );

    const bars = screen.getAllByRole('progressbar');
    expect(bars).toHaveLength(3);

    // Highest cost agent should have 100% width
    expect(bars[0]).toHaveStyle({ width: '100%' });
  });

  it('formats large token counts with k suffix', () => {
    render(
      <ReceiptCostBreakdown
        lineItems={[
          {
            agent: 'test-agent',
            provider: 'Test',
            model: 'test-model',
            tokens_in: 150000,
            tokens_out: 50000,
            cost_usd: 1.5,
            latency_ms: 5000,
            api_calls: 10,
          },
        ]}
        totalCostUsd={1.5}
        totalDurationMs={5000}
      />
    );

    expect(screen.getAllByText('150.0k').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('50.0k').length).toBeGreaterThanOrEqual(1);
  });

  it('formats costs above $1 with 2 decimal places', () => {
    render(
      <ReceiptCostBreakdown
        lineItems={[
          {
            agent: 'expensive-agent',
            provider: 'Test',
            model: 'test-model',
            tokens_in: 500000,
            tokens_out: 200000,
            cost_usd: 12.34,
            latency_ms: 30000,
            api_calls: 50,
          },
        ]}
        totalCostUsd={12.34}
        totalDurationMs={30000}
      />
    );

    // Both the summary and the agent row should show formatted cost
    const costTexts = screen.getAllByText('$12.34');
    expect(costTexts.length).toBeGreaterThanOrEqual(1);
  });

  it('shows line item count in section header', () => {
    render(
      <ReceiptCostBreakdown
        lineItems={SAMPLE_ITEMS}
        totalCostUsd={0.0924}
        totalDurationMs={8500}
      />
    );

    expect(screen.getByText(/AGENT LINE ITEMS \(3\)/)).toBeInTheDocument();
  });

  it('applies custom className', () => {
    render(
      <ReceiptCostBreakdown
        lineItems={SAMPLE_ITEMS}
        totalCostUsd={0.0924}
        totalDurationMs={8500}
        className="custom-class"
      />
    );

    expect(screen.getByTestId('receipt-cost-breakdown')).toHaveClass('custom-class');
  });

  it('handles agent without role gracefully', () => {
    render(
      <ReceiptCostBreakdown
        lineItems={[
          {
            agent: 'no-role-agent',
            provider: 'Test',
            model: 'test-model',
            tokens_in: 1000,
            tokens_out: 500,
            cost_usd: 0.01,
            latency_ms: 1000,
            api_calls: 1,
          },
        ]}
        totalCostUsd={0.01}
        totalDurationMs={1000}
      />
    );

    expect(screen.getByText('no-role-agent')).toBeInTheDocument();
    // Should not crash without a role
    expect(screen.queryByText('PRO')).not.toBeInTheDocument();
    expect(screen.queryByText('CRT')).not.toBeInTheDocument();
  });

  it('aggregates model usage across agents', () => {
    const items: AgentLineItem[] = [
      {
        agent: 'agent-a',
        provider: 'Anthropic',
        model: 'claude-sonnet-4-5-20250514',
        tokens_in: 5000,
        tokens_out: 1000,
        cost_usd: 0.02,
        latency_ms: 2000,
        api_calls: 1,
      },
      {
        agent: 'agent-b',
        provider: 'Anthropic',
        model: 'claude-sonnet-4-5-20250514',
        tokens_in: 3000,
        tokens_out: 800,
        cost_usd: 0.015,
        latency_ms: 1500,
        api_calls: 1,
      },
    ];

    render(
      <ReceiptCostBreakdown
        lineItems={items}
        totalCostUsd={0.035}
        totalDurationMs={3500}
        showModelSummary
      />
    );

    // Model summary table should have one row for claude-sonnet (aggregated)
    const table = screen.getByText(/MODEL USAGE SUMMARY/).parentElement!;
    const rows = within(table).getAllByRole('row');
    // 1 header row + 1 data row (both agents use same model)
    expect(rows).toHaveLength(2);
  });
});
