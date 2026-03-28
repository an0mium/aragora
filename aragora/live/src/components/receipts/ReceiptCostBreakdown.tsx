'use client';

import { useMemo } from 'react';

// ============================================================================
// Types
// ============================================================================

export interface AgentLineItem {
  /** Agent name (e.g. "claude-advocate", "gpt4-critic") */
  agent: string;
  /** Provider name (e.g. "Anthropic", "OpenAI") */
  provider: string;
  /** Model identifier (e.g. "claude-sonnet-4-5-20250514", "gpt-4o") */
  model: string;
  /** Input tokens consumed */
  tokens_in: number;
  /** Output tokens generated */
  tokens_out: number;
  /** Cost in USD */
  cost_usd: number;
  /** Latency in milliseconds for this agent's total processing time */
  latency_ms: number;
  /** Number of API calls made */
  api_calls: number;
  /** Role in the debate (e.g. "proposer", "critic", "judge") */
  role?: string;
  /** Number of rounds this agent participated in */
  rounds?: number;
}

export interface ModelUsageSummary {
  model: string;
  provider: string;
  total_tokens: number;
  total_cost_usd: number;
  agent_count: number;
}

export interface ReceiptCostBreakdownProps {
  /** Per-agent line items */
  lineItems: AgentLineItem[];
  /** Total debate cost in USD */
  totalCostUsd: number;
  /** Total debate duration in milliseconds */
  totalDurationMs: number;
  /** Debate ID for reference */
  debateId?: string;
  /** Number of rounds completed */
  roundsCompleted?: number;
  /** Optional className */
  className?: string;
  /** Whether to show the model summary section */
  showModelSummary?: boolean;
  /** Whether the component is in a loading state */
  loading?: boolean;
}

// ============================================================================
// Helpers
// ============================================================================

function formatCost(usd: number): string {
  if (usd < 0.0001) return '<$0.0001';
  if (usd < 0.01) return `$${usd.toFixed(4)}`;
  if (usd < 1) return `$${usd.toFixed(3)}`;
  return `$${usd.toFixed(2)}`;
}

function formatTokens(count: number): string {
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}k`;
  return count.toLocaleString();
}

function formatLatency(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const mins = Math.floor(ms / 60_000);
  const secs = Math.round((ms % 60_000) / 1000);
  return `${mins}m ${secs}s`;
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const mins = Math.floor(ms / 60_000);
  const secs = Math.round((ms % 60_000) / 1000);
  if (mins < 60) return `${mins}m ${secs}s`;
  const hrs = Math.floor(mins / 60);
  const remMins = mins % 60;
  return `${hrs}h ${remMins}m`;
}

const ROLE_BADGES: Record<string, { text: string; color: string }> = {
  proposer: { text: 'PRO', color: 'text-[var(--acid-green)]' },
  critic: { text: 'CRT', color: 'text-[var(--acid-yellow)]' },
  judge: { text: 'JDG', color: 'text-[var(--acid-cyan)]' },
  mediator: { text: 'MED', color: 'text-purple-400' },
  advocate: { text: 'ADV', color: 'text-[var(--acid-green)]' },
  challenger: { text: 'CHL', color: 'text-orange-400' },
};

// ============================================================================
// Sub-components
// ============================================================================

function SummaryBar({
  totalCostUsd,
  totalDurationMs,
  totalTokensIn,
  totalTokensOut,
  agentCount,
  roundsCompleted,
}: {
  totalCostUsd: number;
  totalDurationMs: number;
  totalTokensIn: number;
  totalTokensOut: number;
  agentCount: number;
  roundsCompleted?: number;
}) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
      <div className="bg-[var(--bg)] border border-[var(--border)] p-3">
        <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase tracking-wider">
          Total Cost
        </div>
        <div className="text-lg font-mono text-[var(--acid-green)] mt-1">
          {formatCost(totalCostUsd)}
        </div>
      </div>
      <div className="bg-[var(--bg)] border border-[var(--border)] p-3">
        <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase tracking-wider">
          Duration
        </div>
        <div className="text-lg font-mono text-[var(--acid-cyan)] mt-1">
          {formatDuration(totalDurationMs)}
        </div>
      </div>
      <div className="bg-[var(--bg)] border border-[var(--border)] p-3">
        <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase tracking-wider">
          Tokens In
        </div>
        <div className="text-lg font-mono text-[var(--text)] mt-1">
          {formatTokens(totalTokensIn)}
        </div>
      </div>
      <div className="bg-[var(--bg)] border border-[var(--border)] p-3">
        <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase tracking-wider">
          Tokens Out
        </div>
        <div className="text-lg font-mono text-[var(--text)] mt-1">
          {formatTokens(totalTokensOut)}
        </div>
      </div>
      <div className="bg-[var(--bg)] border border-[var(--border)] p-3">
        <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase tracking-wider">
          Agents
        </div>
        <div className="text-lg font-mono text-[var(--text)] mt-1">{agentCount}</div>
      </div>
      {roundsCompleted != null && (
        <div className="bg-[var(--bg)] border border-[var(--border)] p-3">
          <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase tracking-wider">
            Rounds
          </div>
          <div className="text-lg font-mono text-[var(--text)] mt-1">{roundsCompleted}</div>
        </div>
      )}
    </div>
  );
}

function AgentRow({ item, maxCost }: { item: AgentLineItem; maxCost: number }) {
  const costBarWidth = maxCost > 0 ? (item.cost_usd / maxCost) * 100 : 0;
  const roleBadge = item.role ? ROLE_BADGES[item.role.toLowerCase()] : null;

  return (
    <div className="bg-[var(--bg)] border border-[var(--border)] p-4 hover:border-[var(--acid-green)]/30 transition-colors">
      {/* Agent header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-mono text-[var(--text)] font-medium">{item.agent}</span>
          {roleBadge && (
            <span
              className={`text-[10px] font-mono ${roleBadge.color} border border-current/30 px-1.5 py-0.5`}
            >
              {roleBadge.text}
            </span>
          )}
        </div>
        <span className="text-sm font-mono text-[var(--acid-green)] font-medium">
          {formatCost(item.cost_usd)}
        </span>
      </div>

      {/* Model and provider */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-[10px] font-mono text-[var(--text-muted)] uppercase tracking-wider">
          {item.provider}
        </span>
        <span className="text-[10px] text-[var(--text-muted)]">/</span>
        <span className="text-xs font-mono text-[var(--text-muted)]">{item.model}</span>
      </div>

      {/* Metrics row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
        <div>
          <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase">Tokens In</div>
          <div className="text-xs font-mono text-[var(--text)]">{formatTokens(item.tokens_in)}</div>
        </div>
        <div>
          <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase">
            Tokens Out
          </div>
          <div className="text-xs font-mono text-[var(--text)]">
            {formatTokens(item.tokens_out)}
          </div>
        </div>
        <div>
          <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase">Latency</div>
          <div className="text-xs font-mono text-[var(--acid-cyan)]">
            {formatLatency(item.latency_ms)}
          </div>
        </div>
        <div>
          <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase">API Calls</div>
          <div className="text-xs font-mono text-[var(--text)]">{item.api_calls}</div>
        </div>
      </div>

      {/* Cost bar */}
      <div className="h-1.5 bg-[var(--surface)] border border-[var(--border)]">
        <div
          className="h-full bg-[var(--acid-green)]/50 transition-all duration-300"
          style={{ width: `${costBarWidth}%` }}
          role="progressbar"
          aria-valuenow={item.cost_usd}
          aria-valuemax={maxCost}
          aria-label={`Cost proportion: ${formatCost(item.cost_usd)}`}
        />
      </div>
    </div>
  );
}

function ModelSummaryTable({ summaries }: { summaries: ModelUsageSummary[] }) {
  if (summaries.length === 0) return null;

  return (
    <div className="mt-6">
      <div className="text-xs font-mono text-[var(--acid-cyan)] mb-3">
        {'>'} MODEL USAGE SUMMARY
      </div>
      <div className="border border-[var(--border)] overflow-hidden">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="bg-[var(--surface)] border-b border-[var(--border)]">
              <th className="text-left p-2 text-[var(--text-muted)] uppercase tracking-wider font-normal">
                Model
              </th>
              <th className="text-left p-2 text-[var(--text-muted)] uppercase tracking-wider font-normal">
                Provider
              </th>
              <th className="text-right p-2 text-[var(--text-muted)] uppercase tracking-wider font-normal">
                Tokens
              </th>
              <th className="text-right p-2 text-[var(--text-muted)] uppercase tracking-wider font-normal">
                Cost
              </th>
              <th className="text-right p-2 text-[var(--text-muted)] uppercase tracking-wider font-normal">
                Agents
              </th>
            </tr>
          </thead>
          <tbody>
            {summaries.map((s) => (
              <tr key={s.model} className="border-b border-[var(--border)] last:border-b-0">
                <td className="p-2 text-[var(--text)]">{s.model}</td>
                <td className="p-2 text-[var(--text-muted)]">{s.provider}</td>
                <td className="p-2 text-right text-[var(--text)]">{formatTokens(s.total_tokens)}</td>
                <td className="p-2 text-right text-[var(--acid-green)]">
                  {formatCost(s.total_cost_usd)}
                </td>
                <td className="p-2 text-right text-[var(--text)]">{s.agent_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function ReceiptCostBreakdown({
  lineItems,
  totalCostUsd,
  totalDurationMs,
  debateId,
  roundsCompleted,
  className,
  showModelSummary = true,
  loading = false,
}: ReceiptCostBreakdownProps) {
  const sortedItems = useMemo(
    () => [...lineItems].sort((a, b) => b.cost_usd - a.cost_usd),
    [lineItems]
  );

  const maxCost = useMemo(
    () => (sortedItems.length > 0 ? Math.max(...sortedItems.map((i) => i.cost_usd)) : 0),
    [sortedItems]
  );

  const totalTokensIn = useMemo(() => lineItems.reduce((s, i) => s + i.tokens_in, 0), [lineItems]);
  const totalTokensOut = useMemo(
    () => lineItems.reduce((s, i) => s + i.tokens_out, 0),
    [lineItems]
  );

  const modelSummaries = useMemo(() => {
    if (!showModelSummary) return [];
    const byModel = new Map<string, ModelUsageSummary>();
    for (const item of lineItems) {
      const key = item.model;
      const existing = byModel.get(key);
      if (existing) {
        existing.total_tokens += item.tokens_in + item.tokens_out;
        existing.total_cost_usd += item.cost_usd;
        existing.agent_count += 1;
      } else {
        byModel.set(key, {
          model: item.model,
          provider: item.provider,
          total_tokens: item.tokens_in + item.tokens_out,
          total_cost_usd: item.cost_usd,
          agent_count: 1,
        });
      }
    }
    return Array.from(byModel.values()).sort((a, b) => b.total_cost_usd - a.total_cost_usd);
  }, [lineItems, showModelSummary]);

  if (loading) {
    return (
      <div className={`bg-[var(--surface)] border border-[var(--border)] p-6 ${className ?? ''}`}>
        <div className="text-xs font-mono text-[var(--acid-green)] mb-4">
          {'>'} RECEIPT COST BREAKDOWN
        </div>
        <div className="animate-pulse space-y-4">
          <div className="grid grid-cols-3 gap-3">
            {[1, 2, 3].map((n) => (
              <div key={n} className="h-16 bg-[var(--bg)] border border-[var(--border)]" />
            ))}
          </div>
          {[1, 2].map((n) => (
            <div key={n} className="h-24 bg-[var(--bg)] border border-[var(--border)]" />
          ))}
        </div>
      </div>
    );
  }

  if (lineItems.length === 0) {
    return (
      <div className={`bg-[var(--surface)] border border-[var(--border)] p-6 ${className ?? ''}`}>
        <div className="text-xs font-mono text-[var(--acid-green)] mb-4">
          {'>'} RECEIPT COST BREAKDOWN
        </div>
        <div className="text-center py-8 text-[var(--text-muted)] font-mono text-sm">
          {'>'} No cost data available for this receipt.
        </div>
      </div>
    );
  }

  return (
    <div
      className={`bg-[var(--surface)] border border-[var(--border)] p-6 ${className ?? ''}`}
      data-testid="receipt-cost-breakdown"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="text-xs font-mono text-[var(--acid-green)]">
          {'>'} RECEIPT COST BREAKDOWN
        </div>
        {debateId && (
          <span className="text-[10px] font-mono text-[var(--text-muted)]">
            {debateId.slice(0, 12)}...
          </span>
        )}
      </div>

      {/* Summary stats */}
      <SummaryBar
        totalCostUsd={totalCostUsd}
        totalDurationMs={totalDurationMs}
        totalTokensIn={totalTokensIn}
        totalTokensOut={totalTokensOut}
        agentCount={lineItems.length}
        roundsCompleted={roundsCompleted}
      />

      {/* Agent line items */}
      <div className="text-xs font-mono text-[var(--acid-cyan)] mb-3">
        {'>'} AGENT LINE ITEMS ({sortedItems.length})
      </div>
      <div className="space-y-2">
        {sortedItems.map((item) => (
          <AgentRow key={item.agent} item={item} maxCost={maxCost} />
        ))}
      </div>

      {/* Model summary */}
      {showModelSummary && <ModelSummaryTable summaries={modelSummaries} />}
    </div>
  );
}
