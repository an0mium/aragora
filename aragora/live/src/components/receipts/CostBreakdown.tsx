'use client';

import { useState, useMemo } from 'react';

export interface AgentCostEntry {
  agent: string;
  cost_usd: number;
  tokens_in: number;
  tokens_out: number;
  call_count: number;
  model?: string;
  provider?: string;
  latency_ms?: number;
}

export interface ModelUsageEntry {
  model: string;
  provider?: string;
  call_count: number;
  tokens_in: number;
  tokens_out: number;
  cost_usd: number;
  avg_latency_ms?: number;
}

export interface RoundCostEntry {
  round: number;
  cost_usd: number;
  tokens_in: number;
  tokens_out: number;
  call_count: number;
  duration_ms?: number;
}

export interface CostBreakdownData {
  total_cost_usd: number;
  total_tokens_in: number;
  total_tokens_out: number;
  total_calls: number;
  duration_seconds?: number;
  per_agent: Record<string, {
    cost_usd?: number;
    tokens_in?: number;
    tokens_out?: number;
    call_count?: number;
    model?: string;
    provider?: string;
    latency_ms?: number;
  }>;
  model_usage?: Record<string, {
    call_count?: number;
    tokens_in?: number;
    tokens_out?: number;
    cost_usd?: number;
    avg_latency_ms?: number;
    provider?: string;
  }>;
  per_round?: RoundCostEntry[];
}

export interface CostBreakdownProps {
  data: CostBreakdownData;
  className?: string;
}

type TabKey = 'summary' | 'agents' | 'models' | 'rounds';

function formatCost(usd: number): string {
  if (usd < 0.01 && usd > 0) return `$${usd.toFixed(4)}`;
  return `$${usd.toFixed(2)}`;
}

function formatTokens(count: number): string {
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}K`;
  return count.toLocaleString();
}

function formatLatency(ms: number): string {
  if (ms >= 1_000) return `${(ms / 1_000).toFixed(1)}s`;
  return `${Math.round(ms)}ms`;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m ${secs.toFixed(0)}s`;
}

function CostBar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div className="h-2 w-full bg-surface rounded overflow-hidden" data-testid="cost-bar">
      <div className={`h-full ${color} rounded`} style={{ width: `${pct}%` }} />
    </div>
  );
}

export function CostBreakdown({ data, className = '' }: CostBreakdownProps) {
  const [activeTab, setActiveTab] = useState<TabKey>('summary');

  const agentEntries: AgentCostEntry[] = useMemo(() => {
    if (!data.per_agent) return [];
    return Object.entries(data.per_agent)
      .map(([agent, info]) => ({
        agent,
        cost_usd: info.cost_usd ?? 0,
        tokens_in: info.tokens_in ?? 0,
        tokens_out: info.tokens_out ?? 0,
        call_count: info.call_count ?? 0,
        model: info.model,
        provider: info.provider,
        latency_ms: info.latency_ms,
      }))
      .sort((a, b) => b.cost_usd - a.cost_usd);
  }, [data.per_agent]);

  const modelEntries: ModelUsageEntry[] = useMemo(() => {
    if (!data.model_usage) return [];
    return Object.entries(data.model_usage)
      .map(([model, info]) => ({
        model,
        provider: info.provider,
        call_count: info.call_count ?? 0,
        tokens_in: info.tokens_in ?? 0,
        tokens_out: info.tokens_out ?? 0,
        cost_usd: info.cost_usd ?? 0,
        avg_latency_ms: info.avg_latency_ms,
      }))
      .sort((a, b) => b.cost_usd - a.cost_usd);
  }, [data.model_usage]);

  const roundEntries = data.per_round ?? [];
  const maxAgentCost = agentEntries.length > 0 ? Math.max(...agentEntries.map((a) => a.cost_usd)) : 0;
  const maxModelCost = modelEntries.length > 0 ? Math.max(...modelEntries.map((m) => m.cost_usd)) : 0;

  const totalTokens = data.total_tokens_in + data.total_tokens_out;
  const costPerCall = data.total_calls > 0 ? data.total_cost_usd / data.total_calls : 0;
  const costPerToken = totalTokens > 0 ? data.total_cost_usd / totalTokens : 0;

  const tabs: { key: TabKey; label: string; count?: number }[] = [
    { key: 'summary', label: 'Summary' },
    { key: 'agents', label: 'Agents', count: agentEntries.length },
    { key: 'models', label: 'Models', count: modelEntries.length },
  ];
  if (roundEntries.length > 0) {
    tabs.push({ key: 'rounds', label: 'Rounds', count: roundEntries.length });
  }

  return (
    <div className={`bg-bg border border-border rounded-lg overflow-hidden ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-border bg-surface/30">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-acid-green font-mono text-sm font-bold">COST BREAKDOWN</span>
          <span className="text-text-muted font-mono text-xs">Line-item receipt</span>
        </div>
        <div className="flex items-center gap-6 mt-3">
          <div>
            <div className="text-xs font-mono text-text-muted">TOTAL COST</div>
            <div className="text-2xl font-mono font-bold text-acid-green" data-testid="total-cost">
              {formatCost(data.total_cost_usd)}
            </div>
          </div>
          <div>
            <div className="text-xs font-mono text-text-muted">TOKENS</div>
            <div className="text-lg font-mono font-bold text-acid-cyan" data-testid="total-tokens">
              {formatTokens(totalTokens)}
            </div>
          </div>
          <div>
            <div className="text-xs font-mono text-text-muted">API CALLS</div>
            <div className="text-lg font-mono font-bold text-text" data-testid="total-calls">
              {data.total_calls}
            </div>
          </div>
          {data.duration_seconds !== undefined && (
            <div>
              <div className="text-xs font-mono text-text-muted">DURATION</div>
              <div className="text-lg font-mono font-bold text-text" data-testid="duration">
                {formatDuration(data.duration_seconds)}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-border">
        <div className="flex">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              data-testid={`tab-${tab.key}`}
              onClick={() => setActiveTab(tab.key)}
              className={`flex-1 px-4 py-3 text-sm font-mono border-b-2 transition-colors ${
                activeTab === tab.key
                  ? 'border-acid-green text-acid-green bg-surface/30'
                  : 'border-transparent text-text-muted hover:text-text'
              }`}
            >
              {tab.label}
              {tab.count !== undefined && (
                <span className="ml-1 text-xs opacity-70">({tab.count})</span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="p-4 max-h-96 overflow-y-auto">
        {/* Summary Tab */}
        {activeTab === 'summary' && (
          <div className="space-y-4">
            {/* Token breakdown */}
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 rounded border border-border bg-surface/20">
                <div className="text-xs font-mono text-text-muted mb-1">TOKENS IN</div>
                <div className="text-sm font-mono text-text">{formatTokens(data.total_tokens_in)}</div>
              </div>
              <div className="p-3 rounded border border-border bg-surface/20">
                <div className="text-xs font-mono text-text-muted mb-1">TOKENS OUT</div>
                <div className="text-sm font-mono text-text">{formatTokens(data.total_tokens_out)}</div>
              </div>
            </div>

            {/* Efficiency metrics */}
            <div className="p-3 rounded border border-border bg-surface/20">
              <div className="text-xs font-mono text-text-muted mb-2">EFFICIENCY METRICS</div>
              <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                <div>
                  <span className="text-text-muted">Cost / call:</span>{' '}
                  <span className="text-acid-cyan">{formatCost(costPerCall)}</span>
                </div>
                <div>
                  <span className="text-text-muted">Cost / 1K tokens:</span>{' '}
                  <span className="text-acid-cyan">{formatCost(costPerToken * 1000)}</span>
                </div>
                <div>
                  <span className="text-text-muted">Agents used:</span>{' '}
                  <span className="text-text">{agentEntries.length}</span>
                </div>
                <div>
                  <span className="text-text-muted">Models used:</span>{' '}
                  <span className="text-text">{modelEntries.length}</span>
                </div>
              </div>
            </div>

            {/* Top agent by cost */}
            {agentEntries.length > 0 && (
              <div className="p-3 rounded border border-border bg-surface/20">
                <div className="text-xs font-mono text-text-muted mb-2">TOP AGENT BY COST</div>
                <div className="flex items-center justify-between text-sm font-mono">
                  <span className="text-acid-cyan">{agentEntries[0].agent}</span>
                  <span className="text-acid-green">{formatCost(agentEntries[0].cost_usd)}</span>
                </div>
                {agentEntries[0].model && (
                  <div className="text-xs font-mono text-text-muted mt-1">
                    Model: {agentEntries[0].model}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Agents Tab */}
        {activeTab === 'agents' && (
          <div className="space-y-3">
            {agentEntries.length === 0 ? (
              <div className="text-center py-6 text-text-muted font-mono text-sm">
                No per-agent cost data available
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="text-text-muted border-b border-border">
                      <th className="text-left py-2 pr-3">Agent</th>
                      <th className="text-right py-2 px-2">Cost</th>
                      <th className="text-right py-2 px-2">Tokens In</th>
                      <th className="text-right py-2 px-2">Tokens Out</th>
                      <th className="text-right py-2 px-2">Calls</th>
                      <th className="text-left py-2 px-2">Model</th>
                      {agentEntries.some((a) => a.latency_ms !== undefined) && (
                        <th className="text-right py-2 pl-2">Latency</th>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {agentEntries.map((entry) => (
                      <tr key={entry.agent} className="border-b border-border/50 hover:bg-surface/30">
                        <td className="py-2 pr-3">
                          <div className="text-acid-cyan">{entry.agent}</div>
                          {entry.provider && (
                            <div className="text-text-muted text-[10px]">{entry.provider}</div>
                          )}
                        </td>
                        <td className="text-right py-2 px-2">
                          <div className="text-acid-green">{formatCost(entry.cost_usd)}</div>
                          <CostBar value={entry.cost_usd} max={maxAgentCost} color="bg-acid-green" />
                        </td>
                        <td className="text-right py-2 px-2 text-text">{formatTokens(entry.tokens_in)}</td>
                        <td className="text-right py-2 px-2 text-text">{formatTokens(entry.tokens_out)}</td>
                        <td className="text-right py-2 px-2 text-text">{entry.call_count}</td>
                        <td className="py-2 px-2 text-text-muted">{entry.model ?? '-'}</td>
                        {agentEntries.some((a) => a.latency_ms !== undefined) && (
                          <td className="text-right py-2 pl-2 text-text">
                            {entry.latency_ms !== undefined ? formatLatency(entry.latency_ms) : '-'}
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Models Tab */}
        {activeTab === 'models' && (
          <div className="space-y-3">
            {modelEntries.length === 0 ? (
              <div className="text-center py-6 text-text-muted font-mono text-sm">
                No model usage data available
              </div>
            ) : (
              modelEntries.map((entry) => (
                <div
                  key={entry.model}
                  className="p-3 rounded border border-border bg-surface/20"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div>
                      <span className="text-sm font-mono text-acid-cyan font-bold">
                        {entry.model}
                      </span>
                      {entry.provider && (
                        <span className="text-xs font-mono text-text-muted ml-2">
                          ({entry.provider})
                        </span>
                      )}
                    </div>
                    <span className="text-sm font-mono text-acid-green font-bold">
                      {formatCost(entry.cost_usd)}
                    </span>
                  </div>
                  <CostBar value={entry.cost_usd} max={maxModelCost} color="bg-acid-cyan" />
                  <div className="flex items-center gap-4 mt-2 text-xs font-mono text-text-muted">
                    <span>{entry.call_count} calls</span>
                    <span>{formatTokens(entry.tokens_in)} in</span>
                    <span>{formatTokens(entry.tokens_out)} out</span>
                    {entry.avg_latency_ms !== undefined && (
                      <span>avg {formatLatency(entry.avg_latency_ms)}</span>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {/* Rounds Tab */}
        {activeTab === 'rounds' && (
          <div className="space-y-2">
            {roundEntries.map((entry) => (
              <div
                key={entry.round}
                className="flex items-center gap-4 p-3 rounded border border-border bg-surface/20 text-xs font-mono"
              >
                <span className="text-acid-cyan font-bold w-20">Round {entry.round}</span>
                <span className="text-acid-green w-16 text-right">{formatCost(entry.cost_usd)}</span>
                <span className="text-text-muted">{entry.call_count} calls</span>
                <span className="text-text-muted">{formatTokens(entry.tokens_in + entry.tokens_out)} tokens</span>
                {entry.duration_ms !== undefined && (
                  <span className="text-text-muted">{formatLatency(entry.duration_ms)}</span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-border bg-surface/30 flex items-center justify-between text-[10px] font-mono text-text-muted">
        <span>Costs estimated from provider token pricing</span>
        <span>{data.total_calls} API call{data.total_calls !== 1 ? 's' : ''} total</span>
      </div>
    </div>
  );
}

export default CostBreakdown;
