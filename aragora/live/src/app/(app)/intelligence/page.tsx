'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useAuth } from '@/context/AuthContext';
import { UsageChart } from '@/components/admin/UsageChart';
import type { DataPoint } from '@/components/admin/UsageChart';

// ---------------------------------------------------------------------------
// Mock data -- will be replaced by real API calls once endpoints are wired
// ---------------------------------------------------------------------------

interface AgentPerformance {
  name: string;
  elo: number;
  winRate: number;
  acceptanceRate: number;
  totalDebates: number;
}

const MOCK_TOP_AGENTS: AgentPerformance[] = [
  { name: 'claude-3-opus', elo: 1847, winRate: 0.72, acceptanceRate: 0.81, totalDebates: 412 },
  { name: 'gpt-4-turbo', elo: 1793, winRate: 0.68, acceptanceRate: 0.76, totalDebates: 389 },
  { name: 'gemini-pro', elo: 1721, winRate: 0.61, acceptanceRate: 0.69, totalDebates: 274 },
  { name: 'mistral-large', elo: 1688, winRate: 0.58, acceptanceRate: 0.65, totalDebates: 198 },
  { name: 'grok-2', elo: 1654, winRate: 0.55, acceptanceRate: 0.62, totalDebates: 153 },
];

const MOCK_TOPICS = [
  { topic: 'Technical Architecture', count: 87, consensusRate: 0.82, avgConfidence: 0.88 },
  { topic: 'Risk Assessment', count: 64, consensusRate: 0.71, avgConfidence: 0.79 },
  { topic: 'Product Strategy', count: 52, consensusRate: 0.65, avgConfidence: 0.74 },
  { topic: 'Hiring Decisions', count: 41, consensusRate: 0.58, avgConfidence: 0.71 },
  { topic: 'Budget Allocation', count: 38, consensusRate: 0.76, avgConfidence: 0.83 },
  { topic: 'Compliance Review', count: 29, consensusRate: 0.89, avgConfidence: 0.91 },
];

const MOCK_INSIGHTS = [
  {
    id: 'insight-1',
    text: 'Teams reach consensus 23% faster on technical decisions when including a devil\'s advocate agent.',
    category: 'consensus',
    impact: 'high' as const,
  },
  {
    id: 'insight-2',
    text: 'Risk assessments with 4+ agents have 15% higher confidence than those with 2.',
    category: 'confidence',
    impact: 'high' as const,
  },
  {
    id: 'insight-3',
    text: 'Monday decisions have lower confidence scores than Friday decisions on average.',
    category: 'timing',
    impact: 'medium' as const,
  },
  {
    id: 'insight-4',
    text: 'Debates that reference KnowledgeMound entries reach consensus in 1.4 fewer rounds.',
    category: 'knowledge',
    impact: 'high' as const,
  },
];

const MOCK_NOMIC = {
  latestCycleStatus: 'completed' as const,
  latestCycleId: 'nomic-cycle-47',
  goalsCompletedThisMonth: 12,
  kmEntriesAdded: 847,
  lastRunTimestamp: '2026-02-21T14:32:00Z',
};

function generateMockChartData(days: number, min: number, max: number): DataPoint[] {
  return Array.from({ length: days }, (_, i) => {
    const date = new Date();
    date.setDate(date.getDate() - (days - 1 - i));
    return {
      label: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      value: Math.floor(Math.random() * (max - min)) + min,
      date: date.toISOString(),
    };
  });
}

const agentQualityData = generateMockChartData(30, 60, 95);
const consensusRateData = generateMockChartData(30, 50, 90);

// ---------------------------------------------------------------------------
// Enterprise gate overlay
// ---------------------------------------------------------------------------

function EnterpriseGateOverlay() {
  return (
    <div className="absolute inset-0 z-40 bg-[var(--bg)]/80 backdrop-blur-sm flex items-center justify-center">
      <div className="card p-8 max-w-md text-center border-acid-yellow/40 bg-[var(--surface)]">
        <div className="font-mono text-4xl text-acid-yellow mb-4">{'>'}_</div>
        <h2 className="font-mono text-xl text-acid-green mb-3">
          Decision Intelligence
        </h2>
        <p className="font-mono text-sm text-[var(--text-muted)] mb-6">
          Requires Enterprise Plan. Unlock AI-powered decision analytics,
          agent performance insights, and self-improvement tracking.
        </p>
        <Link
          href="/settings/billing"
          className="inline-block px-6 py-3 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors"
        >
          Upgrade to Enterprise
        </Link>
        <p className="font-mono text-xs text-[var(--text-muted)] mt-4">
          or contact sales@aragora.ai for a demo
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function ImpactBadge({ impact }: { impact: 'high' | 'medium' | 'low' }) {
  const styles = {
    high: 'bg-acid-green/20 text-acid-green border-acid-green/40',
    medium: 'bg-acid-yellow/20 text-acid-yellow border-acid-yellow/40',
    low: 'bg-[var(--text-muted)]/20 text-[var(--text-muted)] border-[var(--text-muted)]/40',
  };
  return (
    <span className={`px-2 py-0.5 text-[10px] font-mono rounded border ${styles[impact]}`}>
      {impact.toUpperCase()}
    </span>
  );
}

function NomicStatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    completed: 'bg-acid-green/20 text-acid-green border-acid-green/40',
    running: 'bg-acid-cyan/20 text-acid-cyan border-acid-cyan/40 animate-pulse',
    failed: 'bg-red-500/20 text-red-400 border-red-500/40',
    pending: 'bg-acid-yellow/20 text-acid-yellow border-acid-yellow/40',
  };
  return (
    <span className={`px-2 py-0.5 text-xs font-mono rounded border ${styles[status] || styles.pending}`}>
      {status.toUpperCase()}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function DecisionIntelligencePage() {
  const { organization } = useAuth();
  const [activeSection, setActiveSection] = useState<string | null>(null);

  const isEnterprise =
    organization?.tier === 'enterprise' ||
    organization?.tier === 'enterprise_plus';

  const sections = [
    { id: 'agents', label: 'AGENT PERFORMANCE' },
    { id: 'patterns', label: 'DECISION PATTERNS' },
    { id: 'insights', label: 'IMPROVEMENT INSIGHTS' },
    { id: 'nomic', label: 'NOMIC LOOP STATUS' },
  ];

  return (
    <div className="relative min-h-[calc(100vh-4rem)]">
      {/* Enterprise gate */}
      {!isEnterprise && <EnterpriseGateOverlay />}

      <div className="container mx-auto px-4 py-6">
        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <h1 className="text-xl font-mono text-[var(--acid-green)] mb-2">
                {'>'} DECISION INTELLIGENCE
              </h1>
              <p className="text-xs text-[var(--text-muted)] font-mono">
                AI-powered analytics across agent performance, decision patterns, and self-improvement cycles.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="px-2 py-1 text-xs font-mono bg-acid-magenta/20 text-acid-magenta border border-acid-magenta/30">
                ENTERPRISE
              </span>
            </div>
          </div>
        </div>

        {/* Section Navigation */}
        <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
          {sections.map((s) => (
            <button
              key={s.id}
              onClick={() => setActiveSection(activeSection === s.id ? null : s.id)}
              className={`px-3 py-1.5 text-xs font-mono rounded border transition-colors whitespace-nowrap ${
                activeSection === s.id || activeSection === null
                  ? 'bg-[var(--acid-green)]/10 text-[var(--acid-green)] border-[var(--acid-green)]/30'
                  : 'text-[var(--text-muted)] border-transparent hover:border-[var(--acid-green)]/20'
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>

        {/* ============================================================= */}
        {/* SECTION 1: Agent Performance                                  */}
        {/* ============================================================= */}
        {(activeSection === null || activeSection === 'agents') && (
          <section className="mb-8">
            <h2 className="font-mono text-sm text-[var(--acid-cyan)] mb-4">
              {'>'} AGENT PERFORMANCE
            </h2>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Top agents table */}
              <div className="lg:col-span-2 bg-[var(--surface)] border border-[var(--border)] p-4">
                <h3 className="font-mono text-xs text-[var(--text-muted)] mb-3">TOP 5 AGENTS BY ACCEPTANCE RATE</h3>
                <div className="overflow-x-auto">
                  <table className="w-full font-mono text-sm">
                    <thead>
                      <tr className="border-b border-[var(--acid-green)]/20 text-[var(--text-muted)] text-xs">
                        <th className="text-left py-2 pr-4">#</th>
                        <th className="text-left py-2 pr-4">AGENT</th>
                        <th className="text-right py-2 pr-4">ELO</th>
                        <th className="text-right py-2 pr-4">WIN RATE</th>
                        <th className="text-right py-2 pr-4">ACCEPT %</th>
                        <th className="text-right py-2">DEBATES</th>
                      </tr>
                    </thead>
                    <tbody>
                      {MOCK_TOP_AGENTS.map((agent, idx) => (
                        <tr
                          key={agent.name}
                          className="border-b border-[var(--acid-green)]/10 hover:bg-[var(--acid-green)]/5 transition-colors"
                        >
                          <td className="py-2 pr-4 text-[var(--text-muted)]">{idx + 1}</td>
                          <td className="py-2 pr-4 text-[var(--acid-green)]">{agent.name}</td>
                          <td className="py-2 pr-4 text-right text-[var(--acid-cyan)]">{agent.elo}</td>
                          <td className="py-2 pr-4 text-right text-[var(--text)]">
                            {(agent.winRate * 100).toFixed(0)}%
                          </td>
                          <td className="py-2 pr-4 text-right text-[var(--acid-yellow)]">
                            {(agent.acceptanceRate * 100).toFixed(0)}%
                          </td>
                          <td className="py-2 text-right text-[var(--text-muted)]">{agent.totalDebates}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Agent quality trend chart */}
              <div>
                <UsageChart
                  title="AGENT QUALITY TREND"
                  data={agentQualityData}
                  type="line"
                  color="acid-green"
                  height={260}
                  formatValue={(v) => `${v}%`}
                />
              </div>
            </div>
          </section>
        )}

        {/* ============================================================= */}
        {/* SECTION 2: Decision Patterns                                  */}
        {/* ============================================================= */}
        {(activeSection === null || activeSection === 'patterns') && (
          <section className="mb-8">
            <h2 className="font-mono text-sm text-[var(--acid-cyan)] mb-4">
              {'>'} DECISION PATTERNS
            </h2>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Topic list with bar visualization */}
              <div className="lg:col-span-2 bg-[var(--surface)] border border-[var(--border)] p-4">
                <h3 className="font-mono text-xs text-[var(--text-muted)] mb-3">MOST COMMON DEBATE TOPICS</h3>
                <div className="space-y-3">
                  {MOCK_TOPICS.map((t) => {
                    const maxCount = MOCK_TOPICS[0].count;
                    const barWidth = (t.count / maxCount) * 100;
                    return (
                      <div key={t.topic} className="group">
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-mono text-sm text-[var(--text)] group-hover:text-[var(--acid-green)] transition-colors">
                            {t.topic}
                          </span>
                          <div className="flex items-center gap-4">
                            <span className="font-mono text-xs text-[var(--text-muted)]">
                              {t.count} debates
                            </span>
                            <span className="font-mono text-xs text-[var(--acid-cyan)]">
                              {(t.avgConfidence * 100).toFixed(0)}% conf
                            </span>
                          </div>
                        </div>
                        <div className="h-2 bg-[var(--bg)] rounded overflow-hidden">
                          <div
                            className="h-full bg-[var(--acid-green)]/60 rounded transition-all group-hover:bg-[var(--acid-green)]/80"
                            style={{ width: `${barWidth}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Confidence by category */}
                <div className="mt-6 pt-4 border-t border-[var(--acid-green)]/20">
                  <h3 className="font-mono text-xs text-[var(--text-muted)] mb-3">AVERAGE CONFIDENCE BY CATEGORY</h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {MOCK_TOPICS.map((t) => (
                      <div key={t.topic} className="bg-[var(--bg)] border border-[var(--border)] p-3">
                        <div className="font-mono text-[10px] text-[var(--text-muted)] mb-1 truncate">
                          {t.topic}
                        </div>
                        <div className="font-mono text-lg text-[var(--acid-green)]">
                          {(t.avgConfidence * 100).toFixed(0)}%
                        </div>
                        <div className="font-mono text-[10px] text-[var(--text-muted)]">
                          consensus {(t.consensusRate * 100).toFixed(0)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Consensus rate trend chart */}
              <div>
                <UsageChart
                  title="CONSENSUS RATE TREND"
                  data={consensusRateData}
                  type="line"
                  color="acid-cyan"
                  height={260}
                  formatValue={(v) => `${v}%`}
                />
              </div>
            </div>
          </section>
        )}

        {/* ============================================================= */}
        {/* SECTION 3: Improvement Insights                               */}
        {/* ============================================================= */}
        {(activeSection === null || activeSection === 'insights') && (
          <section className="mb-8">
            <h2 className="font-mono text-sm text-[var(--acid-cyan)] mb-4">
              {'>'} IMPROVEMENT INSIGHTS
            </h2>

            <div className="bg-[var(--surface)] border border-[var(--border)] p-6">
              <p className="font-mono text-sm text-[var(--text-muted)] mb-4">
                Aragora analyzed your last 100 debates and found:
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {MOCK_INSIGHTS.map((insight) => (
                  <div
                    key={insight.id}
                    className="bg-[var(--bg)] border border-[var(--border)] p-4 hover:border-[var(--acid-green)]/40 transition-colors"
                  >
                    <div className="flex items-start justify-between gap-3 mb-2">
                      <span className="font-mono text-xs text-[var(--acid-yellow)] uppercase">
                        {insight.category}
                      </span>
                      <ImpactBadge impact={insight.impact} />
                    </div>
                    <p className="font-mono text-sm text-[var(--text)] leading-relaxed">
                      {insight.text}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {/* ============================================================= */}
        {/* SECTION 4: Nomic Loop Status                                  */}
        {/* ============================================================= */}
        {(activeSection === null || activeSection === 'nomic') && (
          <section className="mb-8">
            <h2 className="font-mono text-sm text-[var(--acid-cyan)] mb-4">
              {'>'} NOMIC LOOP STATUS
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Latest cycle */}
              <div className="bg-[var(--surface)] border border-[var(--border)] p-6">
                <div className="font-mono text-xs text-[var(--text-muted)] mb-2">Latest Cycle</div>
                <div className="flex items-center gap-3 mb-2">
                  <span className="font-mono text-lg text-[var(--acid-green)]">
                    {MOCK_NOMIC.latestCycleId}
                  </span>
                  <NomicStatusBadge status={MOCK_NOMIC.latestCycleStatus} />
                </div>
                <div className="font-mono text-xs text-[var(--text-muted)]">
                  {new Date(MOCK_NOMIC.lastRunTimestamp).toLocaleString()}
                </div>
              </div>

              {/* Goals completed */}
              <div className="bg-[var(--surface)] border border-[var(--border)] p-6">
                <div className="font-mono text-xs text-[var(--text-muted)] mb-2">Goals Completed This Month</div>
                <div className="font-mono text-4xl text-[var(--acid-cyan)]">
                  {MOCK_NOMIC.goalsCompletedThisMonth}
                </div>
                <div className="font-mono text-xs text-[var(--text-muted)] mt-1">
                  self-improvement goals executed
                </div>
              </div>

              {/* KM entries */}
              <div className="bg-[var(--surface)] border border-[var(--border)] p-6">
                <div className="font-mono text-xs text-[var(--text-muted)] mb-2">KnowledgeMound Entries Added</div>
                <div className="font-mono text-4xl text-[var(--acid-yellow)]">
                  {MOCK_NOMIC.kmEntriesAdded.toLocaleString()}
                </div>
                <div className="font-mono text-xs text-[var(--text-muted)] mt-1">
                  across 34 adapters
                </div>
              </div>
            </div>

            {/* Nomic phases */}
            <div className="bg-[var(--surface)] border border-[var(--border)] p-4 mt-4">
              <h3 className="font-mono text-xs text-[var(--text-muted)] mb-3">SELF-IMPROVEMENT PIPELINE</h3>
              <div className="flex items-center gap-2 overflow-x-auto pb-2">
                {['Context', 'Debate', 'Design', 'Implement', 'Verify'].map((phase, idx) => (
                  <div key={phase} className="flex items-center gap-2">
                    <div className="flex flex-col items-center min-w-[80px]">
                      <div
                        className={`w-8 h-8 rounded-full flex items-center justify-center font-mono text-xs border ${
                          idx <= 4
                            ? 'bg-[var(--acid-green)]/20 text-[var(--acid-green)] border-[var(--acid-green)]/40'
                            : 'bg-[var(--surface)] text-[var(--text-muted)] border-[var(--acid-green)]/20'
                        }`}
                      >
                        {idx}
                      </div>
                      <span className="font-mono text-[10px] text-[var(--text-muted)] mt-1">{phase}</span>
                    </div>
                    {idx < 4 && (
                      <div className="w-8 h-px bg-[var(--acid-green)]/30 flex-shrink-0" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
