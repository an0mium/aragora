'use client';

import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import {
  useQualityScore,
  useOutcomeAgents,
  useDecisionHistory,
  useCalibrationCurve,
} from '@/hooks/useOutcomeAnalytics';

export default function OutcomeDashboardPage() {
  useBackend();

  const { quality, isLoading: qualityLoading } = useQualityScore();
  const { leaderboard: agents, isLoading: agentsLoading } = useOutcomeAgents();
  const { history, isLoading: historyLoading } = useDecisionHistory();
  const { calibration, isLoading: calibrationLoading } = useCalibrationCurve();

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-acid-green';
    if (score >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/">
              <AsciiBannerCompact connected={true} />
            </Link>
            <div className="flex items-center gap-3">
              <Link href="/analytics/decisions" className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors">
                [DECISION ANALYTICS]
              </Link>
              <Link href="/calibration" className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors">
                [CALIBRATION]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        <div className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">
              {'>'} OUTCOME DASHBOARD
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Track decision quality, agent performance with ELO + Brier calibration,
              consensus quality metrics, and calibration accuracy curves.
            </p>
          </div>

          {/* Quality Score Section */}
          <PanelErrorBoundary panelName="Quality Score">
            <div className="mb-6">
              <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">Decision Quality</h2>
              {qualityLoading ? (
                <div className="text-acid-green font-mono animate-pulse text-sm">Loading...</div>
              ) : quality ? (
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  <div className="p-4 bg-surface border border-border rounded-lg text-center">
                    <div className={`text-3xl font-mono font-bold ${getScoreColor(quality.quality_score)}`}>
                      {(quality.quality_score * 100).toFixed(0)}
                    </div>
                    <div className="text-xs text-text-muted uppercase">Quality Score</div>
                  </div>
                  <div className="p-4 bg-surface border border-border rounded-lg text-center">
                    <div className="text-3xl font-mono font-bold text-acid-green">
                      {(quality.consensus_rate * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-text-muted uppercase">Consensus Rate</div>
                  </div>
                  <div className="p-4 bg-surface border border-border rounded-lg text-center">
                    <div className="text-3xl font-mono font-bold text-blue-400">{quality.avg_rounds.toFixed(1)}</div>
                    <div className="text-xs text-text-muted uppercase">Avg Rounds</div>
                  </div>
                  <div className="p-4 bg-surface border border-border rounded-lg text-center">
                    <div className="text-3xl font-mono font-bold text-purple-400">{quality.total_decisions}</div>
                    <div className="text-xs text-text-muted uppercase">Total Decisions</div>
                  </div>
                  <div className="p-4 bg-surface border border-border rounded-lg text-center">
                    <div className="text-3xl font-mono font-bold text-gold">
                      {(quality.completion_rate * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-text-muted uppercase">Completion Rate</div>
                  </div>
                </div>
              ) : (
                <p className="text-text-muted text-sm p-4 bg-surface border border-border rounded-lg">
                  No quality data available. Complete some debates to see metrics.
                </p>
              )}
            </div>
          </PanelErrorBoundary>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Agent Leaderboard */}
            <PanelErrorBoundary panelName="Agent Leaderboard">
              <div className="p-4 bg-surface border border-border rounded-lg">
                <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                  Agent Leaderboard (ELO + Brier)
                </h2>
                {agentsLoading ? (
                  <div className="text-acid-green font-mono animate-pulse text-sm">Loading...</div>
                ) : agents?.agents && agents.agents.length > 0 ? (
                  <div className="space-y-2 max-h-[400px] overflow-y-auto">
                    {agents.agents.map((agent: { agent_id: string; rank: number; agent_name: string; provider: string; model: string; elo: number; elo_change: number; brier_score: number | null; win_rate: number }) => (
                      <div key={agent.agent_id} className="flex items-center gap-3 p-3 bg-bg rounded">
                        <span className="text-xs font-mono text-text-muted w-6">#{agent.rank}</span>
                        <div className="flex-1">
                          <div className="font-mono text-sm text-text">{agent.agent_name}</div>
                          <div className="text-xs text-text-muted">{agent.provider}/{agent.model}</div>
                        </div>
                        <div className="text-right">
                          <div className="font-mono text-sm text-acid-green">{agent.elo}</div>
                          <div className="text-xs text-text-muted">
                            {agent.elo_change >= 0 ? '+' : ''}{agent.elo_change}
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="font-mono text-sm text-blue-400">
                            {agent.brier_score !== null ? agent.brier_score.toFixed(3) : 'N/A'}
                          </div>
                          <div className="text-xs text-text-muted">Brier</div>
                        </div>
                        <div className="text-right">
                          <div className="font-mono text-sm text-purple-400">
                            {(agent.win_rate * 100).toFixed(0)}%
                          </div>
                          <div className="text-xs text-text-muted">Win</div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-text-muted text-sm">No agent data available.</p>
                )}
              </div>
            </PanelErrorBoundary>

            {/* Calibration Curve */}
            <PanelErrorBoundary panelName="Calibration Curve">
              <div className="p-4 bg-surface border border-border rounded-lg">
                <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                  Calibration Curve
                </h2>
                {calibrationLoading ? (
                  <div className="text-acid-green font-mono animate-pulse text-sm">Loading...</div>
                ) : calibration?.bins && calibration.bins.length > 0 ? (
                  <div>
                    <div className="mb-3 text-xs text-text-muted">
                      ECE: {calibration.ece?.toFixed(4) || 'N/A'} | Bins: {calibration.bins.length}
                    </div>
                    <div className="space-y-1">
                      {calibration.bins.map((bin, i) => {
                        const predicted = bin.predicted_confidence * 100;
                        const actual = bin.actual_accuracy * 100;
                        const gap = Math.abs(predicted - actual);
                        return (
                          <div key={i} className="flex items-center gap-2 text-xs">
                            <span className="w-12 text-text-muted font-mono">{predicted.toFixed(0)}%</span>
                            <div className="flex-1 h-4 bg-bg rounded overflow-hidden relative">
                              <div
                                className="h-full bg-acid-green/30 absolute"
                                style={{ width: `${predicted}%` }}
                              />
                              <div
                                className="h-full bg-blue-400/50 absolute"
                                style={{ width: `${actual}%` }}
                              />
                            </div>
                            <span className="w-12 text-text font-mono">{actual.toFixed(0)}%</span>
                            <span className={`w-10 font-mono ${gap > 10 ? 'text-red-400' : 'text-acid-green'}`}>
                              {gap.toFixed(0)}%
                            </span>
                          </div>
                        );
                      })}
                    </div>
                    <div className="flex items-center gap-4 mt-3 text-xs text-text-muted">
                      <span><span className="inline-block w-3 h-2 bg-acid-green/30 rounded mr-1" />Predicted</span>
                      <span><span className="inline-block w-3 h-2 bg-blue-400/50 rounded mr-1" />Actual</span>
                    </div>
                  </div>
                ) : (
                  <p className="text-text-muted text-sm">No calibration data available yet.</p>
                )}
              </div>
            </PanelErrorBoundary>
          </div>

          {/* Decision History */}
          <PanelErrorBoundary panelName="Decision History">
            <div className="mt-6 p-4 bg-surface border border-border rounded-lg">
              <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                Recent Decision History
              </h2>
              {historyLoading ? (
                <div className="text-acid-green font-mono animate-pulse text-sm">Loading...</div>
              ) : history?.decisions && history.decisions.length > 0 ? (
                <div className="space-y-2 max-h-[300px] overflow-y-auto">
                  {history.decisions.map((d) => (
                    <Link
                      key={d.debate_id}
                      href={`/explainability?debate=${d.debate_id}`}
                      className="flex items-center gap-3 p-3 bg-bg rounded hover:border-acid-green/50 border border-transparent transition-colors"
                    >
                      <span className={`px-2 py-0.5 text-xs font-mono rounded ${
                        d.consensus_reached
                          ? 'bg-acid-green/20 text-acid-green'
                          : 'bg-yellow-500/20 text-yellow-400'
                      }`}>
                        {d.status}
                      </span>
                      <span className="text-sm text-text flex-1 line-clamp-1">{d.task}</span>
                      <span className={`text-xs font-mono ${getScoreColor(d.quality_score)}`}>
                        Q:{(d.quality_score * 100).toFixed(0)}
                      </span>
                      <span className="text-xs text-text-muted">{d.rounds}R</span>
                      <span className="text-xs text-text-muted">
                        {new Date(d.created_at).toLocaleDateString()}
                      </span>
                    </Link>
                  ))}
                </div>
              ) : (
                <p className="text-text-muted text-sm">No decision history available.</p>
              )}
            </div>
          </PanelErrorBoundary>
        </div>

        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">{'='.repeat(40)}</div>
          <p className="text-text-muted">{'>'} ARAGORA // OUTCOME DASHBOARD</p>
        </footer>
      </main>
    </>
  );
}
