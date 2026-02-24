'use client';

import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import {
  useSystemIntelligence,
  useAgentPerformance,
  useInstitutionalMemory,
  useImprovementQueue,
} from '@/hooks/useSystemIntelligence';

export default function SystemIntelligencePage() {
  const { config: _config } = useBackend();

  const { overview, isLoading: overviewLoading } = useSystemIntelligence();
  const { agents: agentPerfAgents, isLoading: agentLoading } = useAgentPerformance();
  const { memory, isLoading: memoryLoading } = useInstitutionalMemory();
  const { items: queueItems, isLoading: queueLoading } = useImprovementQueue();

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
              <Link href="/leaderboard" className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors">
                [LEADERBOARD]
              </Link>
              <Link href="/self-improve" className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors">
                [SELF-IMPROVE]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        <div className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">
              {'>'} SYSTEM INTELLIGENCE
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Aggregated view of agent performance, institutional memory, cross-debate learning,
              and the self-improvement queue. The system&apos;s learning at a glance.
            </p>
          </div>

          {/* Overview Section */}
          <PanelErrorBoundary panelName="Overview">
            <div className="mb-6">
              <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">System Overview</h2>
              {overviewLoading ? (
                <div className="text-acid-green font-mono animate-pulse text-sm">Loading...</div>
              ) : overview ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="p-4 bg-surface border border-border rounded-lg text-center">
                    <div className="text-3xl font-mono font-bold text-acid-green">{overview.totalCycles}</div>
                    <div className="text-xs text-text-muted uppercase">Nomic Cycles</div>
                  </div>
                  <div className="p-4 bg-surface border border-border rounded-lg text-center">
                    <div className="text-3xl font-mono font-bold text-blue-400">
                      {(overview.successRate * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-text-muted uppercase">Success Rate</div>
                  </div>
                  <div className="p-4 bg-surface border border-border rounded-lg text-center">
                    <div className="text-3xl font-mono font-bold text-purple-400">{overview.activeAgents}</div>
                    <div className="text-xs text-text-muted uppercase">Active Agents</div>
                  </div>
                  <div className="p-4 bg-surface border border-border rounded-lg text-center">
                    <div className="text-3xl font-mono font-bold text-gold">{overview.knowledgeItems}</div>
                    <div className="text-xs text-text-muted uppercase">Knowledge Items</div>
                  </div>
                </div>
              ) : (
                <div className="p-4 bg-surface border border-border rounded-lg text-center text-text-muted text-sm">
                  No overview data available.
                </div>
              )}
            </div>
          </PanelErrorBoundary>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Agent Performance */}
            <PanelErrorBoundary panelName="Agent Performance">
              <div className="p-4 bg-surface border border-border rounded-lg">
                <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                  Agent Performance (ELO + Calibration)
                </h2>
                {agentLoading ? (
                  <div className="text-acid-green font-mono animate-pulse text-sm">Loading...</div>
                ) : agentPerfAgents && agentPerfAgents.length > 0 ? (
                  <div className="space-y-2 max-h-[400px] overflow-y-auto">
                    {agentPerfAgents.map((agent) => (
                      <div key={agent.id} className="p-3 bg-bg rounded flex items-center gap-3">
                        <div className="flex-1">
                          <div className="font-mono text-sm text-text">{agent.name}</div>
                          <div className="flex gap-3 text-xs text-text-muted mt-1">
                            {agent.domains.slice(0, 3).map((d) => (
                              <span key={d} className="px-1 py-0.5 bg-surface rounded">{d}</span>
                            ))}
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="font-mono text-sm text-acid-green">{agent.elo}</div>
                          <div className="text-xs text-text-muted">ELO</div>
                        </div>
                        <div className="text-right">
                          <div className="font-mono text-sm text-blue-400">
                            {(agent.calibration * 100).toFixed(0)}%
                          </div>
                          <div className="text-xs text-text-muted">Cal.</div>
                        </div>
                        <div className="text-right">
                          <div className="font-mono text-sm text-purple-400">
                            {(agent.winRate * 100).toFixed(0)}%
                          </div>
                          <div className="text-xs text-text-muted">Win</div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-text-muted text-sm">No agent performance data available.</p>
                )}
              </div>
            </PanelErrorBoundary>

            {/* Institutional Memory */}
            <PanelErrorBoundary panelName="Institutional Memory">
              <div className="p-4 bg-surface border border-border rounded-lg">
                <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                  Institutional Memory
                </h2>
                {memoryLoading ? (
                  <div className="text-acid-green font-mono animate-pulse text-sm">Loading...</div>
                ) : memory ? (
                  <div className="space-y-4">
                    {memory.topPatterns && memory.topPatterns.length > 0 && (
                      <div>
                        <h3 className="text-xs text-text-muted uppercase mb-2">Learned Patterns</h3>
                        <div className="space-y-1">
                          {memory.topPatterns.slice(0, 8).map((p, i) => (
                            <div key={i} className="flex items-center gap-2 text-sm p-2 bg-bg rounded">
                              <span className="text-text flex-1">{p.pattern}</span>
                              <span className="text-xs font-mono text-acid-green">
                                {(p.confidence * 100).toFixed(0)}%
                              </span>
                              <span className="text-xs text-text-muted">{p.frequency}x</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {memory.confidenceChanges && memory.confidenceChanges.length > 0 && (
                      <div>
                        <h3 className="text-xs text-text-muted uppercase mb-2">Confidence Shifts</h3>
                        <div className="space-y-1">
                          {memory.confidenceChanges.slice(0, 5).map((c, i) => (
                            <div key={i} className="flex items-center gap-2 text-sm p-2 bg-bg rounded">
                              <span className="text-text flex-1">{c.topic}</span>
                              <span className="text-red-400 text-xs font-mono">{(c.before * 100).toFixed(0)}%</span>
                              <span className="text-text-muted">&rarr;</span>
                              <span className="text-acid-green text-xs font-mono">{(c.after * 100).toFixed(0)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-text-muted text-sm">No institutional memory data available.</p>
                )}
              </div>
            </PanelErrorBoundary>
          </div>

          {/* Improvement Queue */}
          <PanelErrorBoundary panelName="Improvement Queue">
            <div className="mt-6 p-4 bg-surface border border-border rounded-lg">
              <h2 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                Self-Improvement Queue
              </h2>
              {queueLoading ? (
                <div className="text-acid-green font-mono animate-pulse text-sm">Loading...</div>
              ) : queueItems && queueItems.length > 0 ? (
                <div className="space-y-2 max-h-[300px] overflow-y-auto">
                  {queueItems.map((item, i) => (
                    <div key={i} className="flex items-center gap-3 p-3 bg-bg rounded">
                      <span className={`px-2 py-0.5 text-xs font-mono rounded ${
                        item.priority >= 75 ? 'bg-red-500/20 text-red-400' :
                        item.priority >= 50 ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-blue-400/20 text-blue-400'
                      }`}>
                        {item.priority >= 75 ? 'high' : item.priority >= 50 ? 'medium' : 'low'}
                      </span>
                      <span className="text-sm text-text flex-1">{item.goal}</span>
                      <span className="text-xs text-text-muted">{item.status}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-text-muted text-sm">No items in the improvement queue.</p>
              )}
            </div>
          </PanelErrorBoundary>
        </div>

        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">{'='.repeat(40)}</div>
          <p className="text-text-muted">{'>'} ARAGORA // SYSTEM INTELLIGENCE</p>
        </footer>
      </main>
    </>
  );
}
