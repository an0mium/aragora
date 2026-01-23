'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { ExecutiveSummary } from '@/components/dashboard/ExecutiveSummary';
import { useRightSidebar } from '@/context/RightSidebarContext';
import { fetchRecentDebates, type DebateArtifact } from '@/utils/supabase';
import { getAgentColors } from '@/utils/agentColors';
import { logger } from '@/utils/logger';

export default function DashboardPage() {
  const [recentDebates, setRecentDebates] = useState<DebateArtifact[]>([]);
  const [loadingDebates, setLoadingDebates] = useState(true);

  const { setContext, clearContext } = useRightSidebar();

  // Load recent debates
  useEffect(() => {
    async function loadDebates() {
      try {
        setLoadingDebates(true);
        const data = await fetchRecentDebates(5);
        setRecentDebates(data);
      } catch (e) {
        logger.error('Failed to load recent debates:', e);
      } finally {
        setLoadingDebates(false);
      }
    }

    loadDebates();
  }, []);

  // Set up right sidebar
  useEffect(() => {
    setContext({
      title: 'Executive Dashboard',
      subtitle: 'Real-time KPIs',
      statsContent: (
        <div className="space-y-3">
          <div className="text-xs text-[var(--text-muted)] font-mono">
            Control plane overview for multi-agent robust decisionmaking across your org.
          </div>
          <div className="border-t border-[var(--border)] pt-3">
            <div className="text-xs text-[var(--acid-green)] font-mono mb-1">POSITIONING</div>
            <div className="text-xs text-[var(--text)] font-mono leading-relaxed">
              &quot;Control plane for multi-agent robust decisionmaking across org knowledge and channels&quot;
            </div>
          </div>
        </div>
      ),
      actionsContent: (
        <div className="space-y-2">
          <Link
            href="/arena"
            className="block w-full px-3 py-2 text-xs font-mono text-center bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30 hover:bg-[var(--acid-green)]/20 transition-colors"
          >
            + NEW DEBATE
          </Link>
          <Link
            href="/control-plane"
            className="block w-full px-3 py-2 text-xs font-mono text-center bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
          >
            CONTROL PLANE
          </Link>
          <Link
            href="/admin"
            className="block w-full px-3 py-2 text-xs font-mono text-center bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
          >
            ADMIN
          </Link>
        </div>
      ),
    });

    return () => clearContext();
  }, [setContext, clearContext]);

  const formatTimeAgo = (timestamp: string) => {
    const diff = Date.now() - new Date(timestamp).getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return 'just now';
  };

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-[var(--bg)] text-[var(--text)] relative z-10">
        <div className="container mx-auto px-4 py-6">
          {/* Header */}
          <div className="mb-6">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div>
                <h1 className="text-xl font-mono text-[var(--acid-green)] mb-2">
                  {'>'} EXECUTIVE DASHBOARD
                </h1>
                <p className="text-xs text-[var(--text-muted)] font-mono">
                  Control plane for multi-agent robust decisionmaking across org knowledge and channels
                </p>
              </div>
              <div className="flex items-center gap-2">
                <span className="px-2 py-1 text-xs font-mono bg-green-500/20 text-green-400 border border-green-500/30">
                   LIVE
                </span>
              </div>
            </div>
          </div>

          {/* Executive Summary KPIs */}
          <ExecutiveSummary refreshInterval={30000} />

          {/* Recent Activity Section */}
          <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Recent Debates */}
            <div className="bg-[var(--surface)] border border-[var(--border)]">
              <div className="flex items-center justify-between p-4 border-b border-[var(--border)]">
                <h3 className="text-sm font-mono text-[var(--acid-green)]">
                  {'>'} RECENT DEBATES
                </h3>
                <Link
                  href="/debates"
                  className="text-xs font-mono text-[var(--text-muted)] hover:text-[var(--acid-green)] transition-colors"
                >
                  VIEW ALL
                </Link>
              </div>

              {loadingDebates ? (
                <div className="p-4 text-center text-[var(--text-muted)] font-mono text-sm animate-pulse">
                  Loading...
                </div>
              ) : recentDebates.length === 0 ? (
                <div className="p-4 text-center text-[var(--text-muted)] font-mono text-sm">
                  No recent debates. <Link href="/arena" className="text-[var(--acid-green)] hover:underline">Start one</Link>
                </div>
              ) : (
                <div className="divide-y divide-[var(--border)]">
                  {recentDebates.map((debate) => (
                    <Link
                      key={debate.id}
                      href={`/debate/${debate.id}`}
                      className="block p-4 hover:bg-[var(--bg)] transition-colors"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-mono text-[var(--text)] truncate">
                            {debate.task}
                          </p>
                          <div className="flex items-center gap-2 mt-1">
                            <div className="flex items-center gap-1">
                              {debate.agents.slice(0, 3).map((agent, i) => {
                                const colors = getAgentColors(agent);
                                return (
                                  <span
                                    key={i}
                                    className={`px-1 py-0.5 text-[10px] ${colors.bg} ${colors.text} font-mono`}
                                  >
                                    {agent.split('-')[0][0].toUpperCase()}
                                  </span>
                                );
                              })}
                              {debate.agents.length > 3 && (
                                <span className="text-[10px] text-[var(--text-muted)] font-mono">
                                  +{debate.agents.length - 3}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                        <div className="text-right flex-shrink-0">
                          <div className={`text-xs font-mono ${
                            debate.consensus_reached ? 'text-green-400' : 'text-yellow-400'
                          }`}>
                            {debate.consensus_reached ? '' : ''}{' '}
                            {Math.round(debate.confidence * 100)}%
                          </div>
                          <div className="text-[10px] text-[var(--text-muted)] font-mono mt-1">
                            {formatTimeAgo(debate.created_at)}
                          </div>
                        </div>
                      </div>
                    </Link>
                  ))}
                </div>
              )}
            </div>

            {/* System Status */}
            <div className="bg-[var(--surface)] border border-[var(--border)]">
              <div className="flex items-center justify-between p-4 border-b border-[var(--border)]">
                <h3 className="text-sm font-mono text-[var(--acid-green)]">
                  {'>'} SYSTEM STATUS
                </h3>
                <Link
                  href="/admin"
                  className="text-xs font-mono text-[var(--text-muted)] hover:text-[var(--acid-green)] transition-colors"
                >
                  ADMIN
                </Link>
              </div>

              <div className="p-4 space-y-4">
                {/* Status Items */}
                {[
                  { name: 'Debate Engine', status: 'operational', icon: '' },
                  { name: 'Agent Pool', status: 'operational', icon: '' },
                  { name: 'Knowledge Mound', status: 'operational', icon: '' },
                  { name: 'Channel Integrations', status: 'degraded', icon: '' },
                  { name: 'Audit System', status: 'operational', icon: '' },
                ].map((item) => (
                  <div key={item.name} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-sm">{item.icon}</span>
                      <span className="text-xs font-mono text-[var(--text)]">{item.name}</span>
                    </div>
                    <span
                      className={`px-2 py-0.5 text-[10px] font-mono uppercase ${
                        item.status === 'operational'
                          ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                          : item.status === 'degraded'
                          ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                          : 'bg-red-500/20 text-red-400 border border-red-500/30'
                      }`}
                    >
                      {item.status}
                    </span>
                  </div>
                ))}

                {/* Uptime */}
                <div className="pt-3 mt-3 border-t border-[var(--border)]">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-[var(--text-muted)]">30-day uptime</span>
                    <span className="text-green-400">99.87%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Feature Grid */}
          <div className="mt-8">
            <h3 className="text-sm font-mono text-[var(--acid-green)] mb-4">
              {'>'} QUICK ACCESS
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {[
                { href: '/arena', label: 'Arena', icon: '', desc: 'Start debate' },
                { href: '/debates', label: 'Archive', icon: '', desc: 'Past debates' },
                { href: '/debates/provenance', label: 'Provenance', icon: '', desc: 'Audit trails' },
                { href: '/agents', label: 'Agents', icon: '', desc: 'Team selection' },
                { href: '/knowledge', label: 'Knowledge', icon: '', desc: 'Org data' },
                { href: '/control-plane', label: 'Control', icon: '', desc: 'Orchestration' },
                { href: '/integrations/chat', label: 'Channels', icon: '', desc: 'Slack/Teams' },
                { href: '/leaderboard', label: 'Rankings', icon: '', desc: 'Agent ELO' },
                { href: '/audit', label: 'Audit', icon: '', desc: 'Compliance' },
                { href: '/connectors', label: 'Connectors', icon: '', desc: 'Data sources' },
                { href: '/pulse', label: 'Pulse', icon: '', desc: 'Trending' },
                { href: '/settings', label: 'Settings', icon: '', desc: 'Configure' },
              ].map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="bg-[var(--surface)] border border-[var(--border)] p-3 hover:border-[var(--acid-green)]/50 transition-colors group"
                >
                  <div className="text-xl mb-1">{item.icon}</div>
                  <div className="text-xs font-mono text-[var(--text)] group-hover:text-[var(--acid-green)] transition-colors">
                    {item.label}
                  </div>
                  <div className="text-[10px] font-mono text-[var(--text-muted)]">
                    {item.desc}
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-[var(--acid-green)]/20 mt-8">
          <div className="text-[var(--acid-green)]/50 mb-2">{'═'.repeat(40)}</div>
          <p className="text-[var(--text-muted)]">
            {'>'} ARAGORA CONTROL PLANE // MULTI-AGENT DELIBERATION
          </p>
          <p className="text-[var(--acid-cyan)] mt-2">
            <Link href="/" className="hover:text-[var(--acid-green)] transition-colors">
              [ RETURN TO HOME ]
            </Link>
          </p>
          <div className="text-[var(--acid-green)]/50 mt-4">{'═'.repeat(40)}</div>
        </footer>
      </main>
    </>
  );
}
