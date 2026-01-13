'use client';

import Link from 'next/link';
import { AgentRecommender } from '@/components/AgentRecommender';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';

export default function AgentsPage() {
  const handleTeamSelect = (agents: string[]) => {
    // Copy to clipboard
    const agentString = agents.join(',');
    navigator.clipboard.writeText(agentString).then(() => {
      // Could show a toast notification here
    });
  };

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />
      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/">
              <AsciiBannerCompact connected={true} />
            </Link>
            <div className="flex items-center gap-4">
              <Link
                href="/"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [DASHBOARD]
              </Link>
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="container mx-auto px-4 py-8">
          {/* Page Title */}
          <div className="mb-6">
            <h1 className="text-xl font-mono text-acid-green mb-2">
              {'>'} AGENT RECOMMENDER
            </h1>
            <p className="text-sm font-mono text-text-muted">
              Analyze your debate topic to get optimal agent recommendations.
              View domain leaderboards and best-performing team combinations.
            </p>
          </div>

          {/* Main Panel */}
          <div className="max-w-4xl">
            <AgentRecommender onTeamSelect={handleTeamSelect} />
          </div>

          {/* Help Section */}
          <div className="mt-8 max-w-4xl">
            <details className="group">
              <summary className="text-xs font-mono text-text-muted cursor-pointer hover:text-acid-green">
                [?] AGENT SELECTION GUIDE
              </summary>
              <div className="mt-4 p-4 bg-surface/50 border border-acid-green/20 text-xs font-mono text-text-muted space-y-4">
                <div>
                  <div className="text-acid-green mb-1">ANALYZE TAB</div>
                  <p>
                    Enter your debate topic to get AI-powered agent recommendations.
                    The system detects the domain (technical, ethics, security, etc.)
                    and suggests the best agents for that type of question.
                  </p>
                </div>
                <div>
                  <div className="text-acid-green mb-1">LEADERBOARD TAB</div>
                  <p>
                    View the top-performing agents across different domains.
                    Rankings are based on ELO scores from past debates,
                    win rates, and consensus contribution.
                  </p>
                </div>
                <div>
                  <div className="text-acid-green mb-1">TEAMS TAB</div>
                  <p>
                    See which agent combinations work best together.
                    Team synergy is calculated from historical debates
                    where agents reached consensus effectively.
                  </p>
                </div>
                <div>
                  <div className="text-acid-green mb-1">USING RECOMMENDATIONS</div>
                  <p>
                    Click <span className="text-acid-cyan">[USE TEAM]</span> to copy the agent
                    list to your clipboard. Then paste it into the agents field when
                    starting a debate.
                  </p>
                </div>
              </div>
            </details>
          </div>
        </div>
      </main>
    </>
  );
}
