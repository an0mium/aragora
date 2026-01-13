'use client';

import Link from 'next/link';
import { PulseSchedulerControlPanel } from '@/components/PulseSchedulerControlPanel';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';

export default function PulsePage() {
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
              {'>'} PULSE SCHEDULER
            </h1>
            <p className="text-sm font-mono text-text-muted">
              Automated trending topic debates from HackerNews, Reddit, and Twitter.
              Configure sources, rate limits, and monitoring.
            </p>
          </div>

          {/* Main Panel */}
          <div className="max-w-4xl">
            <PulseSchedulerControlPanel />
          </div>

          {/* Help Section */}
          <div className="mt-8 max-w-4xl">
            <details className="group">
              <summary className="text-xs font-mono text-text-muted cursor-pointer hover:text-acid-green">
                [?] PULSE SCHEDULER GUIDE
              </summary>
              <div className="mt-4 p-4 bg-surface/50 border border-acid-green/20 text-xs font-mono text-text-muted space-y-4">
                <div>
                  <div className="text-acid-green mb-1">WHAT IS PULSE?</div>
                  <p>
                    Pulse automatically monitors trending topics across social platforms and
                    creates debates on relevant subjects. It runs in the background, continuously
                    generating content based on your configuration.
                  </p>
                </div>
                <div>
                  <div className="text-acid-green mb-1">SOURCES</div>
                  <ul className="list-disc list-inside space-y-1">
                    <li><span className="text-acid-cyan">HackerNews</span> - Tech and startup discussions</li>
                    <li><span className="text-acid-cyan">Reddit</span> - Various subreddits (tech, science, AI)</li>
                    <li><span className="text-acid-cyan">Twitter/X</span> - Trending hashtags and topics</li>
                  </ul>
                </div>
                <div>
                  <div className="text-acid-green mb-1">RATE LIMITING</div>
                  <p>
                    Configure how many debates are created per hour to manage API costs and
                    ensure quality. Lower rates allow more thorough debates; higher rates
                    increase coverage.
                  </p>
                </div>
                <div>
                  <div className="text-acid-green mb-1">CATEGORIES</div>
                  <p>
                    Filter topics by category to focus on relevant subjects. Unchecked categories
                    are excluded from automatic debate generation.
                  </p>
                </div>
                <div>
                  <div className="text-acid-green mb-1">VOLUME THRESHOLD</div>
                  <p>
                    Minimum engagement score (comments, upvotes, retweets) required before a
                    topic triggers a debate. Higher thresholds focus on more popular topics.
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
