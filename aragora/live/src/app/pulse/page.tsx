'use client';

import { useState, useCallback } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { PulseSchedulerControlPanel } from '@/components/PulseSchedulerControlPanel';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { TrendingTopicsGrid, TopicDetailDrawer } from '@/components/pulse';
import type { TrendingTopic, DebateConfig } from '@/components/pulse';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8080';

type ViewTab = 'trending' | 'scheduler';

export default function PulsePage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<ViewTab>('trending');
  const [selectedTopic, setSelectedTopic] = useState<TrendingTopic | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [debateStarted, setDebateStarted] = useState<string | null>(null);

  const handleStartDebate = useCallback((topic: TrendingTopic) => {
    setSelectedTopic(topic);
    setIsDrawerOpen(true);
  }, []);

  const handleDebateConfigured = useCallback((topic: TrendingTopic, config: DebateConfig) => {
    setDebateStarted(topic.topic);
    setIsDrawerOpen(false);
    setSelectedTopic(null);

    // Show success message briefly then navigate to debates
    setTimeout(() => {
      setDebateStarted(null);
      router.push('/debates');
    }, 2000);
  }, [router]);

  const handleCloseDrawer = useCallback(() => {
    setIsDrawerOpen(false);
    setSelectedTopic(null);
  }, []);

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
            <div className="flex items-center gap-3">
              <Link
                href="/"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [DASHBOARD]
              </Link>
              <Link
                href="/debates"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [DEBATES]
              </Link>
              <Link
                href="/insights"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [INSIGHTS]
              </Link>
              <Link
                href="/gauntlet"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [GAUNTLET]
              </Link>
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Success Banner */}
        {debateStarted && (
          <div className="bg-acid-green/20 border-b border-acid-green/30 py-2">
            <div className="container mx-auto px-4 text-center">
              <span className="text-sm font-mono text-acid-green">
                Debate started on &quot;{debateStarted.slice(0, 50)}...&quot; - Redirecting to debates...
              </span>
            </div>
          </div>
        )}

        {/* Content */}
        <div className="container mx-auto px-4 py-8">
          {/* Page Title */}
          <div className="mb-6">
            <h1 className="text-xl font-mono text-acid-green mb-2">
              {'>'} PULSE - TRENDING TOPICS
            </h1>
            <p className="text-sm font-mono text-text-muted">
              Discover and debate trending topics from HackerNews, Reddit, Twitter, and more.
            </p>
          </div>

          {/* Tab Navigation */}
          <div className="flex items-center gap-4 mb-6 border-b border-border">
            <button
              onClick={() => setActiveTab('trending')}
              className={`pb-3 px-1 text-sm font-mono transition-colors border-b-2 -mb-px ${
                activeTab === 'trending'
                  ? 'text-acid-green border-acid-green'
                  : 'text-text-muted border-transparent hover:text-text hover:border-text-muted/50'
              }`}
            >
              🔥 TRENDING TOPICS
            </button>
            <button
              onClick={() => setActiveTab('scheduler')}
              className={`pb-3 px-1 text-sm font-mono transition-colors border-b-2 -mb-px ${
                activeTab === 'scheduler'
                  ? 'text-acid-green border-acid-green'
                  : 'text-text-muted border-transparent hover:text-text hover:border-text-muted/50'
              }`}
            >
              ⚙️ AUTO SCHEDULER
            </button>
          </div>

          {/* Tab Content */}
          {activeTab === 'trending' ? (
            <TrendingTopicsGrid
              apiBase={API_BASE}
              autoRefresh={true}
              refreshInterval={60000}
              onStartDebate={handleStartDebate}
              onTopicSelect={setSelectedTopic}
              selectedTopic={selectedTopic}
              showFilters={true}
              maxTopics={50}
            />
          ) : (
            <div className="space-y-8">
              {/* Scheduler Control Panel */}
              <div className="max-w-4xl">
                <PulseSchedulerControlPanel />
              </div>

              {/* Help Section */}
              <div className="max-w-4xl">
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
          )}
        </div>

        {/* Topic Detail Drawer */}
        <TopicDetailDrawer
          topic={selectedTopic}
          isOpen={isDrawerOpen}
          onClose={handleCloseDrawer}
          onStartDebate={handleDebateConfigured}
          apiBase={API_BASE}
        />
      </main>
    </>
  );
}
