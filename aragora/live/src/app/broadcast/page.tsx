'use client';

import Link from 'next/link';
import { useState, useEffect, useCallback } from 'react';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';

interface Episode {
  id: string;
  title: string;
  debate_id: string;
  duration_seconds: number;
  file_size_bytes: number;
  format: string;
  created_at: string;
  agents: string[];
}

interface BroadcastStatus {
  available: boolean;
  tts_backends: string[];
  supported_formats: string[];
  episode_count: number;
  pipeline_available: boolean;
}

export default function BroadcastPage() {
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [status, setStatus] = useState<BroadcastStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedEpisode, setSelectedEpisode] = useState<Episode | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/broadcast/status');
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (e) {
      console.error('Failed to fetch broadcast status:', e);
    }
  }, []);

  const fetchEpisodes = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/broadcast/episodes?limit=50');
      if (res.ok) {
        const data = await res.json();
        setEpisodes(data.episodes || []);
        setError(null);
      } else if (res.status === 503) {
        setError('Broadcast module not available. Install with: pip install aragora[broadcast]');
      } else {
        setError('Failed to load episodes');
      }
    } catch (e) {
      setError('Network error loading episodes');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    fetchEpisodes();
  }, [fetchStatus, fetchEpisodes]);

  const formatDuration = (seconds: number) => {
    if (!seconds) return '--:--';
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatFileSize = (bytes: number) => {
    if (!bytes) return '--';
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
  };

  const formatDate = (iso: string) => {
    try {
      return new Date(iso).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      });
    } catch {
      return iso;
    }
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
              {'>'} BROADCAST CENTER
            </h1>
            <p className="text-sm font-mono text-text-muted">
              Generate and manage podcast episodes from debate transcripts.
              Stream or download audio for any completed debate.
            </p>
          </div>

          {/* Status Panel */}
          {status && (
            <div className="mb-6 p-4 border border-acid-green/30 bg-surface/50">
              <div className="flex items-center gap-6 text-xs font-mono">
                <div className="flex items-center gap-2">
                  <span
                    className={`w-2 h-2 rounded-full ${
                      status.available ? 'bg-acid-green' : 'bg-warning'
                    }`}
                  />
                  <span className="text-text-muted">
                    {status.available ? 'BROADCAST READY' : 'UNAVAILABLE'}
                  </span>
                </div>
                <div className="text-text-muted">
                  EPISODES: <span className="text-acid-cyan">{status.episode_count}</span>
                </div>
                {status.tts_backends.length > 0 && (
                  <div className="text-text-muted">
                    TTS: <span className="text-acid-cyan">{status.tts_backends.join(', ')}</span>
                  </div>
                )}
                <a
                  href="/api/podcast/feed.xml"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="ml-auto text-acid-green hover:underline"
                >
                  [RSS FEED]
                </a>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mb-6 p-4 border border-warning/30 bg-warning/10">
              <p className="text-xs font-mono text-warning">{'>'} {error}</p>
            </div>
          )}

          {/* Main Content */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Episode List */}
            <div className="lg:col-span-2">
              <div className="border border-acid-green/30 bg-surface/50">
                <div className="px-4 py-3 border-b border-acid-green/20 bg-bg/50">
                  <span className="text-xs font-mono text-acid-green uppercase tracking-wider">
                    {'>'} EPISODES ({episodes.length})
                  </span>
                </div>

                {loading ? (
                  <div className="p-8 text-center">
                    <div className="w-6 h-6 border-2 border-acid-green/40 border-t-acid-green rounded-full animate-spin mx-auto" />
                    <p className="mt-2 text-xs font-mono text-text-muted">Loading episodes...</p>
                  </div>
                ) : episodes.length === 0 ? (
                  <div className="p-8 text-center">
                    <p className="text-xs font-mono text-text-muted mb-4">
                      No episodes yet. Generate your first podcast from a debate.
                    </p>
                    <Link
                      href="/debates"
                      className="inline-block px-4 py-2 text-xs font-mono border border-acid-green/40 hover:bg-acid-green/10 transition-colors"
                    >
                      [BROWSE DEBATES]
                    </Link>
                  </div>
                ) : (
                  <div className="divide-y divide-acid-green/10">
                    {episodes.map((ep) => (
                      <button
                        key={ep.id}
                        onClick={() => setSelectedEpisode(ep)}
                        className={`w-full px-4 py-3 text-left hover:bg-acid-green/5 transition-colors ${
                          selectedEpisode?.id === ep.id ? 'bg-acid-green/10' : ''
                        }`}
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex-1 min-w-0">
                            <div className="text-sm font-mono text-text truncate">
                              {ep.title || ep.debate_id}
                            </div>
                            <div className="mt-1 flex items-center gap-3 text-xs font-mono text-text-muted">
                              <span>{formatDuration(ep.duration_seconds)}</span>
                              <span>{formatFileSize(ep.file_size_bytes)}</span>
                              <span>{formatDate(ep.created_at)}</span>
                            </div>
                            {ep.agents && ep.agents.length > 0 && (
                              <div className="mt-1 text-xs font-mono text-acid-cyan">
                                {ep.agents.slice(0, 3).join(' vs ')}
                              </div>
                            )}
                          </div>
                          <div className="text-xs font-mono text-acid-green">
                            ▶
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Player Panel */}
            <div className="lg:col-span-1">
              <div className="border border-acid-green/30 bg-surface/50 sticky top-20">
                <div className="px-4 py-3 border-b border-acid-green/20 bg-bg/50">
                  <span className="text-xs font-mono text-acid-green uppercase tracking-wider">
                    {'>'} NOW PLAYING
                  </span>
                </div>

                {selectedEpisode ? (
                  <div className="p-4 space-y-4">
                    <div>
                      <h3 className="text-sm font-mono text-text mb-1">
                        {selectedEpisode.title || selectedEpisode.debate_id}
                      </h3>
                      <p className="text-xs font-mono text-text-muted">
                        {formatDuration(selectedEpisode.duration_seconds)} •{' '}
                        {formatDate(selectedEpisode.created_at)}
                      </p>
                    </div>

                    <audio
                      controls
                      className="w-full"
                      src={`/api/broadcast/audio/${selectedEpisode.debate_id}.${selectedEpisode.format || 'mp3'}`}
                    />

                    <div className="flex flex-col gap-2">
                      <a
                        href={`/api/broadcast/audio/${selectedEpisode.debate_id}.${selectedEpisode.format || 'mp3'}`}
                        download
                        className="w-full px-3 py-2 text-xs font-mono text-center border border-acid-green/40 hover:bg-acid-green/10 transition-colors"
                      >
                        [DOWNLOAD {selectedEpisode.format?.toUpperCase() || 'MP3'}]
                      </a>
                      <Link
                        href={`/debates/${selectedEpisode.debate_id}`}
                        className="w-full px-3 py-2 text-xs font-mono text-center border border-acid-green/40 hover:bg-acid-green/10 transition-colors"
                      >
                        [VIEW DEBATE]
                      </Link>
                    </div>

                    {selectedEpisode.agents && selectedEpisode.agents.length > 0 && (
                      <div className="pt-2 border-t border-acid-green/20">
                        <div className="text-xs font-mono text-text-muted mb-1">PARTICIPANTS</div>
                        <div className="flex flex-wrap gap-1">
                          {selectedEpisode.agents.map((agent) => (
                            <span
                              key={agent}
                              className="px-2 py-0.5 text-xs font-mono bg-acid-green/10 border border-acid-green/30"
                            >
                              {agent}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-8 text-center">
                    <p className="text-xs font-mono text-text-muted">
                      Select an episode to play
                    </p>
                  </div>
                )}
              </div>

              {/* Quick Links */}
              <div className="mt-4 border border-acid-green/30 bg-surface/50">
                <div className="px-4 py-3 border-b border-acid-green/20 bg-bg/50">
                  <span className="text-xs font-mono text-acid-green uppercase tracking-wider">
                    {'>'} QUICK ACTIONS
                  </span>
                </div>
                <div className="p-4 space-y-2">
                  <Link
                    href="/debates"
                    className="block w-full px-3 py-2 text-xs font-mono text-center border border-acid-green/40 hover:bg-acid-green/10 transition-colors"
                  >
                    [+ GENERATE NEW EPISODE]
                  </Link>
                  <a
                    href="/api/podcast/feed.xml"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block w-full px-3 py-2 text-xs font-mono text-center border border-acid-green/40 hover:bg-acid-green/10 transition-colors"
                  >
                    [SUBSCRIBE TO PODCAST]
                  </a>
                </div>
              </div>
            </div>
          </div>

          {/* Help Section */}
          <div className="mt-8">
            <details className="group">
              <summary className="text-xs font-mono text-text-muted cursor-pointer hover:text-acid-green">
                [?] BROADCAST GUIDE
              </summary>
              <div className="mt-4 p-4 bg-surface/50 border border-acid-green/20 text-xs font-mono text-text-muted space-y-4">
                <div>
                  <div className="text-acid-green mb-1">WHAT IS BROADCAST?</div>
                  <p>
                    Broadcast converts debate transcripts into podcast-style audio using text-to-speech.
                    Each participant gets a unique voice, making debates easy to consume on the go.
                  </p>
                </div>
                <div>
                  <div className="text-acid-green mb-1">TTS BACKENDS</div>
                  <ul className="list-disc list-inside space-y-1">
                    <li><span className="text-acid-cyan">edge-tts</span> - Microsoft Edge TTS (free, high quality)</li>
                    <li><span className="text-acid-cyan">elevenlabs</span> - ElevenLabs API (premium voices)</li>
                    <li><span className="text-acid-cyan">pyttsx3</span> - Local TTS (offline, lower quality)</li>
                  </ul>
                </div>
                <div>
                  <div className="text-acid-green mb-1">GENERATING EPISODES</div>
                  <p>
                    Navigate to any completed debate and click the Broadcast button to generate audio.
                    Episodes are automatically added to the RSS feed for podcast apps.
                  </p>
                </div>
                <div>
                  <div className="text-acid-green mb-1">RSS FEED</div>
                  <p>
                    Subscribe to the RSS feed in your favorite podcast app to automatically receive
                    new episodes. The feed is iTunes-compatible.
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
