'use client';
import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { ErrorWithRetry } from '@/components/ErrorWithRetry';

interface ConnectorStatus {
  name: string;
  is_configured: boolean;
  is_connected: boolean;
  quota_remaining?: number;
  last_error?: string;
}

interface PublishJob {
  id: string;
  debate_id: string;
  platform: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
  result_url?: string;
  error?: string;
}

interface RecentDebate {
  id: string;
  task: string;
  created_at: string;
  has_audio: boolean;
  has_video: boolean;
}

type TabType = 'status' | 'publish' | 'history';

export default function SocialPage() {
  const { config } = useBackend();
  const backendUrl = config.api;
  const [activeTab, setActiveTab] = useState<TabType>('status');
  const [connectors, setConnectors] = useState<ConnectorStatus[]>([]);
  const [recentDebates, setRecentDebates] = useState<RecentDebate[]>([]);
  const [publishHistory, setPublishHistory] = useState<PublishJob[]>([]);
  const [selectedDebate, setSelectedDebate] = useState<string>('');
  const [selectedPlatform, setSelectedPlatform] = useState<string>('');
  const [publishing, setPublishing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchConnectorStatus = useCallback(async () => {
    try {
      const [youtube, twitter] = await Promise.all([
        fetch(`${backendUrl}/api/youtube/status`).then(r => r.ok ? r.json() : null).catch(() => null),
        fetch(`${backendUrl}/api/connectors`).then(r => r.ok ? r.json() : null).catch(() => null),
      ]);

      const connectorList: ConnectorStatus[] = [];

      if (youtube) {
        connectorList.push({
          name: 'YouTube',
          is_configured: youtube.is_configured || false,
          is_connected: youtube.is_connected || false,
          quota_remaining: youtube.quota_remaining,
          last_error: youtube.error,
        });
      } else {
        connectorList.push({
          name: 'YouTube',
          is_configured: false,
          is_connected: false,
        });
      }

      // Add Twitter placeholder
      connectorList.push({
        name: 'Twitter/X',
        is_configured: false,
        is_connected: false,
      });

      // Add Slack from connectors if available
      if (twitter?.connectors) {
        const slackConnector = twitter.connectors.find((c: { type: string }) => c.type === 'slack');
        if (slackConnector) {
          connectorList.push({
            name: 'Slack',
            is_configured: slackConnector.is_configured || false,
            is_connected: slackConnector.status === 'active',
          });
        }
      }

      setConnectors(connectorList);
    } catch (err) {
      console.error('Failed to fetch connector status:', err);
      throw err;
    }
  }, [backendUrl]);

  const fetchRecentDebates = useCallback(async () => {
    try {
      const response = await fetch(`${backendUrl}/api/debates?limit=20`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setRecentDebates(data.debates?.map((d: {
        id: string;
        task?: string;
        created_at?: string;
        metadata?: { has_audio?: boolean; has_video?: boolean };
      }) => ({
        id: d.id,
        task: d.task || 'Untitled debate',
        created_at: d.created_at || new Date().toISOString(),
        has_audio: d.metadata?.has_audio || false,
        has_video: d.metadata?.has_video || false,
      })) || []);
    } catch (err) {
      console.error('Failed to fetch debates:', err);
    }
  }, [backendUrl]);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      await Promise.all([fetchConnectorStatus(), fetchRecentDebates()]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, [fetchConnectorStatus, fetchRecentDebates]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handlePublish = async () => {
    if (!selectedDebate || !selectedPlatform) return;

    setPublishing(true);
    setError(null);

    try {
      const response = await fetch(
        `${backendUrl}/api/debates/${selectedDebate}/publish/${selectedPlatform}`,
        { method: 'POST' }
      );

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || `HTTP ${response.status}`);
      }

      const result = await response.json();
      setPublishHistory(prev => [{
        id: result.job_id || Date.now().toString(),
        debate_id: selectedDebate,
        platform: selectedPlatform,
        status: 'completed',
        created_at: new Date().toISOString(),
        completed_at: new Date().toISOString(),
        result_url: result.url,
      }, ...prev]);

      setSelectedDebate('');
      setSelectedPlatform('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Publish failed');
    } finally {
      setPublishing(false);
    }
  };

  const initiateOAuth = async (platform: string) => {
    try {
      if (platform === 'YouTube') {
        const response = await fetch(`${backendUrl}/api/youtube/auth`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (data.auth_url) {
          window.location.href = data.auth_url;
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to initiate auth');
    }
  };

  const renderStatusTab = () => (
    <div className="space-y-6">
      <h2 className="text-xl font-mono font-bold text-acid-green mb-4">Platform Connections</h2>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {connectors.map((connector) => (
          <div
            key={connector.name}
            className="p-4 bg-surface border border-border rounded-lg"
          >
            <div className="flex items-center justify-between mb-3">
              <span className="font-mono font-bold text-text">{connector.name}</span>
              <span
                className={`px-2 py-1 text-xs font-mono rounded ${
                  connector.is_connected
                    ? 'bg-acid-green/20 text-acid-green border border-acid-green/30'
                    : connector.is_configured
                    ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                    : 'bg-red-500/20 text-red-400 border border-red-500/30'
                }`}
              >
                {connector.is_connected ? 'Connected' : connector.is_configured ? 'Configured' : 'Not configured'}
              </span>
            </div>

            {connector.quota_remaining !== undefined && (
              <div className="text-xs text-text-muted mb-2">
                Quota: {connector.quota_remaining} units remaining
              </div>
            )}

            {connector.last_error && (
              <div className="text-xs text-red-400 mb-2">
                Error: {connector.last_error}
              </div>
            )}

            {!connector.is_connected && connector.name === 'YouTube' && (
              <button
                onClick={() => initiateOAuth('YouTube')}
                className="mt-2 w-full px-3 py-2 bg-red-500/20 border border-red-500/50 text-red-400 font-mono text-sm hover:bg-red-500/30 transition-colors rounded"
              >
                Connect YouTube
              </button>
            )}

            {!connector.is_configured && connector.name !== 'YouTube' && (
              <div className="mt-2 text-xs text-text-muted">
                Configure in Settings &rarr; Integrations
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="p-4 bg-surface border border-border rounded-lg">
        <h3 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">Configuration</h3>
        <p className="text-sm text-text-muted mb-4">
          Social media integrations require API credentials. Configure them in the environment or settings.
        </p>
        <div className="grid gap-2 text-xs font-mono text-text-muted">
          <div>YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET</div>
          <div>TWITTER_API_KEY, TWITTER_API_SECRET</div>
          <div>SLACK_WEBHOOK_URL</div>
        </div>
      </div>
    </div>
  );

  const renderPublishTab = () => (
    <div className="space-y-6">
      <h2 className="text-xl font-mono font-bold text-acid-green mb-4">Publish Debate</h2>

      <div className="p-4 bg-surface border border-border rounded-lg">
        <div className="space-y-4">
          {/* Debate Selection */}
          <div>
            <label className="block text-xs font-mono text-text-muted uppercase mb-2">
              Select Debate
            </label>
            <select
              value={selectedDebate}
              onChange={(e) => setSelectedDebate(e.target.value)}
              className="w-full px-3 py-2 bg-bg border border-border rounded font-mono text-sm focus:outline-none focus:border-acid-green/50"
            >
              <option value="">Choose a debate...</option>
              {recentDebates.map((debate) => (
                <option key={debate.id} value={debate.id}>
                  {debate.task.substring(0, 60)}{debate.task.length > 60 ? '...' : ''}
                  {debate.has_audio && ' [audio]'}
                </option>
              ))}
            </select>
          </div>

          {/* Platform Selection */}
          <div>
            <label className="block text-xs font-mono text-text-muted uppercase mb-2">
              Platform
            </label>
            <div className="flex gap-2">
              {['twitter', 'youtube'].map((platform) => {
                const connector = connectors.find(
                  c => c.name.toLowerCase().includes(platform)
                );
                const isAvailable = connector?.is_connected;

                return (
                  <button
                    key={platform}
                    onClick={() => isAvailable && setSelectedPlatform(platform)}
                    disabled={!isAvailable}
                    className={`flex-1 px-4 py-3 rounded border-2 transition-all font-mono text-sm ${
                      selectedPlatform === platform
                        ? 'border-acid-green bg-acid-green/20 text-acid-green'
                        : isAvailable
                        ? 'border-border text-text hover:border-acid-green/50'
                        : 'border-border/50 text-text-muted cursor-not-allowed opacity-50'
                    }`}
                  >
                    {platform === 'twitter' && 'Twitter/X'}
                    {platform === 'youtube' && 'YouTube'}
                    {!isAvailable && ' (not connected)'}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Publish Button */}
          <button
            onClick={handlePublish}
            disabled={!selectedDebate || !selectedPlatform || publishing}
            className={`w-full px-4 py-3 rounded font-mono font-bold transition-all ${
              !selectedDebate || !selectedPlatform || publishing
                ? 'bg-border text-text-muted cursor-not-allowed'
                : 'bg-acid-green/20 border-2 border-acid-green text-acid-green hover:bg-acid-green/30'
            }`}
          >
            {publishing ? 'Publishing...' : 'Publish'}
          </button>
        </div>
      </div>

      {/* Instructions */}
      <div className="p-4 bg-surface border border-border rounded-lg">
        <h3 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">Notes</h3>
        <ul className="text-sm text-text-muted space-y-2">
          <li>• Twitter: Generates a thread summarizing the debate conclusion</li>
          <li>• YouTube: Requires audio/video to be generated first (via /broadcast)</li>
          <li>• Published content includes attribution to participating agents</li>
          <li>• Rate limits apply per platform API quotas</li>
        </ul>
      </div>
    </div>
  );

  const renderHistoryTab = () => (
    <div className="space-y-6">
      <h2 className="text-xl font-mono font-bold text-acid-green mb-4">Publish History</h2>

      {publishHistory.length === 0 ? (
        <div className="p-8 bg-surface border border-border rounded-lg text-center">
          <p className="text-text-muted font-mono">No publish history yet</p>
        </div>
      ) : (
        <div className="space-y-3">
          {publishHistory.map((job) => (
            <div
              key={job.id}
              className="p-4 bg-surface border border-border rounded-lg"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  <span className="px-2 py-1 text-xs font-mono uppercase bg-blue-500/20 text-blue-400 rounded">
                    {job.platform}
                  </span>
                  <span className="text-sm font-mono text-text">
                    Debate: {job.debate_id.substring(0, 8)}...
                  </span>
                </div>
                <span
                  className={`px-2 py-1 text-xs font-mono rounded ${
                    job.status === 'completed'
                      ? 'bg-acid-green/20 text-acid-green'
                      : job.status === 'failed'
                      ? 'bg-red-500/20 text-red-400'
                      : 'bg-yellow-500/20 text-yellow-400'
                  }`}
                >
                  {job.status}
                </span>
              </div>

              <div className="text-xs text-text-muted">
                {new Date(job.created_at).toLocaleString()}
              </div>

              {job.result_url && (
                <a
                  href={job.result_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-2 inline-block text-sm text-acid-green hover:underline"
                >
                  View on {job.platform} &rarr;
                </a>
              )}

              {job.error && (
                <div className="mt-2 text-xs text-red-400">
                  Error: {job.error}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-bg text-text relative overflow-hidden">
      <Scanlines />
      <CRTVignette />

      <div className="max-w-6xl mx-auto px-4 py-8 relative z-10">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <Link href="/" className="hover:opacity-80 transition-opacity">
            <AsciiBannerCompact />
          </Link>
          <div className="flex items-center gap-4">
            <ThemeToggle />
            <BackendSelector />
          </div>
        </div>

        {/* Title */}
        <div className="mb-8">
          <h1 className="text-3xl font-mono font-bold text-acid-green mb-2">Social Media</h1>
          <p className="text-text-muted font-mono text-sm">
            Publish debates to social platforms and manage integrations
          </p>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6">
            <ErrorWithRetry
              error={error}
              onRetry={loadData}
            />
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-2 mb-6 border-b border-border pb-2">
          <button
            onClick={() => setActiveTab('status')}
            className={`px-4 py-2 font-mono text-sm rounded-t transition-colors ${
              activeTab === 'status'
                ? 'bg-acid-green/10 text-acid-green border-b-2 border-acid-green'
                : 'text-text-muted hover:text-text'
            }`}
          >
            Connections
          </button>
          <button
            onClick={() => setActiveTab('publish')}
            className={`px-4 py-2 font-mono text-sm rounded-t transition-colors ${
              activeTab === 'publish'
                ? 'bg-acid-green/10 text-acid-green border-b-2 border-acid-green'
                : 'text-text-muted hover:text-text'
            }`}
          >
            Publish
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`px-4 py-2 font-mono text-sm rounded-t transition-colors ${
              activeTab === 'history'
                ? 'bg-acid-green/10 text-acid-green border-b-2 border-acid-green'
                : 'text-text-muted hover:text-text'
            }`}
          >
            History
          </button>
        </div>

        {/* Content */}
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-acid-green font-mono animate-pulse">Loading...</div>
          </div>
        ) : (
          <div>
            {activeTab === 'status' && renderStatusTab()}
            {activeTab === 'publish' && renderPublishTab()}
            {activeTab === 'history' && renderHistoryTab()}
          </div>
        )}
      </div>
    </div>
  );
}
