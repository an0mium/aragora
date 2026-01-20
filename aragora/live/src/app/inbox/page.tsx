'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import { GmailConnectionCard } from '@/components/inbox/GmailConnectionCard';
import { SyncProgressBar } from '@/components/inbox/SyncProgressBar';
import { PriorityInboxList } from '@/components/inbox/PriorityInboxList';
import { InboxQueryPanel } from '@/components/inbox/InboxQueryPanel';
import { useAuth } from '@/context/AuthContext';

interface GmailStatus {
  connected: boolean;
  configured: boolean;
  email_address?: string;
  indexed_count?: number;
  last_sync?: string;
}

interface SyncStatus {
  job_status: string;
  job_progress: number;
  job_messages_synced: number;
  job_error?: string;
}

interface PrioritizationConfig {
  vip_senders: string[];
  tier_1_threshold: number;
  tier_2_threshold: number;
  enable_slack_context: boolean;
  enable_calendar_context: boolean;
}

export default function InboxPage() {
  const { config: backendConfig } = useBackend();
  const { user, tokens } = useAuth();
  const [status, setStatus] = useState<GmailStatus | null>(null);
  const [syncStatus, setSyncStatus] = useState<SyncStatus | null>(null);
  const [priConfig, setPriConfig] = useState<PrioritizationConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showConfig, setShowConfig] = useState(false);
  const [newVip, setNewVip] = useState('');

  // Use user ID or default
  const userId = user?.id || 'default';

  const fetchStatus = useCallback(async () => {
    try {
      // Try new email API first
      const response = await fetch(
        `${backendConfig.api}/api/email/config?user_id=${userId}`,
        {
          headers: { Authorization: `Bearer ${tokens?.access_token || ''}` },
        }
      );
      if (response.ok) {
        const data = await response.json();
        setStatus({
          connected: data.gmail_connected || false,
          configured: true,
          email_address: data.email_address,
          indexed_count: data.indexed_count,
          last_sync: data.last_sync,
        });
        setPriConfig({
          vip_senders: data.vip_senders || [],
          tier_1_threshold: data.tier_1_threshold || 0.8,
          tier_2_threshold: data.tier_2_threshold || 0.5,
          enable_slack_context: data.enable_slack_context || false,
          enable_calendar_context: data.enable_calendar_context || false,
        });
      } else {
        // Fallback to legacy Gmail status endpoint
        const legacyResponse = await fetch(
          `${backendConfig.api}/api/gmail/status?user_id=${userId}`,
          {
            headers: { Authorization: `Bearer ${tokens?.access_token || ''}` },
          }
        );
        if (legacyResponse.ok) {
          const data = await legacyResponse.json();
          setStatus(data);
        }
      }
    } catch (err) {
      setError('Failed to fetch Gmail status');
    } finally {
      setLoading(false);
    }
  }, [backendConfig.api, userId, tokens?.access_token]);

  const fetchSyncStatus = useCallback(async () => {
    try {
      const response = await fetch(
        `${backendConfig.api}/api/gmail/sync/status?user_id=${userId}`,
        {
          headers: { Authorization: `Bearer ${tokens?.access_token || ''}` },
        }
      );
      if (response.ok) {
        const data = await response.json();
        setSyncStatus(data);
      }
    } catch (err) {
      // Silently fail for sync status
    }
  }, [backendConfig.api, userId, tokens?.access_token]);

  useEffect(() => {
    fetchStatus();
    fetchSyncStatus();

    // Poll sync status when syncing
    const interval = setInterval(() => {
      if (syncStatus?.job_status === 'running') {
        fetchSyncStatus();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [fetchStatus, fetchSyncStatus, syncStatus?.job_status]);

  const handleConnect = useCallback(async () => {
    try {
      // Use new email API OAuth endpoint
      const response = await fetch(`${backendConfig.api}/api/email/gmail/oauth-url`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${tokens?.access_token || ''}`,
        },
        body: JSON.stringify({
          user_id: userId,
          redirect_uri: `${window.location.origin}/inbox/callback`,
          state: userId,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        // Redirect to OAuth URL
        window.location.href = data.url;
      } else {
        // Fallback to legacy endpoint
        const legacyResponse = await fetch(`${backendConfig.api}/api/gmail/connect`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${tokens?.access_token || ''}`,
          },
          body: JSON.stringify({
            user_id: userId,
            redirect_uri: `${window.location.origin}/inbox/callback`,
            state: userId,
          }),
        });

        if (legacyResponse.ok) {
          const data = await legacyResponse.json();
          window.location.href = data.url;
        } else {
          setError('Failed to start connection');
        }
      }
    } catch (err) {
      setError('Failed to connect to Gmail');
    }
  }, [backendConfig.api, userId, tokens?.access_token]);

  const handleDisconnect = useCallback(async () => {
    try {
      const response = await fetch(`${backendConfig.api}/api/gmail/disconnect`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${tokens?.access_token || ''}`,
        },
        body: JSON.stringify({ user_id: userId }),
      });

      if (response.ok) {
        setStatus({ connected: false, configured: status?.configured || false });
        setSyncStatus(null);
      }
    } catch (err) {
      setError('Failed to disconnect');
    }
  }, [backendConfig.api, userId, tokens?.access_token, status?.configured]);

  const handleSync = useCallback(async (fullSync: boolean = false) => {
    try {
      const response = await fetch(`${backendConfig.api}/api/gmail/sync`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${tokens?.access_token || ''}`,
        },
        body: JSON.stringify({
          user_id: userId,
          full_sync: fullSync,
          max_messages: 500,
          labels: ['INBOX'],
        }),
      });

      if (response.ok) {
        fetchSyncStatus();
      }
    } catch (err) {
      setError('Failed to start sync');
    }
  }, [backendConfig.api, userId, tokens?.access_token, fetchSyncStatus]);

  const handleAddVip = useCallback(async () => {
    if (!newVip.trim()) return;
    try {
      const response = await fetch(`${backendConfig.api}/api/email/vip`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${tokens?.access_token || ''}`,
        },
        body: JSON.stringify({
          user_id: userId,
          sender_email: newVip.trim(),
        }),
      });

      if (response.ok) {
        setPriConfig(prev => prev ? {
          ...prev,
          vip_senders: [...prev.vip_senders, newVip.trim()],
        } : null);
        setNewVip('');
      }
    } catch (err) {
      setError('Failed to add VIP sender');
    }
  }, [backendConfig.api, userId, tokens?.access_token, newVip]);

  const handleRemoveVip = useCallback(async (senderEmail: string) => {
    try {
      const response = await fetch(`${backendConfig.api}/api/email/vip`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${tokens?.access_token || ''}`,
        },
        body: JSON.stringify({
          user_id: userId,
          sender_email: senderEmail,
        }),
      });

      if (response.ok) {
        setPriConfig(prev => prev ? {
          ...prev,
          vip_senders: prev.vip_senders.filter(s => s !== senderEmail),
        } : null);
      }
    } catch (err) {
      setError('Failed to remove VIP sender');
    }
  }, [backendConfig.api, userId, tokens?.access_token]);

  return (
    <div className="min-h-screen bg-background">
      <Scanlines />
      <CRTVignette />

      <header className="border-b border-border bg-surface/50 backdrop-blur-sm sticky top-0 z-40">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="hover:text-accent">
              <AsciiBannerCompact />
            </Link>
            <span className="text-muted font-mono text-sm">{'//'} AI SMART INBOX</span>
          </div>
          <div className="flex items-center gap-3">
            {status?.connected && (
              <button
                onClick={() => setShowConfig(!showConfig)}
                className={`px-3 py-1 text-xs font-mono border rounded ${
                  showConfig
                    ? 'bg-acid-green/20 border-acid-green text-acid-green'
                    : 'bg-transparent border-acid-green/40 text-acid-green hover:bg-acid-green/10'
                }`}
              >
                ⚙ Config
              </button>
            )}
            <BackendSelector />
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        {error && (
          <div className="mb-4 p-3 bg-acid-red/10 border border-acid-red/30 rounded text-acid-red font-mono text-sm">
            {error}
            <button
              onClick={() => setError(null)}
              className="ml-4 text-acid-red/70 hover:text-acid-red"
            >
              [X]
            </button>
          </div>
        )}

        {/* Connection Status */}
        <PanelErrorBoundary panelName="Gmail Connection">
          <GmailConnectionCard
            status={status}
            loading={loading}
            onConnect={handleConnect}
            onDisconnect={handleDisconnect}
          />
        </PanelErrorBoundary>

        {/* Prioritization Config Panel */}
        {showConfig && priConfig && (
          <div className="mt-4 border border-acid-green/30 bg-surface/50 p-4 rounded">
            <h3 className="text-acid-green font-mono text-sm mb-4">Prioritization Settings</h3>

            {/* VIP Senders */}
            <div className="mb-4">
              <label className="text-text-muted text-xs font-mono block mb-2">
                VIP Senders (always prioritized)
              </label>
              <div className="flex gap-2 mb-2">
                <input
                  type="email"
                  value={newVip}
                  onChange={(e) => setNewVip(e.target.value)}
                  placeholder="email@example.com"
                  className="flex-1 px-3 py-2 bg-bg border border-acid-green/30 text-text font-mono text-sm rounded focus:outline-none focus:border-acid-green"
                />
                <button
                  onClick={handleAddVip}
                  className="px-4 py-2 text-sm font-mono bg-acid-green/10 border border-acid-green/40 text-acid-green hover:bg-acid-green/20 rounded"
                >
                  Add
                </button>
              </div>
              <div className="flex flex-wrap gap-2">
                {priConfig.vip_senders.map((sender) => (
                  <span
                    key={sender}
                    className="px-2 py-1 text-xs bg-acid-green/10 border border-acid-green/30 rounded text-acid-green flex items-center gap-2"
                  >
                    {sender}
                    <button
                      onClick={() => handleRemoveVip(sender)}
                      className="hover:text-acid-red"
                    >
                      ×
                    </button>
                  </span>
                ))}
                {priConfig.vip_senders.length === 0 && (
                  <span className="text-text-muted text-xs">No VIP senders configured</span>
                )}
              </div>
            </div>

            {/* Context Integration Status */}
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 border border-acid-green/20 rounded">
                <span className="text-text-muted text-xs font-mono">Slack Context</span>
                <div className={`text-sm font-mono mt-1 ${priConfig.enable_slack_context ? 'text-acid-green' : 'text-text-muted'}`}>
                  {priConfig.enable_slack_context ? '✓ Enabled' : '○ Disabled'}
                </div>
              </div>
              <div className="p-3 border border-acid-green/20 rounded">
                <span className="text-text-muted text-xs font-mono">Calendar Context</span>
                <div className={`text-sm font-mono mt-1 ${priConfig.enable_calendar_context ? 'text-acid-green' : 'text-text-muted'}`}>
                  {priConfig.enable_calendar_context ? '✓ Enabled' : '○ Disabled'}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Sync Progress */}
        {status?.connected && syncStatus && (
          <PanelErrorBoundary panelName="Sync Progress">
            <SyncProgressBar
              syncStatus={syncStatus}
              indexedCount={status.indexed_count || 0}
              lastSync={status.last_sync}
              onSync={() => handleSync(false)}
              onFullSync={() => handleSync(true)}
            />
          </PanelErrorBoundary>
        )}

        {/* Main Content */}
        {status?.connected && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
            {/* Priority Inbox List */}
            <div className="lg:col-span-2">
              <PanelErrorBoundary panelName="Priority Inbox">
                <PriorityInboxList
                  apiBase={backendConfig.api}
                  userId={userId}
                  authToken={tokens?.access_token}
                />
              </PanelErrorBoundary>
            </div>

            {/* Q&A Panel */}
            <div className="lg:col-span-1">
              <PanelErrorBoundary panelName="Inbox Q&A">
                <InboxQueryPanel
                  apiBase={backendConfig.api}
                  userId={userId}
                  authToken={tokens?.access_token}
                />
              </PanelErrorBoundary>
            </div>
          </div>
        )}

        {/* Not Connected State */}
        {!loading && !status?.connected && (
          <div className="mt-8 text-center">
            <div className="text-6xl mb-4">📬</div>
            <h2 className="text-xl font-mono text-accent mb-2">
              Connect Your Gmail
            </h2>
            <p className="text-muted font-mono text-sm mb-6 max-w-md mx-auto">
              Connect your Gmail account to get AI-powered email prioritization
              with our 3-tier scoring system. Critical emails float to the top,
              newsletters and bulk mail sink to the bottom.
            </p>
            <div className="flex flex-wrap justify-center gap-4 mb-6 text-xs font-mono">
              <div className="px-3 py-2 bg-red-500/10 border border-red-500/30 rounded text-red-400">
                🔴 Critical - Needs immediate attention
              </div>
              <div className="px-3 py-2 bg-orange-500/10 border border-orange-500/30 rounded text-orange-400">
                🟠 High - Important, respond today
              </div>
              <div className="px-3 py-2 bg-yellow-500/10 border border-yellow-500/30 rounded text-yellow-400">
                🟡 Medium - Standard priority
              </div>
              <div className="px-3 py-2 bg-blue-500/10 border border-blue-500/30 rounded text-blue-400">
                🔵 Low - Can wait
              </div>
              <div className="px-3 py-2 bg-gray-500/10 border border-gray-500/30 rounded text-gray-400">
                ⚪ Defer - Newsletters, bulk mail
              </div>
            </div>
            {status?.configured !== false ? (
              <button
                onClick={handleConnect}
                className="btn btn-primary px-8 py-3"
              >
                Connect Gmail Account
              </button>
            ) : (
              <div className="text-muted font-mono text-sm">
                Gmail integration is not configured. Set GMAIL_CLIENT_ID and
                GMAIL_CLIENT_SECRET environment variables.
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="border-t border-border bg-surface/50 py-4 mt-8">
        <div className="container mx-auto px-4 flex items-center justify-between text-xs text-muted font-mono">
          <span>ARAGORA AI SMART INBOX</span>
          <div className="flex gap-4">
            <Link href="/documents" className="hover:text-accent">
              DOCUMENTS
            </Link>
            <Link href="/connectors" className="hover:text-accent">
              CONNECTORS
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
