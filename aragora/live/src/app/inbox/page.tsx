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

export default function InboxPage() {
  const { config: backendConfig } = useBackend();
  const { user, tokens } = useAuth();
  const [status, setStatus] = useState<GmailStatus | null>(null);
  const [syncStatus, setSyncStatus] = useState<SyncStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Use user ID or default
  const userId = user?.id || 'default';

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(
        `${backendConfig.api}/api/gmail/status?user_id=${userId}`,
        {
          headers: { Authorization: `Bearer ${tokens?.access_token || ''}` },
        }
      );
      if (response.ok) {
        const data = await response.json();
        setStatus(data);
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
      const response = await fetch(`${backendConfig.api}/api/gmail/connect`, {
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
        setError('Failed to start connection');
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
            <span className="text-muted font-mono text-sm">{'//'} SMART INBOX</span>
          </div>
          <div className="flex items-center gap-3">
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
              Connect your Gmail account to get AI-powered email prioritization,
              smart search, and natural language Q&A over your inbox.
            </p>
            {status?.configured ? (
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
          <span>ARAGORA SMART INBOX</span>
          <Link href="/documents" className="hover:text-accent">
            DOCUMENTS
          </Link>
        </div>
      </footer>
    </div>
  );
}
