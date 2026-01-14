'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';

interface Webhook {
  id: string;
  url: string;
  events: string[];
  active: boolean;
  created_at: string;
  last_triggered?: string;
  failure_count: number;
}

interface WebhookEvent {
  id: string;
  webhook_id: string;
  event_type: string;
  status: 'success' | 'failed' | 'pending';
  response_code?: number;
  triggered_at: string;
}

const EVENT_TYPES = [
  { id: 'debate.started', label: 'Debate Started', description: 'When a new debate begins' },
  { id: 'debate.completed', label: 'Debate Completed', description: 'When a debate reaches consensus or ends' },
  { id: 'consensus.reached', label: 'Consensus Reached', description: 'When agents reach agreement' },
  { id: 'round.completed', label: 'Round Completed', description: 'After each debate round' },
  { id: 'agent.message', label: 'Agent Message', description: 'When an agent sends a message' },
  { id: 'vote.cast', label: 'Vote Cast', description: 'When a vote is recorded' },
];

function StatusBadge({ active }: { active: boolean }) {
  return (
    <span className={`px-2 py-0.5 text-xs font-mono rounded ${
      active ? 'bg-acid-green/20 text-acid-green' : 'bg-text-muted/20 text-text-muted'
    }`}>
      {active ? 'ACTIVE' : 'INACTIVE'}
    </span>
  );
}

function DeliveryStatus({ status }: { status: string }) {
  const colors: Record<string, string> = {
    success: 'bg-acid-green/20 text-acid-green',
    failed: 'bg-warning/20 text-warning',
    pending: 'bg-acid-cyan/20 text-acid-cyan',
  };
  return (
    <span className={`px-2 py-0.5 text-xs font-mono rounded ${colors[status] || colors.pending}`}>
      {status.toUpperCase()}
    </span>
  );
}

export default function WebhooksPage() {
  const { config: backendConfig } = useBackend();
  const [webhooks, setWebhooks] = useState<Webhook[]>([]);
  const [events, setEvents] = useState<WebhookEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'webhooks' | 'events' | 'create'>('webhooks');

  // Create form state
  const [newUrl, setNewUrl] = useState('');
  const [selectedEvents, setSelectedEvents] = useState<string[]>([]);
  const [creating, setCreating] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const [webhooksRes, eventsRes] = await Promise.all([
        fetch(`${backendConfig.api}/api/webhooks`),
        fetch(`${backendConfig.api}/api/webhooks/events?limit=50`),
      ]);

      if (webhooksRes.ok) {
        const data = await webhooksRes.json();
        setWebhooks(data.webhooks || []);
      }

      if (eventsRes.ok) {
        const data = await eventsRes.json();
        setEvents(data.events || []);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch webhook data');
    } finally {
      setLoading(false);
    }
  }, [backendConfig.api]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleCreate = async () => {
    if (!newUrl || selectedEvents.length === 0) return;

    setCreating(true);
    try {
      const res = await fetch(`${backendConfig.api}/api/webhooks`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          url: newUrl,
          events: selectedEvents,
        }),
      });

      if (res.ok) {
        setNewUrl('');
        setSelectedEvents([]);
        setActiveTab('webhooks');
        fetchData();
      } else {
        const data = await res.json();
        setError(data.error || 'Failed to create webhook');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create webhook');
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this webhook?')) return;

    try {
      const res = await fetch(`${backendConfig.api}/api/webhooks/${id}`, {
        method: 'DELETE',
      });

      if (res.ok) {
        fetchData();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete webhook');
    }
  };

  const handleToggle = async (id: string, active: boolean) => {
    try {
      const res = await fetch(`${backendConfig.api}/api/webhooks/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active: !active }),
      });

      if (res.ok) {
        fetchData();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle webhook');
    }
  };

  const toggleEvent = (eventId: string) => {
    setSelectedEvents(prev =>
      prev.includes(eventId)
        ? prev.filter(e => e !== eventId)
        : [...prev, eventId]
    );
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
            <div className="flex items-center gap-3">
              <Link
                href="/"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [DASHBOARD]
              </Link>
              <Link
                href="/api-explorer"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [API]
              </Link>
              <Link
                href="/settings"
                className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors"
              >
                [SETTINGS]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">
              {'>'} WEBHOOKS
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Configure webhooks to receive real-time notifications for debate events.
              Integrate with external systems, Slack, Discord, or custom applications.
            </p>
          </div>

          {error && (
            <div className="mb-6 p-4 border border-warning/30 bg-warning/10 rounded">
              <p className="text-warning font-mono text-sm">{error}</p>
            </div>
          )}

          {/* Tab Navigation */}
          <div className="flex gap-2 mb-6">
            <button
              onClick={() => setActiveTab('webhooks')}
              className={`px-4 py-2 font-mono text-sm border transition-colors ${
                activeTab === 'webhooks'
                  ? 'border-acid-green bg-acid-green/10 text-acid-green'
                  : 'border-acid-green/30 text-text-muted hover:text-text'
              }`}
            >
              [WEBHOOKS] ({webhooks.length})
            </button>
            <button
              onClick={() => setActiveTab('events')}
              className={`px-4 py-2 font-mono text-sm border transition-colors ${
                activeTab === 'events'
                  ? 'border-acid-green bg-acid-green/10 text-acid-green'
                  : 'border-acid-green/30 text-text-muted hover:text-text'
              }`}
            >
              [DELIVERY LOG]
            </button>
            <button
              onClick={() => setActiveTab('create')}
              className={`px-4 py-2 font-mono text-sm border transition-colors ${
                activeTab === 'create'
                  ? 'border-acid-cyan bg-acid-cyan/10 text-acid-cyan'
                  : 'border-acid-cyan/30 text-acid-cyan hover:bg-acid-cyan/5'
              }`}
            >
              [+ NEW]
            </button>
          </div>

          <PanelErrorBoundary panelName="Webhooks">
            {loading ? (
              <div className="p-8 text-center">
                <p className="font-mono text-text-muted">Loading webhooks...</p>
              </div>
            ) : activeTab === 'webhooks' ? (
              <div className="space-y-4">
                {webhooks.length === 0 ? (
                  <div className="p-8 border border-acid-green/20 rounded text-center">
                    <p className="font-mono text-text-muted mb-4">No webhooks configured yet.</p>
                    <button
                      onClick={() => setActiveTab('create')}
                      className="px-4 py-2 border border-acid-cyan/50 text-acid-cyan font-mono text-sm hover:bg-acid-cyan/10 transition-colors"
                    >
                      [CREATE WEBHOOK]
                    </button>
                  </div>
                ) : (
                  webhooks.map(webhook => (
                    <div key={webhook.id} className="p-4 border border-acid-green/20 rounded bg-surface/30">
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <code className="text-acid-cyan text-sm">{webhook.url}</code>
                          <div className="flex items-center gap-2 mt-1">
                            <StatusBadge active={webhook.active} />
                            {webhook.failure_count > 0 && (
                              <span className="text-xs font-mono text-warning">
                                {webhook.failure_count} failures
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <button
                            onClick={() => handleToggle(webhook.id, webhook.active)}
                            className="px-2 py-1 text-xs font-mono border border-acid-green/30 text-text-muted hover:text-text transition-colors"
                          >
                            {webhook.active ? '[DISABLE]' : '[ENABLE]'}
                          </button>
                          <button
                            onClick={() => handleDelete(webhook.id)}
                            className="px-2 py-1 text-xs font-mono border border-warning/30 text-warning hover:bg-warning/10 transition-colors"
                          >
                            [DELETE]
                          </button>
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {webhook.events.map(event => (
                          <span key={event} className="px-2 py-0.5 text-xs font-mono bg-acid-green/10 text-acid-green rounded">
                            {event}
                          </span>
                        ))}
                      </div>
                      {webhook.last_triggered && (
                        <div className="mt-2 text-xs font-mono text-text-muted">
                          Last triggered: {new Date(webhook.last_triggered).toLocaleString()}
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            ) : activeTab === 'events' ? (
              <div className="space-y-2">
                {events.length === 0 ? (
                  <div className="p-8 border border-acid-green/20 rounded text-center">
                    <p className="font-mono text-text-muted">No delivery events recorded yet.</p>
                  </div>
                ) : (
                  events.map(event => (
                    <div key={event.id} className="p-3 border border-acid-green/10 rounded bg-surface/20 flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <DeliveryStatus status={event.status} />
                        <span className="font-mono text-sm text-acid-cyan">{event.event_type}</span>
                        {event.response_code && (
                          <span className="font-mono text-xs text-text-muted">
                            HTTP {event.response_code}
                          </span>
                        )}
                      </div>
                      <span className="font-mono text-xs text-text-muted">
                        {new Date(event.triggered_at).toLocaleString()}
                      </span>
                    </div>
                  ))
                )}
              </div>
            ) : (
              <div className="p-6 border border-acid-cyan/30 rounded bg-surface/30">
                <h3 className="font-mono text-acid-cyan mb-4">Create New Webhook</h3>

                <div className="mb-4">
                  <label className="block font-mono text-sm text-text-muted mb-2">
                    Webhook URL
                  </label>
                  <input
                    type="url"
                    value={newUrl}
                    onChange={(e) => setNewUrl(e.target.value)}
                    placeholder="https://your-server.com/webhook"
                    className="w-full bg-bg border border-acid-green/30 px-3 py-2 text-sm font-mono text-text focus:outline-none focus:border-acid-green"
                  />
                </div>

                <div className="mb-4">
                  <label className="block font-mono text-sm text-text-muted mb-2">
                    Events to Subscribe
                  </label>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {EVENT_TYPES.map(event => (
                      <button
                        key={event.id}
                        onClick={() => toggleEvent(event.id)}
                        className={`p-3 border text-left transition-colors ${
                          selectedEvents.includes(event.id)
                            ? 'border-acid-green bg-acid-green/10'
                            : 'border-acid-green/20 hover:border-acid-green/40'
                        }`}
                      >
                        <div className="font-mono text-sm text-text">{event.label}</div>
                        <div className="font-mono text-xs text-text-muted">{event.description}</div>
                      </button>
                    ))}
                  </div>
                </div>

                <button
                  onClick={handleCreate}
                  disabled={creating || !newUrl || selectedEvents.length === 0}
                  className="px-4 py-2 bg-acid-cyan/20 border border-acid-cyan/50 text-acid-cyan font-mono text-sm hover:bg-acid-cyan/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {creating ? 'Creating...' : '[CREATE WEBHOOK]'}
                </button>
              </div>
            )}
          </PanelErrorBoundary>
        </div>

        {/* Footer */}
        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">
            {'='.repeat(40)}
          </div>
          <p className="text-text-muted">
            {'>'} ARAGORA // WEBHOOKS
          </p>
        </footer>
      </main>
    </>
  );
}
