'use client';

import { useState, useEffect, useCallback } from 'react';
import { useToastContext } from '@/context/ToastContext';
import { API_BASE_URL } from '@/config';

interface Connector {
  id: string;
  job_id: string;
  tenant_id: string;
  type?: string;
  schedule: {
    interval_minutes?: number;
    cron_expression?: string;
    enabled: boolean;
  };
  last_run: string | null;
  next_run: string | null;
  consecutive_failures: number;
  is_running?: boolean;
}

interface SchedulerStats {
  total_jobs: number;
  running_syncs: number;
  pending_syncs: number;
  completed_syncs: number;
  failed_syncs: number;
  success_rate: number;
}

interface SyncHistoryEntry {
  run_id: string;
  job_id: string;
  status: string;
  started_at: string;
  completed_at: string | null;
  items_synced: number;
  error: string | null;
}

const connectorTypeIcons: Record<string, string> = {
  github: '🐙',
  s3: '📦',
  postgres: '🐘',
  mongodb: '🍃',
  fhir: '🏥',
};

const connectorTypeColors: Record<string, string> = {
  github: 'border-purple-500 bg-purple-500/10',
  s3: 'border-orange-500 bg-orange-500/10',
  postgres: 'border-blue-500 bg-blue-500/10',
  mongodb: 'border-green-500 bg-green-500/10',
  fhir: 'border-red-500 bg-red-500/10',
};

function ConnectorCard({
  connector,
  onSync,
  onDelete,
  syncing,
}: {
  connector: Connector;
  onSync: () => void;
  onDelete: () => void;
  syncing: boolean;
}) {
  const connectorType = connector.type || connector.id.split(':')[0] || 'unknown';
  const connectorId = connector.id.split(':').pop() || connector.id;

  return (
    <div
      className={`
        p-5 rounded-lg border-2 transition-all
        ${connectorTypeColors[connectorType] || 'border-gray-500 bg-gray-500/10'}
      `}
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <span className="text-3xl">{connectorTypeIcons[connectorType] || '🔗'}</span>
          <div>
            <h3 className="font-mono font-bold text-text">{connectorId}</h3>
            <span className="text-xs text-text-muted font-mono uppercase">
              {connectorType}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {connector.is_running && (
            <span className="px-2 py-1 text-xs bg-acid-green/20 text-acid-green border border-acid-green/50 rounded font-mono animate-pulse">
              SYNCING
            </span>
          )}
          {connector.consecutive_failures > 0 && (
            <span className="px-2 py-1 text-xs bg-red-500/20 text-red-400 border border-red-500/50 rounded font-mono">
              {connector.consecutive_failures} FAILURES
            </span>
          )}
        </div>
      </div>

      {/* Schedule Info */}
      <div className="mb-4 p-3 bg-bg/50 rounded">
        <div className="grid grid-cols-2 gap-2 text-xs font-mono">
          <div>
            <span className="text-text-muted">Schedule:</span>
            <span className="ml-2 text-text">
              {connector.schedule.cron_expression ||
                `Every ${connector.schedule.interval_minutes || 60}m`}
            </span>
          </div>
          <div>
            <span className="text-text-muted">Status:</span>
            <span
              className={`ml-2 ${
                connector.schedule.enabled ? 'text-acid-green' : 'text-text-muted'
              }`}
            >
              {connector.schedule.enabled ? 'ENABLED' : 'DISABLED'}
            </span>
          </div>
          {connector.last_run && (
            <div>
              <span className="text-text-muted">Last Run:</span>
              <span className="ml-2 text-text">
                {new Date(connector.last_run).toLocaleString()}
              </span>
            </div>
          )}
          {connector.next_run && (
            <div>
              <span className="text-text-muted">Next Run:</span>
              <span className="ml-2 text-text">
                {new Date(connector.next_run).toLocaleString()}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <button
          onClick={onSync}
          disabled={syncing || connector.is_running}
          className="flex-1 px-3 py-2 bg-acid-green/20 border border-acid-green/50 text-acid-green font-mono text-sm hover:bg-acid-green/30 disabled:opacity-50 disabled:cursor-not-allowed transition-colors rounded"
        >
          {syncing ? 'SYNCING...' : 'SYNC NOW'}
        </button>
        <button
          onClick={onDelete}
          className="px-3 py-2 bg-red-500/20 border border-red-500/50 text-red-400 font-mono text-sm hover:bg-red-500/30 transition-colors rounded"
        >
          DELETE
        </button>
      </div>
    </div>
  );
}

function AddConnectorModal({
  onClose,
  onAdd,
}: {
  onClose: () => void;
  onAdd: (type: string, config: Record<string, string>) => void;
}) {
  const [type, setType] = useState('github');
  const [config, setConfig] = useState<Record<string, string>>({});

  const configFields: Record<string, { label: string; placeholder: string; required?: boolean }[]> = {
    github: [
      { label: 'owner', placeholder: 'Organization/User', required: true },
      { label: 'repo', placeholder: 'Repository name', required: true },
      { label: 'token', placeholder: 'GitHub token (optional)' },
    ],
    s3: [
      { label: 'bucket', placeholder: 'Bucket name', required: true },
      { label: 'prefix', placeholder: 'Path prefix (optional)' },
      { label: 'region', placeholder: 'AWS region' },
    ],
    postgres: [
      { label: 'host', placeholder: 'Database host', required: true },
      { label: 'database', placeholder: 'Database name', required: true },
      { label: 'schema', placeholder: 'Schema (default: public)' },
    ],
    mongodb: [
      { label: 'connection_string', placeholder: 'MongoDB URI', required: true },
      { label: 'database', placeholder: 'Database name', required: true },
    ],
    fhir: [
      { label: 'base_url', placeholder: 'FHIR server URL', required: true },
      { label: 'organization_id', placeholder: 'Organization ID', required: true },
      { label: 'client_id', placeholder: 'OAuth client ID' },
    ],
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onAdd(type, config);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-bg/80 backdrop-blur-sm">
      <div className="w-full max-w-lg bg-surface border border-border rounded-lg shadow-2xl">
        <div className="flex items-center justify-between p-4 border-b border-border">
          <h2 className="text-lg font-mono font-bold text-text">Add Connector</h2>
          <button onClick={onClose} className="text-text-muted hover:text-text">
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-4">
          {/* Connector Type */}
          <div className="mb-4">
            <label className="block text-xs font-mono text-text-muted uppercase mb-2">
              Connector Type
            </label>
            <div className="grid grid-cols-5 gap-2">
              {Object.entries(connectorTypeIcons).map(([t, icon]) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => {
                    setType(t);
                    setConfig({});
                  }}
                  className={`
                    p-3 rounded border-2 transition-all text-center
                    ${
                      type === t
                        ? 'border-acid-green bg-acid-green/20'
                        : 'border-border hover:border-text'
                    }
                  `}
                >
                  <span className="text-2xl">{icon}</span>
                  <span className="block text-xs font-mono mt-1 capitalize">{t}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Config Fields */}
          <div className="space-y-3 mb-6">
            {configFields[type]?.map((field) => (
              <div key={field.label}>
                <label className="block text-xs font-mono text-text-muted uppercase mb-1">
                  {field.label}
                  {field.required && <span className="text-red-400 ml-1">*</span>}
                </label>
                <input
                  type="text"
                  value={config[field.label] || ''}
                  onChange={(e) =>
                    setConfig({ ...config, [field.label]: e.target.value })
                  }
                  placeholder={field.placeholder}
                  required={field.required}
                  className="w-full px-3 py-2 bg-bg border border-border rounded text-sm font-mono text-text focus:border-acid-green focus:outline-none"
                />
              </div>
            ))}
          </div>

          {/* Actions */}
          <div className="flex gap-2">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 bg-surface border border-border text-text font-mono text-sm hover:border-text transition-colors rounded"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 px-4 py-2 bg-acid-green text-bg font-mono text-sm font-bold hover:bg-acid-green/80 transition-colors rounded"
            >
              Add Connector
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default function ConnectorsPage() {
  const { showToast } = useToastContext();
  const [connectors, setConnectors] = useState<Connector[]>([]);
  const [stats, setStats] = useState<SchedulerStats | null>(null);
  const [history, setHistory] = useState<SyncHistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAddModal, setShowAddModal] = useState(false);
  const [syncingConnectors, setSyncingConnectors] = useState<Set<string>>(new Set());

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      const [connectorsRes, statsRes, historyRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/connectors`),
        fetch(`${API_BASE_URL}/api/connectors/scheduler/stats`),
        fetch(`${API_BASE_URL}/api/connectors/sync/history?limit=10`),
      ]);

      if (connectorsRes.ok) {
        const data = await connectorsRes.json();
        setConnectors(data.connectors || []);
      }

      if (statsRes.ok) {
        const data = await statsRes.json();
        setStats(data);
      }

      if (historyRes.ok) {
        const data = await historyRes.json();
        setHistory(data.history || []);
      }
    } catch (error) {
      showToast('Failed to load connector data', 'error');
    } finally {
      setLoading(false);
    }
  }, [showToast]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleAddConnector = async (type: string, config: Record<string, string>) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/connectors`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type, config }),
      });

      if (!response.ok) throw new Error('Failed to add connector');

      showToast('Connector added successfully', 'success');
      setShowAddModal(false);
      fetchData();
    } catch (error) {
      showToast('Failed to add connector', 'error');
    }
  };

  const handleSync = async (connectorId: string) => {
    try {
      setSyncingConnectors((prev) => new Set(prev).add(connectorId));

      const response = await fetch(
        `${API_BASE_URL}/api/connectors/${connectorId}/sync`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ full_sync: false }),
        }
      );

      if (!response.ok) throw new Error('Failed to trigger sync');

      showToast('Sync started', 'success');
      setTimeout(fetchData, 2000); // Refresh after 2s
    } catch (error) {
      showToast('Failed to trigger sync', 'error');
    } finally {
      setSyncingConnectors((prev) => {
        const next = new Set(prev);
        next.delete(connectorId);
        return next;
      });
    }
  };

  const handleDelete = async (connectorId: string) => {
    if (!confirm('Are you sure you want to delete this connector?')) return;

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/connectors/${connectorId}`,
        { method: 'DELETE' }
      );

      if (!response.ok) throw new Error('Failed to delete connector');

      showToast('Connector deleted', 'success');
      fetchData();
    } catch (error) {
      showToast('Failed to delete connector', 'error');
    }
  };

  return (
    <main className="min-h-screen bg-bg p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-mono font-bold text-text mb-2">
              Enterprise Connectors
            </h1>
            <p className="text-text-muted">
              Connect and sync data from external sources
            </p>
          </div>

          <button
            onClick={() => setShowAddModal(true)}
            className="px-6 py-3 bg-acid-green text-bg font-mono font-bold hover:bg-acid-green/80 transition-colors rounded flex items-center gap-2"
          >
            <span>+</span>
            <span>Add Connector</span>
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="max-w-7xl mx-auto mb-8">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="p-4 bg-surface border border-border rounded-lg">
              <div className="text-2xl font-mono font-bold text-text">
                {stats.total_jobs}
              </div>
              <div className="text-xs text-text-muted font-mono uppercase">
                Total Connectors
              </div>
            </div>
            <div className="p-4 bg-surface border border-border rounded-lg">
              <div className="text-2xl font-mono font-bold text-acid-green">
                {stats.running_syncs}
              </div>
              <div className="text-xs text-text-muted font-mono uppercase">
                Running Syncs
              </div>
            </div>
            <div className="p-4 bg-surface border border-border rounded-lg">
              <div className="text-2xl font-mono font-bold text-text">
                {stats.completed_syncs}
              </div>
              <div className="text-xs text-text-muted font-mono uppercase">
                Completed
              </div>
            </div>
            <div className="p-4 bg-surface border border-border rounded-lg">
              <div className="text-2xl font-mono font-bold text-red-400">
                {stats.failed_syncs}
              </div>
              <div className="text-xs text-text-muted font-mono uppercase">
                Failed
              </div>
            </div>
            <div className="p-4 bg-surface border border-border rounded-lg">
              <div className="text-2xl font-mono font-bold text-text">
                {(stats.success_rate * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-text-muted font-mono uppercase">
                Success Rate
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Connectors Grid */}
        <div className="lg:col-span-2">
          <h2 className="text-lg font-mono font-bold text-text mb-4">
            Active Connectors
          </h2>

          {loading && (
            <div className="flex items-center justify-center py-12">
              <div className="animate-pulse text-text-muted font-mono">
                Loading connectors...
              </div>
            </div>
          )}

          {!loading && connectors.length === 0 && (
            <div className="text-center py-12 bg-surface border border-border rounded-lg">
              <div className="text-4xl mb-4">🔌</div>
              <h3 className="text-lg font-mono font-bold text-text mb-2">
                No connectors configured
              </h3>
              <p className="text-text-muted mb-4">
                Add your first connector to start syncing data
              </p>
              <button
                onClick={() => setShowAddModal(true)}
                className="inline-flex items-center gap-2 px-4 py-2 bg-acid-green text-bg font-mono font-bold hover:bg-acid-green/80 transition-colors rounded"
              >
                Add Connector
              </button>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {connectors.map((connector) => (
              <ConnectorCard
                key={connector.job_id}
                connector={connector}
                onSync={() => handleSync(connector.id)}
                onDelete={() => handleDelete(connector.id)}
                syncing={syncingConnectors.has(connector.id)}
              />
            ))}
          </div>
        </div>

        {/* Sync History */}
        <div>
          <h2 className="text-lg font-mono font-bold text-text mb-4">
            Recent Syncs
          </h2>

          <div className="bg-surface border border-border rounded-lg overflow-hidden">
            {history.length === 0 ? (
              <div className="p-4 text-center text-text-muted text-sm">
                No sync history yet
              </div>
            ) : (
              <div className="divide-y divide-border">
                {history.map((entry) => (
                  <div key={entry.run_id} className="p-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-mono text-text truncate">
                        {entry.job_id.split(':').pop()}
                      </span>
                      <span
                        className={`text-xs font-mono px-2 py-0.5 rounded ${
                          entry.status === 'completed'
                            ? 'bg-green-500/20 text-green-400'
                            : entry.status === 'failed'
                            ? 'bg-red-500/20 text-red-400'
                            : 'bg-yellow-500/20 text-yellow-400'
                        }`}
                      >
                        {entry.status.toUpperCase()}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-xs text-text-muted font-mono">
                      <span>
                        {new Date(entry.started_at).toLocaleTimeString()}
                      </span>
                      <span>{entry.items_synced} items</span>
                    </div>
                    {entry.error && (
                      <div className="mt-1 text-xs text-red-400 truncate">
                        {entry.error}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Add Connector Modal */}
      {showAddModal && (
        <AddConnectorModal
          onClose={() => setShowAddModal(false)}
          onAdd={handleAddConnector}
        />
      )}
    </main>
  );
}
