'use client';

import { useState, useEffect, useCallback } from 'react';
import { useBackend } from '@/components/BackendSelector';
import { IntegrationType, INTEGRATION_CONFIGS } from './IntegrationSetupWizard';

interface IntegrationStatus {
  type: IntegrationType;
  enabled: boolean;
  lastActivity?: string;
  messagesSent: number;
  errors: number;
  status: 'connected' | 'degraded' | 'disconnected' | 'not_configured';
}

interface IntegrationStatusDashboardProps {
  onConfigure: (type: IntegrationType) => void;
  onEdit: (type: IntegrationType, config: Record<string, unknown>) => void;
}

function StatusIndicator({ status }: { status: IntegrationStatus['status'] }) {
  const styles: Record<string, { bg: string; text: string; label: string }> = {
    connected: { bg: 'bg-acid-green/20', text: 'text-acid-green', label: 'CONNECTED' },
    degraded: { bg: 'bg-warning/20', text: 'text-warning', label: 'DEGRADED' },
    disconnected: { bg: 'bg-crimson/20', text: 'text-crimson', label: 'DISCONNECTED' },
    not_configured: { bg: 'bg-text-muted/20', text: 'text-text-muted', label: 'NOT CONFIGURED' },
  };

  const style = styles[status] || styles.not_configured;

  return (
    <span className={`px-2 py-0.5 text-xs font-mono rounded ${style.bg} ${style.text}`}>
      {style.label}
    </span>
  );
}

export function IntegrationStatusDashboard({ onConfigure, onEdit }: IntegrationStatusDashboardProps) {
  const { config: backendConfig } = useBackend();
  const [integrations, setIntegrations] = useState<IntegrationStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const res = await fetch(`${backendConfig.api}/api/integrations/status`);

      if (res.ok) {
        const data = await res.json();
        setIntegrations(data.integrations || []);
      } else if (res.status === 404) {
        // API not implemented yet, use mock data
        setIntegrations(getMockIntegrations());
      } else {
        throw new Error('Failed to fetch integration status');
      }
    } catch (err) {
      // Use mock data for demo
      setIntegrations(getMockIntegrations());
      setError(err instanceof Error ? err.message : 'Failed to load integration status');
    } finally {
      setLoading(false);
    }
  }, [backendConfig.api]);

  useEffect(() => {
    fetchStatus();
    // Refresh every 30 seconds
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const handleDisable = async (type: IntegrationType) => {
    if (!confirm(`Disable ${INTEGRATION_CONFIGS[type].title} integration?`)) return;

    try {
      const res = await fetch(`${backendConfig.api}/api/integrations/${type}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: false }),
      });

      if (res.ok) {
        fetchStatus();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to disable integration');
    }
  };

  const handleDelete = async (type: IntegrationType) => {
    if (!confirm(`Delete ${INTEGRATION_CONFIGS[type].title} configuration? This cannot be undone.`)) return;

    try {
      const res = await fetch(`${backendConfig.api}/api/integrations/${type}`, {
        method: 'DELETE',
      });

      if (res.ok) {
        fetchStatus();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete integration');
    }
  };

  const handleTestConnection = async (type: IntegrationType) => {
    try {
      const res = await fetch(`${backendConfig.api}/api/integrations/${type}/test`, {
        method: 'POST',
      });

      if (res.ok) {
        const data = await res.json();
        alert(data.success ? 'Connection test successful!' : `Test failed: ${data.error}`);
      }
    } catch {
      alert('Connection test failed');
    }
  };

  // Calculate stats
  const connectedCount = integrations.filter(i => i.status === 'connected').length;
  const totalMessages = integrations.reduce((sum, i) => sum + i.messagesSent, 0);
  const totalErrors = integrations.reduce((sum, i) => sum + i.errors, 0);

  if (loading) {
    return (
      <div className="p-6 border border-acid-green/20 rounded bg-surface/30">
        <p className="font-mono text-text-muted text-center">Loading integration status...</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Stats Bar */}
      <div className="grid grid-cols-3 gap-4">
        <div className="p-4 border border-acid-green/20 rounded bg-surface/30 text-center">
          <div className="font-mono text-2xl text-acid-green">{connectedCount}</div>
          <div className="font-mono text-xs text-text-muted">Connected</div>
        </div>
        <div className="p-4 border border-acid-green/20 rounded bg-surface/30 text-center">
          <div className="font-mono text-2xl text-acid-cyan">{totalMessages.toLocaleString()}</div>
          <div className="font-mono text-xs text-text-muted">Messages Sent</div>
        </div>
        <div className="p-4 border border-acid-green/20 rounded bg-surface/30 text-center">
          <div className="font-mono text-2xl text-warning">{totalErrors}</div>
          <div className="font-mono text-xs text-text-muted">Errors (24h)</div>
        </div>
      </div>

      {error && (
        <div className="p-3 border border-warning/30 bg-warning/10 rounded">
          <p className="text-warning font-mono text-sm">{error}</p>
        </div>
      )}

      {/* Integration List */}
      <div className="space-y-3">
        {integrations.map(integration => {
          const config = INTEGRATION_CONFIGS[integration.type];
          const isConfigured = integration.status !== 'not_configured';

          return (
            <div
              key={integration.type}
              className={`p-4 border rounded transition-colors ${
                isConfigured
                  ? 'border-acid-green/30 bg-surface/40'
                  : 'border-acid-green/10 bg-surface/20'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-lg text-acid-cyan">{config.icon}</span>
                  <div>
                    <div className="flex items-center gap-2">
                      <h4 className="font-mono text-text">{config.title}</h4>
                      <StatusIndicator status={integration.status} />
                    </div>
                    <p className="font-mono text-xs text-text-muted mt-0.5">
                      {config.description}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {isConfigured ? (
                    <>
                      <button
                        onClick={() => handleTestConnection(integration.type)}
                        className="px-2 py-1 text-xs font-mono border border-acid-cyan/30 text-acid-cyan hover:bg-acid-cyan/10 transition-colors"
                      >
                        [TEST]
                      </button>
                      <button
                        onClick={() => onEdit(integration.type, {})}
                        className="px-2 py-1 text-xs font-mono border border-acid-green/30 text-text-muted hover:text-text transition-colors"
                      >
                        [EDIT]
                      </button>
                      <button
                        onClick={() => handleDisable(integration.type)}
                        className="px-2 py-1 text-xs font-mono border border-warning/30 text-warning hover:bg-warning/10 transition-colors"
                      >
                        [DISABLE]
                      </button>
                      <button
                        onClick={() => handleDelete(integration.type)}
                        className="px-2 py-1 text-xs font-mono border border-crimson/30 text-crimson hover:bg-crimson/10 transition-colors"
                      >
                        [DELETE]
                      </button>
                    </>
                  ) : (
                    <button
                      onClick={() => onConfigure(integration.type)}
                      className="px-3 py-1 text-xs font-mono border border-acid-green/50 text-acid-green hover:bg-acid-green/10 transition-colors"
                    >
                      [CONFIGURE]
                    </button>
                  )}
                </div>
              </div>

              {isConfigured && (
                <div className="mt-3 pt-3 border-t border-acid-green/10 flex gap-4 text-xs font-mono">
                  <span className="text-text-muted">
                    Messages: <span className="text-acid-cyan">{integration.messagesSent}</span>
                  </span>
                  {integration.errors > 0 && (
                    <span className="text-text-muted">
                      Errors: <span className="text-warning">{integration.errors}</span>
                    </span>
                  )}
                  {integration.lastActivity && (
                    <span className="text-text-muted">
                      Last activity: {new Date(integration.lastActivity).toLocaleString()}
                    </span>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Refresh Button */}
      <div className="text-center">
        <button
          onClick={fetchStatus}
          className="text-xs font-mono text-text-muted hover:text-text transition-colors"
        >
          [REFRESH STATUS]
        </button>
      </div>
    </div>
  );
}

// Mock data for development/demo
function getMockIntegrations(): IntegrationStatus[] {
  return [
    { type: 'slack', enabled: true, status: 'connected', messagesSent: 142, errors: 0, lastActivity: new Date().toISOString() },
    { type: 'discord', enabled: true, status: 'connected', messagesSent: 87, errors: 2, lastActivity: new Date(Date.now() - 3600000).toISOString() },
    { type: 'telegram', enabled: false, status: 'not_configured', messagesSent: 0, errors: 0 },
    { type: 'email', enabled: true, status: 'connected', messagesSent: 256, errors: 1, lastActivity: new Date(Date.now() - 7200000).toISOString() },
    { type: 'teams', enabled: false, status: 'not_configured', messagesSent: 0, errors: 0 },
    { type: 'whatsapp', enabled: false, status: 'not_configured', messagesSent: 0, errors: 0 },
    { type: 'matrix', enabled: false, status: 'not_configured', messagesSent: 0, errors: 0 },
  ];
}

export default IntegrationStatusDashboard;
