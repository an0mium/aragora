'use client';

import { useState, useEffect, useCallback } from 'react';
import { useBackend } from '@/components/BackendSelector';
import { useAuth } from '@/context/AuthContext';
import {
  IntegrationType,
  INTEGRATION_CONFIGS,
  MASKED_SECRET_FIELD_VALUE,
} from './IntegrationSetupWizard';

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

const MASKED_SECRET_DISPLAY_VALUE = '••••••••';

async function readResponseMessage(response: Response, fallback: string) {
  const text = await response.text().catch(() => '');
  if (!text) {
    return fallback;
  }

  try {
    const data = JSON.parse(text) as { error?: unknown; message?: unknown };
    if (typeof data.error === 'string' && data.error) {
      return data.error;
    }
    if (typeof data.message === 'string' && data.message) {
      return data.message;
    }
  } catch {
    return text;
  }

  return text;
}

function mapStatusLoadError(status: number, message: string) {
  if (status === 401 || status === 403) {
    return 'Sign in to load live integration status. Demo data is not shown here.';
  }
  if (status === 404) {
    return 'This server does not expose live integration status. Demo data is not shown here.';
  }
  return message;
}

function buildEditConfig(
  type: IntegrationType,
  integration: Record<string, unknown>
): Record<string, unknown> {
  const settings = (
    integration.settings && typeof integration.settings === 'object'
      ? integration.settings
      : {}
  ) as Record<string, unknown>;
  const config = INTEGRATION_CONFIGS[type];
  const nextConfig: Record<string, unknown> = {};

  for (const field of config.fields) {
    const rawValue = settings[field.key];
    if (rawValue === undefined) {
      continue;
    }
    nextConfig[field.key] =
      field.type === 'password' && rawValue === MASKED_SECRET_DISPLAY_VALUE
        ? MASKED_SECRET_FIELD_VALUE
        : rawValue;
  }

  for (const option of config.notificationOptions) {
    if (integration[option.key] !== undefined) {
      nextConfig[option.key] = integration[option.key];
      continue;
    }
    if (settings[option.key] !== undefined) {
      nextConfig[option.key] = settings[option.key];
    }
  }

  return nextConfig;
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
  const { tokens } = useAuth();
  const [integrations, setIntegrations] = useState<IntegrationStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingType, setEditingType] = useState<IntegrationType | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      if (!tokens?.access_token) {
        setIntegrations([]);
        setError('Sign in to load live integration status. Demo data is not shown here.');
        return;
      }

      const headers: HeadersInit = {};
      headers['Authorization'] = `Bearer ${tokens.access_token}`;

      const res = await fetch(`${backendConfig.api}/api/integrations/status`, { headers });

      if (res.ok) {
        const data = await res.json();
        setIntegrations(data.integrations || []);
      } else {
        const message = await readResponseMessage(res, 'Failed to fetch integration status');
        setIntegrations([]);
        setError(mapStatusLoadError(res.status, message));
      }
    } catch (err) {
      setIntegrations([]);
      setError(err instanceof Error ? err.message : 'Failed to load integration status');
    } finally {
      setLoading(false);
    }
  }, [backendConfig.api, tokens?.access_token]);

  useEffect(() => {
    fetchStatus();
    // Refresh every 30 seconds
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const handleDisable = async (type: IntegrationType) => {
    if (!confirm(`Disable ${INTEGRATION_CONFIGS[type].title} integration?`)) return;

    try {
      if (!tokens?.access_token) {
        throw new Error('Sign in to update integrations.');
      }
      const headers: HeadersInit = { 'Content-Type': 'application/json' };
      headers['Authorization'] = `Bearer ${tokens.access_token}`;

      const res = await fetch(`${backendConfig.api}/api/integrations/${type}`, {
        method: 'PATCH',
        headers,
        body: JSON.stringify({ enabled: false }),
      });

      if (res.ok) {
        fetchStatus();
        return;
      }

      throw new Error(await readResponseMessage(res, 'Failed to disable integration'));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to disable integration');
    }
  };

  const handleDelete = async (type: IntegrationType) => {
    if (!confirm(`Delete ${INTEGRATION_CONFIGS[type].title} configuration? This cannot be undone.`)) return;

    try {
      if (!tokens?.access_token) {
        throw new Error('Sign in to delete integrations.');
      }
      const headers: HeadersInit = {};
      headers['Authorization'] = `Bearer ${tokens.access_token}`;

      const res = await fetch(`${backendConfig.api}/api/integrations/${type}`, {
        method: 'DELETE',
        headers,
      });

      if (res.ok) {
        fetchStatus();
        return;
      }

      throw new Error(await readResponseMessage(res, 'Failed to delete integration'));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete integration');
    }
  };

  const handleTestConnection = async (type: IntegrationType) => {
    try {
      if (!tokens?.access_token) {
        throw new Error('Sign in to test integrations.');
      }
      const headers: HeadersInit = {};
      headers['Authorization'] = `Bearer ${tokens.access_token}`;

      const res = await fetch(`${backendConfig.api}/api/integrations/${type}/test`, {
        method: 'POST',
        headers,
      });

      if (res.ok) {
        const data = await res.json();
        alert(data.success ? 'Connection test successful!' : `Test failed: ${data.error}`);
        return;
      }

      throw new Error(await readResponseMessage(res, 'Connection test failed'));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Connection test failed');
    }
  };

  const handleEditClick = async (type: IntegrationType) => {
    try {
      if (!tokens?.access_token) {
        throw new Error('Sign in to edit integrations.');
      }

      setEditingType(type);
      setError(null);

      const res = await fetch(`${backendConfig.api}/api/integrations/${type}`, {
        headers: {
          Authorization: `Bearer ${tokens.access_token}`,
        },
      });

      if (!res.ok) {
        throw new Error(await readResponseMessage(res, `Failed to load ${INTEGRATION_CONFIGS[type].title} configuration`));
      }

      const data = await res.json();
      onEdit(type, buildEditConfig(type, data.integration || {}));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load integration configuration');
    } finally {
      setEditingType(null);
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
      {integrations.length > 0 && (
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
      )}

      {error && (
        <div className="p-3 border border-warning/30 bg-warning/10 rounded">
          <p className="text-warning font-mono text-sm">{error}</p>
        </div>
      )}

      {/* Integration List */}
      <div className="space-y-3">
        {integrations.length === 0 && (
          <div className="p-6 border border-acid-green/20 rounded bg-surface/20 text-center">
            <p className="font-mono text-sm text-text-muted">
              No live integration status is available.
            </p>
          </div>
        )}

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
                        onClick={() => handleEditClick(integration.type)}
                        disabled={editingType === integration.type}
                        className="px-2 py-1 text-xs font-mono border border-acid-green/30 text-text-muted hover:text-text transition-colors"
                      >
                        {editingType === integration.type ? '[LOADING...]' : '[EDIT]'}
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

export default IntegrationStatusDashboard;
