'use client';

import { useState, useEffect, useCallback, useRef } from 'react';

interface TeamsWorkspace {
  tenant_id: string;
  tenant_name: string;
  connected_at: string;
  is_active: boolean;
}

interface TeamsTenantApiRecord {
  tenant_id?: string;
  tenant_name?: string;
  connected_at?: string;
  installed_at_iso?: string;
  is_active?: boolean;
}

interface TeamsAppWizardProps {
  onClose: () => void;
  onComplete: () => void;
  apiBaseUrl?: string;
}

type WizardStep = 'check' | 'consent' | 'test' | 'complete';

function normalizeWorkspace(record: TeamsTenantApiRecord): TeamsWorkspace | null {
  if (!record.tenant_id || !record.tenant_name) {
    return null;
  }

  return {
    tenant_id: record.tenant_id,
    tenant_name: record.tenant_name,
    connected_at: record.connected_at ?? record.installed_at_iso ?? '',
    is_active: record.is_active ?? true,
  };
}

export function TeamsAppWizard({
  onClose,
  onComplete,
  apiBaseUrl = ''
}: TeamsAppWizardProps) {
  const [step, setStep] = useState<WizardStep>('check');
  const [error, setError] = useState<string | null>(null);
  const [isConfigured, setIsConfigured] = useState<boolean | null>(null);
  const [workspace, setWorkspace] = useState<TeamsWorkspace | null>(null);
  const [testStatus, setTestStatus] = useState<'idle' | 'testing' | 'success' | 'failed'>('idle');
  const [loading, setLoading] = useState(false);
  const popupPollRef = useRef<number | null>(null);

  const clearPopupPoll = useCallback(() => {
    if (popupPollRef.current !== null) {
      window.clearInterval(popupPollRef.current);
      popupPollRef.current = null;
    }
  }, []);

  const loadConnectedWorkspace = useCallback(async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/v1/sme/teams/tenants`);
      if (!response.ok) {
        return false;
      }

      const data = await response.json();
      const records = Array.isArray(data.workspaces) ? data.workspaces : [];
      const activeRecord =
        records.find((record: TeamsTenantApiRecord) => record.is_active !== false) ?? records[0];
      const nextWorkspace = activeRecord ? normalizeWorkspace(activeRecord) : null;

      if (!nextWorkspace) {
        return false;
      }

      setWorkspace(nextWorkspace);
      setStep('test');
      return true;
    } catch {
      return false;
    }
  }, [apiBaseUrl]);

  const refreshWorkspaceAfterConsent = useCallback(async () => {
    setLoading(true);
    setError(null);
    const connected = await loadConnectedWorkspace();
    if (!connected) {
      setError(
        'Teams consent finished, but no connected tenant was found yet. Try again in a moment.'
      );
    }
    setLoading(false);
  }, [loadConnectedWorkspace]);

  const checkConfiguration = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiBaseUrl}/api/v1/integrations/teams/status`);
      if (!response.ok) {
        setIsConfigured(false);
        return;
      }

      const data = await response.json();
      const configured = Boolean(data.app_id_configured && data.password_configured);
      setIsConfigured(configured);

      if (!configured) {
        return;
      }

      const hasWorkspace = await loadConnectedWorkspace();
      if (!hasWorkspace) {
        setStep('consent');
      }
    } catch {
      setIsConfigured(false);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, loadConnectedWorkspace]);

  useEffect(() => {
    if (step === 'check') {
      void checkConfiguration();
    }
  }, [step, checkConfiguration]);

  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (event.data?.type === 'teams-oauth-complete') {
        const nextWorkspace = normalizeWorkspace(event.data.workspace ?? {});
        if (nextWorkspace) {
          setWorkspace(nextWorkspace);
          setStep('test');
        }
      } else if (event.data?.type === 'teams-oauth-error') {
        setError(event.data.error || 'OAuth flow failed');
      }
    };

    window.addEventListener('message', handleMessage);
    return () => {
      clearPopupPoll();
      window.removeEventListener('message', handleMessage);
    };
  }, [clearPopupPoll]);

  const startAdminConsent = () => {
    const width = 600;
    const height = 700;
    const left = window.screenX + (window.outerWidth - width) / 2;
    const top = window.screenY + (window.outerHeight - height) / 2;

    const popup = window.open(
      `${apiBaseUrl}/api/v1/sme/teams/oauth/start?host=${encodeURIComponent(window.location.host)}`,
      'teams-oauth',
      `width=${width},height=${height},left=${left},top=${top},popup=yes`
    );

    if (!popup) {
      setError('Popup blocked. Please allow popups and try again.');
      return;
    }

    clearPopupPoll();
    popupPollRef.current = window.setInterval(() => {
      if (!popup.closed) {
        return;
      }

      clearPopupPoll();
      void refreshWorkspaceAfterConsent();
    }, 1000);
  };

  const handleClose = () => {
    clearPopupPoll();
    onClose();
  };

  const testConnection = async () => {
    if (!workspace) {
      setError('Connect a Teams tenant before running a test.');
      return;
    }

    setTestStatus('testing');
    setError(null);

    try {
      const response = await fetch(
        `${apiBaseUrl}/api/v1/sme/teams/tenants/${encodeURIComponent(workspace.tenant_id)}/test`,
        {
          method: 'POST',
        }
      );

      if (!response.ok) {
        throw new Error('Connection test failed');
      }

      setTestStatus('success');
      setTimeout(() => setStep('complete'), 1500);
    } catch (err) {
      setTestStatus('failed');
      setError(err instanceof Error ? err.message : 'Connection test failed');
    }
  };

  const renderStep = () => {
    switch (step) {
      case 'check':
        return (
          <div className="text-center py-8">
            {loading ? (
              <>
                <div className="animate-pulse font-mono text-acid-cyan mb-4">
                  [CHECKING CONFIGURATION...]
                </div>
                <p className="font-mono text-sm text-text-muted">
                  Verifying Microsoft Teams connector configuration
                </p>
              </>
            ) : isConfigured === false ? (
              <>
                <div className="font-mono text-warning text-4xl mb-4">!</div>
                <h3 className="font-mono text-lg text-text mb-2">
                  Teams OAuth Not Configured
                </h3>
                <p className="font-mono text-sm text-text-muted mb-4">
                  The server needs Microsoft Entra ID app credentials.
                </p>
                <div className="bg-bg/50 border border-acid-green/20 p-4 rounded text-left">
                  <p className="font-mono text-xs text-text-muted mb-2">
                    1. Register an app in{' '}
                    <a
                      href="https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps/ApplicationsListBlade"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-acid-cyan hover:underline"
                    >
                      Azure Portal
                    </a>
                  </p>
                  <p className="font-mono text-xs text-text-muted mb-2">
                    2. Add the following environment variables:
                  </p>
                  <pre className="font-mono text-xs text-acid-green bg-bg p-2 rounded overflow-x-auto">
{`TEAMS_APP_ID=your_application_id
TEAMS_APP_PASSWORD=your_client_secret
TEAMS_TENANT_ID=your_tenant_id`}
                  </pre>
                  <p className="font-mono text-xs text-text-muted mt-2">
                    3. Configure API permissions: ChannelMessage.Send, Team.ReadBasic.All
                  </p>
                </div>
              </>
            ) : (
              <div className="font-mono text-acid-green">
                Configuration verified. Proceeding to admin consent...
              </div>
            )}
          </div>
        );

      case 'consent':
        return (
          <div className="text-center py-8">
            <div className="font-mono text-acid-cyan text-4xl mb-4">T#</div>
            <h3 className="font-mono text-lg text-text mb-2">
              Admin Consent Required
            </h3>
            <p className="font-mono text-sm text-text-muted mb-6">
              A Microsoft 365 admin must grant consent for Aragora to access your organization.
              After the popup closes, this wizard will refresh and attach the connected tenant.
            </p>
            <button
              onClick={startAdminConsent}
              disabled={loading}
              className="px-6 py-3 bg-acid-green/20 border border-acid-green text-acid-green font-mono text-sm hover:bg-acid-green/30 transition-colors disabled:opacity-50"
            >
              {loading ? '[WAITING FOR TENANT...]' : '[START ADMIN CONSENT]'}
            </button>
            <p className="font-mono text-xs text-text-muted mt-4">
              Required permissions: Send channel messages, Read team info
            </p>
          </div>
        );

      case 'test':
        return (
          <div className="text-center py-8">
            {workspace && (
              <div className="flex items-center justify-center gap-2 mb-4 p-3 bg-acid-green/10 border border-acid-green/30 rounded">
                <span className="font-mono text-acid-green">Connected to:</span>
                <span className="font-mono text-text">{workspace.tenant_name}</span>
              </div>
            )}
            <h3 className="font-mono text-lg text-text mb-2">
              Test Connection
            </h3>
            <p className="font-mono text-sm text-text-muted mb-6">
              Run a tenant health check to verify the stored Teams credentials are usable.
            </p>
            <p className="font-mono text-xs text-text-muted mb-6">
              Channel subscriptions are configured separately after the tenant is connected.
            </p>

            <button
              onClick={testConnection}
              disabled={testStatus === 'testing'}
              className={`px-6 py-3 font-mono text-sm border transition-colors ${
                testStatus === 'success'
                  ? 'bg-acid-green/20 border-acid-green text-acid-green'
                  : testStatus === 'failed'
                  ? 'bg-warning/20 border-warning text-warning'
                  : 'bg-acid-cyan/20 border-acid-cyan text-acid-cyan hover:bg-acid-cyan/30'
              }`}
            >
              {testStatus === 'testing' && '[RUNNING CONNECTION TEST...]'}
              {testStatus === 'success' && '[TEST SUCCESSFUL!]'}
              {testStatus === 'failed' && '[TEST FAILED - TRY AGAIN]'}
              {testStatus === 'idle' && '[RUN CONNECTION TEST]'}
            </button>

            {testStatus === 'success' && (
              <p className="font-mono text-sm text-acid-green mt-4">
                Teams credentials validated successfully.
              </p>
            )}
          </div>
        );

      case 'complete':
        return (
          <div className="text-center py-8">
            <div className="font-mono text-acid-green text-4xl mb-4">✓</div>
            <h3 className="font-mono text-lg text-text mb-2">
              Teams Integration Complete!
            </h3>
            <p className="font-mono text-sm text-text-muted mb-6">
              Aragora is now connected to Microsoft Teams. Configure channel subscriptions from the
              Teams workspace controls when you are ready to route receipts.
            </p>
          </div>
        );
    }
  };

  const canGoBack = step !== 'check' && step !== 'complete';
  const canGoNext = step === 'test' && testStatus === 'success';

  const handleBack = () => {
    if (step === 'consent') setStep('check');
    else if (step === 'test') setStep('consent');
  };

  const handleNext = () => {
    if (step === 'test' && testStatus === 'success') {
      setStep('complete');
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-bg/80 backdrop-blur-sm"
        onClick={handleClose}
      />

      <div className="relative bg-surface border border-acid-green/30 rounded-lg w-full max-w-xl max-h-[90vh] overflow-hidden">
        <div className="p-4 border-b border-acid-green/20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="font-mono text-acid-cyan text-xl">T#</span>
            <div>
              <h2 className="font-mono text-acid-green text-lg">
                Microsoft Teams Setup
              </h2>
              <p className="font-mono text-xs text-text-muted">
                Connect Aragora to your Microsoft 365 organization
              </p>
            </div>
          </div>
          <button
            onClick={handleClose}
            className="text-text-muted hover:text-text font-mono"
          >
            [X]
          </button>
        </div>

        <div className="p-6">
          {error && (
            <div className="mb-4 p-3 border border-warning/30 bg-warning/10 rounded">
              <p className="text-warning font-mono text-sm">{error}</p>
            </div>
          )}

          {renderStep()}
        </div>

        <div className="p-4 border-t border-acid-green/20 flex justify-between">
          {canGoBack ? (
            <button
              onClick={handleBack}
              className="px-4 py-2 border border-acid-green/30 text-text-muted font-mono text-sm hover:text-text transition-colors"
            >
              [BACK]
            </button>
          ) : (
            <button
              onClick={handleClose}
              className="px-4 py-2 border border-acid-green/30 text-text-muted font-mono text-sm hover:text-text transition-colors"
            >
              [CANCEL]
            </button>
          )}

          {step === 'complete' ? (
            <button
              onClick={onComplete}
              className="px-4 py-2 bg-acid-green/20 border border-acid-green text-acid-green font-mono text-sm hover:bg-acid-green/30 transition-colors"
            >
              [DONE]
            </button>
          ) : canGoNext ? (
            <button
              onClick={handleNext}
              disabled={loading}
              className="px-4 py-2 border border-acid-green/50 text-acid-green font-mono text-sm hover:bg-acid-green/10 transition-colors disabled:opacity-50"
            >
              {loading ? '[SAVING...]' : '[NEXT]'}
            </button>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export default TeamsAppWizard;
