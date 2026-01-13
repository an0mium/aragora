'use client';

import { useEffect, useState, useCallback } from 'react';
import { useAuth } from '@/context/AuthContext';
import { API_BASE_URL } from '@/config';

const API_BASE = API_BASE_URL;

interface ApiKeyInfo {
  prefix: string;
  created_at: string | null;
  expires_at: string | null;
  has_key: boolean;
}

interface UsageStats {
  total_requests: number;
  requests_today: number;
  requests_this_month: number;
  tokens_used: number;
  cost_usd: number;
}

export default function DeveloperPortal() {
  const { isAuthenticated, tokens, user } = useAuth();
  const [apiKeyInfo, setApiKeyInfo] = useState<ApiKeyInfo | null>(null);
  const [newApiKey, setNewApiKey] = useState<string | null>(null);
  const [usageStats, setUsageStats] = useState<UsageStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);
  const [revoking, setRevoking] = useState(false);
  const [copied, setCopied] = useState(false);
  const accessToken = tokens?.access_token;

  const fetchApiKeyInfo = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/auth/me`, {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
      if (res.ok) {
        const data = await res.json();
        const user = data.user;
        setApiKeyInfo({
          prefix: user.api_key_prefix || null,
          created_at: user.api_key_created_at || null,
          expires_at: user.api_key_expires_at || null,
          has_key: !!user.api_key_prefix,
        });
      }
    } catch (err) {
      console.error('Failed to fetch API key info:', err);
    } finally {
      setLoading(false);
    }
  }, [accessToken]);

  const fetchUsageStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/billing/usage`, {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
      if (res.ok) {
        const data = await res.json();
        setUsageStats({
          total_requests: data.usage?.total_api_calls || 0,
          requests_today: data.usage?.api_calls_today || 0,
          requests_this_month: data.usage?.debates_used || 0,
          tokens_used: data.usage?.tokens_used || 0,
          cost_usd: data.usage?.estimated_cost_usd || 0,
        });
      }
    } catch (err) {
      console.error('Failed to fetch usage stats:', err);
    }
  }, [accessToken]);

  useEffect(() => {
    if (isAuthenticated && accessToken) {
      fetchApiKeyInfo();
      fetchUsageStats();
    } else {
      setLoading(false);
    }
  }, [isAuthenticated, accessToken, fetchApiKeyInfo, fetchUsageStats]);

  const generateApiKey = async () => {
    setGenerating(true);
    setError(null);
    setNewApiKey(null);
    try {
      const res = await fetch(`${API_BASE}/api/auth/api-key`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json',
        },
      });
      const data = await res.json();
      if (res.ok) {
        setNewApiKey(data.api_key);
        setApiKeyInfo({
          prefix: data.prefix,
          created_at: new Date().toISOString(),
          expires_at: data.expires_at,
          has_key: true,
        });
      } else {
        setError(data.error || 'Failed to generate API key');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setGenerating(false);
    }
  };

  const revokeApiKey = async () => {
    if (!confirm('Are you sure you want to revoke your API key? This cannot be undone.')) {
      return;
    }
    setRevoking(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/auth/api-key`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });
      if (res.ok) {
        setApiKeyInfo({
          prefix: '',
          created_at: null,
          expires_at: null,
          has_key: false,
        });
        setNewApiKey(null);
      } else {
        const data = await res.json();
        setError(data.error || 'Failed to revoke API key');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setRevoking(false);
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-background p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-2xl font-mono text-acid-green mb-4">DEVELOPER PORTAL</h1>
          <p className="text-text-muted font-mono">Please log in to access the developer portal.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-2xl font-mono text-acid-green mb-2">DEVELOPER PORTAL</h1>
        <p className="text-text-muted font-mono text-sm mb-8">
          Manage your API keys and monitor usage
        </p>

        {/* API Key Management */}
        <div className="border border-acid-green/30 bg-surface/30 p-6 mb-6">
          <h2 className="text-lg font-mono text-acid-cyan mb-4">API KEY</h2>

          {loading ? (
            <div className="text-xs font-mono text-text-muted">Loading...</div>
          ) : apiKeyInfo?.has_key ? (
            <div className="space-y-4">
              {/* Current Key Info */}
              <div className="space-y-2">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-text-muted">Key Prefix</span>
                  <span className="text-acid-green">{apiKeyInfo.prefix}...</span>
                </div>
                {apiKeyInfo.created_at && (
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-text-muted">Created</span>
                    <span className="text-text">{new Date(apiKeyInfo.created_at).toLocaleDateString()}</span>
                  </div>
                )}
                {apiKeyInfo.expires_at && (
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-text-muted">Expires</span>
                    <span className="text-text">{new Date(apiKeyInfo.expires_at).toLocaleDateString()}</span>
                  </div>
                )}
              </div>

              {/* New Key Display */}
              {newApiKey && (
                <div className="border border-warning/50 bg-warning/10 p-4">
                  <div className="text-xs font-mono text-warning mb-2">
                    Save this key now - it will not be shown again!
                  </div>
                  <div className="flex items-center gap-2">
                    <code className="flex-1 text-xs font-mono bg-background p-2 border border-acid-green/20 text-acid-green break-all">
                      {newApiKey}
                    </code>
                    <button
                      onClick={() => copyToClipboard(newApiKey)}
                      className="px-3 py-2 text-xs font-mono border border-acid-green/50 text-acid-green hover:bg-acid-green/10 transition-colors"
                    >
                      {copied ? 'COPIED!' : 'COPY'}
                    </button>
                  </div>
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-2 pt-2">
                <button
                  onClick={generateApiKey}
                  disabled={generating}
                  className="px-4 py-2 text-xs font-mono border border-acid-cyan/50 text-acid-cyan hover:bg-acid-cyan/10 transition-colors disabled:opacity-50"
                >
                  {generating ? 'GENERATING...' : 'REGENERATE KEY'}
                </button>
                <button
                  onClick={revokeApiKey}
                  disabled={revoking}
                  className="px-4 py-2 text-xs font-mono border border-warning/50 text-warning hover:bg-warning/10 transition-colors disabled:opacity-50"
                >
                  {revoking ? 'REVOKING...' : 'REVOKE KEY'}
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              <p className="text-xs font-mono text-text-muted">
                You don&apos;t have an API key yet. Generate one to access the Aragora API.
              </p>
              <button
                onClick={generateApiKey}
                disabled={generating}
                className="px-4 py-2 text-xs font-mono border border-acid-green/50 text-acid-green hover:bg-acid-green/10 transition-colors disabled:opacity-50"
              >
                {generating ? 'GENERATING...' : 'GENERATE API KEY'}
              </button>
            </div>
          )}

          {error && (
            <div className="mt-4 text-xs font-mono text-warning">{error}</div>
          )}
        </div>

        {/* Usage Statistics */}
        <div className="border border-acid-green/30 bg-surface/30 p-6 mb-6">
          <h2 className="text-lg font-mono text-acid-cyan mb-4">USAGE STATISTICS</h2>

          {usageStats ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="border border-acid-green/20 p-3">
                <div className="text-xs font-mono text-text-muted mb-1">DEBATES</div>
                <div className="text-xl font-mono text-acid-green">{usageStats.requests_this_month}</div>
              </div>
              <div className="border border-acid-green/20 p-3">
                <div className="text-xs font-mono text-text-muted mb-1">TOKENS USED</div>
                <div className="text-xl font-mono text-acid-green">{usageStats.tokens_used.toLocaleString()}</div>
              </div>
              <div className="border border-acid-green/20 p-3">
                <div className="text-xs font-mono text-text-muted mb-1">API CALLS</div>
                <div className="text-xl font-mono text-acid-green">{usageStats.total_requests}</div>
              </div>
              <div className="border border-acid-green/20 p-3">
                <div className="text-xs font-mono text-text-muted mb-1">EST. COST</div>
                <div className="text-xl font-mono text-acid-cyan">${usageStats.cost_usd.toFixed(2)}</div>
              </div>
            </div>
          ) : (
            <div className="text-xs font-mono text-text-muted">Loading usage data...</div>
          )}
        </div>

        {/* Quick Start Guide */}
        <div className="border border-acid-green/30 bg-surface/30 p-6 mb-6">
          <h2 className="text-lg font-mono text-acid-cyan mb-4">QUICK START</h2>

          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-mono text-text mb-2">Authentication</h3>
              <p className="text-xs font-mono text-text-muted mb-2">
                Include your API key in the Authorization header:
              </p>
              <pre className="text-xs font-mono bg-background p-3 border border-acid-green/20 text-acid-green overflow-x-auto">
{`curl -H "Authorization: Bearer YOUR_API_KEY" \\
  ${API_BASE}/api/debates`}
              </pre>
            </div>

            <div>
              <h3 className="text-sm font-mono text-text mb-2">Create a Debate</h3>
              <pre className="text-xs font-mono bg-background p-3 border border-acid-green/20 text-acid-green overflow-x-auto">
{`curl -X POST ${API_BASE}/api/debates \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "topic": "Should AI be regulated?",
    "agents": ["claude", "gpt4"],
    "rounds": 3
  }'`}
              </pre>
            </div>

            <div>
              <h3 className="text-sm font-mono text-text mb-2">SDK Installation</h3>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-text-muted">Python:</span>
                  <code className="text-xs font-mono bg-background px-2 py-1 border border-acid-green/20 text-acid-green">
                    pip install aragora
                  </code>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-text-muted">JavaScript:</span>
                  <code className="text-xs font-mono bg-background px-2 py-1 border border-acid-green/20 text-acid-green">
                    npm install aragora
                  </code>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* API Reference Link */}
        <div className="border border-acid-green/30 bg-surface/30 p-6">
          <h2 className="text-lg font-mono text-acid-cyan mb-4">DOCUMENTATION</h2>
          <div className="flex flex-wrap gap-4">
            <a
              href="/docs/api"
              className="px-4 py-2 text-xs font-mono border border-acid-green/50 text-acid-green hover:bg-acid-green/10 transition-colors"
            >
              API REFERENCE
            </a>
            <a
              href="/docs/sdk"
              className="px-4 py-2 text-xs font-mono border border-acid-green/50 text-acid-green hover:bg-acid-green/10 transition-colors"
            >
              SDK GUIDE
            </a>
            <a
              href="/docs/webhooks"
              className="px-4 py-2 text-xs font-mono border border-acid-green/50 text-acid-green hover:bg-acid-green/10 transition-colors"
            >
              WEBHOOKS
            </a>
            <a
              href="https://github.com/aragora/examples"
              target="_blank"
              rel="noopener noreferrer"
              className="px-4 py-2 text-xs font-mono border border-acid-green/50 text-acid-green hover:bg-acid-green/10 transition-colors"
            >
              EXAMPLES
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
