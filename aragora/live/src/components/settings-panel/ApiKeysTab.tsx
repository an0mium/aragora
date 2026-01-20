'use client';

import { useState } from 'react';
import type { UserPreferences, ApiKey } from './types';

export interface ApiKeysTabProps {
  preferences: UserPreferences;
  onGenerateKey: (name: string) => Promise<string>;
  onRevokeKey: (prefix: string) => void;
  apiBase?: string;
}

function formatRelativeTime(dateString: string | null): string {
  if (!dateString) return 'Never';
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

function formatExpirationTime(dateString: string | null | undefined): { text: string; isExpiringSoon: boolean; isExpired: boolean } {
  if (!dateString) return { text: 'Never expires', isExpiringSoon: false, isExpired: false };
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = date.getTime() - now.getTime();
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMs < 0) return { text: 'Expired', isExpiringSoon: false, isExpired: true };
  if (diffDays < 7) return { text: `Expires in ${diffDays}d`, isExpiringSoon: true, isExpired: false };
  if (diffDays < 30) return { text: `Expires in ${diffDays}d`, isExpiringSoon: false, isExpired: false };
  return { text: `Expires ${date.toLocaleDateString()}`, isExpiringSoon: false, isExpired: false };
}

function UsageBar({ used, total, label }: { used: number; total: number; label: string }) {
  const percentage = total > 0 ? Math.min((used / total) * 100, 100) : 0;
  const isHigh = percentage > 80;
  const isMedium = percentage > 50;

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px] font-mono text-text-muted">
        <span>{label}</span>
        <span>{used.toLocaleString()} / {total.toLocaleString()}</span>
      </div>
      <div className="h-1.5 bg-bg rounded-full overflow-hidden">
        <div
          className={`h-full transition-all ${
            isHigh ? 'bg-crimson' : isMedium ? 'bg-acid-yellow' : 'bg-acid-green'
          }`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

function ApiKeyCard({
  apiKey,
  onRevoke,
  apiBase
}: {
  apiKey: ApiKey;
  onRevoke: () => void;
  apiBase?: string;
}) {
  const [showCurl, setShowCurl] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const expiration = formatExpirationTime(apiKey.expires_at);

  const curlExample = `curl -X POST ${apiBase || 'https://api.aragora.ai'}/api/v1/debates \\
  -H "Authorization: Bearer ${apiKey.prefix}..." \\
  -H "Content-Type: application/json" \\
  -d '{"task": "Your debate topic here"}'`;

  const handleCopy = async (text: string, type: string) => {
    await navigator.clipboard.writeText(text);
    setCopied(type);
    setTimeout(() => setCopied(null), 2000);
  };

  return (
    <div className={`p-4 bg-surface rounded border ${
      expiration.isExpired ? 'border-crimson/40' :
      expiration.isExpiringSoon ? 'border-acid-yellow/40' :
      'border-acid-green/20'
    }`}>
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="font-mono text-sm text-text font-medium">{apiKey.name}</div>
          <div className="flex items-center gap-2 mt-1">
            <code className="font-mono text-xs text-text-muted bg-bg px-1.5 py-0.5 rounded">
              {apiKey.prefix}...
            </code>
            <span className="text-text-muted text-[10px]">
              Created {new Date(apiKey.created_at).toLocaleDateString()}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowCurl(!showCurl)}
            className="px-2 py-1 text-[10px] font-mono text-acid-cyan hover:bg-acid-cyan/10 rounded transition-colors"
            title="Show cURL example"
          >
            {showCurl ? 'Hide' : 'cURL'}
          </button>
          <button
            onClick={onRevoke}
            className="px-2 py-1 text-[10px] font-mono text-crimson hover:bg-crimson/10 rounded transition-colors"
          >
            Revoke
          </button>
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-4 gap-3 mb-3">
        <div className="text-center">
          <div className="text-lg font-mono text-acid-green">
            {apiKey.usage?.total_requests?.toLocaleString() || '0'}
          </div>
          <div className="text-[10px] text-text-muted">Total Requests</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-mono text-text">
            {apiKey.usage?.requests_today?.toLocaleString() || '0'}
          </div>
          <div className="text-[10px] text-text-muted">Today</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-mono text-text">
            {apiKey.usage?.avg_latency_ms ? `${apiKey.usage.avg_latency_ms}ms` : '-'}
          </div>
          <div className="text-[10px] text-text-muted">Avg Latency</div>
        </div>
        <div className="text-center">
          <div className="text-lg font-mono text-text">
            {formatRelativeTime(apiKey.last_used)}
          </div>
          <div className="text-[10px] text-text-muted">Last Used</div>
        </div>
      </div>

      {/* Rate Limit Bar */}
      {apiKey.usage && apiKey.usage.rate_limit_total > 0 && (
        <div className="mb-3">
          <UsageBar
            used={apiKey.usage.rate_limit_total - apiKey.usage.rate_limit_remaining}
            total={apiKey.usage.rate_limit_total}
            label="Rate Limit"
          />
        </div>
      )}

      {/* Expiration Warning */}
      {(expiration.isExpired || expiration.isExpiringSoon) && (
        <div className={`text-xs font-mono px-2 py-1 rounded mb-3 ${
          expiration.isExpired ? 'bg-crimson/10 text-crimson' : 'bg-acid-yellow/10 text-acid-yellow'
        }`}>
          {expiration.text}
        </div>
      )}

      {/* cURL Example */}
      {showCurl && (
        <div className="mt-3 pt-3 border-t border-border">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-mono text-text-muted">cURL Example</span>
            <button
              onClick={() => handleCopy(curlExample, 'curl')}
              className={`px-2 py-0.5 text-[10px] font-mono rounded transition-colors ${
                copied === 'curl'
                  ? 'bg-acid-green/20 text-acid-green'
                  : 'text-acid-cyan hover:bg-acid-cyan/10'
              }`}
            >
              {copied === 'curl' ? 'Copied!' : 'Copy'}
            </button>
          </div>
          <pre className="bg-bg p-3 rounded font-mono text-[10px] text-text overflow-x-auto whitespace-pre-wrap">
            {curlExample}
          </pre>
        </div>
      )}
    </div>
  );
}

export function ApiKeysTab({ preferences, onGenerateKey, onRevokeKey, apiBase }: ApiKeysTabProps) {
  const [newKeyName, setNewKeyName] = useState('');
  const [generatedKey, setGeneratedKey] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerateKey = async () => {
    if (!newKeyName.trim() || isGenerating) return;
    setIsGenerating(true);
    try {
      const key = await onGenerateKey(newKeyName);
      setGeneratedKey(key);
      setNewKeyName('');
    } finally {
      setIsGenerating(false);
    }
  };

  // Calculate aggregate stats
  const totalRequests = preferences.api_keys.reduce(
    (sum, key) => sum + (key.usage?.total_requests || 0), 0
  );
  const activeKeys = preferences.api_keys.filter(
    key => !key.expires_at || new Date(key.expires_at) > new Date()
  ).length;

  return (
    <div className="space-y-6" role="tabpanel" id="panel-api" aria-labelledby="tab-api">
      {/* Summary Stats */}
      {preferences.api_keys.length > 0 && (
        <div className="grid grid-cols-3 gap-4 p-4 bg-surface/50 border border-acid-green/20 rounded">
          <div className="text-center">
            <div className="text-2xl font-mono text-acid-green">{activeKeys}</div>
            <div className="text-xs text-text-muted">Active Keys</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-mono text-text">{totalRequests.toLocaleString()}</div>
            <div className="text-xs text-text-muted">Total Requests</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-mono text-text">{preferences.api_keys.length}</div>
            <div className="text-xs text-text-muted">Total Keys</div>
          </div>
        </div>
      )}

      {/* Generate Key */}
      <div className="card p-6">
        <h3 className="font-mono text-acid-green mb-4">Generate API Key</h3>
        <div className="flex gap-3">
          <input
            type="text"
            value={newKeyName}
            onChange={(e) => setNewKeyName(e.target.value)}
            placeholder="Key name (e.g., CI/CD, Development)"
            className="flex-1 bg-surface border border-acid-green/30 rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-acid-green"
            aria-label="API key name"
            onKeyDown={(e) => e.key === 'Enter' && handleGenerateKey()}
          />
          <button
            onClick={handleGenerateKey}
            disabled={!newKeyName.trim() || isGenerating}
            className="px-4 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isGenerating ? 'Generating...' : 'Generate'}
          </button>
        </div>

        {generatedKey && (
          <div className="mt-4 p-4 bg-acid-yellow/10 border border-acid-yellow/30 rounded">
            <div className="font-mono text-xs text-acid-yellow mb-2">
              Copy this key now - it won&apos;t be shown again!
            </div>
            <div className="flex gap-2">
              <code className="flex-1 bg-surface p-2 rounded font-mono text-sm text-text break-all">
                {generatedKey}
              </code>
              <button
                onClick={() => {
                  navigator.clipboard.writeText(generatedKey);
                  setGeneratedKey(null);
                }}
                className="px-3 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30"
              >
                Copy
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Keys List */}
      <div className="card p-6">
        <h3 className="font-mono text-acid-green mb-4">
          Your API Keys ({preferences.api_keys.length})
        </h3>
        {preferences.api_keys.length === 0 ? (
          <p className="font-mono text-sm text-text-muted">
            No API keys generated yet. Create one to access the Aragora API programmatically.
          </p>
        ) : (
          <div className="space-y-4">
            {preferences.api_keys.map((key) => (
              <ApiKeyCard
                key={key.prefix}
                apiKey={key}
                onRevoke={() => onRevokeKey(key.prefix)}
                apiBase={apiBase}
              />
            ))}
          </div>
        )}
      </div>

      {/* Documentation */}
      <div className="card p-6">
        <h3 className="font-mono text-acid-green mb-2">API Documentation</h3>
        <p className="font-mono text-sm text-text-muted mb-4">
          Use your API key to authenticate requests to the Aragora API.
        </p>
        <div className="grid gap-3 sm:grid-cols-2">
          <a
            href="/docs/api"
            className="flex items-center gap-2 p-3 bg-surface border border-acid-green/20 rounded hover:border-acid-green/40 transition-colors"
          >
            <span className="text-acid-green">{">"}</span>
            <span className="font-mono text-sm text-text">Full API Reference</span>
          </a>
          <a
            href="/docs/api#rate-limits"
            className="flex items-center gap-2 p-3 bg-surface border border-acid-green/20 rounded hover:border-acid-green/40 transition-colors"
          >
            <span className="text-acid-green">{">"}</span>
            <span className="font-mono text-sm text-text">Rate Limits & Quotas</span>
          </a>
        </div>
      </div>
    </div>
  );
}

export default ApiKeysTab;
