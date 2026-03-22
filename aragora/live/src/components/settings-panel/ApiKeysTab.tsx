'use client';

import { useState } from 'react';
import type { UserPreferences, ApiKey } from './types';

export interface ApiKeysTabProps {
  preferences: UserPreferences;
  onGenerateKey: () => Promise<string>;
  onRevokeKey: (prefix: string) => Promise<void>;
  apiBase?: string;
  loading?: boolean;
  error?: string | null;
  singleKeyMode?: boolean;
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

function ApiKeyCard({
  apiKey,
  onRevoke,
  apiBase
}: {
  apiKey: ApiKey;
  onRevoke: () => Promise<void>;
  apiBase?: string;
}) {
  const [showCurl, setShowCurl] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [isRevoking, setIsRevoking] = useState(false);
  const expiration = formatExpirationTime(apiKey.expires_at);
  const createdAtLabel = apiKey.created_at
    ? new Date(apiKey.created_at).toLocaleDateString()
    : 'Unknown';

  const curlExample = `curl -X POST ${apiBase || 'https://api.aragora.ai'}/api/v1/debates \\
  -H "Authorization: Bearer ${apiKey.prefix}..." \\
  -H "Content-Type: application/json" \\
  -d '{"task": "Your debate topic here"}'`;

  const handleCopy = async (text: string, type: string) => {
    await navigator.clipboard.writeText(text);
    setCopied(type);
    setTimeout(() => setCopied(null), 2000);
  };

  const handleRevoke = async () => {
    if (isRevoking) return;
    setIsRevoking(true);
    try {
      await onRevoke();
    } finally {
      setIsRevoking(false);
    }
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
          <div className="font-mono text-sm text-text font-medium">
            {apiKey.name || 'Active key'}
          </div>
          <div className="flex items-center gap-2 mt-1">
            <code className="font-mono text-xs text-text-muted bg-bg px-1.5 py-0.5 rounded">
              {apiKey.prefix}...
            </code>
            <span className="text-text-muted text-[10px]">
              Created {createdAtLabel}
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
            onClick={handleRevoke}
            disabled={isRevoking}
            className="px-2 py-1 text-[10px] font-mono text-crimson hover:bg-crimson/10 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isRevoking ? 'Revoking...' : 'Revoke'}
          </button>
        </div>
      </div>

      <div className="grid gap-3 mb-3 sm:grid-cols-2">
        <div className="rounded border border-border bg-bg/60 p-3">
          <div className="text-[10px] font-mono text-text-muted uppercase tracking-wide">
            Status
          </div>
          <div className={`mt-1 font-mono text-sm ${
            expiration.isExpired ? 'text-crimson' : 'text-acid-green'
          }`}>
            {expiration.isExpired ? 'Expired' : 'Active'}
          </div>
        </div>
        <div className="rounded border border-border bg-bg/60 p-3">
          <div className="text-[10px] font-mono text-text-muted uppercase tracking-wide">
            Expiration
          </div>
          <div className={`mt-1 font-mono text-sm ${
            expiration.isExpired
              ? 'text-crimson'
              : expiration.isExpiringSoon
              ? 'text-acid-yellow'
              : 'text-text'
          }`}>
            {expiration.text}
          </div>
        </div>
      </div>

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

export function ApiKeysTab({
  preferences,
  onGenerateKey,
  onRevokeKey,
  apiBase,
  loading = false,
  error = null,
  singleKeyMode = false,
}: ApiKeysTabProps) {
  const [generatedKey, setGeneratedKey] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerateKey = async () => {
    if (isGenerating) return;
    setIsGenerating(true);
    try {
      const key = await onGenerateKey();
      setGeneratedKey(key);
    } finally {
      setIsGenerating(false);
    }
  };

  const activeKeys = preferences.api_keys.filter(
    key => !key.expires_at || new Date(key.expires_at) > new Date()
  ).length;
  const hasExistingKey = preferences.api_keys.length > 0;

  return (
    <div className="space-y-6" role="tabpanel" id="panel-api" aria-labelledby="tab-api">
      {error && (
        <div className="rounded border border-crimson/40 bg-crimson/10 px-4 py-3 font-mono text-sm text-crimson">
          {error}
        </div>
      )}

      {/* Generate Key */}
      <div className="card p-6">
        <h3 className="font-mono text-acid-green mb-2">Personal API Key</h3>
        <p className="mb-4 font-mono text-sm text-text-muted">
          {singleKeyMode
            ? 'This backend currently supports one active personal API key per account. Generating a new key rotates the current key, and the full value is only shown once.'
            : 'Generate an API key for authenticated requests to the Aragora API.'}
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={handleGenerateKey}
            disabled={isGenerating || loading}
            className="px-4 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isGenerating ? 'Generating...' : hasExistingKey ? 'Rotate key' : 'Generate key'}
          </button>
          {singleKeyMode && (
            <span className="font-mono text-xs text-text-muted">
              Active keys: {activeKeys} / 1
            </span>
          )}
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
          {singleKeyMode ? 'Active API Key' : `Your API Keys (${preferences.api_keys.length})`}
        </h3>
        {loading ? (
          <p className="font-mono text-sm text-text-muted">
            Loading API keys...
          </p>
        ) : preferences.api_keys.length === 0 ? (
          <p className="font-mono text-sm text-text-muted">
            No API key generated yet. Create one to access the Aragora API programmatically.
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
