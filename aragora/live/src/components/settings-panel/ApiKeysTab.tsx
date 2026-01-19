'use client';

import { useState } from 'react';
import type { UserPreferences } from './types';

export interface ApiKeysTabProps {
  preferences: UserPreferences;
  onGenerateKey: (name: string) => Promise<string>;
  onRevokeKey: (prefix: string) => void;
}

export function ApiKeysTab({ preferences, onGenerateKey, onRevokeKey }: ApiKeysTabProps) {
  const [newKeyName, setNewKeyName] = useState('');
  const [generatedKey, setGeneratedKey] = useState<string | null>(null);

  const handleGenerateKey = async () => {
    if (!newKeyName.trim()) return;
    const key = await onGenerateKey(newKeyName);
    setGeneratedKey(key);
    setNewKeyName('');
  };

  return (
    <div className="space-y-6" role="tabpanel" id="panel-api" aria-labelledby="tab-api">
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
          />
          <button
            onClick={handleGenerateKey}
            disabled={!newKeyName.trim()}
            className="px-4 py-2 bg-acid-green/20 border border-acid-green/40 text-acid-green font-mono text-sm rounded hover:bg-acid-green/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Generate
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

      <div className="card p-6">
        <h3 className="font-mono text-acid-green mb-4">
          Your API Keys ({preferences.api_keys.length})
        </h3>
        {preferences.api_keys.length === 0 ? (
          <p className="font-mono text-sm text-text-muted">
            No API keys generated yet. Create one to access the Aragora API programmatically.
          </p>
        ) : (
          <div className="space-y-3">
            {preferences.api_keys.map((key) => (
              <div
                key={key.prefix}
                className="flex items-center justify-between p-3 bg-surface rounded border border-acid-green/20"
              >
                <div>
                  <div className="font-mono text-sm text-text">{key.name}</div>
                  <div className="font-mono text-xs text-text-muted">
                    {key.prefix} &middot; Created {new Date(key.created_at).toLocaleDateString()}
                  </div>
                </div>
                <button
                  onClick={() => onRevokeKey(key.prefix)}
                  className="px-3 py-1 text-acid-red font-mono text-xs hover:bg-acid-red/10 rounded transition-colors"
                >
                  Revoke
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="card p-6">
        <h3 className="font-mono text-acid-green mb-2">API Documentation</h3>
        <p className="font-mono text-sm text-text-muted mb-4">
          Use your API key to authenticate requests to the Aragora API.
        </p>
        <pre className="bg-surface p-4 rounded font-mono text-xs text-text overflow-x-auto">
{`curl -X POST https://api.aragora.ai/api/v1/debates \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"task": "Evaluate rate limiting strategies"}'`}
        </pre>
      </div>
    </div>
  );
}

export default ApiKeysTab;
