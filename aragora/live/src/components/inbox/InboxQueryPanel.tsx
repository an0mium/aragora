'use client';

import { useState } from 'react';

interface InboxQueryPanelProps {
  apiBase: string;
  userId: string;
  authToken?: string;
}

export function InboxQueryPanel({ apiBase, userId, authToken }: InboxQueryPanelProps) {
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState<string | null>(null);
  const [isQuerying, setIsQuerying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsQuerying(true);
    setError(null);
    setAnswer(null);

    try {
      const response = await fetch(`${apiBase}/api/inbox/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
        body: JSON.stringify({ query: query.trim(), user_id: userId }),
      });

      if (!response.ok) throw new Error('Failed to query inbox');
      const data = await response.json();
      setAnswer(data.answer || 'No answer found');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="border border-acid-green/30 bg-surface/50 p-4 rounded">
      <h3 className="text-acid-green font-mono text-sm mb-4">Ask About Your Inbox</h3>

      <form onSubmit={handleSubmit} className="space-y-3">
        <div>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about your emails..."
            rows={3}
            className="w-full px-3 py-2 bg-bg border border-acid-green/30 text-text font-mono text-sm rounded focus:outline-none focus:border-acid-green resize-none"
          />
        </div>
        <button
          type="submit"
          disabled={isQuerying || !query.trim()}
          className="w-full px-3 py-2 text-sm font-mono bg-acid-green/10 border border-acid-green/40 text-acid-green hover:bg-acid-green/20 disabled:opacity-50 rounded"
        >
          {isQuerying ? 'Thinking...' : 'Ask'}
        </button>
      </form>

      {error && (
        <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded">
          <p className="text-red-400 text-xs">{error}</p>
        </div>
      )}

      {answer && (
        <div className="mt-4 p-3 bg-acid-green/5 border border-acid-green/20 rounded">
          <p className="text-text text-sm whitespace-pre-wrap">{answer}</p>
        </div>
      )}
    </div>
  );
}
