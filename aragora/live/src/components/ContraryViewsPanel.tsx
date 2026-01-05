'use client';

import { useState, useEffect } from 'react';

interface ContraryView {
  agent: string;
  position: string;
  confidence: number;
  reasoning: string;
  debate_id?: string;
}

interface ContraryViewsPanelProps {
  apiBase?: string;
}

export function ContraryViewsPanel({ apiBase = '' }: ContraryViewsPanelProps) {
  const [views, setViews] = useState<ContraryView[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    const fetchViews = async () => {
      try {
        const response = await fetch(`${apiBase}/api/consensus/contrarian-views`);
        if (response.ok) {
          const data = await response.json();
          setViews(data.views || data || []);
        } else {
          setError('Failed to fetch contrary views');
        }
      } catch (e) {
        setError('Network error');
      } finally {
        setLoading(false);
      }
    };

    fetchViews();
    // Refresh every 30 seconds
    const interval = setInterval(fetchViews, 30000);
    return () => clearInterval(interval);
  }, [apiBase]);

  if (!isExpanded) {
    return (
      <div
        className="border border-acid-green/30 bg-surface/50 p-3 cursor-pointer hover:border-acid-green/50 transition-colors"
        onClick={() => setIsExpanded(true)}
      >
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-mono text-acid-green">
            {'>'} CONTRARY_VIEWS [{views.length}]
          </h3>
          <span className="text-xs text-text-muted">[EXPAND]</span>
        </div>
      </div>
    );
  }

  return (
    <div className="border border-acid-green/30 bg-surface/50 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-mono text-acid-green">
          {'>'} CONTRARY_VIEWS
        </h3>
        <button
          onClick={() => setIsExpanded(false)}
          className="text-xs text-text-muted hover:text-acid-green"
        >
          [COLLAPSE]
        </button>
      </div>

      {loading && (
        <div className="text-xs text-text-muted font-mono animate-pulse">
          Loading dissenting opinions...
        </div>
      )}

      {error && (
        <div className="text-xs text-warning font-mono">{error}</div>
      )}

      {!loading && !error && views.length === 0 && (
        <div className="text-xs text-text-muted font-mono">
          No contrary views recorded yet.
        </div>
      )}

      <div className="space-y-3 max-h-64 overflow-y-auto">
        {views.map((view, idx) => (
          <div
            key={idx}
            className="border border-warning/30 bg-warning/5 p-3 space-y-2"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-warning">
                {view.agent}
              </span>
              <span className="text-xs font-mono text-text-muted">
                {Math.round(view.confidence * 100)}% confident
              </span>
            </div>
            <p className="text-xs text-text leading-relaxed">
              {view.position}
            </p>
            {view.reasoning && (
              <p className="text-xs text-text-muted italic">
                "{view.reasoning}"
              </p>
            )}
          </div>
        ))}
      </div>

      <div className="mt-3 text-[10px] text-text-muted font-mono">
        Dissenting opinions that didn't achieve consensus
      </div>
    </div>
  );
}
