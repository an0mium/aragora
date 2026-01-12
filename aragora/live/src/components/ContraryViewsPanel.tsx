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
        className="panel panel-compact cursor-pointer hover:border-acid-green/30 transition-colors"
        onClick={() => setIsExpanded(true)}
      >
        <div className="flex items-center justify-between">
          <h3 className="panel-title-sm flex items-center gap-2">
            <span className="text-acid-green">{'>'}</span>
            CONTRARY_VIEWS
            {views.length > 0 && <span className="panel-badge">{views.length}</span>}
          </h3>
          <span className="panel-toggle">[EXPAND]</span>
        </div>
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <h3 className="panel-title-sm flex items-center gap-2">
          <span className="text-acid-green">{'>'}</span>
          CONTRARY_VIEWS
        </h3>
        <button
          onClick={() => setIsExpanded(false)}
          className="panel-toggle hover:text-acid-green transition-colors"
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
                &quot;{view.reasoning}&quot;
              </p>
            )}
          </div>
        ))}
      </div>

      <div className="mt-3 text-[10px] text-text-muted font-mono">
        Dissenting opinions that didn&apos;t achieve consensus
      </div>
    </div>
  );
}
