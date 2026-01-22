'use client';

import { useState, useEffect, useCallback } from 'react';
import { API_BASE_URL } from '@/config';

interface Breakpoint {
  id: string;
  debate_id: string;
  type: string;
  reason: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  created_at: string;
  context?: Record<string, unknown>;
  options?: string[];
}

interface BreakpointsPanelProps {
  apiBase?: string;
  onBreakpointResolved?: (id: string) => void;
}

export function BreakpointsPanel({ apiBase = API_BASE_URL, onBreakpointResolved }: BreakpointsPanelProps) {
  const [breakpoints, setBreakpoints] = useState<Breakpoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [resolving, setResolving] = useState<string | null>(null);
  const [, _setSelectedAction] = useState<Record<string, string>>({});

  const fetchBreakpoints = useCallback(async () => {
    try {
      const response = await fetch(`${apiBase}/api/breakpoints/pending`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      setBreakpoints(data.breakpoints || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch breakpoints');
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    fetchBreakpoints();
    // Poll for updates every 10 seconds
    const interval = setInterval(fetchBreakpoints, 10000);
    return () => clearInterval(interval);
  }, [fetchBreakpoints]);

  const resolveBreakpoint = async (id: string, action: string) => {
    setResolving(id);
    try {
      const response = await fetch(`${apiBase}/api/breakpoints/${id}/resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, reasoning: `User selected: ${action}` }),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      // Remove resolved breakpoint from list
      setBreakpoints(prev => prev.filter(bp => bp.id !== id));
      onBreakpointResolved?.(id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to resolve breakpoint');
    } finally {
      setResolving(null);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-500 border-red-500/50 bg-red-500/10';
      case 'high': return 'text-orange-500 border-orange-500/50 bg-orange-500/10';
      case 'medium': return 'text-yellow-500 border-yellow-500/50 bg-yellow-500/10';
      default: return 'text-acid-cyan border-acid-cyan/50 bg-acid-cyan/10';
    }
  };

  if (loading) {
    return (
      <div className="card p-6 animate-pulse">
        <div className="h-8 bg-surface rounded w-1/3 mb-4" />
        <div className="space-y-3">
          <div className="h-24 bg-surface rounded" />
          <div className="h-24 bg-surface rounded" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-mono text-acid-green">
          Pending Breakpoints
          {breakpoints.length > 0 && (
            <span className="ml-2 px-2 py-0.5 bg-acid-green/20 text-acid-green text-xs rounded">
              {breakpoints.length}
            </span>
          )}
        </h2>
        <button
          onClick={fetchBreakpoints}
          className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
        >
          [REFRESH]
        </button>
      </div>

      {error && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-sm font-mono">
          Error: {error}
        </div>
      )}

      {breakpoints.length === 0 ? (
        <div className="card p-8 text-center">
          <div className="text-4xl mb-4">&#x2713;</div>
          <p className="text-text-muted font-mono">No pending breakpoints</p>
          <p className="text-text-muted/60 font-mono text-xs mt-2">
            Debates are running smoothly without intervention
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {breakpoints.map((bp) => (
            <div
              key={bp.id}
              className={`card p-4 border-l-4 ${getSeverityColor(bp.severity)}`}
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`text-xs font-mono uppercase px-2 py-0.5 rounded ${getSeverityColor(bp.severity)}`}>
                      {bp.severity}
                    </span>
                    <span className="text-xs font-mono text-text-muted">
                      {bp.type}
                    </span>
                  </div>
                  <h3 className="font-mono text-acid-green">{bp.reason}</h3>
                </div>
                <span className="text-xs font-mono text-text-muted">
                  {new Date(bp.created_at).toLocaleTimeString()}
                </span>
              </div>

              <div className="text-xs font-mono text-text-muted mb-3">
                Debate: <span className="text-acid-cyan">{bp.debate_id}</span>
              </div>

              {bp.context && Object.keys(bp.context).length > 0 && (
                <details className="mb-3">
                  <summary className="text-xs font-mono text-acid-cyan cursor-pointer hover:text-acid-green">
                    [VIEW CONTEXT]
                  </summary>
                  <pre className="mt-2 p-2 bg-surface rounded text-xs overflow-x-auto">
                    {JSON.stringify(bp.context, null, 2)}
                  </pre>
                </details>
              )}

              <div className="flex items-center gap-2 flex-wrap">
                {(bp.options || ['continue', 'pause', 'abort']).map((option) => (
                  <button
                    key={option}
                    onClick={() => resolveBreakpoint(bp.id, option)}
                    disabled={resolving === bp.id}
                    className={`px-3 py-1.5 text-xs font-mono rounded border transition-colors
                      ${option === 'abort'
                        ? 'border-red-500/50 text-red-400 hover:bg-red-500/20'
                        : option === 'continue'
                        ? 'border-acid-green/50 text-acid-green hover:bg-acid-green/20'
                        : 'border-acid-cyan/50 text-acid-cyan hover:bg-acid-cyan/20'
                      }
                      ${resolving === bp.id ? 'opacity-50 cursor-not-allowed' : ''}
                    `}
                  >
                    {resolving === bp.id ? '...' : option.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
