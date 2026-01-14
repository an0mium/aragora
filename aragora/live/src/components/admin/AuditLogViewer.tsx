'use client';

import { useState, useEffect, useCallback } from 'react';

interface AuditEvent {
  id: string;
  timestamp: string;
  category: string;
  action: string;
  actor_id: string;
  actor_type: string;
  outcome: string;
  resource_type: string;
  resource_id: string;
  org_id: string | null;
  ip_address: string | null;
  details: Record<string, unknown>;
  hash: string;
}

interface AuditStats {
  total_events: number;
  events_by_category: Record<string, number>;
  events_by_outcome: Record<string, number>;
  recent_events_24h: number;
  integrity_verified: boolean;
}

interface AuditLogViewerProps {
  apiBase?: string;
}

type OutcomeFilter = 'all' | 'success' | 'failure' | 'error';
type CategoryFilter = 'all' | 'auth' | 'data' | 'admin' | 'system' | 'billing';

const CATEGORY_COLORS: Record<string, string> = {
  auth: 'text-acid-cyan border-acid-cyan/30 bg-acid-cyan/10',
  data: 'text-purple border-purple/30 bg-purple/10',
  admin: 'text-crimson border-crimson/30 bg-crimson/10',
  system: 'text-gold border-gold/30 bg-gold/10',
  billing: 'text-acid-green border-acid-green/30 bg-acid-green/10',
};

const OUTCOME_COLORS: Record<string, string> = {
  success: 'text-acid-green',
  failure: 'text-crimson',
  error: 'text-orange-400',
  pending: 'text-gold',
};

export function AuditLogViewer({ apiBase = '/api' }: AuditLogViewerProps) {
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [stats, setStats] = useState<AuditStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<AuditEvent | null>(null);

  // Filters
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<CategoryFilter>('all');
  const [outcomeFilter, setOutcomeFilter] = useState<OutcomeFilter>('all');
  const [dateRange, setDateRange] = useState<{ start: string; end: string }>({
    start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    end: new Date().toISOString().split('T')[0],
  });

  // Pagination
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const limit = 50;

  const fetchEvents = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        limit: String(limit),
        offset: String(offset),
        start_date: new Date(dateRange.start).toISOString(),
        end_date: new Date(dateRange.end + 'T23:59:59').toISOString(),
      });

      if (categoryFilter !== 'all') {
        params.set('category', categoryFilter);
      }
      if (outcomeFilter !== 'all') {
        params.set('outcome', outcomeFilter);
      }
      if (searchQuery) {
        params.set('search', searchQuery);
      }

      const response = await fetch(`${apiBase}/audit/events?${params}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch audit events: ${response.status}`);
      }
      const data = await response.json();
      setEvents(data.events || []);
      setHasMore(data.events?.length === limit);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load audit events');
    } finally {
      setLoading(false);
    }
  }, [apiBase, offset, dateRange, categoryFilter, outcomeFilter, searchQuery]);

  const fetchStats = useCallback(async () => {
    try {
      const response = await fetch(`${apiBase}/audit/stats`);
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch {
      // Stats are optional, don't show error
    }
  }, [apiBase]);

  useEffect(() => {
    fetchEvents();
    fetchStats();
  }, [fetchEvents, fetchStats]);

  const handleExport = async (format: 'json' | 'csv' | 'soc2') => {
    try {
      const response = await fetch(`${apiBase}/audit/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          format,
          start_date: new Date(dateRange.start).toISOString(),
          end_date: new Date(dateRange.end + 'T23:59:59').toISOString(),
        }),
      });

      if (!response.ok) {
        throw new Error('Export failed');
      }

      const blob = await response.blob();
      const filename = response.headers.get('Content-Disposition')?.split('filename=')[1]?.replace(/"/g, '') ||
        `audit_export.${format}`;

      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      alert('Export failed: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  const handleVerifyIntegrity = async () => {
    try {
      const response = await fetch(`${apiBase}/audit/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start_date: new Date(dateRange.start).toISOString(),
          end_date: new Date(dateRange.end + 'T23:59:59').toISOString(),
        }),
      });

      const data = await response.json();
      if (data.verified) {
        alert('Audit log integrity verified successfully');
      } else {
        alert(`Integrity check failed: ${data.total_errors} errors found`);
      }
    } catch (err) {
      alert('Verification failed: ' + (err instanceof Error ? err.message : 'Unknown error'));
    }
  };

  const formatTimestamp = (ts: string) => {
    try {
      return new Date(ts).toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return ts;
    }
  };

  return (
    <div className="bg-surface border border-acid-green/30">
      {/* Header */}
      <div className="px-4 py-3 border-b border-acid-green/20 bg-bg/50 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-acid-green uppercase tracking-wider">
            {'>'} AUDIT LOG VIEWER
          </span>
          {stats && (
            <span className="text-xs font-mono text-text-muted">
              {stats.total_events.toLocaleString()} total events
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleVerifyIntegrity}
            className="px-2 py-1 text-xs font-mono text-acid-cyan border border-acid-cyan/40 hover:bg-acid-cyan/10 transition-colors"
          >
            VERIFY
          </button>
          <div className="relative group">
            <button className="px-2 py-1 text-xs font-mono text-acid-green border border-acid-green/40 hover:bg-acid-green/10 transition-colors">
              EXPORT
            </button>
            <div className="absolute right-0 top-full mt-1 bg-surface border border-acid-green/30 hidden group-hover:block z-10">
              <button
                onClick={() => handleExport('json')}
                className="block w-full px-3 py-2 text-xs font-mono text-left hover:bg-acid-green/10"
              >
                JSON
              </button>
              <button
                onClick={() => handleExport('csv')}
                className="block w-full px-3 py-2 text-xs font-mono text-left hover:bg-acid-green/10"
              >
                CSV
              </button>
              <button
                onClick={() => handleExport('soc2')}
                className="block w-full px-3 py-2 text-xs font-mono text-left hover:bg-acid-green/10"
              >
                SOC2
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Summary */}
      {stats && (
        <div className="px-4 py-3 border-b border-acid-green/10 bg-bg/30">
          <div className="flex items-center gap-6 flex-wrap">
            <div>
              <span className="text-xs font-mono text-text-muted">24H: </span>
              <span className="text-xs font-mono text-acid-green">{stats.recent_events_24h}</span>
            </div>
            {Object.entries(stats.events_by_category || {}).slice(0, 4).map(([cat, count]) => (
              <div key={cat}>
                <span className="text-xs font-mono text-text-muted">{cat.toUpperCase()}: </span>
                <span className={`text-xs font-mono ${CATEGORY_COLORS[cat]?.split(' ')[0] || 'text-text-primary'}`}>
                  {count}
                </span>
              </div>
            ))}
            <div>
              <span className="text-xs font-mono text-text-muted">INTEGRITY: </span>
              <span className={`text-xs font-mono ${stats.integrity_verified ? 'text-acid-green' : 'text-crimson'}`}>
                {stats.integrity_verified ? 'OK' : 'CHECK'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="px-4 py-3 border-b border-acid-green/10 space-y-3">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex-1 min-w-[200px]">
            <input
              type="text"
              placeholder="Search events..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setOffset(0);
              }}
              className="w-full bg-bg border border-border px-3 py-2 text-xs font-mono text-text-primary placeholder-text-muted focus:border-acid-green/50 focus:outline-none"
            />
          </div>

          <div className="flex items-center gap-2">
            <label className="text-xs font-mono text-text-muted">FROM:</label>
            <input
              type="date"
              value={dateRange.start}
              onChange={(e) => {
                setDateRange((prev) => ({ ...prev, start: e.target.value }));
                setOffset(0);
              }}
              className="bg-bg border border-border px-2 py-1.5 text-xs font-mono text-text-primary focus:border-acid-green/50 focus:outline-none"
            />
          </div>

          <div className="flex items-center gap-2">
            <label className="text-xs font-mono text-text-muted">TO:</label>
            <input
              type="date"
              value={dateRange.end}
              onChange={(e) => {
                setDateRange((prev) => ({ ...prev, end: e.target.value }));
                setOffset(0);
              }}
              className="bg-bg border border-border px-2 py-1.5 text-xs font-mono text-text-primary focus:border-acid-green/50 focus:outline-none"
            />
          </div>
        </div>

        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <label className="text-xs font-mono text-text-muted">CATEGORY:</label>
            <select
              value={categoryFilter}
              onChange={(e) => {
                setCategoryFilter(e.target.value as CategoryFilter);
                setOffset(0);
              }}
              className="bg-bg border border-border px-2 py-1.5 text-xs font-mono text-text-primary focus:border-acid-green/50 focus:outline-none"
            >
              <option value="all">ALL</option>
              <option value="auth">AUTH</option>
              <option value="data">DATA</option>
              <option value="admin">ADMIN</option>
              <option value="system">SYSTEM</option>
              <option value="billing">BILLING</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-xs font-mono text-text-muted">OUTCOME:</label>
            <select
              value={outcomeFilter}
              onChange={(e) => {
                setOutcomeFilter(e.target.value as OutcomeFilter);
                setOffset(0);
              }}
              className="bg-bg border border-border px-2 py-1.5 text-xs font-mono text-text-primary focus:border-acid-green/50 focus:outline-none"
            >
              <option value="all">ALL</option>
              <option value="success">SUCCESS</option>
              <option value="failure">FAILURE</option>
              <option value="error">ERROR</option>
            </select>
          </div>
        </div>
      </div>

      {/* Content */}
      {loading ? (
        <div className="p-8">
          <div className="flex items-center justify-center gap-2">
            <div className="w-2 h-2 bg-acid-green rounded-full animate-pulse" />
            <span className="text-xs font-mono text-acid-green">LOADING EVENTS...</span>
          </div>
        </div>
      ) : error ? (
        <div className="p-4">
          <div className="flex items-center gap-2">
            <span className="text-crimson text-xs font-mono">ERROR:</span>
            <span className="text-text-primary text-xs font-mono">{error}</span>
          </div>
          <button
            onClick={fetchEvents}
            className="mt-3 px-3 py-1.5 text-xs font-mono bg-crimson/20 text-crimson border border-crimson/40 hover:bg-crimson/30 transition-colors"
          >
            RETRY
          </button>
        </div>
      ) : events.length === 0 ? (
        <div className="text-center py-8">
          <span className="text-xs font-mono text-text-muted">No audit events found</span>
        </div>
      ) : (
        <>
          <div className="divide-y divide-border">
            {events.map((event) => (
              <AuditEventRow
                key={event.id}
                event={event}
                isSelected={selectedEvent?.id === event.id}
                onClick={() =>
                  setSelectedEvent(selectedEvent?.id === event.id ? null : event)
                }
                formatTimestamp={formatTimestamp}
              />
            ))}
          </div>

          {/* Pagination */}
          <div className="px-4 py-3 border-t border-acid-green/20 flex items-center justify-between">
            <button
              onClick={() => setOffset(Math.max(0, offset - limit))}
              disabled={offset === 0}
              className={`px-3 py-1.5 text-xs font-mono border ${
                offset === 0
                  ? 'text-text-muted border-border cursor-not-allowed'
                  : 'text-acid-green border-acid-green/40 hover:bg-acid-green/10'
              }`}
            >
              PREV
            </button>
            <span className="text-xs font-mono text-text-muted">
              Showing {offset + 1}-{offset + events.length}
            </span>
            <button
              onClick={() => setOffset(offset + limit)}
              disabled={!hasMore}
              className={`px-3 py-1.5 text-xs font-mono border ${
                !hasMore
                  ? 'text-text-muted border-border cursor-not-allowed'
                  : 'text-acid-green border-acid-green/40 hover:bg-acid-green/10'
              }`}
            >
              NEXT
            </button>
          </div>
        </>
      )}

      {/* Detail Panel */}
      {selectedEvent && (
        <AuditEventDetailPanel
          event={selectedEvent}
          onClose={() => setSelectedEvent(null)}
          formatTimestamp={formatTimestamp}
        />
      )}
    </div>
  );
}

interface AuditEventRowProps {
  event: AuditEvent;
  isSelected: boolean;
  onClick: () => void;
  formatTimestamp: (ts: string) => string;
}

function AuditEventRow({ event, isSelected, onClick, formatTimestamp }: AuditEventRowProps) {
  return (
    <div
      onClick={onClick}
      className={`px-4 py-3 cursor-pointer transition-colors flex items-center gap-4 ${
        isSelected ? 'bg-acid-green/10' : 'hover:bg-bg/50'
      }`}
    >
      <div className="flex-shrink-0 w-32">
        <span className="text-xs font-mono text-text-muted">{formatTimestamp(event.timestamp)}</span>
      </div>

      <div className="flex-shrink-0">
        <span
          className={`px-1.5 py-0.5 text-xs font-mono border ${
            CATEGORY_COLORS[event.category] || 'text-text-primary border-border bg-bg/30'
          }`}
        >
          {event.category.toUpperCase()}
        </span>
      </div>

      <div className="flex-1 min-w-0">
        <span className="text-xs font-mono text-text-primary truncate block">{event.action}</span>
      </div>

      <div className="flex-shrink-0 w-24">
        <span className="text-xs font-mono text-acid-cyan truncate block">
          {event.actor_id?.slice(0, 12) || 'system'}
        </span>
      </div>

      <div className="flex-shrink-0">
        <span className={`text-xs font-mono ${OUTCOME_COLORS[event.outcome] || 'text-text-primary'}`}>
          {event.outcome.toUpperCase()}
        </span>
      </div>
    </div>
  );
}

interface AuditEventDetailPanelProps {
  event: AuditEvent;
  onClose: () => void;
  formatTimestamp: (ts: string) => string;
}

function AuditEventDetailPanel({ event, onClose, formatTimestamp }: AuditEventDetailPanelProps) {
  return (
    <div className="border-t border-acid-green/20 bg-bg/50 p-4">
      <div className="flex items-center justify-between mb-4">
        <span className="text-xs font-mono text-acid-green uppercase tracking-wider">
          EVENT DETAILS
        </span>
        <button
          onClick={onClose}
          className="px-2 py-1 text-xs font-mono text-text-muted hover:text-crimson border border-border hover:border-crimson/40 transition-colors"
        >
          CLOSE
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="space-y-3">
          <DetailField label="ID" value={event.id} />
          <DetailField label="TIMESTAMP" value={formatTimestamp(event.timestamp)} />
          <DetailField label="CATEGORY" value={event.category.toUpperCase()} />
          <DetailField label="ACTION" value={event.action} />
        </div>

        <div className="space-y-3">
          <DetailField label="ACTOR ID" value={event.actor_id || 'N/A'} />
          <DetailField label="ACTOR TYPE" value={event.actor_type || 'N/A'} />
          <DetailField label="OUTCOME" value={event.outcome.toUpperCase()} />
          <DetailField label="IP ADDRESS" value={event.ip_address || 'N/A'} />
        </div>

        <div className="space-y-3">
          <DetailField label="RESOURCE TYPE" value={event.resource_type || 'N/A'} />
          <DetailField label="RESOURCE ID" value={event.resource_id || 'N/A'} />
          <DetailField label="ORG ID" value={event.org_id || 'N/A'} />
          <DetailField
            label="HASH"
            value={event.hash ? `${event.hash.slice(0, 16)}...` : 'N/A'}
            mono
          />
        </div>
      </div>

      {Object.keys(event.details || {}).length > 0 && (
        <div className="mt-4 pt-4 border-t border-border">
          <h4 className="text-xs font-mono text-text-muted mb-2">DETAILS</h4>
          <pre className="text-xs font-mono text-text-primary bg-surface p-3 border border-border overflow-x-auto max-h-48">
            {JSON.stringify(event.details, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

interface DetailFieldProps {
  label: string;
  value: string;
  mono?: boolean;
}

function DetailField({ label, value, mono }: DetailFieldProps) {
  return (
    <div>
      <span className="text-xs font-mono text-text-muted block">{label}</span>
      <span className={`text-xs font-mono text-text-primary ${mono ? 'font-mono' : ''}`}>
        {value}
      </span>
    </div>
  );
}

export default AuditLogViewer;
