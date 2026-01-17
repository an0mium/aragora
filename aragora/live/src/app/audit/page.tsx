'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import { useAuth } from '@/context/AuthContext';

interface AuditSession {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  document_ids: string[];
  progress: number;
  findings_count: number;
  findings_by_severity: Record<string, number>;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  duration_seconds?: number;
}

interface AuditStats {
  total_sessions: number;
  active_sessions: number;
  total_findings: number;
  critical_findings: number;
  documents_audited: number;
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    completed: 'bg-acid-green/20 text-acid-green border-acid-green/40',
    running: 'bg-acid-yellow/20 text-acid-yellow border-acid-yellow/40',
    pending: 'bg-acid-blue/20 text-acid-blue border-acid-blue/40',
    paused: 'bg-acid-purple/20 text-acid-purple border-acid-purple/40',
    failed: 'bg-acid-red/20 text-acid-red border-acid-red/40',
    cancelled: 'bg-muted/20 text-muted border-muted/40',
  };
  return (
    <span className={`px-2 py-0.5 text-xs font-mono rounded border ${colors[status] || colors.pending}`}>
      {status.toUpperCase()}
    </span>
  );
}

function SeverityBadge({ severity, count }: { severity: string; count: number }) {
  const colors: Record<string, string> = {
    critical: 'bg-acid-red/20 text-acid-red',
    high: 'bg-acid-orange/20 text-acid-orange',
    medium: 'bg-acid-yellow/20 text-acid-yellow',
    low: 'bg-acid-blue/20 text-acid-blue',
    info: 'bg-muted/20 text-muted',
  };
  if (count === 0) return null;
  return (
    <span className={`px-1.5 py-0.5 text-xs font-mono rounded ${colors[severity]}`}>
      {count}
    </span>
  );
}

function formatDuration(seconds?: number): string {
  if (!seconds) return '-';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
}

function formatDate(dateStr?: string): string {
  if (!dateStr) return '-';
  return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export default function AuditDashboardPage() {
  const { config: backendConfig } = useBackend();
  const { user } = useAuth();
  const [sessions, setSessions] = useState<AuditSession[]>([]);
  const [stats, setStats] = useState<AuditStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const fetchSessions = useCallback(async () => {
    try {
      const response = await fetch(`${backendConfig.url}/api/audit/sessions`, {
        headers: { 'Authorization': `Bearer ${user?.token || ''}` },
      });
      if (response.ok) {
        const data = await response.json();
        setSessions(data.sessions || []);
        setStats(data.stats || null);
      }
    } catch (err) {
      setError('Failed to fetch audit sessions');
    } finally {
      setLoading(false);
    }
  }, [backendConfig.url, user?.token]);

  useEffect(() => {
    fetchSessions();
    const interval = setInterval(fetchSessions, 5000);
    return () => clearInterval(interval);
  }, [fetchSessions]);

  const handlePause = async (sessionId: string) => {
    await fetch(`${backendConfig.url}/api/audit/sessions/${sessionId}/pause`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${user?.token || ''}` },
    });
    fetchSessions();
  };

  const handleResume = async (sessionId: string) => {
    await fetch(`${backendConfig.url}/api/audit/sessions/${sessionId}/resume`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${user?.token || ''}` },
    });
    fetchSessions();
  };

  const handleCancel = async (sessionId: string) => {
    await fetch(`${backendConfig.url}/api/audit/sessions/${sessionId}`, {
      method: 'DELETE',
      headers: { 'Authorization': `Bearer ${user?.token || ''}` },
    });
    fetchSessions();
  };

  const filteredSessions = sessions.filter((s) =>
    statusFilter === 'all' || s.status === statusFilter
  );

  return (
    <div className="min-h-screen bg-background">
      <Scanlines />
      <CRTVignette />

      <header className="border-b border-border bg-surface/50 backdrop-blur-sm sticky top-0 z-40">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="hover:text-accent"><AsciiBannerCompact /></Link>
            <span className="text-muted font-mono text-sm">// AUDIT DASHBOARD</span>
          </div>
          <div className="flex items-center gap-3">
            <BackendSelector />
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-6">
        {/* Stats */}
        <div className="grid grid-cols-5 gap-4 mb-6">
          <div className="card p-4">
            <div className="text-xs text-muted font-mono mb-1">TOTAL AUDITS</div>
            <div className="text-2xl font-bold text-accent">{stats?.total_sessions || 0}</div>
          </div>
          <div className="card p-4">
            <div className="text-xs text-muted font-mono mb-1">ACTIVE</div>
            <div className="text-2xl font-bold text-acid-yellow">{stats?.active_sessions || 0}</div>
          </div>
          <div className="card p-4">
            <div className="text-xs text-muted font-mono mb-1">FINDINGS</div>
            <div className="text-2xl font-bold text-foreground">{stats?.total_findings || 0}</div>
          </div>
          <div className="card p-4">
            <div className="text-xs text-muted font-mono mb-1">CRITICAL</div>
            <div className="text-2xl font-bold text-acid-red">{stats?.critical_findings || 0}</div>
          </div>
          <div className="card p-4">
            <div className="text-xs text-muted font-mono mb-1">DOCS AUDITED</div>
            <div className="text-2xl font-bold text-foreground">{stats?.documents_audited || 0}</div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center justify-between mb-4">
          <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} className="input">
            <option value="all">All Status</option>
            <option value="running">Running</option>
            <option value="completed">Completed</option>
            <option value="paused">Paused</option>
            <option value="failed">Failed</option>
          </select>
          <Link href="/documents" className="btn btn-primary">📁 SELECT DOCUMENTS</Link>
        </div>

        {/* Sessions List */}
        <PanelErrorBoundary title="Audit Sessions">
          <div className="space-y-4">
            {loading ? (
              <div className="card p-8 text-center animate-pulse text-muted font-mono">LOADING...</div>
            ) : filteredSessions.length === 0 ? (
              <div className="card p-8 text-center">
                <div className="text-4xl mb-3">🔍</div>
                <div className="text-muted font-mono">NO AUDIT SESSIONS</div>
                <div className="text-sm text-muted mt-2">Select documents to start an audit</div>
              </div>
            ) : (
              filteredSessions.map((session) => (
                <div key={session.id} className="card p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center gap-3">
                        <Link href={`/audit/${session.id}`} className="font-mono text-lg hover:text-accent">
                          {session.name || session.id.slice(0, 8)}
                        </Link>
                        <StatusBadge status={session.status} />
                      </div>
                      <div className="text-sm text-muted mt-1">
                        {session.document_ids.length} documents • Created {formatDate(session.created_at)}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {session.status === 'running' && (
                        <button onClick={() => handlePause(session.id)} className="btn btn-sm btn-ghost">⏸️ Pause</button>
                      )}
                      {session.status === 'paused' && (
                        <button onClick={() => handleResume(session.id)} className="btn btn-sm btn-ghost">▶️ Resume</button>
                      )}
                      {['pending', 'running', 'paused'].includes(session.status) && (
                        <button onClick={() => handleCancel(session.id)} className="btn btn-sm btn-ghost text-acid-red">✕</button>
                      )}
                      <Link href={`/audit/${session.id}`} className="btn btn-sm btn-primary">View →</Link>
                    </div>
                  </div>

                  {/* Progress bar */}
                  {['running', 'paused'].includes(session.status) && (
                    <div className="mb-3">
                      <div className="w-full bg-surface rounded-full h-2">
                        <div className="bg-accent h-2 rounded-full transition-all" style={{ width: `${session.progress * 100}%` }} />
                      </div>
                      <div className="text-xs text-muted mt-1 font-mono">{Math.round(session.progress * 100)}%</div>
                    </div>
                  )}

                  {/* Findings summary */}
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-muted">Findings:</span>
                      <span className="font-mono">{session.findings_count}</span>
                    </div>
                    {session.findings_by_severity && (
                      <div className="flex items-center gap-1">
                        <SeverityBadge severity="critical" count={session.findings_by_severity.critical || 0} />
                        <SeverityBadge severity="high" count={session.findings_by_severity.high || 0} />
                        <SeverityBadge severity="medium" count={session.findings_by_severity.medium || 0} />
                        <SeverityBadge severity="low" count={session.findings_by_severity.low || 0} />
                      </div>
                    )}
                    {session.duration_seconds && (
                      <div className="text-sm text-muted ml-auto">
                        Duration: {formatDuration(session.duration_seconds)}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </PanelErrorBoundary>
      </main>

      <footer className="border-t border-border bg-surface/50 py-4 mt-8">
        <div className="container mx-auto px-4 flex items-center justify-between text-xs text-muted font-mono">
          <span>ARAGORA AUDIT ENGINE</span>
          <Link href="/documents" className="hover:text-accent">DOCUMENTS →</Link>
        </div>
      </footer>
    </div>
  );
}
