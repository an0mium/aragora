'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { ErrorWithRetry } from '@/components/ErrorWithRetry';

interface QueueStats {
  pending: number;
  running: number;
  completed: number;
  failed: number;
  total: number;
  avg_wait_time_ms?: number;
  avg_processing_time_ms?: number;
}

interface Worker {
  id: string;
  status: 'idle' | 'busy' | 'offline';
  current_job_id?: string;
  jobs_processed: number;
  last_heartbeat?: string;
}

interface Job {
  id: string;
  type: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  priority: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error?: string;
  result?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

const statusColors: Record<string, string> = {
  pending: 'text-acid-yellow',
  running: 'text-acid-cyan animate-pulse',
  completed: 'text-acid-green',
  failed: 'text-crimson',
  cancelled: 'text-text-muted',
  idle: 'text-acid-green',
  busy: 'text-acid-cyan',
  offline: 'text-text-muted',
};

export default function QueuePage() {
  const { config: backendConfig } = useBackend();
  const [stats, setStats] = useState<QueueStats | null>(null);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      setRefreshing(true);
      const [statsRes, workersRes, jobsRes] = await Promise.all([
        fetch(`${backendConfig.api}/api/queue/stats`),
        fetch(`${backendConfig.api}/api/queue/workers`),
        fetch(`${backendConfig.api}/api/queue/jobs?limit=50`),
      ]);

      if (statsRes.ok) {
        const data = await statsRes.json();
        setStats(data.stats || data);
      }

      if (workersRes.ok) {
        const data = await workersRes.json();
        setWorkers(data.workers || []);
      }

      if (jobsRes.ok) {
        const data = await jobsRes.json();
        setJobs(data.jobs || []);
      }

      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch queue data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [backendConfig.api]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000); // Refresh every 5s
    return () => clearInterval(interval);
  }, [fetchData]);

  const handleRetryJob = async (jobId: string) => {
    try {
      const res = await fetch(`${backendConfig.api}/api/queue/jobs/${jobId}/retry`, {
        method: 'POST',
      });
      if (res.ok) {
        fetchData();
      }
    } catch (err) {
      console.error('Failed to retry job:', err);
    }
  };

  const handleCancelJob = async (jobId: string) => {
    try {
      const res = await fetch(`${backendConfig.api}/api/queue/jobs/${jobId}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        fetchData();
      }
    } catch (err) {
      console.error('Failed to cancel job:', err);
    }
  };

  const filteredJobs = statusFilter === 'all'
    ? jobs
    : jobs.filter(j => j.status === statusFilter);

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        {/* Header */}
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/">
              <AsciiBannerCompact connected={true} />
            </Link>
            <div className="flex items-center gap-4">
              <BackendSelector />
              <ThemeToggle />
            </div>
          </div>
        </header>

        <div className="container mx-auto px-4 py-8">
          {/* Title */}
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-2xl font-mono font-bold text-acid-green mb-2">
                [QUEUE_MONITOR]
              </h1>
              <p className="text-text-muted font-mono text-sm">
                Job queue status and worker management
              </p>
            </div>
            <button
              onClick={fetchData}
              disabled={refreshing}
              className="px-4 py-2 font-mono text-sm border border-acid-green/50 text-acid-green hover:bg-acid-green/10 transition-colors disabled:opacity-50"
            >
              {refreshing ? '[REFRESHING...]' : '[REFRESH]'}
            </button>
          </div>

          {error && (
            <ErrorWithRetry error={error} onRetry={fetchData} className="mb-6" />
          )}

          {loading ? (
            <div className="text-center py-12">
              <div className="text-acid-green font-mono animate-pulse">
                Loading queue data...
              </div>
            </div>
          ) : (
            <>
              {/* Stats Cards */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
                <div className="card p-4 text-center">
                  <div className="text-3xl font-mono text-acid-yellow">{stats?.pending || 0}</div>
                  <div className="text-xs font-mono text-text-muted">Pending</div>
                </div>
                <div className="card p-4 text-center">
                  <div className="text-3xl font-mono text-acid-cyan">{stats?.running || 0}</div>
                  <div className="text-xs font-mono text-text-muted">Running</div>
                </div>
                <div className="card p-4 text-center">
                  <div className="text-3xl font-mono text-acid-green">{stats?.completed || 0}</div>
                  <div className="text-xs font-mono text-text-muted">Completed</div>
                </div>
                <div className="card p-4 text-center">
                  <div className="text-3xl font-mono text-crimson">{stats?.failed || 0}</div>
                  <div className="text-xs font-mono text-text-muted">Failed</div>
                </div>
                <div className="card p-4 text-center">
                  <div className="text-3xl font-mono text-text">{stats?.total || 0}</div>
                  <div className="text-xs font-mono text-text-muted">Total</div>
                </div>
              </div>

              {/* Workers Section */}
              <div className="card p-4 mb-8">
                <h2 className="text-lg font-mono font-bold text-acid-green mb-4">[WORKERS]</h2>
                {workers.length === 0 ? (
                  <div className="text-text-muted font-mono text-sm">
                    No workers registered. Redis queue may not be configured.
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {workers.map((worker) => (
                      <div key={worker.id} className="bg-bg p-3 rounded border border-border">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-mono text-sm truncate">{worker.id}</span>
                          <span className={`font-mono text-xs uppercase ${statusColors[worker.status]}`}>
                            {worker.status}
                          </span>
                        </div>
                        <div className="text-xs text-text-muted font-mono">
                          Jobs processed: {worker.jobs_processed}
                        </div>
                        {worker.current_job_id && (
                          <div className="text-xs text-acid-cyan font-mono mt-1">
                            Current: {worker.current_job_id}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Jobs Section */}
              <div className="card p-4">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-mono font-bold text-acid-green">[JOBS]</h2>
                  <div className="flex gap-2">
                    {['all', 'pending', 'running', 'completed', 'failed'].map((status) => (
                      <button
                        key={status}
                        onClick={() => setStatusFilter(status)}
                        className={`px-3 py-1 font-mono text-xs border transition-colors ${
                          statusFilter === status
                            ? 'border-acid-green bg-acid-green/20 text-acid-green'
                            : 'border-border text-text-muted hover:border-acid-green/50'
                        }`}
                      >
                        {status.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>

                {filteredJobs.length === 0 ? (
                  <div className="text-text-muted font-mono text-sm text-center py-8">
                    No jobs found{statusFilter !== 'all' ? ` with status "${statusFilter}"` : ''}.
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full font-mono text-sm">
                      <thead>
                        <tr className="border-b border-border">
                          <th className="text-left py-2 px-2 text-text-muted">ID</th>
                          <th className="text-left py-2 px-2 text-text-muted">Type</th>
                          <th className="text-left py-2 px-2 text-text-muted">Status</th>
                          <th className="text-left py-2 px-2 text-text-muted">Priority</th>
                          <th className="text-left py-2 px-2 text-text-muted">Created</th>
                          <th className="text-left py-2 px-2 text-text-muted">Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredJobs.map((job) => (
                          <tr key={job.id} className="border-b border-border/50 hover:bg-surface/50">
                            <td className="py-2 px-2 truncate max-w-[120px]" title={job.id}>
                              {job.id.slice(0, 8)}...
                            </td>
                            <td className="py-2 px-2">{job.type}</td>
                            <td className={`py-2 px-2 ${statusColors[job.status]}`}>
                              {job.status.toUpperCase()}
                            </td>
                            <td className="py-2 px-2">{job.priority}</td>
                            <td className="py-2 px-2 text-text-muted">
                              {new Date(job.created_at).toLocaleTimeString()}
                            </td>
                            <td className="py-2 px-2">
                              <div className="flex gap-2">
                                {job.status === 'failed' && (
                                  <button
                                    onClick={() => handleRetryJob(job.id)}
                                    className="text-acid-cyan hover:text-acid-green text-xs"
                                  >
                                    [RETRY]
                                  </button>
                                )}
                                {(job.status === 'pending' || job.status === 'running') && (
                                  <button
                                    onClick={() => handleCancelJob(job.id)}
                                    className="text-crimson hover:text-crimson/80 text-xs"
                                  >
                                    [CANCEL]
                                  </button>
                                )}
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              {/* Performance Stats */}
              {stats?.avg_wait_time_ms !== undefined && (
                <div className="grid grid-cols-2 gap-4 mt-8">
                  <div className="card p-4">
                    <div className="text-text-muted font-mono text-xs mb-1">Avg Wait Time</div>
                    <div className="text-xl font-mono text-acid-green">
                      {(stats.avg_wait_time_ms / 1000).toFixed(2)}s
                    </div>
                  </div>
                  <div className="card p-4">
                    <div className="text-text-muted font-mono text-xs mb-1">Avg Processing Time</div>
                    <div className="text-xl font-mono text-acid-cyan">
                      {((stats.avg_processing_time_ms || 0) / 1000).toFixed(2)}s
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </main>
    </>
  );
}
