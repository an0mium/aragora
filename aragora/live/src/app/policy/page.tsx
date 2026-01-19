'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { ErrorWithRetry } from '@/components/ErrorWithRetry';

interface Policy {
  id: string;
  name: string;
  description: string;
  type: 'content' | 'output' | 'behavior' | 'custom';
  severity: 'low' | 'medium' | 'high' | 'critical';
  enabled: boolean;
  rules: PolicyRule[];
  created_at: string;
  updated_at?: string;
  violation_count?: number;
}

interface PolicyRule {
  id: string;
  pattern?: string;
  action: 'warn' | 'block' | 'flag' | 'redact';
  message: string;
}

interface Violation {
  id: string;
  policy_id: string;
  policy_name?: string;
  content_snippet: string;
  severity: string;
  status: 'open' | 'resolved' | 'ignored';
  created_at: string;
  resolved_at?: string;
}

interface ComplianceStats {
  total_policies: number;
  active_policies: number;
  total_violations: number;
  open_violations: number;
  resolved_violations: number;
  compliance_score: number;
}

const severityColors: Record<string, string> = {
  low: 'text-text-muted',
  medium: 'text-acid-yellow',
  high: 'text-warning',
  critical: 'text-crimson',
};

const typeIcons: Record<string, string> = {
  content: '#',
  output: '>',
  behavior: '!',
  custom: '*',
};

export default function PolicyPage() {
  const { config: backendConfig } = useBackend();
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [stats, setStats] = useState<ComplianceStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'policies' | 'violations'>('policies');
  const [selectedPolicy, setSelectedPolicy] = useState<Policy | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const [policiesRes, violationsRes, statsRes] = await Promise.all([
        fetch(`${backendConfig.api}/api/policies`),
        fetch(`${backendConfig.api}/api/compliance/violations?limit=50`),
        fetch(`${backendConfig.api}/api/compliance/stats`),
      ]);

      if (policiesRes.ok) {
        const data = await policiesRes.json();
        setPolicies(data.policies || []);
      }

      if (violationsRes.ok) {
        const data = await violationsRes.json();
        setViolations(data.violations || []);
      }

      if (statsRes.ok) {
        const data = await statsRes.json();
        setStats(data.stats || data);
      }

      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch policy data');
    } finally {
      setLoading(false);
    }
  }, [backendConfig.api]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleTogglePolicy = async (policyId: string) => {
    try {
      const res = await fetch(`${backendConfig.api}/api/policies/${policyId}/toggle`, {
        method: 'POST',
      });
      if (res.ok) {
        fetchData();
      }
    } catch (err) {
      console.error('Failed to toggle policy:', err);
    }
  };

  const handleResolveViolation = async (violationId: string) => {
    try {
      const res = await fetch(`${backendConfig.api}/api/compliance/violations/${violationId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'resolved' }),
      });
      if (res.ok) {
        fetchData();
      }
    } catch (err) {
      console.error('Failed to resolve violation:', err);
    }
  };

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
                [POLICY_ADMIN]
              </h1>
              <p className="text-text-muted font-mono text-sm">
                Compliance policies and violation tracking
              </p>
            </div>
            <button
              onClick={() => setShowCreateModal(true)}
              className="px-4 py-2 font-mono text-sm bg-acid-green/20 border border-acid-green text-acid-green hover:bg-acid-green/30 transition-colors"
            >
              [+ NEW POLICY]
            </button>
          </div>

          {error && (
            <ErrorWithRetry message={error} onRetry={fetchData} className="mb-6" />
          )}

          {loading ? (
            <div className="text-center py-12">
              <div className="text-acid-green font-mono animate-pulse">
                Loading policy data...
              </div>
            </div>
          ) : (
            <>
              {/* Stats Cards */}
              {stats && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                  <div className="card p-4 text-center">
                    <div className="text-3xl font-mono text-acid-green">
                      {stats.compliance_score?.toFixed(0) || 100}%
                    </div>
                    <div className="text-xs font-mono text-text-muted">Compliance Score</div>
                  </div>
                  <div className="card p-4 text-center">
                    <div className="text-3xl font-mono text-accent">
                      {stats.active_policies}/{stats.total_policies}
                    </div>
                    <div className="text-xs font-mono text-text-muted">Active Policies</div>
                  </div>
                  <div className="card p-4 text-center">
                    <div className="text-3xl font-mono text-crimson">{stats.open_violations}</div>
                    <div className="text-xs font-mono text-text-muted">Open Violations</div>
                  </div>
                  <div className="card p-4 text-center">
                    <div className="text-3xl font-mono text-acid-cyan">{stats.resolved_violations}</div>
                    <div className="text-xs font-mono text-text-muted">Resolved</div>
                  </div>
                </div>
              )}

              {/* Tabs */}
              <div className="flex gap-4 mb-6 border-b border-border">
                <button
                  onClick={() => setActiveTab('policies')}
                  className={`px-4 py-2 font-mono text-sm border-b-2 transition-colors ${
                    activeTab === 'policies'
                      ? 'border-acid-green text-acid-green'
                      : 'border-transparent text-text-muted hover:text-text'
                  }`}
                >
                  POLICIES ({policies.length})
                </button>
                <button
                  onClick={() => setActiveTab('violations')}
                  className={`px-4 py-2 font-mono text-sm border-b-2 transition-colors ${
                    activeTab === 'violations'
                      ? 'border-acid-green text-acid-green'
                      : 'border-transparent text-text-muted hover:text-text'
                  }`}
                >
                  VIOLATIONS ({violations.filter(v => v.status === 'open').length} open)
                </button>
              </div>

              {/* Policies Tab */}
              {activeTab === 'policies' && (
                <div className="space-y-4">
                  {policies.length === 0 ? (
                    <div className="card p-8 text-center">
                      <div className="text-text-muted font-mono">
                        No policies defined. Create your first compliance policy.
                      </div>
                    </div>
                  ) : (
                    policies.map((policy) => (
                      <div
                        key={policy.id}
                        className="card p-4 hover:border-acid-green/50 transition-colors cursor-pointer"
                        onClick={() => setSelectedPolicy(selectedPolicy?.id === policy.id ? null : policy)}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex items-start gap-3">
                            <span className="text-acid-green font-mono text-lg">
                              {typeIcons[policy.type] || '#'}
                            </span>
                            <div>
                              <div className="flex items-center gap-2">
                                <h3 className="font-mono font-bold text-text">{policy.name}</h3>
                                <span className={`text-xs font-mono uppercase ${severityColors[policy.severity]}`}>
                                  [{policy.severity}]
                                </span>
                              </div>
                              <p className="text-sm text-text-muted mt-1">{policy.description}</p>
                              {policy.violation_count !== undefined && policy.violation_count > 0 && (
                                <span className="text-xs text-crimson font-mono mt-2 inline-block">
                                  {policy.violation_count} violation{policy.violation_count !== 1 ? 's' : ''}
                                </span>
                              )}
                            </div>
                          </div>
                          <div className="flex items-center gap-3">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleTogglePolicy(policy.id);
                              }}
                              className={`px-3 py-1 font-mono text-xs border transition-colors ${
                                policy.enabled
                                  ? 'border-acid-green text-acid-green hover:bg-acid-green/10'
                                  : 'border-text-muted text-text-muted hover:border-text'
                              }`}
                            >
                              {policy.enabled ? '[ENABLED]' : '[DISABLED]'}
                            </button>
                          </div>
                        </div>

                        {/* Expanded details */}
                        {selectedPolicy?.id === policy.id && (
                          <div className="mt-4 pt-4 border-t border-border">
                            <h4 className="font-mono text-sm text-acid-green mb-2">Rules:</h4>
                            {policy.rules && policy.rules.length > 0 ? (
                              <div className="space-y-2">
                                {policy.rules.map((rule) => (
                                  <div key={rule.id} className="bg-bg p-2 rounded text-sm font-mono">
                                    <span className="text-acid-cyan">[{rule.action.toUpperCase()}]</span>
                                    {' '}{rule.message}
                                    {rule.pattern && (
                                      <span className="text-text-muted ml-2">/{rule.pattern}/</span>
                                    )}
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="text-text-muted text-sm">No rules defined</div>
                            )}
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              )}

              {/* Violations Tab */}
              {activeTab === 'violations' && (
                <div className="card">
                  {violations.length === 0 ? (
                    <div className="p-8 text-center">
                      <div className="text-text-muted font-mono">
                        No violations recorded. Your content is compliant.
                      </div>
                    </div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full font-mono text-sm">
                        <thead>
                          <tr className="border-b border-border">
                            <th className="text-left py-3 px-4 text-text-muted">Policy</th>
                            <th className="text-left py-3 px-4 text-text-muted">Content</th>
                            <th className="text-left py-3 px-4 text-text-muted">Severity</th>
                            <th className="text-left py-3 px-4 text-text-muted">Status</th>
                            <th className="text-left py-3 px-4 text-text-muted">Date</th>
                            <th className="text-left py-3 px-4 text-text-muted">Actions</th>
                          </tr>
                        </thead>
                        <tbody>
                          {violations.map((violation) => (
                            <tr key={violation.id} className="border-b border-border/50 hover:bg-surface/50">
                              <td className="py-3 px-4">{violation.policy_name || violation.policy_id}</td>
                              <td className="py-3 px-4 text-text-muted max-w-[200px] truncate">
                                {violation.content_snippet}
                              </td>
                              <td className={`py-3 px-4 ${severityColors[violation.severity]}`}>
                                {violation.severity.toUpperCase()}
                              </td>
                              <td className="py-3 px-4">
                                <span className={violation.status === 'open' ? 'text-crimson' : 'text-acid-green'}>
                                  {violation.status.toUpperCase()}
                                </span>
                              </td>
                              <td className="py-3 px-4 text-text-muted">
                                {new Date(violation.created_at).toLocaleDateString()}
                              </td>
                              <td className="py-3 px-4">
                                {violation.status === 'open' && (
                                  <button
                                    onClick={() => handleResolveViolation(violation.id)}
                                    className="text-acid-cyan hover:text-acid-green text-xs"
                                  >
                                    [RESOLVE]
                                  </button>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>

        {/* Create Policy Modal (simplified) */}
        {showCreateModal && (
          <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
            <div className="card p-6 w-full max-w-md">
              <h2 className="text-lg font-mono font-bold text-acid-green mb-4">[NEW POLICY]</h2>
              <p className="text-text-muted text-sm mb-4">
                Policy creation requires admin privileges and API configuration.
              </p>
              <button
                onClick={() => setShowCreateModal(false)}
                className="w-full px-4 py-2 font-mono text-sm border border-border text-text hover:border-acid-green/50 transition-colors"
              >
                [CLOSE]
              </button>
            </div>
          </div>
        )}
      </main>
    </>
  );
}
