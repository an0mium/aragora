'use client';

import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { ErrorWithRetry } from '@/components/ErrorWithRetry';
import { useToastContext } from '@/context/ToastContext';

interface PolicyRule {
  id: string;
  pattern?: string;
  action: 'warn' | 'block' | 'flag' | 'redact';
  message: string;
}

interface Policy {
  id: string;
  name: string;
  description: string;
  type: 'content' | 'output' | 'behavior' | 'custom';
  severity: 'low' | 'medium' | 'high' | 'critical';
  enabled: boolean;
  rules: PolicyRule[];
  framework_id?: string;
  vertical_id?: string;
  workspace_id?: string;
  created_at: string;
  updated_at?: string;
  violation_count?: number;
}

interface Violation {
  id: string;
  policy_id: string;
  policy_name?: string;
  rule_id?: string;
  rule_name?: string;
  content_snippet: string;
  severity: string;
  status: 'open' | 'investigating' | 'resolved' | 'false_positive' | 'ignored';
  description?: string;
  source?: string;
  created_at: string;
  resolved_at?: string;
  resolution_notes?: string;
}

interface ComplianceStats {
  policies: {
    total: number;
    enabled: number;
    disabled: number;
  };
  violations: {
    total: number;
    open: number;
    by_severity: {
      critical: number;
      high: number;
      medium: number;
      low: number;
    };
  };
  risk_score: number;
}

const severityColors: Record<string, string> = {
  low: 'text-text-muted border-text-muted/30',
  medium: 'text-acid-yellow border-acid-yellow/30',
  high: 'text-warning border-warning/30',
  critical: 'text-crimson border-crimson/30',
};

const severityBgColors: Record<string, string> = {
  low: 'bg-text-muted/10',
  medium: 'bg-acid-yellow/10',
  high: 'bg-warning/10',
  critical: 'bg-crimson/10',
};

const statusColors: Record<string, string> = {
  open: 'text-crimson bg-crimson/10 border-crimson/30',
  investigating: 'text-acid-yellow bg-acid-yellow/10 border-acid-yellow/30',
  resolved: 'text-acid-green bg-acid-green/10 border-acid-green/30',
  false_positive: 'text-text-muted bg-text-muted/10 border-text-muted/30',
  ignored: 'text-text-muted bg-text-muted/10 border-text-muted/30',
};

const typeIcons: Record<string, string> = {
  content: '#',
  output: '>',
  behavior: '!',
  custom: '*',
};

const actionColors: Record<string, string> = {
  warn: 'text-acid-yellow',
  block: 'text-crimson',
  flag: 'text-acid-cyan',
  redact: 'text-warning',
};

// Create/Edit Policy Modal
function PolicyModal({
  policy,
  onClose,
  onSave,
}: {
  policy?: Policy | null;
  onClose: () => void;
  onSave: (data: Partial<Policy>) => Promise<void>;
}) {
  const [name, setName] = useState(policy?.name || '');
  const [description, setDescription] = useState(policy?.description || '');
  const [type, setType] = useState<Policy['type']>(policy?.type || 'content');
  const [severity, setSeverity] = useState<Policy['severity']>(policy?.severity || 'medium');
  const [frameworkId, setFrameworkId] = useState(policy?.framework_id || 'default');
  const [verticalId, setVerticalId] = useState(policy?.vertical_id || 'general');
  const [rules, setRules] = useState<PolicyRule[]>(policy?.rules || []);
  const [saving, setSaving] = useState(false);

  const handleAddRule = () => {
    setRules([
      ...rules,
      {
        id: `rule-${Date.now()}`,
        pattern: '',
        action: 'warn',
        message: '',
      },
    ]);
  };

  const handleRemoveRule = (ruleId: string) => {
    setRules(rules.filter((r) => r.id !== ruleId));
  };

  const handleUpdateRule = (ruleId: string, updates: Partial<PolicyRule>) => {
    setRules(rules.map((r) => (r.id === ruleId ? { ...r, ...updates } : r)));
  };

  const handleSubmit = async () => {
    if (!name.trim()) return;
    setSaving(true);
    try {
      await onSave({
        name,
        description,
        type,
        severity,
        framework_id: frameworkId,
        vertical_id: verticalId,
        rules,
      });
      onClose();
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 overflow-y-auto p-4">
      <div className="card p-6 w-full max-w-2xl my-4">
        <h2 className="text-lg font-mono font-bold text-acid-green mb-4">
          {policy ? '[EDIT POLICY]' : '[NEW POLICY]'}
        </h2>

        <div className="space-y-4 max-h-[70vh] overflow-y-auto pr-2">
          {/* Name */}
          <div>
            <label className="block text-xs font-mono text-text-muted mb-1">Name *</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Policy name"
              className="w-full bg-bg border border-border px-3 py-2 text-sm font-mono text-text focus:outline-none focus:border-acid-green"
            />
          </div>

          {/* Description */}
          <div>
            <label className="block text-xs font-mono text-text-muted mb-1">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe what this policy enforces..."
              rows={2}
              className="w-full bg-bg border border-border px-3 py-2 text-sm font-mono text-text focus:outline-none focus:border-acid-green"
            />
          </div>

          {/* Type & Severity */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-mono text-text-muted mb-1">Type</label>
              <select
                value={type}
                onChange={(e) => setType(e.target.value as Policy['type'])}
                className="w-full bg-bg border border-border px-3 py-2 text-sm font-mono text-text focus:outline-none focus:border-acid-green"
              >
                <option value="content">Content</option>
                <option value="output">Output</option>
                <option value="behavior">Behavior</option>
                <option value="custom">Custom</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-mono text-text-muted mb-1">Severity</label>
              <select
                value={severity}
                onChange={(e) => setSeverity(e.target.value as Policy['severity'])}
                className="w-full bg-bg border border-border px-3 py-2 text-sm font-mono text-text focus:outline-none focus:border-acid-green"
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
                <option value="critical">Critical</option>
              </select>
            </div>
          </div>

          {/* Framework & Vertical */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-mono text-text-muted mb-1">Framework ID</label>
              <input
                type="text"
                value={frameworkId}
                onChange={(e) => setFrameworkId(e.target.value)}
                placeholder="default"
                className="w-full bg-bg border border-border px-3 py-2 text-sm font-mono text-text focus:outline-none focus:border-acid-green"
              />
            </div>
            <div>
              <label className="block text-xs font-mono text-text-muted mb-1">Vertical ID</label>
              <input
                type="text"
                value={verticalId}
                onChange={(e) => setVerticalId(e.target.value)}
                placeholder="general"
                className="w-full bg-bg border border-border px-3 py-2 text-sm font-mono text-text focus:outline-none focus:border-acid-green"
              />
            </div>
          </div>

          {/* Rules */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-mono text-text-muted">Rules</label>
              <button
                type="button"
                onClick={handleAddRule}
                className="text-xs font-mono text-acid-green hover:text-acid-green/80"
              >
                [+ ADD RULE]
              </button>
            </div>
            <div className="space-y-2">
              {rules.map((rule, idx) => (
                <div key={rule.id} className="bg-bg border border-border p-3 rounded space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono text-text-muted">Rule {idx + 1}</span>
                    <button
                      type="button"
                      onClick={() => handleRemoveRule(rule.id)}
                      className="text-xs font-mono text-crimson hover:text-crimson/80"
                    >
                      [X]
                    </button>
                  </div>
                  <div className="grid grid-cols-3 gap-2">
                    <div>
                      <select
                        value={rule.action}
                        onChange={(e) =>
                          handleUpdateRule(rule.id, { action: e.target.value as PolicyRule['action'] })
                        }
                        className="w-full bg-surface border border-border px-2 py-1 text-xs font-mono text-text focus:outline-none focus:border-acid-green"
                      >
                        <option value="warn">Warn</option>
                        <option value="block">Block</option>
                        <option value="flag">Flag</option>
                        <option value="redact">Redact</option>
                      </select>
                    </div>
                    <div className="col-span-2">
                      <input
                        type="text"
                        value={rule.pattern || ''}
                        onChange={(e) => handleUpdateRule(rule.id, { pattern: e.target.value })}
                        placeholder="Regex pattern (optional)"
                        className="w-full bg-surface border border-border px-2 py-1 text-xs font-mono text-text focus:outline-none focus:border-acid-green"
                      />
                    </div>
                  </div>
                  <input
                    type="text"
                    value={rule.message}
                    onChange={(e) => handleUpdateRule(rule.id, { message: e.target.value })}
                    placeholder="Violation message"
                    className="w-full bg-surface border border-border px-2 py-1 text-xs font-mono text-text focus:outline-none focus:border-acid-green"
                  />
                </div>
              ))}
              {rules.length === 0 && (
                <div className="text-text-muted text-xs text-center py-2">
                  No rules defined. Add rules to define policy behavior.
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-2 mt-4 pt-4 border-t border-border">
          <button
            onClick={handleSubmit}
            disabled={saving || !name.trim()}
            className="flex-1 px-4 py-2 font-mono text-sm bg-acid-green/20 border border-acid-green text-acid-green hover:bg-acid-green/30 transition-colors disabled:opacity-50"
          >
            {saving ? '[SAVING...]' : policy ? '[SAVE CHANGES]' : '[CREATE POLICY]'}
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 font-mono text-sm border border-border text-text-muted hover:border-text-muted transition-colors"
          >
            [CANCEL]
          </button>
        </div>
      </div>
    </div>
  );
}

// Violation Details Modal
function ViolationModal({
  violation,
  onClose,
  onUpdateStatus,
}: {
  violation: Violation;
  onClose: () => void;
  onUpdateStatus: (status: string, notes?: string) => Promise<void>;
}) {
  const [notes, setNotes] = useState('');
  const [updating, setUpdating] = useState(false);

  const handleUpdate = async (status: string) => {
    setUpdating(true);
    try {
      await onUpdateStatus(status, notes);
      onClose();
    } finally {
      setUpdating(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="card p-6 w-full max-w-lg">
        <h2 className="text-lg font-mono font-bold text-acid-green mb-4">[VIOLATION DETAILS]</h2>

        <div className="space-y-3">
          <div>
            <span className="text-xs font-mono text-text-muted">Policy:</span>
            <p className="font-mono text-text">{violation.policy_name || violation.policy_id}</p>
          </div>
          <div>
            <span className="text-xs font-mono text-text-muted">Severity:</span>
            <span className={`ml-2 px-2 py-0.5 text-xs font-mono border ${severityColors[violation.severity]}`}>
              {violation.severity.toUpperCase()}
            </span>
          </div>
          <div>
            <span className="text-xs font-mono text-text-muted">Status:</span>
            <span className={`ml-2 px-2 py-0.5 text-xs font-mono border ${statusColors[violation.status]}`}>
              {violation.status.toUpperCase()}
            </span>
          </div>
          {violation.description && (
            <div>
              <span className="text-xs font-mono text-text-muted">Description:</span>
              <p className="text-sm text-text">{violation.description}</p>
            </div>
          )}
          <div>
            <span className="text-xs font-mono text-text-muted">Content:</span>
            <div className="bg-bg p-2 rounded mt-1 text-sm font-mono text-text-muted">
              {violation.content_snippet}
            </div>
          </div>
          <div>
            <span className="text-xs font-mono text-text-muted">Detected:</span>
            <p className="text-sm text-text">{new Date(violation.created_at).toLocaleString()}</p>
          </div>

          {violation.status === 'open' && (
            <div>
              <label className="text-xs font-mono text-text-muted block mb-1">Resolution Notes</label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Add notes about the resolution..."
                rows={2}
                className="w-full bg-bg border border-border px-3 py-2 text-sm font-mono text-text focus:outline-none focus:border-acid-green"
              />
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t border-border">
          {violation.status === 'open' && (
            <>
              <button
                onClick={() => handleUpdate('investigating')}
                disabled={updating}
                className="px-3 py-1.5 font-mono text-xs bg-acid-yellow/20 border border-acid-yellow text-acid-yellow hover:bg-acid-yellow/30 transition-colors disabled:opacity-50"
              >
                [INVESTIGATE]
              </button>
              <button
                onClick={() => handleUpdate('resolved')}
                disabled={updating}
                className="px-3 py-1.5 font-mono text-xs bg-acid-green/20 border border-acid-green text-acid-green hover:bg-acid-green/30 transition-colors disabled:opacity-50"
              >
                [RESOLVE]
              </button>
              <button
                onClick={() => handleUpdate('false_positive')}
                disabled={updating}
                className="px-3 py-1.5 font-mono text-xs border border-text-muted text-text-muted hover:border-text transition-colors disabled:opacity-50"
              >
                [FALSE POSITIVE]
              </button>
            </>
          )}
          {violation.status === 'investigating' && (
            <button
              onClick={() => handleUpdate('resolved')}
              disabled={updating}
              className="px-3 py-1.5 font-mono text-xs bg-acid-green/20 border border-acid-green text-acid-green hover:bg-acid-green/30 transition-colors disabled:opacity-50"
            >
              [RESOLVE]
            </button>
          )}
          <button
            onClick={onClose}
            className="ml-auto px-3 py-1.5 font-mono text-xs border border-border text-text-muted hover:border-text-muted transition-colors"
          >
            [CLOSE]
          </button>
        </div>
      </div>
    </div>
  );
}

// Compliance Check Modal
function ComplianceCheckModal({
  apiBase,
  onClose,
}: {
  apiBase: string;
  onClose: () => void;
}) {
  const [content, setContent] = useState('');
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<{
    compliant: boolean;
    score: number;
    issue_count: number;
    issues?: Array<{ description: string; severity: string; framework: string }>;
  } | null>(null);

  const handleCheck = async () => {
    if (!content.trim()) return;
    setChecking(true);
    try {
      const res = await fetch(`${apiBase}/api/compliance/check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content }),
      });
      if (res.ok) {
        const data = await res.json();
        setResult(data);
      }
    } catch (error) {
      console.error('Failed to check compliance:', error);
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50">
      <div className="card p-6 w-full max-w-lg">
        <h2 className="text-lg font-mono font-bold text-acid-green mb-4">[COMPLIANCE CHECK]</h2>

        <div className="space-y-4">
          <div>
            <label className="text-xs font-mono text-text-muted block mb-1">Content to Check</label>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Enter content to check against compliance policies..."
              rows={5}
              className="w-full bg-bg border border-border px-3 py-2 text-sm font-mono text-text focus:outline-none focus:border-acid-green"
            />
          </div>

          {result && (
            <div className={`p-4 rounded border ${result.compliant ? 'border-acid-green bg-acid-green/10' : 'border-crimson bg-crimson/10'}`}>
              <div className="flex items-center justify-between mb-2">
                <span className={`font-mono font-bold ${result.compliant ? 'text-acid-green' : 'text-crimson'}`}>
                  {result.compliant ? 'COMPLIANT' : 'NON-COMPLIANT'}
                </span>
                <span className="font-mono text-sm text-text-muted">
                  Score: {result.score.toFixed(0)}%
                </span>
              </div>
              {result.issue_count > 0 && (
                <p className="text-sm text-text-muted">
                  {result.issue_count} issue{result.issue_count !== 1 ? 's' : ''} found
                </p>
              )}
            </div>
          )}
        </div>

        <div className="flex gap-2 mt-4 pt-4 border-t border-border">
          <button
            onClick={handleCheck}
            disabled={checking || !content.trim()}
            className="flex-1 px-4 py-2 font-mono text-sm bg-acid-green/20 border border-acid-green text-acid-green hover:bg-acid-green/30 transition-colors disabled:opacity-50"
          >
            {checking ? '[CHECKING...]' : '[CHECK]'}
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 font-mono text-sm border border-border text-text-muted hover:border-text-muted transition-colors"
          >
            [CLOSE]
          </button>
        </div>
      </div>
    </div>
  );
}

export default function PolicyPage() {
  const { config: backendConfig } = useBackend();
  const { showToast } = useToastContext();
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [stats, setStats] = useState<ComplianceStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'policies' | 'violations'>('policies');
  const [selectedPolicy, setSelectedPolicy] = useState<Policy | null>(null);
  const [showPolicyModal, setShowPolicyModal] = useState(false);
  const [editingPolicy, setEditingPolicy] = useState<Policy | null>(null);
  const [selectedViolation, setSelectedViolation] = useState<Violation | null>(null);
  const [showComplianceCheck, setShowComplianceCheck] = useState(false);
  const [violationFilter, setViolationFilter] = useState<'all' | 'open' | 'resolved'>('all');
  const [severityFilter, setSeverityFilter] = useState<'all' | 'critical' | 'high' | 'medium' | 'low'>('all');

  const fetchData = useCallback(async () => {
    try {
      const [policiesRes, violationsRes, statsRes] = await Promise.all([
        fetch(`${backendConfig.api}/api/policies`),
        fetch(`${backendConfig.api}/api/compliance/violations?limit=100`),
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
        setStats(data);
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

  const handleCreatePolicy = async (data: Partial<Policy>) => {
    const res = await fetch(`${backendConfig.api}/api/policies`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (res.ok) {
      showToast('Policy created successfully', 'success');
      fetchData();
    } else {
      showToast('Failed to create policy', 'error');
    }
  };

  const handleUpdatePolicy = async (policyId: string, data: Partial<Policy>) => {
    const res = await fetch(`${backendConfig.api}/api/policies/${policyId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (res.ok) {
      showToast('Policy updated successfully', 'success');
      fetchData();
    } else {
      showToast('Failed to update policy', 'error');
    }
  };

  const handleDeletePolicy = async (policyId: string) => {
    if (!confirm('Are you sure you want to delete this policy?')) return;
    const res = await fetch(`${backendConfig.api}/api/policies/${policyId}`, {
      method: 'DELETE',
    });
    if (res.ok) {
      showToast('Policy deleted successfully', 'success');
      fetchData();
    } else {
      showToast('Failed to delete policy', 'error');
    }
  };

  const handleTogglePolicy = async (policyId: string) => {
    const res = await fetch(`${backendConfig.api}/api/policies/${policyId}/toggle`, {
      method: 'POST',
    });
    if (res.ok) {
      fetchData();
    }
  };

  const handleUpdateViolation = async (violationId: string, status: string, notes?: string) => {
    const res = await fetch(`${backendConfig.api}/api/compliance/violations/${violationId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status, resolution_notes: notes }),
    });
    if (res.ok) {
      showToast('Violation updated successfully', 'success');
      fetchData();
    } else {
      showToast('Failed to update violation', 'error');
    }
  };

  // Filter violations
  const filteredViolations = violations.filter((v) => {
    if (violationFilter !== 'all' && v.status !== violationFilter && (violationFilter !== 'resolved' || v.status !== 'false_positive')) {
      return false;
    }
    if (severityFilter !== 'all' && v.severity !== severityFilter) {
      return false;
    }
    return true;
  });

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
              <h1 className="text-2xl font-mono font-bold text-acid-green mb-2">[POLICY_ADMIN]</h1>
              <p className="text-text-muted font-mono text-sm">Compliance policies and violation tracking</p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setShowComplianceCheck(true)}
                className="px-4 py-2 font-mono text-sm border border-acid-cyan text-acid-cyan hover:bg-acid-cyan/10 transition-colors"
              >
                [CHECK CONTENT]
              </button>
              <button
                onClick={() => {
                  setEditingPolicy(null);
                  setShowPolicyModal(true);
                }}
                className="px-4 py-2 font-mono text-sm bg-acid-green/20 border border-acid-green text-acid-green hover:bg-acid-green/30 transition-colors"
              >
                [+ NEW POLICY]
              </button>
            </div>
          </div>

          {error && <ErrorWithRetry error={error} onRetry={fetchData} className="mb-6" />}

          {loading ? (
            <div className="text-center py-12">
              <div className="text-acid-green font-mono animate-pulse">Loading policy data...</div>
            </div>
          ) : (
            <>
              {/* Stats Cards */}
              {stats && (
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
                  <div className="card p-4 text-center">
                    <div className={`text-3xl font-mono ${stats.risk_score < 25 ? 'text-acid-green' : stats.risk_score < 50 ? 'text-acid-yellow' : stats.risk_score < 75 ? 'text-warning' : 'text-crimson'}`}>
                      {100 - stats.risk_score}%
                    </div>
                    <div className="text-xs font-mono text-text-muted">Compliance Score</div>
                  </div>
                  <div className="card p-4 text-center">
                    <div className="text-3xl font-mono text-accent">
                      {stats.policies.enabled}/{stats.policies.total}
                    </div>
                    <div className="text-xs font-mono text-text-muted">Active Policies</div>
                  </div>
                  <div className="card p-4 text-center">
                    <div className="text-3xl font-mono text-crimson">{stats.violations.open}</div>
                    <div className="text-xs font-mono text-text-muted">Open Violations</div>
                  </div>
                  <div className="card p-4 text-center">
                    <div className="text-3xl font-mono text-warning">{stats.violations.by_severity.critical + stats.violations.by_severity.high}</div>
                    <div className="text-xs font-mono text-text-muted">Critical/High</div>
                  </div>
                  <div className="card p-4 text-center">
                    <div className="text-3xl font-mono text-acid-cyan">{stats.violations.total - stats.violations.open}</div>
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
                  VIOLATIONS ({violations.filter((v) => v.status === 'open').length} open)
                </button>
              </div>

              {/* Policies Tab */}
              {activeTab === 'policies' && (
                <div className="space-y-4">
                  {policies.length === 0 ? (
                    <div className="card p-8 text-center">
                      <div className="text-text-muted font-mono">No policies defined. Create your first compliance policy.</div>
                    </div>
                  ) : (
                    policies.map((policy) => (
                      <div
                        key={policy.id}
                        className={`card p-4 transition-colors ${selectedPolicy?.id === policy.id ? 'border-acid-green/50' : 'hover:border-acid-green/30'}`}
                      >
                        <div className="flex items-start justify-between">
                          <div
                            className="flex items-start gap-3 flex-1 cursor-pointer"
                            onClick={() => setSelectedPolicy(selectedPolicy?.id === policy.id ? null : policy)}
                          >
                            <span className="text-acid-green font-mono text-lg">{typeIcons[policy.type] || '#'}</span>
                            <div className="flex-1">
                              <div className="flex items-center gap-2 flex-wrap">
                                <h3 className="font-mono font-bold text-text">{policy.name}</h3>
                                <span className={`text-xs font-mono uppercase px-2 py-0.5 border ${severityColors[policy.severity]} ${severityBgColors[policy.severity]}`}>
                                  {policy.severity}
                                </span>
                                <span className="text-xs font-mono text-text-muted">[{policy.type}]</span>
                              </div>
                              <p className="text-sm text-text-muted mt-1">{policy.description}</p>
                              {policy.violation_count !== undefined && policy.violation_count > 0 && (
                                <span className="text-xs text-crimson font-mono mt-2 inline-block">
                                  {policy.violation_count} violation{policy.violation_count !== 1 ? 's' : ''}
                                </span>
                              )}
                            </div>
                          </div>
                          <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
                            <button
                              onClick={() => handleTogglePolicy(policy.id)}
                              className={`px-3 py-1 font-mono text-xs border transition-colors ${
                                policy.enabled
                                  ? 'border-acid-green text-acid-green hover:bg-acid-green/10'
                                  : 'border-text-muted text-text-muted hover:border-text'
                              }`}
                            >
                              {policy.enabled ? '[ON]' : '[OFF]'}
                            </button>
                            <button
                              onClick={() => {
                                setEditingPolicy(policy);
                                setShowPolicyModal(true);
                              }}
                              className="px-3 py-1 font-mono text-xs border border-acid-cyan text-acid-cyan hover:bg-acid-cyan/10 transition-colors"
                            >
                              [EDIT]
                            </button>
                            <button
                              onClick={() => handleDeletePolicy(policy.id)}
                              className="px-3 py-1 font-mono text-xs border border-crimson text-crimson hover:bg-crimson/10 transition-colors"
                            >
                              [DEL]
                            </button>
                          </div>
                        </div>

                        {/* Expanded details */}
                        {selectedPolicy?.id === policy.id && (
                          <div className="mt-4 pt-4 border-t border-border">
                            <h4 className="font-mono text-sm text-acid-green mb-2">Rules ({policy.rules?.length || 0}):</h4>
                            {policy.rules && policy.rules.length > 0 ? (
                              <div className="space-y-2">
                                {policy.rules.map((rule) => (
                                  <div key={rule.id} className="bg-bg p-2 rounded text-sm font-mono">
                                    <span className={`${actionColors[rule.action]}`}>[{rule.action.toUpperCase()}]</span> {rule.message}
                                    {rule.pattern && <span className="text-text-muted ml-2">/{rule.pattern}/</span>}
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div className="text-text-muted text-sm">No rules defined</div>
                            )}
                            <div className="mt-3 text-xs text-text-muted font-mono">
                              Framework: {policy.framework_id || 'default'} | Vertical: {policy.vertical_id || 'general'}
                            </div>
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              )}

              {/* Violations Tab */}
              {activeTab === 'violations' && (
                <div className="space-y-4">
                  {/* Filters */}
                  <div className="flex gap-4 flex-wrap">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-text-muted">Status:</span>
                      {(['all', 'open', 'resolved'] as const).map((f) => (
                        <button
                          key={f}
                          onClick={() => setViolationFilter(f)}
                          className={`px-2 py-1 text-xs font-mono border transition-colors ${
                            violationFilter === f
                              ? 'border-acid-green text-acid-green bg-acid-green/10'
                              : 'border-border text-text-muted hover:border-text-muted'
                          }`}
                        >
                          {f.toUpperCase()}
                        </button>
                      ))}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-text-muted">Severity:</span>
                      {(['all', 'critical', 'high', 'medium', 'low'] as const).map((f) => (
                        <button
                          key={f}
                          onClick={() => setSeverityFilter(f)}
                          className={`px-2 py-1 text-xs font-mono border transition-colors ${
                            severityFilter === f
                              ? 'border-acid-green text-acid-green bg-acid-green/10'
                              : 'border-border text-text-muted hover:border-text-muted'
                          }`}
                        >
                          {f.toUpperCase()}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="card">
                    {filteredViolations.length === 0 ? (
                      <div className="p-8 text-center">
                        <div className="text-text-muted font-mono">
                          {violations.length === 0 ? 'No violations recorded. Your content is compliant.' : 'No violations match the current filters.'}
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
                            {filteredViolations.map((violation) => (
                              <tr key={violation.id} className="border-b border-border/50 hover:bg-surface/50">
                                <td className="py-3 px-4">{violation.policy_name || violation.policy_id}</td>
                                <td className="py-3 px-4 text-text-muted max-w-[200px] truncate">{violation.content_snippet}</td>
                                <td className="py-3 px-4">
                                  <span className={`text-xs font-mono px-2 py-0.5 border ${severityColors[violation.severity]}`}>
                                    {violation.severity.toUpperCase()}
                                  </span>
                                </td>
                                <td className="py-3 px-4">
                                  <span className={`text-xs font-mono px-2 py-0.5 border ${statusColors[violation.status]}`}>
                                    {violation.status.toUpperCase()}
                                  </span>
                                </td>
                                <td className="py-3 px-4 text-text-muted">{new Date(violation.created_at).toLocaleDateString()}</td>
                                <td className="py-3 px-4">
                                  <button
                                    onClick={() => setSelectedViolation(violation)}
                                    className="text-acid-cyan hover:text-acid-green text-xs"
                                  >
                                    [VIEW]
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Modals */}
        {showPolicyModal && (
          <PolicyModal
            policy={editingPolicy}
            onClose={() => {
              setShowPolicyModal(false);
              setEditingPolicy(null);
            }}
            onSave={async (data) => {
              if (editingPolicy) {
                await handleUpdatePolicy(editingPolicy.id, data);
              } else {
                await handleCreatePolicy(data);
              }
            }}
          />
        )}

        {selectedViolation && (
          <ViolationModal
            violation={selectedViolation}
            onClose={() => setSelectedViolation(null)}
            onUpdateStatus={(status, notes) => handleUpdateViolation(selectedViolation.id, status, notes)}
          />
        )}

        {showComplianceCheck && (
          <ComplianceCheckModal
            apiBase={backendConfig.api}
            onClose={() => setShowComplianceCheck(false)}
          />
        )}
      </main>
    </>
  );
}
