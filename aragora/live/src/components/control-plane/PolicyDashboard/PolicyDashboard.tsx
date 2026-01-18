'use client';

import { useState, useCallback, useEffect, useMemo } from 'react';
import { useApi } from '@/hooks/useApi';
import { useBackend } from '@/components/BackendSelector';
import { ComplianceFrameworkList, type ComplianceFramework } from './ComplianceFrameworkList';
import { ViolationTracker, type ComplianceViolation } from './ViolationTracker';
import { RiskOverview } from './RiskOverview';

export type PolicyTab = 'overview' | 'frameworks' | 'violations' | 'risk';

export interface PolicyDashboardProps {
  defaultTab?: PolicyTab;
  className?: string;
  onFrameworkSelect?: (framework: ComplianceFramework) => void;
  onViolationSelect?: (violation: ComplianceViolation) => void;
}

export function PolicyDashboard({
  defaultTab = 'overview',
  className = '',
  onFrameworkSelect,
  onViolationSelect,
}: PolicyDashboardProps) {
  const { config: backendConfig } = useBackend();
  const api = useApi(backendConfig?.api);

  const [activeTab, setActiveTab] = useState<PolicyTab>(defaultTab);
  const [frameworks, setFrameworks] = useState<ComplianceFramework[]>([]);
  const [violations, setViolations] = useState<ComplianceViolation[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedVertical, setSelectedVertical] = useState<string | null>(null);

  // Available verticals
  const verticals = useMemo(() => [
    { id: 'software', name: 'Software Engineering' },
    { id: 'legal', name: 'Legal' },
    { id: 'healthcare', name: 'Healthcare' },
    { id: 'accounting', name: 'Accounting & Finance' },
    { id: 'research', name: 'Research' },
  ], []);

  // Load frameworks from API
  const loadFrameworks = useCallback(async () => {
    try {
      const allFrameworks: ComplianceFramework[] = [];

      for (const vertical of verticals) {
        try {
          const response = await api.get(`/api/verticals/${vertical.id}/compliance`) as {
            compliance_frameworks: Array<{
              framework_id: string;
              name: string;
              description: string;
              level: 'mandatory' | 'recommended' | 'optional';
              rules?: Array<{ rule_id: string }>;
              enabled?: boolean;
            }>;
          };

          (response.compliance_frameworks || []).forEach((fw) => {
            allFrameworks.push({
              framework_id: fw.framework_id,
              name: fw.name,
              description: fw.description,
              level: fw.level,
              vertical_id: vertical.id,
              rules_count: fw.rules?.length || 0,
              enabled: fw.enabled ?? true,
            });
          });
        } catch {
          // Skip verticals that fail
        }
      }

      if (allFrameworks.length === 0) {
        // Mock data
        setFrameworks([
          { framework_id: 'owasp', name: 'OWASP Top 10', description: 'Web application security risks', level: 'mandatory', vertical_id: 'software', rules_count: 10, enabled: true },
          { framework_id: 'cwe', name: 'CWE', description: 'Common Weakness Enumeration', level: 'recommended', vertical_id: 'software', rules_count: 25, enabled: true },
          { framework_id: 'gdpr', name: 'GDPR', description: 'General Data Protection Regulation', level: 'mandatory', vertical_id: 'legal', rules_count: 15, enabled: true },
          { framework_id: 'hipaa', name: 'HIPAA', description: 'Health Insurance Portability Act', level: 'mandatory', vertical_id: 'healthcare', rules_count: 18, enabled: true },
          { framework_id: 'sox', name: 'SOX', description: 'Sarbanes-Oxley Act', level: 'mandatory', vertical_id: 'accounting', rules_count: 12, enabled: true },
          { framework_id: 'irb', name: 'IRB', description: 'Institutional Review Board', level: 'mandatory', vertical_id: 'research', rules_count: 8, enabled: true },
        ]);
      } else {
        setFrameworks(allFrameworks);
      }
    } catch (err) {
      console.error('Failed to load frameworks:', err);
    }
  }, [api, verticals]);

  // Load violations
  const loadViolations = useCallback(async () => {
    try {
      const response = await api.get('/api/compliance/violations') as { violations: ComplianceViolation[] };
      setViolations(response.violations || []);
    } catch {
      setViolations([
        { id: 'v_001', rule_id: 'owasp_a03', rule_name: 'SQL Injection Prevention', framework_id: 'owasp', vertical_id: 'software', severity: 'critical', status: 'open', description: 'Potential SQL injection in user input', source: 'src/api/users.py:42', detected_at: new Date().toISOString() },
        { id: 'v_002', rule_id: 'hipaa_phi', rule_name: 'PHI Protection', framework_id: 'hipaa', vertical_id: 'healthcare', severity: 'high', status: 'investigating', description: 'PHI detected in logs', source: 'logs/app.log:1542', detected_at: new Date(Date.now() - 86400000).toISOString() },
        { id: 'v_003', rule_id: 'gdpr_consent', rule_name: 'Consent Management', framework_id: 'gdpr', vertical_id: 'legal', severity: 'medium', status: 'resolved', description: 'Missing consent banner', source: 'forms/signup.tsx', detected_at: new Date(Date.now() - 172800000).toISOString(), resolved_at: new Date(Date.now() - 86400000).toISOString() },
      ]);
    }
  }, [api]);

  // Load data
  useEffect(() => {
    const load = async () => {
      setLoading(true);
      await Promise.all([loadFrameworks(), loadViolations()]);
      setLoading(false);
    };
    load();
  }, [loadFrameworks, loadViolations]);

  // Stats
  const stats = useMemo(() => ({
    totalFrameworks: frameworks.length,
    enabledFrameworks: frameworks.filter((f) => f.enabled).length,
    totalRules: frameworks.reduce((acc, f) => acc + f.rules_count, 0),
    openViolations: violations.filter((v) => v.status !== 'resolved').length,
    criticalViolations: violations.filter((v) => v.severity === 'critical' && v.status !== 'resolved').length,
    riskScore: Math.min(100,
      violations.filter((v) => v.status !== 'resolved').reduce((acc, v) => {
        const weights = { critical: 25, high: 10, medium: 5, low: 2 };
        return acc + (weights[v.severity] || 0);
      }, 0)
    ),
  }), [frameworks, violations]);

  const tabs: Array<{ id: PolicyTab; label: string; badge?: number }> = [
    { id: 'overview', label: 'Overview' },
    { id: 'frameworks', label: 'Frameworks', badge: stats.enabledFrameworks },
    { id: 'violations', label: 'Violations', badge: stats.openViolations },
    { id: 'risk', label: 'Risk', badge: stats.criticalViolations > 0 ? stats.criticalViolations : undefined },
  ];

  return (
    <div className={`bg-surface border border-border rounded-lg overflow-hidden ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-border bg-bg">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-mono font-bold text-acid-green">POLICY & COMPLIANCE</h2>
          <span className="text-xs text-text-muted font-mono">[{stats.totalFrameworks} FRAMEWORKS]</span>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-border">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-xs font-mono uppercase flex items-center gap-2 ${
              activeTab === tab.id
                ? 'text-acid-green border-b-2 border-acid-green bg-bg'
                : 'text-text-muted hover:text-text'
            }`}
          >
            {tab.label}
            {tab.badge !== undefined && (
              <span className={`px-1.5 py-0.5 rounded text-xs ${
                tab.id === 'violations' && tab.badge > 0 ? 'bg-yellow-900/30 text-yellow-400' :
                tab.id === 'risk' && tab.badge > 0 ? 'bg-red-900/30 text-red-400' :
                'bg-surface text-text-muted'
              }`}>
                {tab.badge}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Stats bar */}
      <div className="grid grid-cols-5 gap-4 p-4 border-b border-border bg-bg">
        <div className="text-center">
          <div className="text-xl font-mono text-acid-green">{stats.totalFrameworks}</div>
          <div className="text-xs text-text-muted">Frameworks</div>
        </div>
        <div className="text-center">
          <div className="text-xl font-mono text-acid-cyan">{stats.totalRules}</div>
          <div className="text-xs text-text-muted">Rules</div>
        </div>
        <div className="text-center">
          <div className="text-xl font-mono text-yellow-400">{stats.openViolations}</div>
          <div className="text-xs text-text-muted">Open Issues</div>
        </div>
        <div className="text-center">
          <div className="text-xl font-mono text-red-400">{stats.criticalViolations}</div>
          <div className="text-xs text-text-muted">Critical</div>
        </div>
        <div className="text-center">
          <div className={`text-xl font-mono ${
            stats.riskScore > 70 ? 'text-red-400' : stats.riskScore > 40 ? 'text-yellow-400' : 'text-green-400'
          }`}>{stats.riskScore}</div>
          <div className="text-xs text-text-muted">Risk Score</div>
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        {loading ? (
          <div className="text-center py-8 text-text-muted font-mono">Loading...</div>
        ) : (
          <>
            {activeTab === 'overview' && (
              <RiskOverview
                frameworks={frameworks}
                violations={violations}
                verticals={verticals}
              />
            )}
            {activeTab === 'frameworks' && (
              <ComplianceFrameworkList
                frameworks={frameworks}
                onSelectFramework={onFrameworkSelect}
                verticals={verticals}
                selectedVertical={selectedVertical}
                onVerticalChange={setSelectedVertical}
              />
            )}
            {activeTab === 'violations' && (
              <ViolationTracker
                violations={violations}
                onSelectViolation={onViolationSelect}
                verticals={verticals}
                selectedVertical={selectedVertical}
                onVerticalChange={setSelectedVertical}
              />
            )}
            {activeTab === 'risk' && (
              <RiskOverview
                frameworks={frameworks}
                violations={violations}
                verticals={verticals}
                showDetails
              />
            )}
          </>
        )}
      </div>
    </div>
  );
}
