'use client';

import { useState, useCallback } from 'react';
import { useAuthFetch } from '@/hooks/useAuthenticatedFetch';
import { ScanProgressView } from './ScanProgressView';
import { FindingsSummary } from './FindingsSummary';
import { ReportExporter } from './ReportExporter';

type WizardStep = 'configure' | 'scanning' | 'results';

type ScanType = 'quick' | 'full' | 'secrets';

interface ScanConfig {
  scanType: ScanType;
  repoPath: string;
  includeSecrets: boolean;
  includeHistory: boolean;
  historyDepth: number;
}

interface ScanResult {
  scan_id: string;
  status: 'running' | 'completed' | 'failed';
  repository: string;
  files_scanned: number;
  scanned_label?: string;
  lines_scanned?: number;
  risk_score?: number;
  summary: {
    critical: number;
    high: number;
    medium: number;
    low: number;
    info?: number;
  };
  findings: Finding[];
  error?: string;
}

interface Finding {
  id: string;
  title: string;
  description: string;
  category: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  confidence: number;
  file_path: string;
  line_number: number;
  code_snippet?: string;
  cwe_id?: string;
  recommendation?: string;
}

const DEFAULT_REPOSITORY_ID = 'default';
const POLL_INTERVAL_MS = 5000;
const MAX_POLL_ATTEMPTS = 60;

const SCAN_TYPES: Array<{ id: ScanType; name: string; description: string; icon: string }> = [
  {
    id: 'quick',
    name: 'Quick Scan',
    description: 'Fast pattern-based scan for common vulnerabilities (~30 seconds)',
    icon: '⚡',
  },
  {
    id: 'full',
    name: 'Full Scan',
    description: 'Comprehensive dependency + code analysis (~2-5 minutes)',
    icon: '🔍',
  },
  {
    id: 'secrets',
    name: 'Secrets Scan',
    description: 'Detect hardcoded secrets, API keys, and credentials',
    icon: '🔐',
  },
];

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' && value.trim() ? value : fallback;
}

function asNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function asObjectArray(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => asRecord(item))
    .filter((item): item is Record<string, unknown> => item !== null);
}

function hasOwn(obj: Record<string, unknown> | null, key: string): boolean {
  return Boolean(obj) && Object.prototype.hasOwnProperty.call(obj, key);
}

function normalizeSeverity(value: unknown): Finding['severity'] {
  switch (typeof value === 'string' ? value.toLowerCase() : '') {
    case 'critical':
    case 'high':
    case 'medium':
    case 'low':
    case 'info':
      return value as Finding['severity'];
    default:
      return 'info';
  }
}

function computeRiskScore(summary: ScanResult['summary']): number {
  const score =
    summary.critical * 40 +
    summary.high * 20 +
    summary.medium * 10 +
    summary.low * 5 +
    (summary.info || 0);
  return Math.min(100, score);
}

function titleCase(value: string): string {
  return value
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function buildScanStartEndpoint(scanType: ScanType, repositoryId: string): string {
  switch (scanType) {
    case 'quick':
      return '/api/v1/codebase/quick-scan';
    case 'secrets':
      return `/api/v1/codebase/${repositoryId}/scan/secrets`;
    case 'full':
    default:
      return `/api/v1/codebase/${repositoryId}/scan`;
  }
}

function buildScanStatusEndpoint(
  scanType: ScanType,
  repositoryId: string,
  scanId: string
): string {
  switch (scanType) {
    case 'quick':
      return `/api/v1/codebase/quick-scan/${scanId}`;
    case 'secrets':
      return `/api/v1/codebase/${repositoryId}/scan/secrets/${scanId}`;
    case 'full':
    default:
      return `/api/v1/codebase/${repositoryId}/scan/${scanId}`;
  }
}

function buildScanRequest(config: ScanConfig, repositoryId: string): {
  endpoint: string;
  body: Record<string, unknown>;
} {
  const body: Record<string, unknown> = {
    repo_path: config.repoPath,
  };

  if (config.scanType === 'quick') {
    body.include_secrets = config.includeSecrets;
    body.severity_threshold = 'medium';
  }

  if (config.scanType === 'secrets') {
    body.include_history = config.includeHistory;
    body.history_depth = config.historyDepth;
  }

  return {
    endpoint: buildScanStartEndpoint(config.scanType, repositoryId),
    body,
  };
}

function extractScanPayload(data: Record<string, unknown>): Record<string, unknown> {
  return asRecord(data.scan_result) ?? data;
}

function normalizeQuickScanResult(
  payload: Record<string, unknown>,
  fallbackRepository: string
): ScanResult {
  const summary = asRecord(payload.summary);
  const findings = asObjectArray(payload.findings).map((finding, index) => ({
    id: asString(finding.id, `finding-${index + 1}`),
    title: asString(finding.title, 'Security finding'),
    description: asString(finding.description, 'No description provided.'),
    category: asString(finding.category, 'security'),
    severity: normalizeSeverity(finding.severity),
    confidence: asNumber(finding.confidence, 0),
    file_path: asString(finding.file_path, 'unknown'),
    line_number: asNumber(finding.line_number, 1),
    code_snippet: asString(finding.code_snippet) || undefined,
    cwe_id: asString(finding.cwe_id) || undefined,
    recommendation: asString(finding.recommendation) || undefined,
  }));

  const normalizedSummary: ScanResult['summary'] = {
    critical: asNumber(summary?.critical, 0),
    high: asNumber(summary?.high, 0),
    medium: asNumber(summary?.medium, 0),
    low: asNumber(summary?.low, 0),
    info: asNumber(summary?.info, 0),
  };

  return {
    scan_id: asString(payload.scan_id, 'scan_unknown'),
    status: (asString(payload.status, 'completed') as ScanResult['status']),
    repository: asString(payload.repository, fallbackRepository),
    files_scanned: asNumber(payload.files_scanned, 0),
    scanned_label: 'Files Scanned',
    lines_scanned: hasOwn(payload, 'lines_scanned')
      ? asNumber(payload.lines_scanned, 0)
      : undefined,
    risk_score: hasOwn(payload, 'risk_score')
      ? asNumber(payload.risk_score, computeRiskScore(normalizedSummary))
      : computeRiskScore(normalizedSummary),
    summary: normalizedSummary,
    findings,
    error: asString(payload.error) || undefined,
  };
}

function normalizeVulnerabilityScanResult(
  payload: Record<string, unknown>,
  fallbackRepository: string
): ScanResult {
  const summary = asRecord(payload.summary);
  const findings = asObjectArray(payload.vulnerabilities).map((finding, index) => {
    const cweIds = Array.isArray(finding.cwe_ids)
      ? finding.cwe_ids.filter((value): value is string => typeof value === 'string')
      : [];
    const packageName = asString(finding.package_name);
    const recommendedVersion = asString(finding.recommended_version);
    const remediationGuidance = asString(finding.remediation_guidance);

    return {
      id: asString(finding.id, `vulnerability-${index + 1}`),
      title: asString(finding.title, packageName ? `Vulnerability in ${packageName}` : 'Dependency vulnerability'),
      description: asString(finding.description, 'No description provided.'),
      category: 'dependency_vulnerability',
      severity: normalizeSeverity(finding.severity),
      confidence: 1,
      file_path: asString(finding.file_path, packageName || 'dependency'),
      line_number: asNumber(finding.line_number, 1),
      cwe_id: cweIds[0],
      recommendation:
        remediationGuidance ||
        (recommendedVersion && packageName
          ? `Upgrade ${packageName} to ${recommendedVersion}.`
          : undefined),
    };
  });

  const normalizedSummary: ScanResult['summary'] = {
    critical: asNumber(summary?.critical_count, 0),
    high: asNumber(summary?.high_count, 0),
    medium: asNumber(summary?.medium_count, 0),
    low: asNumber(summary?.low_count, 0),
    info: 0,
  };

  return {
    scan_id: asString(payload.scan_id, 'scan_unknown'),
    status: (asString(payload.status, 'completed') as ScanResult['status']),
    repository: asString(payload.repository, fallbackRepository),
    files_scanned: asNumber(summary?.total_dependencies, 0),
    scanned_label: 'Dependencies Scanned',
    risk_score: computeRiskScore(normalizedSummary),
    summary: normalizedSummary,
    findings,
    error: asString(payload.error) || undefined,
  };
}

function normalizeSecretsScanResult(
  payload: Record<string, unknown>,
  fallbackRepository: string
): ScanResult {
  const summary = asRecord(payload.summary);
  const findings = asObjectArray(payload.secrets).map((finding, index) => ({
    id: asString(finding.id, `secret-${index + 1}`),
    title: `${titleCase(asString(finding.secret_type, 'secret'))} detected`,
    description: asString(
      finding.context_line,
      asString(finding.matched_text, 'Potential secret detected in repository.')
    ),
    category: asString(finding.secret_type, 'secret'),
    severity: normalizeSeverity(finding.severity),
    confidence: asNumber(finding.confidence, 0),
    file_path: asString(finding.file_path, 'unknown'),
    line_number: asNumber(finding.line_number, 1),
    code_snippet: asString(finding.context_line) || undefined,
    recommendation:
      asString(finding.remediation) || 'Rotate the secret and move it out of source control.',
  }));

  const normalizedSummary: ScanResult['summary'] = {
    critical: asNumber(summary?.critical_count, 0),
    high: asNumber(summary?.high_count, 0),
    medium: asNumber(summary?.medium_count, 0),
    low: asNumber(summary?.low_count, 0),
    info: 0,
  };

  return {
    scan_id: asString(payload.scan_id, 'scan_unknown'),
    status: (asString(payload.status, 'completed') as ScanResult['status']),
    repository: asString(payload.repository, fallbackRepository),
    files_scanned: asNumber(payload.files_scanned, 0),
    scanned_label: 'Files Scanned',
    risk_score: computeRiskScore(normalizedSummary),
    summary: normalizedSummary,
    findings,
    error: asString(payload.error) || undefined,
  };
}

function normalizeScanResult(
  data: Record<string, unknown>,
  fallbackRepository: string
): ScanResult {
  const payload = extractScanPayload(data);
  const summary = asRecord(payload.summary);

  if (asObjectArray(payload.findings).length > 0 || hasOwn(summary, 'critical')) {
    return normalizeQuickScanResult(payload, fallbackRepository);
  }

  if (
    asObjectArray(payload.secrets).length > 0 ||
    hasOwn(summary, 'total_secrets')
  ) {
    return normalizeSecretsScanResult(payload, fallbackRepository);
  }

  if (
    asObjectArray(payload.vulnerabilities).length > 0 ||
    hasOwn(summary, 'total_dependencies') ||
    hasOwn(summary, 'critical_count')
  ) {
    return normalizeVulnerabilityScanResult(payload, fallbackRepository);
  }

  return normalizeQuickScanResult(payload, fallbackRepository);
}

export function SecurityScanWizard() {
  const { authFetch, isAuthenticated, isLoading: authLoading } = useAuthFetch();
  const [step, setStep] = useState<WizardStep>('configure');
  const [config, setConfig] = useState<ScanConfig>({
    scanType: 'quick',
    repoPath: process.cwd?.() || '.',
    includeSecrets: true,
    includeHistory: false,
    historyDepth: 100,
  });
  const [scanResult, setScanResult] = useState<ScanResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pollForResult = useCallback(async (scanId: string, scanType: ScanType): Promise<ScanResult> => {
    let lastError: string | null = null;

    for (let attempt = 0; attempt < MAX_POLL_ATTEMPTS; attempt++) {
      await sleep(POLL_INTERVAL_MS);

      try {
        const data = await authFetch<Record<string, unknown>>(
          buildScanStatusEndpoint(scanType, DEFAULT_REPOSITORY_ID, scanId),
          { method: 'GET' }
        );

        if (!data) {
          throw new Error('Sign in to run security scans.');
        }

        const result = normalizeScanResult(data, config.repoPath);
        if (result.status === 'completed') {
          return result;
        }
        if (result.status === 'failed') {
          throw new Error(result.error || 'Security scan failed.');
        }
      } catch (err) {
        lastError = err instanceof Error ? err.message : 'Failed to poll scan status.';
      }
    }

    throw new Error(
      lastError || 'Scan timed out before the backend returned a terminal result.'
    );
  }, [authFetch, config.repoPath]);

  const startScan = useCallback(async () => {
    setStep('scanning');
    setError(null);
    setScanResult(null);

    if (authLoading) {
      setError('Authentication is still loading. Try again in a moment.');
      setStep('configure');
      return;
    }

    if (!isAuthenticated) {
      setError('Sign in to run security scans.');
      setStep('configure');
      return;
    }

    try {
      const { endpoint, body } = buildScanRequest(config, DEFAULT_REPOSITORY_ID);
      const data = await authFetch<Record<string, unknown>>(endpoint, {
        method: 'POST',
        body: JSON.stringify(body),
      });

      if (!data) {
        throw new Error('Sign in to run security scans.');
      }

      const payload = extractScanPayload(data);
      const status = asString(payload.status, 'completed');

      if (status === 'running') {
        const scanId = asString(payload.scan_id);
        if (!scanId) {
          throw new Error('Backend started a scan without returning a scan ID.');
        }

        const result = await pollForResult(scanId, config.scanType);
        setScanResult(result);
        setStep('results');
        return;
      }

      const result = normalizeScanResult(payload, config.repoPath);
      if (result.status === 'failed') {
        throw new Error(result.error || 'Security scan failed.');
      }

      setScanResult(result);
      setStep('results');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start security scan.');
      setStep('configure');
    }
  }, [authFetch, authLoading, config, isAuthenticated, pollForResult]);

  const resetWizard = () => {
    setStep('configure');
    setScanResult(null);
    setError(null);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-theme-data text-[var(--acid-green)]">
            {'>'} SECURITY SCAN
          </h1>
          <p className="text-sm text-[var(--text-muted)] mt-1">
            Scan your codebase for vulnerabilities, secrets, and security issues
          </p>
        </div>
        {step !== 'configure' && (
          <button
            onClick={resetWizard}
            className="px-4 py-2 text-sm font-theme-data text-[var(--text-muted)] hover:text-[var(--text)] border border-[var(--border)] rounded hover:border-[var(--acid-green)]/30 transition-colors"
          >
            New Scan
          </button>
        )}
      </div>

      {/* Progress Indicator */}
      <div className="flex items-center gap-2">
        <StepIndicator step={1} active={step === 'configure'} completed={step !== 'configure'} label="Configure" />
        <div className="flex-1 h-px bg-[var(--border)]" />
        <StepIndicator step={2} active={step === 'scanning'} completed={step === 'results'} label="Scanning" />
        <div className="flex-1 h-px bg-[var(--border)]" />
        <StepIndicator step={3} active={step === 'results'} completed={false} label="Results" />
      </div>

      {error && (
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-sm font-theme-data">
          {error}
        </div>
      )}

      {/* Step Content */}
      {step === 'configure' && (
        <ConfigureStep
          config={config}
          onChange={setConfig}
          onStart={startScan}
        />
      )}

      {step === 'scanning' && (
        <ScanProgressView scanType={config.scanType} />
      )}

      {step === 'results' && scanResult && (
        <div className="space-y-6">
          <FindingsSummary result={scanResult} />
          <ReportExporter result={scanResult} />
        </div>
      )}
    </div>
  );
}

interface StepIndicatorProps {
  step: number;
  active: boolean;
  completed: boolean;
  label: string;
}

function StepIndicator({ step, active, completed, label }: StepIndicatorProps) {
  return (
    <div className="flex items-center gap-2">
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center font-theme-data text-sm border transition-colors ${
          active
            ? 'bg-[var(--acid-green)] text-[var(--bg)] border-[var(--acid-green)]'
            : completed
            ? 'bg-[var(--acid-green)]/20 text-[var(--acid-green)] border-[var(--acid-green)]'
            : 'bg-[var(--surface)] text-[var(--text-muted)] border-[var(--border)]'
        }`}
      >
        {completed ? '✓' : step}
      </div>
      <span className={`text-xs font-theme-data ${active ? 'text-[var(--acid-green)]' : 'text-[var(--text-muted)]'}`}>
        {label}
      </span>
    </div>
  );
}

interface ConfigureStepProps {
  config: ScanConfig;
  onChange: (config: ScanConfig) => void;
  onStart: () => void;
}

function ConfigureStep({ config, onChange, onStart }: ConfigureStepProps) {
  return (
    <div className="space-y-6">
      {/* Scan Type Selection */}
      <div className="bg-[var(--surface)] border border-[var(--border)] rounded p-4">
        <h3 className="text-sm font-theme-data text-[var(--acid-green)] mb-4">
          {'>'} SELECT SCAN TYPE
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {SCAN_TYPES.map((type) => (
            <button
              key={type.id}
              onClick={() => onChange({ ...config, scanType: type.id })}
              className={`p-4 text-left border rounded transition-colors ${
                config.scanType === type.id
                  ? 'border-[var(--acid-green)] bg-[var(--acid-green)]/10'
                  : 'border-[var(--border)] hover:border-[var(--acid-green)]/30'
              }`}
            >
              <div className="text-2xl mb-2">{type.icon}</div>
              <div className="font-theme-data text-sm text-[var(--text)]">{type.name}</div>
              <div className="text-xs text-[var(--text-muted)] mt-1">{type.description}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Repository Path */}
      <div className="bg-[var(--surface)] border border-[var(--border)] rounded p-4">
        <h3 className="text-sm font-theme-data text-[var(--acid-green)] mb-4">
          {'>'} REPOSITORY PATH
        </h3>
        <input
          type="text"
          value={config.repoPath}
          onChange={(e) => onChange({ ...config, repoPath: e.target.value })}
          placeholder="/path/to/repository"
          className="w-full px-3 py-2 bg-[var(--bg)] border border-[var(--border)] rounded font-theme-data text-sm text-[var(--text)] focus:border-[var(--acid-green)] focus:outline-none"
        />
        <p className="text-xs text-[var(--text-muted)] mt-2">
          Enter the absolute path to your repository or leave as default
        </p>
      </div>

      {/* Advanced Options */}
      {config.scanType === 'secrets' && (
        <div className="bg-[var(--surface)] border border-[var(--border)] rounded p-4">
          <h3 className="text-sm font-theme-data text-[var(--acid-green)] mb-4">
            {'>'} SECRETS SCAN OPTIONS
          </h3>
          <div className="space-y-3">
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={config.includeHistory}
                onChange={(e) => onChange({ ...config, includeHistory: e.target.checked })}
                className="w-4 h-4 accent-[var(--acid-green)]"
              />
              <span className="text-sm text-[var(--text)]">Scan git history for leaked secrets</span>
            </label>
            {config.includeHistory && (
              <div className="ml-7">
                <label className="text-xs text-[var(--text-muted)] block mb-1">History depth (commits)</label>
                <input
                  type="number"
                  value={config.historyDepth}
                  onChange={(e) => onChange({ ...config, historyDepth: parseInt(e.target.value) || 100 })}
                  min={10}
                  max={1000}
                  className="w-24 px-2 py-1 bg-[var(--bg)] border border-[var(--border)] rounded font-theme-data text-sm text-[var(--text)]"
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Start Button */}
      <div className="flex justify-end">
        <button
          onClick={onStart}
          className="px-6 py-3 bg-[var(--acid-green)] text-[var(--bg)] font-theme-data text-sm rounded hover:bg-[var(--acid-green)]/80 transition-colors flex items-center gap-2"
        >
          <span>Start Scan</span>
          <span>→</span>
        </button>
      </div>
    </div>
  );
}

export default SecurityScanWizard;
