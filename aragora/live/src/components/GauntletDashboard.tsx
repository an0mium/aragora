'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { API_BASE_URL } from '@/config';

interface GauntletResult {
  gauntlet_id: string;
  input_summary: string;
  input_hash: string;
  verdict: string;
  confidence: number;
  robustness_score: number;
  critical_count: number;
  high_count: number;
  medium_count: number;
  low_count: number;
  total_findings: number;
  created_at: string;
  duration_seconds?: number;
}

interface HeatmapCell {
  category: string;
  severity: string;
  count: number;
}

interface HeatmapData {
  cells: HeatmapCell[];
  categories: string[];
  severities: string[];
  total_findings: number;
}

interface GauntletDashboardProps {
  apiBase?: string;
  authToken?: string;
  onResultSelect?: (result: GauntletResult) => void;
}

const VERDICT_CONFIG: Record<string, { bg: string; border: string; text: string; icon: string }> = {
  PASS: { bg: 'bg-acid-green/20', border: 'border-acid-green', text: 'text-acid-green', icon: '\u2713' },
  APPROVED: { bg: 'bg-acid-green/20', border: 'border-acid-green', text: 'text-acid-green', icon: '\u2713' },
  CONDITIONAL: { bg: 'bg-acid-yellow/20', border: 'border-acid-yellow', text: 'text-acid-yellow', icon: '\u26A0' },
  APPROVED_WITH_CONDITIONS: { bg: 'bg-acid-yellow/20', border: 'border-acid-yellow', text: 'text-acid-yellow', icon: '\u26A0' },
  NEEDS_REVIEW: { bg: 'bg-warning/20', border: 'border-warning', text: 'text-warning', icon: '\u2691' },
  FAIL: { bg: 'bg-acid-red/20', border: 'border-acid-red', text: 'text-acid-red', icon: '\u2717' },
  REJECTED: { bg: 'bg-acid-red/20', border: 'border-acid-red', text: 'text-acid-red', icon: '\u2717' },
  UNKNOWN: { bg: 'bg-text-muted/20', border: 'border-text-muted', text: 'text-text-muted', icon: '?' },
};

const SEVERITY_COLORS: Record<string, string> = {
  critical: 'bg-acid-red',
  high: 'bg-warning',
  medium: 'bg-acid-yellow',
  low: 'bg-acid-cyan',
};

function StatCard({
  label,
  value,
  subValue,
  color = 'acid-green',
  icon,
}: {
  label: string;
  value: number | string;
  subValue?: string;
  color?: string;
  icon?: string;
}) {
  return (
    <div className="p-4 bg-surface/50 border border-border rounded-lg">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-mono text-text-muted">{label}</span>
        {icon && <span className="text-lg">{icon}</span>}
      </div>
      <div className={`text-2xl font-mono text-${color}`}>{value}</div>
      {subValue && (
        <div className="text-xs font-mono text-text-muted mt-1">{subValue}</div>
      )}
    </div>
  );
}

function VerdictBadge({ verdict }: { verdict: string }) {
  const config = VERDICT_CONFIG[verdict] || VERDICT_CONFIG.UNKNOWN;
  return (
    <span className={`px-2 py-1 rounded text-xs font-mono ${config.bg} ${config.border} ${config.text} border`}>
      {config.icon} {verdict}
    </span>
  );
}

function HeatmapVisualization({ data }: { data: HeatmapData }) {
  if (!data.cells.length) {
    return (
      <div className="text-center py-8 text-text-muted font-mono text-sm">
        No findings to display
      </div>
    );
  }

  const maxCount = Math.max(...data.cells.map(c => c.count), 1);

  return (
    <div className="space-y-4">
      {/* Heatmap grid */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr>
              <th className="text-xs font-mono text-text-muted text-left p-2">Category</th>
              {data.severities.map(sev => (
                <th key={sev} className="text-xs font-mono text-text-muted text-center p-2 capitalize">
                  {sev}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.categories.map(category => (
              <tr key={category}>
                <td className="text-xs font-mono text-text p-2 capitalize">
                  {category.replace(/_/g, ' ')}
                </td>
                {data.severities.map(severity => {
                  const cell = data.cells.find(c => c.category === category && c.severity === severity);
                  const count = cell?.count || 0;
                  const intensity = count / maxCount;
                  const bgColor = SEVERITY_COLORS[severity] || 'bg-text-muted';

                  return (
                    <td key={severity} className="p-1 text-center">
                      <div
                        className={`w-12 h-12 mx-auto rounded flex items-center justify-center font-mono text-sm transition-all ${
                          count > 0 ? `${bgColor}` : 'bg-surface'
                        }`}
                        style={{ opacity: count > 0 ? 0.3 + intensity * 0.7 : 0.2 }}
                        title={`${category} - ${severity}: ${count}`}
                      >
                        {count > 0 && <span className="text-bg font-bold">{count}</span>}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 justify-center text-xs font-mono">
        {data.severities.map(severity => (
          <div key={severity} className="flex items-center gap-2">
            <div className={`w-4 h-4 rounded ${SEVERITY_COLORS[severity]}`} style={{ opacity: 0.7 }} />
            <span className="text-text-muted capitalize">{severity}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ResultRow({
  result,
  onClick,
  isSelected,
  onExport,
  onCompare,
}: {
  result: GauntletResult;
  onClick: () => void;
  isSelected: boolean;
  onExport: (format: string) => void;
  onCompare: () => void;
}) {
  const [showActions, setShowActions] = useState(false);

  return (
    <div
      className={`
        p-4 border-l-4 transition-all cursor-pointer
        ${isSelected ? 'bg-surface border-acid-green' : 'bg-surface/30 border-transparent hover:border-acid-green/50'}
      `}
      onClick={onClick}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <VerdictBadge verdict={result.verdict} />
            <span className="text-xs font-mono text-text-muted">
              {result.gauntlet_id.slice(-12)}
            </span>
          </div>
          <p className="text-sm font-mono text-text truncate">
            {result.input_summary}
          </p>
          <div className="flex items-center gap-4 mt-2 text-xs font-mono text-text-muted">
            <span>{new Date(result.created_at).toLocaleString()}</span>
            {result.duration_seconds && (
              <span>{result.duration_seconds}s</span>
            )}
          </div>
        </div>

        <div className="flex flex-col items-end gap-2">
          <div className="flex items-center gap-2">
            {result.critical_count > 0 && (
              <span className="px-2 py-0.5 bg-acid-red/20 text-acid-red text-xs font-mono rounded">
                {result.critical_count} CRIT
              </span>
            )}
            {result.high_count > 0 && (
              <span className="px-2 py-0.5 bg-warning/20 text-warning text-xs font-mono rounded">
                {result.high_count} HIGH
              </span>
            )}
          </div>

          {showActions && (
            <div className="flex gap-1">
              <button
                onClick={(e) => { e.stopPropagation(); onExport('html'); }}
                className="px-2 py-1 text-xs font-mono bg-acid-green/10 text-acid-green hover:bg-acid-green/20 rounded transition-colors"
              >
                HTML
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); onExport('md'); }}
                className="px-2 py-1 text-xs font-mono bg-acid-cyan/10 text-acid-cyan hover:bg-acid-cyan/20 rounded transition-colors"
              >
                MD
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); onCompare(); }}
                className="px-2 py-1 text-xs font-mono bg-accent/10 text-accent hover:bg-accent/20 rounded transition-colors"
              >
                CMP
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function CompareView({
  result1,
  result2,
  apiBase,
  onClose,
}: {
  result1: GauntletResult;
  result2: GauntletResult;
  apiBase: string;
  onClose: () => void;
}) {
  const [comparison, setComparison] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [compareError, setCompareError] = useState<string | null>(null);

  useEffect(() => {
    const fetchComparison = async () => {
      try {
        setCompareError(null);
        const response = await fetch(
          `${apiBase}/api/gauntlet/${result1.gauntlet_id}/compare/${result2.gauntlet_id}`
        );
        if (response.ok) {
          const data = await response.json();
          setComparison(data);
        } else {
          setCompareError('Failed to load comparison. Please try again.');
        }
      } catch (err) {
        console.error('Failed to fetch comparison:', err);
        setCompareError('Unable to compare results. Please check your connection.');
      } finally {
        setLoading(false);
      }
    };
    fetchComparison();
  }, [apiBase, result1.gauntlet_id, result2.gauntlet_id]);

  const calcDiff = (a: number, b: number) => {
    const diff = a - b;
    return diff > 0 ? `+${diff}` : diff.toString();
  };

  return (
    <div className="bg-surface border border-acid-cyan/30 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-mono text-acid-cyan text-sm">COMPARISON VIEW</h3>
        <button
          onClick={onClose}
          className="text-text-muted hover:text-acid-green font-mono text-sm"
        >
          [CLOSE]
        </button>
      </div>

      {loading ? (
        <div className="text-center py-8 text-acid-green font-mono animate-pulse">
          Loading comparison...
        </div>
      ) : compareError ? (
        <div className="p-4 bg-warning/10 border border-warning/30 rounded text-warning font-mono text-sm">
          {compareError}
        </div>
      ) : (
        <div className="grid md:grid-cols-2 gap-6">
          {/* Result 1 */}
          <div className="p-4 border border-acid-green/30 rounded-lg">
            <div className="text-xs font-mono text-acid-green mb-2">RUN A</div>
            <VerdictBadge verdict={result1.verdict} />
            <div className="mt-4 space-y-2 text-sm font-mono">
              <div className="flex justify-between">
                <span className="text-text-muted">Critical:</span>
                <span className="text-acid-red">{result1.critical_count}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">High:</span>
                <span className="text-warning">{result1.high_count}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Confidence:</span>
                <span className="text-text">{(result1.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Robustness:</span>
                <span className="text-text">{(result1.robustness_score * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>

          {/* Result 2 */}
          <div className="p-4 border border-accent/30 rounded-lg">
            <div className="text-xs font-mono text-accent mb-2">RUN B</div>
            <VerdictBadge verdict={result2.verdict} />
            <div className="mt-4 space-y-2 text-sm font-mono">
              <div className="flex justify-between">
                <span className="text-text-muted">Critical:</span>
                <span className="text-acid-red">{result2.critical_count}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">High:</span>
                <span className="text-warning">{result2.high_count}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Confidence:</span>
                <span className="text-text">{(result2.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Robustness:</span>
                <span className="text-text">{(result2.robustness_score * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>

          {/* Delta */}
          <div className="md:col-span-2 p-4 bg-bg/50 border border-border rounded-lg">
            <div className="text-xs font-mono text-text-muted mb-3">DELTA (A - B)</div>
            <div className="grid grid-cols-4 gap-4 text-center">
              <div>
                <div className={`text-lg font-mono ${result1.critical_count - result2.critical_count > 0 ? 'text-acid-red' : result1.critical_count - result2.critical_count < 0 ? 'text-acid-green' : 'text-text-muted'}`}>
                  {calcDiff(result1.critical_count, result2.critical_count)}
                </div>
                <div className="text-xs font-mono text-text-muted">Critical</div>
              </div>
              <div>
                <div className={`text-lg font-mono ${result1.high_count - result2.high_count > 0 ? 'text-warning' : result1.high_count - result2.high_count < 0 ? 'text-acid-green' : 'text-text-muted'}`}>
                  {calcDiff(result1.high_count, result2.high_count)}
                </div>
                <div className="text-xs font-mono text-text-muted">High</div>
              </div>
              <div>
                <div className={`text-lg font-mono ${result1.total_findings - result2.total_findings > 0 ? 'text-acid-yellow' : result1.total_findings - result2.total_findings < 0 ? 'text-acid-green' : 'text-text-muted'}`}>
                  {calcDiff(result1.total_findings, result2.total_findings)}
                </div>
                <div className="text-xs font-mono text-text-muted">Total</div>
              </div>
              <div>
                <div className={`text-lg font-mono ${result1.robustness_score - result2.robustness_score > 0 ? 'text-acid-green' : result1.robustness_score - result2.robustness_score < 0 ? 'text-acid-red' : 'text-text-muted'}`}>
                  {((result1.robustness_score - result2.robustness_score) * 100).toFixed(0)}%
                </div>
                <div className="text-xs font-mono text-text-muted">Robustness</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function GauntletDashboard({
  apiBase = API_BASE_URL,
  authToken,
  onResultSelect,
}: GauntletDashboardProps) {
  const [results, setResults] = useState<GauntletResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<GauntletResult | null>(null);
  const [compareResult, setCompareResult] = useState<GauntletResult | null>(null);
  const [heatmapData, setHeatmapData] = useState<HeatmapData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [heatmapError, setHeatmapError] = useState<string | null>(null);
  const [verdictFilter, setVerdictFilter] = useState<string | null>(null);
  const [showCompare, setShowCompare] = useState(false);

  const fetchResults = useCallback(async () => {
    try {
      setLoading(true);
      const headers: Record<string, string> = {};
      if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
      }

      const url = new URL(`${apiBase}/api/gauntlet/results`);
      url.searchParams.set('limit', '50');
      if (verdictFilter) {
        url.searchParams.set('verdict', verdictFilter);
      }

      const response = await fetch(url.toString(), { headers });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      setResults(data.results || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch results');
    } finally {
      setLoading(false);
    }
  }, [apiBase, authToken, verdictFilter]);

  const fetchHeatmap = useCallback(async (gauntletId: string) => {
    try {
      setHeatmapError(null);
      const response = await fetch(`${apiBase}/api/gauntlet/${gauntletId}/heatmap`);
      if (response.ok) {
        const data = await response.json();
        setHeatmapData(data);
      } else {
        setHeatmapError('Failed to load heatmap data.');
      }
    } catch (err) {
      console.error('Failed to fetch heatmap:', err);
      setHeatmapError('Unable to load heatmap. Please try again.');
    }
  }, [apiBase]);

  useEffect(() => {
    fetchResults();
  }, [fetchResults]);

  useEffect(() => {
    if (selectedResult) {
      fetchHeatmap(selectedResult.gauntlet_id);
    } else {
      setHeatmapData(null);
    }
  }, [selectedResult, fetchHeatmap]);

  const handleResultSelect = (result: GauntletResult) => {
    if (showCompare && selectedResult) {
      setCompareResult(result);
    } else {
      setSelectedResult(result);
      if (onResultSelect) {
        onResultSelect(result);
      }
    }
  };

  const handleExport = (gauntletId: string, format: string) => {
    window.open(`${apiBase}/api/gauntlet/${gauntletId}/receipt?format=${format}`, '_blank');
  };

  const handleStartCompare = (result: GauntletResult) => {
    setSelectedResult(result);
    setShowCompare(true);
  };

  // Summary stats
  const summary = useMemo(() => {
    const passed = results.filter(r => ['PASS', 'APPROVED'].includes(r.verdict)).length;
    const conditional = results.filter(r => ['CONDITIONAL', 'APPROVED_WITH_CONDITIONS', 'NEEDS_REVIEW'].includes(r.verdict)).length;
    const failed = results.filter(r => ['FAIL', 'REJECTED'].includes(r.verdict)).length;
    const totalCritical = results.reduce((sum, r) => sum + (r.critical_count || 0), 0);
    const totalHigh = results.reduce((sum, r) => sum + (r.high_count || 0), 0);
    const avgRobustness = results.length > 0
      ? results.reduce((sum, r) => sum + (r.robustness_score || 0), 0) / results.length
      : 0;

    return { passed, conditional, failed, totalCritical, totalHigh, avgRobustness, total: results.length };
  }, [results]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="font-mono text-acid-green text-lg flex items-center gap-2">
          <span className="text-xl">\u2694\uFE0F</span> GAUNTLET DASHBOARD
        </h2>
        <button
          onClick={fetchResults}
          disabled={loading}
          className="px-4 py-2 text-sm font-mono border border-acid-green/50 text-acid-green hover:bg-acid-green/10 transition-colors disabled:opacity-50"
        >
          {loading ? 'LOADING...' : 'REFRESH'}
        </button>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
        <StatCard label="TOTAL RUNS" value={summary.total} icon="\u{1F3C3}" />
        <StatCard label="PASSED" value={summary.passed} color="acid-green" icon="\u2713" />
        <StatCard label="CONDITIONAL" value={summary.conditional} color="acid-yellow" icon="\u26A0" />
        <StatCard label="FAILED" value={summary.failed} color="acid-red" icon="\u2717" />
        <StatCard label="CRITICAL ISSUES" value={summary.totalCritical} color="acid-red" icon="\u{1F6A8}" />
        <StatCard label="HIGH ISSUES" value={summary.totalHigh} color="warning" icon="\u26A0" />
        <StatCard
          label="AVG ROBUSTNESS"
          value={`${(summary.avgRobustness * 100).toFixed(0)}%`}
          color={summary.avgRobustness > 0.7 ? 'acid-green' : summary.avgRobustness > 0.4 ? 'acid-yellow' : 'acid-red'}
          icon="\u{1F6E1}"
        />
      </div>

      {/* Filters */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-mono text-text-muted">FILTER:</span>
        {['PASS', 'CONDITIONAL', 'FAIL'].map(verdict => (
          <button
            key={verdict}
            onClick={() => setVerdictFilter(verdictFilter === verdict ? null : verdict)}
            className={`px-2 py-1 text-xs font-mono border rounded transition-colors ${
              verdictFilter === verdict
                ? `${VERDICT_CONFIG[verdict]?.bg} ${VERDICT_CONFIG[verdict]?.border} ${VERDICT_CONFIG[verdict]?.text}`
                : 'bg-surface border-border text-text-muted hover:border-acid-green/30'
            }`}
          >
            {verdict}
          </button>
        ))}
        {verdictFilter && (
          <button
            onClick={() => setVerdictFilter(null)}
            className="text-xs font-mono text-text-muted hover:text-acid-green"
          >
            [CLEAR]
          </button>
        )}
        <div className="flex-1" />
        <button
          onClick={() => setShowCompare(!showCompare)}
          className={`px-3 py-1 text-xs font-mono border rounded transition-colors ${
            showCompare
              ? 'bg-accent/20 border-accent text-accent'
              : 'border-border text-text-muted hover:border-accent/30'
          }`}
        >
          {showCompare ? 'COMPARING...' : 'COMPARE MODE'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="p-4 bg-warning/10 border border-warning/30 rounded-lg">
          <div className="text-warning font-mono text-sm">{error}</div>
        </div>
      )}

      {/* Comparison View */}
      {showCompare && selectedResult && compareResult && (
        <CompareView
          result1={selectedResult}
          result2={compareResult}
          apiBase={apiBase}
          onClose={() => {
            setShowCompare(false);
            setCompareResult(null);
          }}
        />
      )}

      {/* Main Content */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Results List */}
        <div className="lg:col-span-2 bg-surface border border-border rounded-lg overflow-hidden">
          <div className="px-4 py-3 border-b border-border bg-bg/50">
            <span className="text-xs font-mono text-acid-green uppercase">
              {'>'} RECENT GAUNTLET RUNS ({results.length})
            </span>
            {showCompare && selectedResult && !compareResult && (
              <span className="ml-4 text-xs font-mono text-accent animate-pulse">
                Select another run to compare...
              </span>
            )}
          </div>
          <div className="max-h-[500px] overflow-y-auto divide-y divide-border">
            {loading && results.length === 0 && (
              <div className="p-8 text-center text-acid-green font-mono animate-pulse">
                Loading results...
              </div>
            )}
            {!loading && results.length === 0 && (
              <div className="p-8 text-center text-text-muted font-mono">
                No gauntlet runs found
              </div>
            )}
            {results.map(result => (
              <ResultRow
                key={result.gauntlet_id}
                result={result}
                onClick={() => handleResultSelect(result)}
                isSelected={selectedResult?.gauntlet_id === result.gauntlet_id}
                onExport={(format) => handleExport(result.gauntlet_id, format)}
                onCompare={() => handleStartCompare(result)}
              />
            ))}
          </div>
        </div>

        {/* Detail Panel */}
        <div className="space-y-4">
          {/* Selected Result Details */}
          <div className="bg-surface border border-acid-cyan/30 rounded-lg p-4">
            <h3 className="font-mono text-acid-cyan text-sm mb-4">RESULT DETAILS</h3>
            {selectedResult ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <VerdictBadge verdict={selectedResult.verdict} />
                  <span className="text-xs font-mono text-text-muted">
                    {selectedResult.gauntlet_id.slice(-12)}
                  </span>
                </div>

                <div className="p-3 bg-bg/50 rounded text-xs font-mono text-text">
                  {selectedResult.input_summary}
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="p-2 bg-bg/50 rounded">
                    <div className="text-xs text-text-muted">Confidence</div>
                    <div className="text-lg font-mono text-acid-green">
                      {(selectedResult.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div className="p-2 bg-bg/50 rounded">
                    <div className="text-xs text-text-muted">Robustness</div>
                    <div className="text-lg font-mono text-acid-cyan">
                      {(selectedResult.robustness_score * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  {selectedResult.critical_count > 0 && (
                    <span className="px-2 py-1 bg-acid-red/20 text-acid-red text-xs font-mono rounded">
                      {selectedResult.critical_count} Critical
                    </span>
                  )}
                  {selectedResult.high_count > 0 && (
                    <span className="px-2 py-1 bg-warning/20 text-warning text-xs font-mono rounded">
                      {selectedResult.high_count} High
                    </span>
                  )}
                  {(selectedResult.medium_count || 0) > 0 && (
                    <span className="px-2 py-1 bg-acid-yellow/20 text-acid-yellow text-xs font-mono rounded">
                      {selectedResult.medium_count} Medium
                    </span>
                  )}
                  {(selectedResult.low_count || 0) > 0 && (
                    <span className="px-2 py-1 bg-acid-cyan/20 text-acid-cyan text-xs font-mono rounded">
                      {selectedResult.low_count} Low
                    </span>
                  )}
                </div>

                <div className="flex gap-2">
                  <a
                    href={`${apiBase}/api/gauntlet/${selectedResult.gauntlet_id}/receipt?format=html`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-1 py-2 text-center text-xs font-mono bg-acid-green/10 border border-acid-green/30 text-acid-green hover:bg-acid-green/20 rounded transition-colors"
                  >
                    VIEW RECEIPT
                  </a>
                  <a
                    href={`${apiBase}/api/gauntlet/${selectedResult.gauntlet_id}/heatmap?format=svg`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-1 py-2 text-center text-xs font-mono bg-acid-cyan/10 border border-acid-cyan/30 text-acid-cyan hover:bg-acid-cyan/20 rounded transition-colors"
                  >
                    SVG HEATMAP
                  </a>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-text-muted font-mono text-sm">
                Select a result to view details
              </div>
            )}
          </div>

          {/* Heatmap */}
          {selectedResult && (heatmapData || heatmapError) && (
            <div className="bg-surface border border-acid-yellow/30 rounded-lg p-4">
              <h3 className="font-mono text-acid-yellow text-sm mb-4">RISK HEATMAP</h3>
              {heatmapError ? (
                <div className="p-3 bg-warning/10 border border-warning/30 rounded text-warning font-mono text-sm">
                  {heatmapError}
                </div>
              ) : heatmapData ? (
                <HeatmapVisualization data={heatmapData} />
              ) : null}
            </div>
          )}
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-xs font-mono pt-4 border-t border-border">
        {Object.entries(VERDICT_CONFIG).slice(0, 5).map(([verdict, config]) => (
          <div key={verdict} className="flex items-center gap-2">
            <div className={`w-4 h-4 rounded ${config.bg} ${config.border} border flex items-center justify-center`}>
              <span className={config.text}>{config.icon}</span>
            </div>
            <span className="text-text-muted">{verdict}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default GauntletDashboard;
