'use client';

import { useState, useEffect, useCallback } from 'react';

interface ExportStats {
  available_exporters: string[];
  export_directory: string;
  exported_files: ExportedFile[];
  sft_available?: boolean;
}

interface ExportedFile {
  name: string;
  size_bytes: number;
  created_at: string;
  modified_at: string;
}

interface ExportFormat {
  description: string;
  schema: Record<string, unknown>;
  use_case: string;
}

interface FormatsResponse {
  formats: Record<string, ExportFormat>;
  output_formats: string[];
  endpoints: Record<string, string>;
}

interface ExportResult {
  export_type: string;
  total_records: number;
  parameters: Record<string, unknown>;
  exported_at: string;
  format: string;
  records?: unknown[];
  data?: string;
}

type ExportType = 'sft' | 'dpo' | 'gauntlet';
type OutputFormat = 'json' | 'jsonl';

interface TrainingExportPanelProps {
  apiBase?: string;
}

export function TrainingExportPanel({ apiBase = '/api' }: TrainingExportPanelProps) {
  const [stats, setStats] = useState<ExportStats | null>(null);
  const [formats, setFormats] = useState<FormatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [lastExport, setLastExport] = useState<ExportResult | null>(null);

  // Export parameters
  const [exportType, setExportType] = useState<ExportType>('sft');
  const [outputFormat, setOutputFormat] = useState<OutputFormat>('json');

  // SFT parameters
  const [minConfidence, setMinConfidence] = useState(0.7);
  const [minSuccessRate, setMinSuccessRate] = useState(0.6);
  const [limit, setLimit] = useState(1000);
  const [includeCritiques, setIncludeCritiques] = useState(true);
  const [includePatterns, setIncludePatterns] = useState(true);
  const [includeDebates, setIncludeDebates] = useState(true);

  // DPO parameters
  const [minConfidenceDiff, setMinConfidenceDiff] = useState(0.1);

  // Gauntlet parameters
  const [persona, setPersona] = useState('all');
  const [minSeverity, setMinSeverity] = useState(0.5);

  const fetchStats = useCallback(async () => {
    try {
      const response = await fetch(`${apiBase}/training/stats`);
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (err) {
      console.error('Failed to fetch training stats:', err);
    }
  }, [apiBase]);

  const fetchFormats = useCallback(async () => {
    try {
      const response = await fetch(`${apiBase}/training/formats`);
      if (response.ok) {
        const data = await response.json();
        setFormats(data);
      }
    } catch (err) {
      console.error('Failed to fetch training formats:', err);
    }
  }, [apiBase]);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        await Promise.all([fetchStats(), fetchFormats()]);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load training data');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [fetchStats, fetchFormats]);

  const handleExport = async () => {
    setExporting(true);
    setError(null);
    setLastExport(null);

    try {
      const params = new URLSearchParams();
      params.set('format', outputFormat);

      if (exportType === 'sft') {
        params.set('min_confidence', String(minConfidence));
        params.set('min_success_rate', String(minSuccessRate));
        params.set('limit', String(limit));
        params.set('include_critiques', String(includeCritiques));
        params.set('include_patterns', String(includePatterns));
        params.set('include_debates', String(includeDebates));
      } else if (exportType === 'dpo') {
        params.set('min_confidence_diff', String(minConfidenceDiff));
        params.set('limit', String(limit));
      } else if (exportType === 'gauntlet') {
        params.set('persona', persona);
        params.set('min_severity', String(minSeverity));
        params.set('limit', String(limit));
      }

      const response = await fetch(`${apiBase}/training/export/${exportType}?${params}`);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Export failed: ${response.status}`);
      }

      const result: ExportResult = await response.json();
      setLastExport(result);

      // Refresh stats after export
      fetchStats();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed');
    } finally {
      setExporting(false);
    }
  };

  const downloadExport = () => {
    if (!lastExport) return;

    let content: string;
    let filename: string;

    if (lastExport.format === 'jsonl' && lastExport.data) {
      content = lastExport.data;
      filename = `${lastExport.export_type}_export_${Date.now()}.jsonl`;
    } else {
      content = JSON.stringify(lastExport.records || [], null, 2);
      filename = `${lastExport.export_type}_export_${Date.now()}.json`;
    }

    const blob = new Blob([content], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (dateStr: string) => {
    try {
      return new Date(dateStr).toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return dateStr;
    }
  };

  if (loading) {
    return (
      <div className="bg-surface border border-acid-green/30 p-8">
        <div className="flex items-center justify-center gap-2">
          <div className="w-2 h-2 bg-acid-green rounded-full animate-pulse" />
          <span className="text-xs font-mono text-acid-green">LOADING TRAINING DATA...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-surface border border-acid-green/30">
      {/* Header */}
      <div className="px-4 py-3 border-b border-acid-green/20 bg-bg/50 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-acid-green uppercase tracking-wider">
            {'>'} TRAINING DATA EXPORT
          </span>
          {stats && (
            <span className="text-xs font-mono text-text-muted">
              {stats.available_exporters.length} exporter{stats.available_exporters.length !== 1 ? 's' : ''} available
            </span>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="px-4 py-3 bg-crimson/10 border-b border-crimson/30">
          <span className="text-xs font-mono text-crimson">{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 p-4">
        {/* Export Configuration */}
        <div className="space-y-4">
          <div className="border border-border p-4">
            <h3 className="text-xs font-mono text-acid-green uppercase mb-4">EXPORT CONFIGURATION</h3>

            {/* Export Type */}
            <div className="space-y-2 mb-4">
              <label className="text-xs font-mono text-text-muted">EXPORT TYPE</label>
              <div className="flex gap-2">
                {(['sft', 'dpo', 'gauntlet'] as ExportType[]).map((type) => (
                  <button
                    key={type}
                    onClick={() => setExportType(type)}
                    disabled={stats && !stats.available_exporters.includes(type)}
                    className={`px-3 py-1.5 text-xs font-mono border transition-colors ${
                      exportType === type
                        ? 'bg-acid-green/20 text-acid-green border-acid-green/40'
                        : stats && !stats.available_exporters.includes(type)
                          ? 'text-text-muted border-border cursor-not-allowed opacity-50'
                          : 'text-text-primary border-border hover:border-acid-green/40'
                    }`}
                  >
                    {type.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            {/* Output Format */}
            <div className="space-y-2 mb-4">
              <label className="text-xs font-mono text-text-muted">OUTPUT FORMAT</label>
              <div className="flex gap-2">
                {(['json', 'jsonl'] as OutputFormat[]).map((format) => (
                  <button
                    key={format}
                    onClick={() => setOutputFormat(format)}
                    className={`px-3 py-1.5 text-xs font-mono border transition-colors ${
                      outputFormat === format
                        ? 'bg-acid-cyan/20 text-acid-cyan border-acid-cyan/40'
                        : 'text-text-primary border-border hover:border-acid-cyan/40'
                    }`}
                  >
                    {format.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            {/* SFT Parameters */}
            {exportType === 'sft' && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs font-mono text-text-muted block mb-1">MIN CONFIDENCE</label>
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.1"
                      value={minConfidence}
                      onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
                      className="w-full bg-bg border border-border px-2 py-1.5 text-xs font-mono text-text-primary focus:border-acid-green/50 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-mono text-text-muted block mb-1">MIN SUCCESS RATE</label>
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.1"
                      value={minSuccessRate}
                      onChange={(e) => setMinSuccessRate(parseFloat(e.target.value))}
                      className="w-full bg-bg border border-border px-2 py-1.5 text-xs font-mono text-text-primary focus:border-acid-green/50 focus:outline-none"
                    />
                  </div>
                </div>
                <div>
                  <label className="text-xs font-mono text-text-muted block mb-1">LIMIT</label>
                  <input
                    type="number"
                    min="1"
                    max="10000"
                    value={limit}
                    onChange={(e) => setLimit(parseInt(e.target.value, 10))}
                    className="w-full bg-bg border border-border px-2 py-1.5 text-xs font-mono text-text-primary focus:border-acid-green/50 focus:outline-none"
                  />
                </div>
                <div className="flex flex-wrap gap-3">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={includeCritiques}
                      onChange={(e) => setIncludeCritiques(e.target.checked)}
                      className="accent-acid-green"
                    />
                    <span className="text-xs font-mono text-text-primary">Critiques</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={includePatterns}
                      onChange={(e) => setIncludePatterns(e.target.checked)}
                      className="accent-acid-green"
                    />
                    <span className="text-xs font-mono text-text-primary">Patterns</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={includeDebates}
                      onChange={(e) => setIncludeDebates(e.target.checked)}
                      className="accent-acid-green"
                    />
                    <span className="text-xs font-mono text-text-primary">Debates</span>
                  </label>
                </div>
              </div>
            )}

            {/* DPO Parameters */}
            {exportType === 'dpo' && (
              <div className="space-y-3">
                <div>
                  <label className="text-xs font-mono text-text-muted block mb-1">MIN CONFIDENCE DIFF</label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.05"
                    value={minConfidenceDiff}
                    onChange={(e) => setMinConfidenceDiff(parseFloat(e.target.value))}
                    className="w-full bg-bg border border-border px-2 py-1.5 text-xs font-mono text-text-primary focus:border-acid-green/50 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="text-xs font-mono text-text-muted block mb-1">LIMIT</label>
                  <input
                    type="number"
                    min="1"
                    max="5000"
                    value={limit}
                    onChange={(e) => setLimit(parseInt(e.target.value, 10))}
                    className="w-full bg-bg border border-border px-2 py-1.5 text-xs font-mono text-text-primary focus:border-acid-green/50 focus:outline-none"
                  />
                </div>
              </div>
            )}

            {/* Gauntlet Parameters */}
            {exportType === 'gauntlet' && (
              <div className="space-y-3">
                <div>
                  <label className="text-xs font-mono text-text-muted block mb-1">PERSONA</label>
                  <select
                    value={persona}
                    onChange={(e) => setPersona(e.target.value)}
                    className="w-full bg-bg border border-border px-2 py-1.5 text-xs font-mono text-text-primary focus:border-acid-green/50 focus:outline-none"
                  >
                    <option value="all">ALL</option>
                    <option value="gdpr">GDPR</option>
                    <option value="hipaa">HIPAA</option>
                    <option value="ai_act">AI ACT</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs font-mono text-text-muted block mb-1">MIN SEVERITY</label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    value={minSeverity}
                    onChange={(e) => setMinSeverity(parseFloat(e.target.value))}
                    className="w-full bg-bg border border-border px-2 py-1.5 text-xs font-mono text-text-primary focus:border-acid-green/50 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="text-xs font-mono text-text-muted block mb-1">LIMIT</label>
                  <input
                    type="number"
                    min="1"
                    max="5000"
                    value={limit}
                    onChange={(e) => setLimit(parseInt(e.target.value, 10))}
                    className="w-full bg-bg border border-border px-2 py-1.5 text-xs font-mono text-text-primary focus:border-acid-green/50 focus:outline-none"
                  />
                </div>
              </div>
            )}

            {/* Export Button */}
            <div className="mt-4 pt-4 border-t border-border">
              <button
                onClick={handleExport}
                disabled={exporting || (stats && !stats.available_exporters.includes(exportType))}
                className={`w-full px-4 py-2 text-xs font-mono border transition-colors ${
                  exporting
                    ? 'bg-acid-green/10 text-acid-green border-acid-green/40 cursor-wait'
                    : stats && !stats.available_exporters.includes(exportType)
                      ? 'bg-bg text-text-muted border-border cursor-not-allowed'
                      : 'bg-acid-green/20 text-acid-green border-acid-green/40 hover:bg-acid-green/30'
                }`}
              >
                {exporting ? 'EXPORTING...' : `EXPORT ${exportType.toUpperCase()}`}
              </button>
            </div>
          </div>

          {/* Format Info */}
          {formats && formats.formats[exportType] && (
            <div className="border border-border p-4">
              <h3 className="text-xs font-mono text-purple uppercase mb-2">FORMAT INFO</h3>
              <p className="text-xs font-mono text-text-primary mb-2">
                {formats.formats[exportType].description}
              </p>
              <p className="text-xs font-mono text-text-muted">
                {formats.formats[exportType].use_case}
              </p>
            </div>
          )}
        </div>

        {/* Results & Stats */}
        <div className="space-y-4">
          {/* Last Export Result */}
          {lastExport && (
            <div className="border border-acid-green/40 bg-acid-green/5 p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xs font-mono text-acid-green uppercase">EXPORT COMPLETE</h3>
                <button
                  onClick={downloadExport}
                  className="px-3 py-1 text-xs font-mono text-acid-cyan border border-acid-cyan/40 hover:bg-acid-cyan/10 transition-colors"
                >
                  DOWNLOAD
                </button>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span className="text-xs font-mono text-text-muted">Type:</span>
                  <span className="text-xs font-mono text-text-primary">{lastExport.export_type.toUpperCase()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-xs font-mono text-text-muted">Records:</span>
                  <span className="text-xs font-mono text-acid-green">{lastExport.total_records.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-xs font-mono text-text-muted">Format:</span>
                  <span className="text-xs font-mono text-text-primary">{lastExport.format.toUpperCase()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-xs font-mono text-text-muted">Exported:</span>
                  <span className="text-xs font-mono text-text-primary">{formatDate(lastExport.exported_at)}</span>
                </div>
              </div>
            </div>
          )}

          {/* Available Exporters */}
          {stats && (
            <div className="border border-border p-4">
              <h3 className="text-xs font-mono text-acid-cyan uppercase mb-3">AVAILABLE EXPORTERS</h3>
              <div className="flex flex-wrap gap-2">
                {['sft', 'dpo', 'gauntlet'].map((type) => (
                  <span
                    key={type}
                    className={`px-2 py-1 text-xs font-mono border ${
                      stats.available_exporters.includes(type)
                        ? 'text-acid-green border-acid-green/40 bg-acid-green/10'
                        : 'text-text-muted border-border bg-bg/30'
                    }`}
                  >
                    {type.toUpperCase()}
                    {stats.available_exporters.includes(type) ? ' ✓' : ' ✗'}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Exported Files */}
          {stats && stats.exported_files.length > 0 && (
            <div className="border border-border p-4">
              <h3 className="text-xs font-mono text-gold uppercase mb-3">RECENT EXPORTS</h3>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {stats.exported_files.slice(0, 10).map((file, idx) => (
                  <div key={idx} className="flex items-center justify-between py-1 border-b border-border last:border-0">
                    <span className="text-xs font-mono text-text-primary truncate max-w-[200px]" title={file.name}>
                      {file.name}
                    </span>
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-mono text-text-muted">{formatBytes(file.size_bytes)}</span>
                      <span className="text-xs font-mono text-text-muted">{formatDate(file.created_at)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Export Directory */}
          {stats && (
            <div className="border border-border p-4">
              <h3 className="text-xs font-mono text-text-muted uppercase mb-2">EXPORT DIRECTORY</h3>
              <code className="text-xs font-mono text-text-primary break-all">{stats.export_directory}</code>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default TrainingExportPanel;
