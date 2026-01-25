'use client';
import { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { ErrorWithRetry } from '@/components/ErrorWithRetry';
import { DeliveryModal } from '@/components/receipts';
import { logger } from '@/utils/logger';

interface GauntletResult {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  verdict?: 'PASS' | 'CONDITIONAL' | 'FAIL';
  confidence?: number;
  created_at: string;
  completed_at?: string;
  input_summary?: string;
  risk_summary?: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
  vulnerabilities_found?: number;
}

interface ProvenanceRecord {
  timestamp: string;
  event_type: string;
  agent?: string;
  description: string;
  evidence_hash: string;
}

interface ConsensusProof {
  reached: boolean;
  confidence: number;
  supporting_agents: string[];
  dissenting_agents: string[];
  method: string;
  evidence_hash: string;
}

interface DecisionReceipt {
  receipt_id: string;
  gauntlet_id: string;
  timestamp: string;
  input_summary: string;
  input_hash: string;
  risk_summary: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
  attacks_attempted: number;
  attacks_successful: number;
  probes_run: number;
  vulnerabilities_found: number;
  verdict: 'PASS' | 'CONDITIONAL' | 'FAIL';
  confidence: number;
  robustness_score: number;
  vulnerability_details: Array<{
    id: string;
    category: string;
    severity: string;
    description: string;
  }>;
  verdict_reasoning: string;
  dissenting_views: string[];
  consensus_proof?: ConsensusProof;
  provenance_chain: ProvenanceRecord[];
  artifact_hash: string;
}

type TabType = 'list' | 'detail';

export default function ReceiptsPage() {
  const { config } = useBackend();
  const backendUrl = config.api;
  const [activeTab, setActiveTab] = useState<TabType>('list');
  const [results, setResults] = useState<GauntletResult[]>([]);
  const [selectedReceipt, setSelectedReceipt] = useState<DecisionReceipt | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [receiptLoading, setReceiptLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'PASS' | 'CONDITIONAL' | 'FAIL'>('all');
  const [deliveryModalOpen, setDeliveryModalOpen] = useState(false);

  const fetchResults = useCallback(async () => {
    try {
      const response = await fetch(`${backendUrl}/api/gauntlet/results?limit=50`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setResults(data.results || []);
    } catch (err) {
      logger.error('Failed to fetch results:', err);
      throw err;
    }
  }, [backendUrl]);

  const fetchReceipt = useCallback(async (gauntletId: string) => {
    setReceiptLoading(true);
    try {
      const response = await fetch(`${backendUrl}/api/gauntlet/${gauntletId}/receipt`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setSelectedReceipt(data);
      setSelectedId(gauntletId);
      setActiveTab('detail');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load receipt');
    } finally {
      setReceiptLoading(false);
    }
  }, [backendUrl]);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      await fetchResults();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, [fetchResults]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const downloadReceipt = async (format: 'json' | 'html' | 'markdown') => {
    if (!selectedId) return;

    try {
      const response = await fetch(`${backendUrl}/api/gauntlet/${selectedId}/receipt?format=${format}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const blob = await response.blob();

      const ext = format === 'markdown' ? 'md' : format;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `receipt-${selectedId}.${ext}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed');
    }
  };

  const getVerdictColor = (verdict?: string) => {
    switch (verdict) {
      case 'PASS': return 'text-acid-green bg-acid-green/20 border-acid-green/30';
      case 'CONDITIONAL': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30';
      case 'FAIL': return 'text-red-400 bg-red-500/20 border-red-500/30';
      default: return 'text-text-muted bg-surface border-border';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical': return 'text-red-500 bg-red-500/20';
      case 'high': return 'text-orange-400 bg-orange-500/20';
      case 'medium': return 'text-yellow-400 bg-yellow-500/20';
      case 'low': return 'text-blue-400 bg-blue-500/20';
      default: return 'text-text-muted bg-surface';
    }
  };

  const filteredResults = filter === 'all'
    ? results
    : results.filter(r => r.verdict === filter);

  const renderResultsList = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-mono font-bold text-acid-green">Decision Receipts</h2>
        <div className="flex gap-2">
          {(['all', 'PASS', 'CONDITIONAL', 'FAIL'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1 text-xs font-mono rounded border transition-colors ${
                filter === f
                  ? 'bg-acid-green/20 border-acid-green text-acid-green'
                  : 'border-border text-text-muted hover:border-acid-green/50'
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {filteredResults.length === 0 ? (
        <div className="p-8 bg-surface border border-border rounded-lg text-center">
          <p className="text-text-muted font-mono">No completed gauntlet runs found</p>
          <Link href="/gauntlet" className="mt-4 inline-block text-acid-green hover:underline">
            Run a gauntlet test &rarr;
          </Link>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredResults.map((result) => (
            <button
              key={result.id}
              onClick={() => result.status === 'completed' && fetchReceipt(result.id)}
              disabled={result.status !== 'completed'}
              className={`w-full p-4 bg-surface border border-border rounded-lg text-left transition-all ${
                result.status === 'completed'
                  ? 'hover:border-acid-green/50 cursor-pointer'
                  : 'opacity-50 cursor-not-allowed'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-sm text-text-muted">
                    {result.id.substring(0, 12)}...
                  </span>
                  {result.verdict && (
                    <span className={`px-2 py-0.5 text-xs font-mono rounded border ${getVerdictColor(result.verdict)}`}>
                      {result.verdict}
                    </span>
                  )}
                </div>
                <span className="text-xs text-text-muted">
                  {new Date(result.created_at).toLocaleDateString()}
                </span>
              </div>

              {result.input_summary && (
                <p className="text-sm text-text mb-2 line-clamp-1">{result.input_summary}</p>
              )}

              {result.risk_summary && (
                <div className="flex gap-3 text-xs font-mono">
                  {result.risk_summary.critical > 0 && (
                    <span className="text-red-400">C:{result.risk_summary.critical}</span>
                  )}
                  {result.risk_summary.high > 0 && (
                    <span className="text-orange-400">H:{result.risk_summary.high}</span>
                  )}
                  {result.risk_summary.medium > 0 && (
                    <span className="text-yellow-400">M:{result.risk_summary.medium}</span>
                  )}
                  {result.risk_summary.low > 0 && (
                    <span className="text-blue-400">L:{result.risk_summary.low}</span>
                  )}
                </div>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );

  const renderReceiptDetail = () => {
    if (receiptLoading) {
      return (
        <div className="flex items-center justify-center py-12">
          <div className="text-acid-green font-mono animate-pulse">Loading receipt...</div>
        </div>
      );
    }

    if (!selectedReceipt) {
      return <p className="text-text-muted">No receipt selected</p>;
    }

    const r = selectedReceipt;

    return (
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-mono font-bold text-acid-green">Decision Receipt</h2>
            <div className="text-xs text-text-muted font-mono mt-1">
              ID: {r.receipt_id} | Artifact: {r.artifact_hash?.substring(0, 16)}...
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => { setActiveTab('list'); setSelectedReceipt(null); }}
              className="px-3 py-1 text-sm font-mono border border-border rounded hover:border-acid-green/50"
            >
              Back
            </button>
            <button
              onClick={() => setDeliveryModalOpen(true)}
              className="px-3 py-1 text-sm font-mono bg-blue-500/20 border border-blue-500 text-blue-400 rounded hover:bg-blue-500/30"
            >
              Deliver
            </button>
            <div className="relative group">
              <button className="px-3 py-1 text-sm font-mono bg-acid-green/20 border border-acid-green text-acid-green rounded">
                Export
              </button>
              <div className="absolute right-0 mt-1 w-32 bg-surface border border-border rounded shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
                <button onClick={() => downloadReceipt('json')} className="w-full px-3 py-2 text-left text-sm hover:bg-bg">JSON</button>
                <button onClick={() => downloadReceipt('html')} className="w-full px-3 py-2 text-left text-sm hover:bg-bg">HTML</button>
                <button onClick={() => downloadReceipt('markdown')} className="w-full px-3 py-2 text-left text-sm hover:bg-bg">Markdown</button>
              </div>
            </div>
          </div>
        </div>

        {/* Verdict Card */}
        <div className={`p-4 rounded-lg border-2 ${getVerdictColor(r.verdict)}`}>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl font-mono font-bold">{r.verdict}</div>
              <div className="text-sm opacity-80">Confidence: {(r.confidence * 100).toFixed(1)}%</div>
            </div>
            <div className="text-right">
              <div className="text-sm">Robustness Score</div>
              <div className="text-2xl font-mono font-bold">{(r.robustness_score * 100).toFixed(0)}%</div>
            </div>
          </div>
          {r.verdict_reasoning && (
            <p className="mt-3 text-sm opacity-90">{r.verdict_reasoning}</p>
          )}
        </div>

        {/* Risk Summary */}
        <div className="p-4 bg-surface border border-border rounded-lg">
          <h3 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">Risk Summary</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-3xl font-mono font-bold text-red-500">{r.risk_summary.critical}</div>
              <div className="text-xs text-text-muted">Critical</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-mono font-bold text-orange-400">{r.risk_summary.high}</div>
              <div className="text-xs text-text-muted">High</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-mono font-bold text-yellow-400">{r.risk_summary.medium}</div>
              <div className="text-xs text-text-muted">Medium</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-mono font-bold text-blue-400">{r.risk_summary.low}</div>
              <div className="text-xs text-text-muted">Low</div>
            </div>
          </div>
          <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-4 text-center border-t border-border pt-4">
            <div>
              <div className="text-xl font-mono">{r.attacks_attempted}</div>
              <div className="text-xs text-text-muted">Attacks Attempted</div>
            </div>
            <div>
              <div className="text-xl font-mono">{r.attacks_successful}</div>
              <div className="text-xs text-text-muted">Successful</div>
            </div>
            <div>
              <div className="text-xl font-mono">{r.probes_run}</div>
              <div className="text-xs text-text-muted">Probes Run</div>
            </div>
          </div>
        </div>

        {/* Consensus Proof */}
        {r.consensus_proof && (
          <div className="p-4 bg-surface border border-border rounded-lg">
            <h3 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">Consensus Proof</h3>
            <div className="flex items-center gap-4 mb-3">
              <span className={`px-2 py-1 text-xs font-mono rounded ${r.consensus_proof.reached ? 'bg-acid-green/20 text-acid-green' : 'bg-red-500/20 text-red-400'}`}>
                {r.consensus_proof.reached ? 'Consensus Reached' : 'No Consensus'}
              </span>
              <span className="text-sm text-text-muted">
                Method: {r.consensus_proof.method} | Confidence: {(r.consensus_proof.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-text-muted mb-1">Supporting Agents</div>
                <div className="flex flex-wrap gap-1">
                  {r.consensus_proof.supporting_agents.map((agent, i) => (
                    <span key={i} className="px-2 py-0.5 text-xs font-mono bg-acid-green/10 text-acid-green rounded">
                      {agent}
                    </span>
                  ))}
                </div>
              </div>
              {r.consensus_proof.dissenting_agents.length > 0 && (
                <div>
                  <div className="text-xs text-text-muted mb-1">Dissenting Agents</div>
                  <div className="flex flex-wrap gap-1">
                    {r.consensus_proof.dissenting_agents.map((agent, i) => (
                      <span key={i} className="px-2 py-0.5 text-xs font-mono bg-red-500/10 text-red-400 rounded">
                        {agent}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Vulnerability Details */}
        {r.vulnerability_details && r.vulnerability_details.length > 0 && (
          <div className="p-4 bg-surface border border-border rounded-lg">
            <h3 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
              Vulnerabilities ({r.vulnerability_details.length})
            </h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {r.vulnerability_details.map((vuln, i) => (
                <div key={i} className="p-2 bg-bg rounded">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`px-1.5 py-0.5 text-xs font-mono rounded ${getSeverityColor(vuln.severity)}`}>
                      {vuln.severity.toUpperCase()}
                    </span>
                    <span className="text-xs text-text-muted">{vuln.category}</span>
                    <span className="text-xs text-text-muted font-mono">{vuln.id}</span>
                  </div>
                  <p className="text-sm text-text">{vuln.description}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Provenance Chain */}
        {r.provenance_chain && r.provenance_chain.length > 0 && (
          <div className="p-4 bg-surface border border-border rounded-lg">
            <h3 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">Provenance Chain</h3>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {r.provenance_chain.map((record, i) => (
                <div key={i} className="flex items-start gap-3 text-xs">
                  <div className="w-16 text-text-muted shrink-0">
                    {new Date(record.timestamp).toLocaleTimeString()}
                  </div>
                  <div className="px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded font-mono shrink-0">
                    {record.event_type}
                  </div>
                  {record.agent && (
                    <div className="text-acid-green shrink-0">{record.agent}</div>
                  )}
                  <div className="text-text flex-1">{record.description}</div>
                  {record.evidence_hash && (
                    <div className="text-text-muted font-mono shrink-0" title={record.evidence_hash}>
                      #{record.evidence_hash.substring(0, 8)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Input & Integrity */}
        <div className="p-4 bg-surface border border-border rounded-lg">
          <h3 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">Input & Integrity</h3>
          <div className="space-y-2 text-sm">
            <div>
              <span className="text-text-muted">Input Summary: </span>
              <span className="text-text">{r.input_summary}</span>
            </div>
            <div className="font-mono text-xs">
              <span className="text-text-muted">Input Hash: </span>
              <span className="text-text">{r.input_hash}</span>
            </div>
            <div className="font-mono text-xs">
              <span className="text-text-muted">Artifact Hash: </span>
              <span className="text-text">{r.artifact_hash}</span>
            </div>
            <div className="font-mono text-xs">
              <span className="text-text-muted">Timestamp: </span>
              <span className="text-text">{r.timestamp}</span>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-bg text-text relative overflow-hidden">
      <Scanlines />
      <CRTVignette />

      <div className="max-w-6xl mx-auto px-4 py-8 relative z-10">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <Link href="/" className="hover:opacity-80 transition-opacity">
            <AsciiBannerCompact />
          </Link>
          <div className="flex items-center gap-4">
            <ThemeToggle />
            <BackendSelector />
          </div>
        </div>

        {/* Title */}
        <div className="mb-8">
          <h1 className="text-3xl font-mono font-bold text-acid-green mb-2">Decision Receipts</h1>
          <p className="text-text-muted font-mono text-sm">
            Audit-ready compliance documentation from gauntlet validations
          </p>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6">
            <ErrorWithRetry error={error} onRetry={loadData} />
          </div>
        )}

        {/* Content */}
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-acid-green font-mono animate-pulse">Loading...</div>
          </div>
        ) : (
          <div>
            {activeTab === 'list' && renderResultsList()}
            {activeTab === 'detail' && renderReceiptDetail()}
          </div>
        )}
      </div>

      {/* Delivery Modal */}
      {selectedReceipt && (
        <DeliveryModal
          isOpen={deliveryModalOpen}
          onClose={() => setDeliveryModalOpen(false)}
          receiptId={selectedReceipt.receipt_id}
          receiptSummary={selectedReceipt.input_summary}
          apiUrl={backendUrl}
          onDeliverySuccess={() => {
            // Could refresh history or show notification
          }}
        />
      )}
    </div>
  );
}
