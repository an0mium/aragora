'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { useRightSidebar } from '@/context/RightSidebarContext';
import { API_BASE_URL } from '@/config';
import { DecisionPackageView } from '@/components/debates/DecisionPackageView';
import { CostBreakdown } from '@/components/debates/CostBreakdown';
import { logger } from '@/utils/logger';

type Tab = 'overview' | 'arguments' | 'receipt' | 'export';

interface DecisionPackage {
  id: string;
  question: string;
  verdict: string;
  confidence: number;
  consensus_reached: boolean;
  explanation: string;
  final_answer: string;
  agents: string[];
  rounds: number;
  arguments: Array<{
    agent: string;
    round: number;
    position: string;
    content: string;
  }>;
  cost_breakdown: Array<{
    agent: string;
    tokens: number;
    cost: number;
  }>;
  total_cost: number;
  receipt: {
    hash: string;
    timestamp: string;
    signers: string[];
  } | null;
  next_steps: string[];
  created_at: string;
  duration_seconds: number;
}

export default function DebateDetailPage() {
  const params = useParams();
  const id = params.id as string;

  const [pkg, setPkg] = useState<DecisionPackage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>('overview');
  const [copied, setCopied] = useState(false);

  const { setContext, clearContext } = useRightSidebar();

  useEffect(() => {
    async function fetchPackage() {
      try {
        setLoading(true);
        setError(null);
        const res = await fetch(`${API_BASE_URL}/api/v1/debates/${id}/package`);
        if (res.status === 404) {
          setError('not_found');
          return;
        }
        if (!res.ok) {
          setError(`Failed to load debate (HTTP ${res.status})`);
          return;
        }
        const data = await res.json();
        setPkg(data);
      } catch (e) {
        logger.error('Failed to fetch debate package:', e);
        setError('Network error. Please try again.');
      } finally {
        setLoading(false);
      }
    }

    if (id) fetchPackage();
  }, [id]);

  // Right sidebar context
  useEffect(() => {
    if (!pkg) return;

    setContext({
      title: 'Decision Detail',
      subtitle: pkg.id.slice(0, 8),
      statsContent: (
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Confidence</span>
            <span className="text-sm font-mono text-[var(--acid-green)]">
              {Math.round(pkg.confidence * 100)}%
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Agents</span>
            <span className="text-sm font-mono text-[var(--acid-cyan)]">
              {pkg.agents.length}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Rounds</span>
            <span className="text-sm font-mono text-[var(--text)]">{pkg.rounds}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-xs text-[var(--text-muted)]">Duration</span>
            <span className="text-sm font-mono text-[var(--text)]">
              {pkg.duration_seconds ? `${Math.round(pkg.duration_seconds)}s` : '--'}
            </span>
          </div>
          {pkg.receipt && (
            <div className="flex justify-between items-center">
              <span className="text-xs text-[var(--text-muted)]">Receipt</span>
              <span className="text-xs font-mono text-[var(--acid-green)]">SIGNED</span>
            </div>
          )}
        </div>
      ),
      actionsContent: (
        <div className="space-y-2">
          <button
            onClick={handleShare}
            className="block w-full px-3 py-2 text-xs font-mono text-center bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30 hover:bg-[var(--acid-green)]/20 transition-colors"
          >
            SHARE LINK
          </button>
          <Link
            href="/debates"
            className="block w-full px-3 py-2 text-xs font-mono text-center bg-[var(--surface)] text-[var(--text-muted)] border border-[var(--border)] hover:border-[var(--acid-green)]/30 transition-colors"
          >
            BACK TO ARCHIVE
          </Link>
        </div>
      ),
    });

    return () => clearContext();
  }, [pkg, setContext, clearContext]);

  const handleShare = async () => {
    const url = `${window.location.origin}/debates/${id}`;
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      logger.error('Failed to copy link:', err);
    }
  };

  const tabs: { key: Tab; label: string }[] = [
    { key: 'overview', label: 'OVERVIEW' },
    { key: 'arguments', label: 'ARGUMENTS' },
    { key: 'receipt', label: 'RECEIPT' },
    { key: 'export', label: 'EXPORT' },
  ];

  // Loading state
  if (loading) {
    return (
      <>
        <Scanlines opacity={0.02} />
        <CRTVignette />
        <main className="min-h-screen bg-[var(--bg)] text-[var(--text)] relative z-10">
          <div className="flex items-center justify-center py-20">
            <div className="text-[var(--acid-green)] font-mono animate-pulse">
              {'>'} LOADING DECISION PACKAGE...
            </div>
          </div>
        </main>
      </>
    );
  }

  // Not found state
  if (error === 'not_found') {
    return (
      <>
        <Scanlines opacity={0.02} />
        <CRTVignette />
        <main className="min-h-screen bg-[var(--bg)] text-[var(--text)] relative z-10">
          <div className="container mx-auto px-4 py-20 text-center">
            <div className="text-[var(--warning)] font-mono text-lg mb-4">
              {'>'} DEBATE NOT FOUND
            </div>
            <p className="text-[var(--text-muted)] font-mono text-sm mb-6">
              The debate with ID {id} does not exist or has been removed.
            </p>
            <Link
              href="/debates"
              className="px-4 py-2 text-xs font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30 hover:bg-[var(--acid-green)]/20 transition-colors"
            >
              BACK TO ARCHIVE
            </Link>
          </div>
        </main>
      </>
    );
  }

  // Error state
  if (error) {
    return (
      <>
        <Scanlines opacity={0.02} />
        <CRTVignette />
        <main className="min-h-screen bg-[var(--bg)] text-[var(--text)] relative z-10">
          <div className="container mx-auto px-4 py-20 text-center">
            <div className="text-[var(--warning)] font-mono text-lg mb-4">
              {'>'} ERROR
            </div>
            <p className="text-[var(--text-muted)] font-mono text-sm mb-6">{error}</p>
            <Link
              href="/debates"
              className="px-4 py-2 text-xs font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30 hover:bg-[var(--acid-green)]/20 transition-colors"
            >
              BACK TO ARCHIVE
            </Link>
          </div>
        </main>
      </>
    );
  }

  if (!pkg) return null;

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-[var(--bg)] text-[var(--text)] relative z-10">
        <div className="container mx-auto px-4 py-6">
          {/* Breadcrumb */}
          <div className="mb-4 text-xs font-mono text-[var(--text-muted)]">
            <Link href="/debates" className="hover:text-[var(--acid-green)] transition-colors">
              Debates
            </Link>
            <span className="mx-2">/</span>
            <span className="text-[var(--acid-green)]">{pkg.id.slice(0, 8)}</span>
          </div>

          {/* Verdict Banner */}
          <div className="bg-[var(--surface)] border border-[var(--acid-green)]/40 p-6 mb-6">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1">
                <h1 className="text-lg font-mono text-[var(--acid-green)] mb-3">
                  {'>'} {pkg.question}
                </h1>
                <div className="flex items-center gap-3 flex-wrap">
                  <span
                    className={`px-2 py-1 text-xs font-mono border ${
                      pkg.consensus_reached
                        ? 'bg-[var(--acid-green)]/10 text-[var(--acid-green)] border-[var(--acid-green)]/40'
                        : 'bg-[var(--warning)]/10 text-[var(--warning)] border-[var(--warning)]/40'
                    }`}
                  >
                    {pkg.consensus_reached ? 'CONSENSUS' : 'NO CONSENSUS'}
                  </span>
                  <span className="px-2 py-1 text-xs font-mono bg-[var(--acid-cyan)]/10 text-[var(--acid-cyan)] border border-[var(--acid-cyan)]/40">
                    {Math.round(pkg.confidence * 100)}% CONFIDENCE
                  </span>
                  <span className="text-xs font-mono text-[var(--text-muted)]">
                    {new Date(pkg.created_at).toLocaleString()}
                  </span>
                </div>
              </div>
              <button
                onClick={handleShare}
                className="px-3 py-2 text-xs font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30 hover:bg-[var(--acid-green)]/20 transition-colors flex-shrink-0"
              >
                {copied ? 'COPIED!' : 'SHARE'}
              </button>
            </div>

            {/* Final answer */}
            {pkg.final_answer && (
              <div className="mt-4 p-3 bg-[var(--bg)] border border-[var(--border)]">
                <div className="text-xs font-mono text-[var(--text-muted)] mb-1">VERDICT</div>
                <p className="text-sm font-mono text-[var(--text)]">{pkg.final_answer}</p>
              </div>
            )}
          </div>

          {/* Tabs */}
          <div className="flex border-b border-[var(--border)] mb-6">
            {tabs.map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key)}
                className={`px-4 py-2 text-xs font-mono transition-colors border-b-2 -mb-px ${
                  activeTab === tab.key
                    ? 'text-[var(--acid-green)] border-[var(--acid-green)]'
                    : 'text-[var(--text-muted)] border-transparent hover:text-[var(--text)]'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          {activeTab === 'overview' && (
            <DecisionPackageView pkg={pkg} />
          )}

          {activeTab === 'arguments' && (
            <div className="space-y-3">
              {pkg.arguments && pkg.arguments.length > 0 ? (
                pkg.arguments.map((arg, i) => (
                  <div
                    key={i}
                    className="bg-[var(--surface)] border border-[var(--border)] p-4"
                  >
                    <div className="flex items-center gap-3 mb-2">
                      <span className="px-1.5 py-0.5 text-xs font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)]">
                        {arg.agent}
                      </span>
                      <span className="text-xs font-mono text-[var(--text-muted)]">
                        Round {arg.round}
                      </span>
                      {arg.position && (
                        <span className="text-xs font-mono text-[var(--acid-cyan)]">
                          {arg.position}
                        </span>
                      )}
                    </div>
                    <p className="text-sm font-mono text-[var(--text)] whitespace-pre-wrap">
                      {arg.content}
                    </p>
                  </div>
                ))
              ) : (
                <div className="text-center py-12 text-[var(--text-muted)] font-mono text-sm">
                  {'>'} No argument transcript available for this debate.
                </div>
              )}
            </div>
          )}

          {activeTab === 'receipt' && (
            <div className="bg-[var(--surface)] border border-[var(--border)] p-6">
              {pkg.receipt ? (
                <div className="space-y-4">
                  <div className="text-xs font-mono text-[var(--acid-green)] mb-4">
                    {'>'} CRYPTOGRAPHIC RECEIPT
                  </div>
                  <div className="space-y-3">
                    <div>
                      <div className="text-xs font-mono text-[var(--text-muted)] mb-1">
                        SHA-256 HASH
                      </div>
                      <div className="text-xs font-mono text-[var(--acid-cyan)] break-all bg-[var(--bg)] p-2 border border-[var(--border)]">
                        {pkg.receipt.hash}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs font-mono text-[var(--text-muted)] mb-1">
                        TIMESTAMP
                      </div>
                      <div className="text-sm font-mono text-[var(--text)]">
                        {new Date(pkg.receipt.timestamp).toLocaleString()}
                      </div>
                    </div>
                    {pkg.receipt.signers && pkg.receipt.signers.length > 0 && (
                      <div>
                        <div className="text-xs font-mono text-[var(--text-muted)] mb-1">
                          SIGNERS
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {pkg.receipt.signers.map((signer, i) => (
                            <span
                              key={i}
                              className="px-1.5 py-0.5 text-xs font-mono bg-[var(--acid-green)]/10 text-[var(--acid-green)] border border-[var(--acid-green)]/30"
                            >
                              {signer}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-[var(--text-muted)] font-mono text-sm">
                  {'>'} No receipt generated for this debate.
                </div>
              )}
            </div>
          )}

          {activeTab === 'export' && (
            <div className="space-y-4">
              <div className="bg-[var(--surface)] border border-[var(--border)] p-6">
                <div className="text-xs font-mono text-[var(--acid-green)] mb-4">
                  {'>'} EXPORT OPTIONS
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <button
                    onClick={() => {
                      const blob = new Blob([JSON.stringify(pkg, null, 2)], { type: 'application/json' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `debate-${pkg.id.slice(0, 8)}.json`;
                      a.click();
                      URL.revokeObjectURL(url);
                    }}
                    className="px-4 py-3 text-xs font-mono bg-[var(--surface)] text-[var(--acid-green)] border border-[var(--acid-green)]/30 hover:bg-[var(--acid-green)]/10 transition-colors text-left"
                  >
                    <div className="text-[var(--acid-green)]">JSON</div>
                    <div className="text-[var(--text-muted)] mt-1">
                      Full decision package with metadata
                    </div>
                  </button>
                  <button
                    onClick={handleShare}
                    className="px-4 py-3 text-xs font-mono bg-[var(--surface)] text-[var(--acid-cyan)] border border-[var(--acid-cyan)]/30 hover:bg-[var(--acid-cyan)]/10 transition-colors text-left"
                  >
                    <div className="text-[var(--acid-cyan)]">PERMALINK</div>
                    <div className="text-[var(--text-muted)] mt-1">
                      {copied ? 'Copied to clipboard!' : 'Copy shareable link'}
                    </div>
                  </button>
                </div>
              </div>

              <CostBreakdown
                costBreakdown={pkg.cost_breakdown}
                totalCost={pkg.total_cost}
              />
            </div>
          )}
        </div>
      </main>
    </>
  );
}
