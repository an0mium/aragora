'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { AsciiBannerCompact } from '@/components/AsciiBanner';
import { ThemeToggle } from '@/components/ThemeToggle';
import { BackendSelector, useBackend } from '@/components/BackendSelector';
import { PanelErrorBoundary } from '@/components/PanelErrorBoundary';
import { useSWRFetch } from '@/hooks/useSWRFetch';

interface FlowNode {
  id: string;
  type: 'debate' | 'knowledge_mound' | 'adapter';
  label: string;
  count: number;
}

interface FlowEdge {
  from: string;
  to: string;
  count: number;
  label: string;
}

interface FlowData {
  nodes: FlowNode[];
  edges: FlowEdge[];
  total_debates: number;
  total_knowledge_items: number;
  total_injections: number;
}

interface ConfidencePoint {
  timestamp: string;
  topic: string;
  confidence: number;
  source: string;
}

interface ConfidenceHistory {
  points: ConfidencePoint[];
  count: number;
}

interface AdapterHealth {
  adapter_name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  items_stored: number;
  last_sync: string | null;
  error_count: number;
}

interface AdapterHealthResponse {
  adapters: AdapterHealth[];
}

export default function KnowledgeFlowPage() {
  const { config } = useBackend();
  const [activeTab, setActiveTab] = useState<'flow' | 'confidence' | 'adapters'>('flow');

  const { data: flowData, isLoading: flowLoading } = useSWRFetch<{ data: FlowData }>(
    '/api/v1/knowledge/flow',
    { refreshInterval: 30000, baseUrl: config.api }
  );

  const { data: confidenceData, isLoading: confLoading } = useSWRFetch<{ data: ConfidenceHistory }>(
    activeTab === 'confidence' ? '/api/v1/knowledge/flow/confidence-history' : null,
    { refreshInterval: 30000, baseUrl: config.api }
  );

  const { data: adapterData, isLoading: adapterLoading } = useSWRFetch<{ data: AdapterHealthResponse }>(
    activeTab === 'adapters' ? '/api/v1/knowledge/adapters/health' : null,
    { refreshInterval: 15000, baseUrl: config.api }
  );

  const flow = flowData?.data;
  const confidence = confidenceData?.data;
  const adapters = adapterData?.data?.adapters || [];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-acid-green bg-acid-green/20';
      case 'degraded': return 'text-yellow-400 bg-yellow-500/20';
      case 'unhealthy': return 'text-red-400 bg-red-500/20';
      default: return 'text-text-muted bg-surface';
    }
  };

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        <header className="border-b border-acid-green/30 bg-surface/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <Link href="/">
              <AsciiBannerCompact connected={true} />
            </Link>
            <div className="flex items-center gap-3">
              <Link href="/knowledge" className="text-xs font-mono text-text-muted hover:text-acid-green transition-colors">
                [KNOWLEDGE]
              </Link>
              <BackendSelector compact />
              <ThemeToggle />
            </div>
          </div>
        </header>

        <div className="container mx-auto px-4 py-6">
          <div className="mb-6">
            <h1 className="text-2xl font-mono text-acid-green mb-2">
              {'>'} KNOWLEDGE FLYWHEEL
            </h1>
            <p className="text-text-muted font-mono text-sm">
              Visualize the Debate &rarr; Knowledge Mound &rarr; Debate learning loop.
              Track confidence changes, adapter health, and knowledge flow across the system.
            </p>
          </div>

          {/* Tabs */}
          <div className="flex gap-2 mb-6">
            {([
              { key: 'flow', label: 'Flow Visualization' },
              { key: 'confidence', label: 'Confidence History' },
              { key: 'adapters', label: 'Adapter Health' },
            ] as const).map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className={`px-4 py-2 text-sm font-mono rounded border transition-colors ${
                  activeTab === key
                    ? 'bg-acid-green/20 border-acid-green text-acid-green'
                    : 'border-border text-text-muted hover:border-acid-green/50'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <PanelErrorBoundary panelName="Knowledge Flow">
            {activeTab === 'flow' && (
              <div>
                {flowLoading ? (
                  <div className="text-acid-green font-mono animate-pulse text-center py-12">Loading flow data...</div>
                ) : !flow ? (
                  <div className="p-8 bg-surface border border-border rounded-lg text-center">
                    <p className="text-text-muted font-mono">No flow data available. Run debates with Knowledge Mound enabled.</p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {/* Summary Stats */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="p-4 bg-surface border border-border rounded-lg text-center">
                        <div className="text-3xl font-mono font-bold text-acid-green">{flow.total_debates}</div>
                        <div className="text-xs text-text-muted uppercase">Total Debates</div>
                      </div>
                      <div className="p-4 bg-surface border border-border rounded-lg text-center">
                        <div className="text-3xl font-mono font-bold text-blue-400">{flow.total_knowledge_items}</div>
                        <div className="text-xs text-text-muted uppercase">Knowledge Items</div>
                      </div>
                      <div className="p-4 bg-surface border border-border rounded-lg text-center">
                        <div className="text-3xl font-mono font-bold text-purple-400">{flow.total_injections}</div>
                        <div className="text-xs text-text-muted uppercase">Cross-Debate Injections</div>
                      </div>
                    </div>

                    {/* Flow Nodes */}
                    <div className="p-4 bg-surface border border-border rounded-lg">
                      <h3 className="text-sm font-mono font-bold text-text-muted uppercase mb-4">Knowledge Flow Graph</h3>
                      <div className="flex flex-wrap items-center justify-center gap-4">
                        {flow.nodes.map((node) => (
                          <div
                            key={node.id}
                            className={`p-3 rounded-lg border text-center min-w-[120px] ${
                              node.type === 'debate' ? 'border-acid-green bg-acid-green/10' :
                              node.type === 'knowledge_mound' ? 'border-blue-400 bg-blue-400/10' :
                              'border-purple-400 bg-purple-400/10'
                            }`}
                          >
                            <div className="text-lg font-mono font-bold">{node.count}</div>
                            <div className="text-xs text-text-muted">{node.label}</div>
                          </div>
                        ))}
                      </div>
                      {flow.edges.length > 0 && (
                        <div className="mt-4 space-y-1">
                          {flow.edges.map((edge, i) => (
                            <div key={i} className="flex items-center gap-2 text-xs font-mono text-text-muted">
                              <span>{edge.from}</span>
                              <span className="text-acid-green">&rarr;</span>
                              <span>{edge.to}</span>
                              <span className="text-acid-green ml-2">({edge.count} items)</span>
                              {edge.label && <span className="text-text-muted">- {edge.label}</span>}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'confidence' && (
              <div>
                {confLoading ? (
                  <div className="text-acid-green font-mono animate-pulse text-center py-12">Loading...</div>
                ) : !confidence || confidence.points.length === 0 ? (
                  <div className="p-8 bg-surface border border-border rounded-lg text-center">
                    <p className="text-text-muted font-mono">No confidence history available yet.</p>
                  </div>
                ) : (
                  <div className="p-4 bg-surface border border-border rounded-lg">
                    <h3 className="text-sm font-mono font-bold text-text-muted uppercase mb-3">
                      Confidence Changes Over Time ({confidence.count} records)
                    </h3>
                    <div className="space-y-2 max-h-[500px] overflow-y-auto">
                      {confidence.points.map((pt, i) => (
                        <div key={i} className="flex items-center gap-3 text-sm p-2 bg-bg rounded">
                          <span className="text-text-muted text-xs w-32 shrink-0">
                            {new Date(pt.timestamp).toLocaleString()}
                          </span>
                          <span className="text-text flex-1">{pt.topic}</span>
                          <span className="text-xs text-text-muted">{pt.source}</span>
                          <span className={`font-mono text-sm ${pt.confidence >= 0.7 ? 'text-acid-green' : pt.confidence >= 0.4 ? 'text-yellow-400' : 'text-red-400'}`}>
                            {(pt.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'adapters' && (
              <div>
                {adapterLoading ? (
                  <div className="text-acid-green font-mono animate-pulse text-center py-12">Loading...</div>
                ) : adapters.length === 0 ? (
                  <div className="p-8 bg-surface border border-border rounded-lg text-center">
                    <p className="text-text-muted font-mono">No adapter health data available.</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {adapters.map((adapter) => (
                      <div key={adapter.adapter_name} className="p-4 bg-surface border border-border rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-mono text-sm text-text">{adapter.adapter_name}</span>
                          <span className={`px-2 py-0.5 text-xs font-mono rounded ${getStatusColor(adapter.status)}`}>
                            {adapter.status.toUpperCase()}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <span className="text-text-muted">Items: </span>
                            <span className="text-text">{adapter.items_stored}</span>
                          </div>
                          <div>
                            <span className="text-text-muted">Errors: </span>
                            <span className={adapter.error_count > 0 ? 'text-red-400' : 'text-text'}>
                              {adapter.error_count}
                            </span>
                          </div>
                          {adapter.last_sync && (
                            <div className="col-span-2">
                              <span className="text-text-muted">Last sync: </span>
                              <span className="text-text">{new Date(adapter.last_sync).toLocaleString()}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </PanelErrorBoundary>
        </div>

        <footer className="text-center text-xs font-mono py-8 border-t border-acid-green/20 mt-8">
          <div className="text-acid-green/50 mb-2">{'='.repeat(40)}</div>
          <p className="text-text-muted">{'>'} ARAGORA // KNOWLEDGE FLYWHEEL</p>
        </footer>
      </main>
    </>
  );
}
