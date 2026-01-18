'use client';

import { useState, useEffect, useCallback, useMemo } from 'react';
import { API_BASE_URL } from '@/config';

interface KnowledgeNode {
  id: string;
  nodeType: string;
  content: string;
  confidence: number;
  tier: string;
  sourceType: string;
  documentId?: string;
  debateId?: string;
  agentId?: string;
  topics: string[];
  createdAt: string;
  updatedAt: string;
  metadata?: Record<string, unknown>;
}

interface KnowledgeRelationship {
  id: string;
  sourceId: string;
  targetId: string;
  relationshipType: string;
  strength: number;
  createdAt: string;
}

interface KnowledgeStats {
  totalNodes: number;
  nodesByType: Record<string, number>;
  nodesByTier: Record<string, number>;
  nodesBySource: Record<string, number>;
  totalRelationships: number;
}

const SOURCE_COLORS: Record<string, { bg: string; text: string }> = {
  continuum: { bg: 'bg-blue-900/30', text: 'text-blue-400' },
  consensus: { bg: 'bg-green-900/30', text: 'text-green-400' },
  fact: { bg: 'bg-yellow-900/30', text: 'text-yellow-400' },
  evidence: { bg: 'bg-purple-900/30', text: 'text-purple-400' },
  critique: { bg: 'bg-orange-900/30', text: 'text-orange-400' },
  document: { bg: 'bg-cyan-900/30', text: 'text-cyan-400' },
};

const TIER_COLORS: Record<string, { bg: string; text: string }> = {
  fast: { bg: 'bg-red-900/30', text: 'text-red-400' },
  medium: { bg: 'bg-yellow-900/30', text: 'text-yellow-400' },
  slow: { bg: 'bg-blue-900/30', text: 'text-blue-400' },
  glacial: { bg: 'bg-purple-900/30', text: 'text-purple-400' },
};

const NODE_TYPE_ICONS: Record<string, string> = {
  memory: '🧠',
  consensus: '🤝',
  fact: '📌',
  evidence: '📄',
  critique: '💬',
  claim: '💡',
  entity: '🏷️',
};

export default function KnowledgeMoundPage() {
  const [nodes, setNodes] = useState<KnowledgeNode[]>([]);
  const [stats, setStats] = useState<KnowledgeStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [sourceFilter, setSourceFilter] = useState<string>('all');
  const [tierFilter, setTierFilter] = useState<string>('all');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [selectedNode, setSelectedNode] = useState<KnowledgeNode | null>(null);
  const [relationships, setRelationships] = useState<KnowledgeRelationship[]>([]);

  const fetchNodes = useCallback(async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (searchQuery) params.set('query', searchQuery);
      if (sourceFilter !== 'all') params.set('source', sourceFilter);
      if (tierFilter !== 'all') params.set('tier', tierFilter);
      if (typeFilter !== 'all') params.set('type', typeFilter);
      params.set('limit', '50');

      const response = await fetch(`${API_BASE_URL}/api/knowledge/query?${params}`);
      if (!response.ok) {
        setNodes(getMockNodes());
        setStats(getMockStats());
        return;
      }
      const data = await response.json();
      setNodes(data.items || []);
    } catch {
      setNodes(getMockNodes());
      setStats(getMockStats());
    } finally {
      setLoading(false);
    }
  }, [searchQuery, sourceFilter, tierFilter, typeFilter]);

  const fetchStats = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/knowledge/stats`);
      if (!response.ok) {
        setStats(getMockStats());
        return;
      }
      const data = await response.json();
      setStats(data);
    } catch {
      setStats(getMockStats());
    }
  }, []);

  const fetchRelationships = useCallback(async (nodeId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/knowledge/nodes/${nodeId}/relationships`);
      if (!response.ok) {
        setRelationships(getMockRelationships(nodeId));
        return;
      }
      const data = await response.json();
      setRelationships(data.relationships || []);
    } catch {
      setRelationships(getMockRelationships(nodeId));
    }
  }, []);

  useEffect(() => {
    fetchNodes();
    fetchStats();
  }, [fetchNodes, fetchStats]);

  useEffect(() => {
    if (selectedNode) {
      fetchRelationships(selectedNode.id);
    } else {
      setRelationships([]);
    }
  }, [selectedNode, fetchRelationships]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    fetchNodes();
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.5) return 'text-yellow-400';
    return 'text-red-400';
  };

  const formatDate = (dateStr: string): string => {
    const date = new Date(dateStr);
    const diff = Date.now() - date.getTime();
    const hours = Math.floor(diff / (1000 * 60 * 60));
    if (hours < 1) return 'Just now';
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  };

  return (
    <main className="min-h-screen bg-bg p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-mono font-bold text-text mb-2">
            Knowledge Mound
          </h1>
          <p className="text-text-muted">
            Explore the unified knowledge superstructure across all memory systems
          </p>
        </div>

        {/* Stats Bar */}
        {stats && (
          <div className="grid grid-cols-6 gap-4 mb-6">
            <div className="p-4 bg-surface border border-border rounded-lg text-center">
              <div className="text-2xl font-mono text-acid-green">{stats.totalNodes}</div>
              <div className="text-xs text-text-muted">Total Nodes</div>
            </div>
            <div className="p-4 bg-surface border border-border rounded-lg text-center">
              <div className="text-2xl font-mono text-blue-400">{stats.nodesBySource['continuum'] || 0}</div>
              <div className="text-xs text-text-muted">Continuum</div>
            </div>
            <div className="p-4 bg-surface border border-border rounded-lg text-center">
              <div className="text-2xl font-mono text-green-400">{stats.nodesBySource['consensus'] || 0}</div>
              <div className="text-xs text-text-muted">Consensus</div>
            </div>
            <div className="p-4 bg-surface border border-border rounded-lg text-center">
              <div className="text-2xl font-mono text-yellow-400">{stats.nodesBySource['fact'] || 0}</div>
              <div className="text-xs text-text-muted">Facts</div>
            </div>
            <div className="p-4 bg-surface border border-border rounded-lg text-center">
              <div className="text-2xl font-mono text-purple-400">{stats.nodesBySource['evidence'] || 0}</div>
              <div className="text-xs text-text-muted">Evidence</div>
            </div>
            <div className="p-4 bg-surface border border-border rounded-lg text-center">
              <div className="text-2xl font-mono text-acid-cyan">{stats.totalRelationships}</div>
              <div className="text-xs text-text-muted">Relationships</div>
            </div>
          </div>
        )}

        {/* Search and Filters */}
        <form onSubmit={handleSearch} className="mb-6">
          <div className="flex gap-3 mb-4">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search knowledge semantically..."
              className="flex-1 px-4 py-3 bg-surface border border-border rounded-lg text-text focus:border-acid-green focus:outline-none font-mono"
            />
            <button
              type="submit"
              className="px-6 py-3 bg-acid-green text-bg font-mono font-bold rounded-lg hover:bg-acid-green/80 transition-colors"
            >
              Search
            </button>
          </div>

          <div className="flex flex-wrap gap-3">
            <select
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value)}
              className="px-3 py-2 bg-surface border border-border rounded text-text focus:border-acid-green focus:outline-none"
            >
              <option value="all">All Sources</option>
              <option value="continuum">Continuum Memory</option>
              <option value="consensus">Consensus</option>
              <option value="fact">Facts</option>
              <option value="evidence">Evidence</option>
              <option value="critique">Critique</option>
            </select>

            <select
              value={tierFilter}
              onChange={(e) => setTierFilter(e.target.value)}
              className="px-3 py-2 bg-surface border border-border rounded text-text focus:border-acid-green focus:outline-none"
            >
              <option value="all">All Tiers</option>
              <option value="fast">Fast (1h)</option>
              <option value="medium">Medium (24h)</option>
              <option value="slow">Slow (7d)</option>
              <option value="glacial">Glacial (30d)</option>
            </select>

            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
              className="px-3 py-2 bg-surface border border-border rounded text-text focus:border-acid-green focus:outline-none"
            >
              <option value="all">All Types</option>
              <option value="memory">Memory</option>
              <option value="consensus">Consensus</option>
              <option value="fact">Fact</option>
              <option value="evidence">Evidence</option>
              <option value="critique">Critique</option>
              <option value="claim">Claim</option>
            </select>
          </div>
        </form>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Nodes List */}
          <div>
            <h2 className="text-sm font-mono text-acid-green uppercase mb-4">
              Knowledge Nodes ({nodes.length})
            </h2>

            {loading ? (
              <div className="text-center py-8 text-text-muted font-mono">Loading...</div>
            ) : nodes.length === 0 ? (
              <div className="text-center py-8 text-text-muted">No knowledge nodes found</div>
            ) : (
              <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
                {nodes.map((node) => {
                  const sourceColors = SOURCE_COLORS[node.sourceType] || { bg: 'bg-gray-900/30', text: 'text-gray-400' };
                  const tierColors = TIER_COLORS[node.tier] || { bg: 'bg-gray-900/30', text: 'text-gray-400' };

                  return (
                    <div
                      key={node.id}
                      onClick={() => setSelectedNode(node)}
                      className={`p-4 bg-surface border rounded-lg cursor-pointer transition-all ${
                        selectedNode?.id === node.id
                          ? 'border-acid-green'
                          : 'border-border hover:border-text-muted'
                      }`}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <span className="text-lg">{NODE_TYPE_ICONS[node.nodeType] || '📦'}</span>
                          <span className="font-mono text-xs text-text-muted">{node.id}</span>
                        </div>
                        <div className="flex gap-2">
                          <span className={`px-2 py-0.5 text-xs font-mono rounded ${sourceColors.bg} ${sourceColors.text}`}>
                            {node.sourceType}
                          </span>
                          <span className={`px-2 py-0.5 text-xs font-mono rounded ${tierColors.bg} ${tierColors.text}`}>
                            {node.tier}
                          </span>
                        </div>
                      </div>

                      <p className="text-sm text-text line-clamp-2 mb-2">{node.content}</p>

                      <div className="flex items-center justify-between text-xs text-text-muted">
                        <span className={getConfidenceColor(node.confidence)}>
                          {Math.round(node.confidence * 100)}% confidence
                        </span>
                        <span>{formatDate(node.createdAt)}</span>
                      </div>

                      {node.topics && node.topics.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-2">
                          {node.topics.slice(0, 3).map((topic) => (
                            <span
                              key={topic}
                              className="px-2 py-0.5 text-xs bg-bg text-text-muted rounded"
                            >
                              {topic}
                            </span>
                          ))}
                          {node.topics.length > 3 && (
                            <span className="text-xs text-text-muted">+{node.topics.length - 3}</span>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Node Details / Provenance */}
          <div>
            <h2 className="text-sm font-mono text-acid-green uppercase mb-4">
              Node Details & Provenance
            </h2>

            {selectedNode ? (
              <div className="bg-surface border border-border rounded-lg p-4">
                {/* Node Header */}
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-2xl">{NODE_TYPE_ICONS[selectedNode.nodeType] || '📦'}</span>
                  <div>
                    <div className="font-mono font-bold text-text">{selectedNode.nodeType}</div>
                    <div className="text-xs text-text-muted">{selectedNode.id}</div>
                  </div>
                </div>

                {/* Content */}
                <div className="p-3 bg-bg border border-border rounded-lg mb-4">
                  <p className="text-sm text-text whitespace-pre-wrap">{selectedNode.content}</p>
                </div>

                {/* Metadata Grid */}
                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="p-2 bg-bg rounded">
                    <div className="text-xs text-text-muted">Confidence</div>
                    <div className={`font-mono font-bold ${getConfidenceColor(selectedNode.confidence)}`}>
                      {Math.round(selectedNode.confidence * 100)}%
                    </div>
                  </div>
                  <div className="p-2 bg-bg rounded">
                    <div className="text-xs text-text-muted">Tier</div>
                    <div className="font-mono font-bold text-text">{selectedNode.tier}</div>
                  </div>
                  <div className="p-2 bg-bg rounded">
                    <div className="text-xs text-text-muted">Source</div>
                    <div className="font-mono font-bold text-text">{selectedNode.sourceType}</div>
                  </div>
                  <div className="p-2 bg-bg rounded">
                    <div className="text-xs text-text-muted">Created</div>
                    <div className="font-mono text-text text-sm">{formatDate(selectedNode.createdAt)}</div>
                  </div>
                </div>

                {/* Provenance */}
                <div className="border-t border-border pt-4">
                  <h3 className="text-xs font-mono text-acid-green uppercase mb-3">Provenance</h3>
                  <div className="space-y-2 text-sm">
                    {selectedNode.debateId && (
                      <div className="flex items-center gap-2">
                        <span className="text-text-muted">Debate:</span>
                        <span className="font-mono text-acid-cyan">{selectedNode.debateId}</span>
                      </div>
                    )}
                    {selectedNode.documentId && (
                      <div className="flex items-center gap-2">
                        <span className="text-text-muted">Document:</span>
                        <span className="font-mono text-acid-cyan">{selectedNode.documentId}</span>
                      </div>
                    )}
                    {selectedNode.agentId && (
                      <div className="flex items-center gap-2">
                        <span className="text-text-muted">Agent:</span>
                        <span className="font-mono text-acid-cyan">{selectedNode.agentId}</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Relationships */}
                {relationships.length > 0 && (
                  <div className="border-t border-border pt-4 mt-4">
                    <h3 className="text-xs font-mono text-acid-green uppercase mb-3">
                      Relationships ({relationships.length})
                    </h3>
                    <div className="space-y-2">
                      {relationships.map((rel) => (
                        <div
                          key={rel.id}
                          className="p-2 bg-bg rounded-lg flex items-center justify-between"
                        >
                          <div>
                            <span className="text-xs text-text-muted">{rel.relationshipType}</span>
                            <div className="font-mono text-xs text-acid-cyan">
                              {rel.sourceId === selectedNode.id ? rel.targetId : rel.sourceId}
                            </div>
                          </div>
                          <div className="text-xs text-text-muted">
                            {Math.round(rel.strength * 100)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Additional Metadata */}
                {selectedNode.metadata && Object.keys(selectedNode.metadata).length > 0 && (
                  <div className="border-t border-border pt-4 mt-4">
                    <h3 className="text-xs font-mono text-acid-green uppercase mb-3">Metadata</h3>
                    <pre className="p-2 bg-bg rounded text-xs text-text-muted overflow-x-auto">
                      {JSON.stringify(selectedNode.metadata, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-surface border border-border rounded-lg p-8 text-center text-text-muted">
                Select a node to view details and provenance
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

// Mock data for development
function getMockNodes(): KnowledgeNode[] {
  return [
    {
      id: 'kn_001',
      nodeType: 'consensus',
      content: 'Rate limiting should be implemented at the API gateway level with a sliding window algorithm for better accuracy.',
      confidence: 0.92,
      tier: 'slow',
      sourceType: 'consensus',
      debateId: 'debate_001',
      topics: ['api-design', 'security', 'performance'],
      createdAt: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
      updatedAt: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString(),
    },
    {
      id: 'kn_002',
      nodeType: 'memory',
      content: 'User authentication flow uses JWT tokens with 1-hour expiry and refresh token rotation.',
      confidence: 0.85,
      tier: 'medium',
      sourceType: 'continuum',
      topics: ['authentication', 'security'],
      createdAt: new Date(Date.now() - 1000 * 60 * 60 * 5).toISOString(),
      updatedAt: new Date(Date.now() - 1000 * 60 * 60 * 5).toISOString(),
    },
    {
      id: 'kn_003',
      nodeType: 'fact',
      content: 'GDPR Article 17 requires data controllers to erase personal data without undue delay when requested.',
      confidence: 0.98,
      tier: 'glacial',
      sourceType: 'fact',
      documentId: 'gdpr_2016',
      topics: ['gdpr', 'compliance', 'data-protection'],
      createdAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 30).toISOString(),
      updatedAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 30).toISOString(),
    },
    {
      id: 'kn_004',
      nodeType: 'evidence',
      content: 'Performance test results show 99th percentile latency of 45ms under 1000 RPS load.',
      confidence: 0.75,
      tier: 'fast',
      sourceType: 'evidence',
      agentId: 'performance_agent',
      topics: ['performance', 'testing'],
      createdAt: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
      updatedAt: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
    },
    {
      id: 'kn_005',
      nodeType: 'critique',
      content: 'Suggesting alternative approach: use circuit breaker pattern instead of simple retry for external API calls.',
      confidence: 0.68,
      tier: 'medium',
      sourceType: 'critique',
      debateId: 'debate_002',
      agentId: 'claude',
      topics: ['resilience', 'patterns'],
      createdAt: new Date(Date.now() - 1000 * 60 * 60 * 12).toISOString(),
      updatedAt: new Date(Date.now() - 1000 * 60 * 60 * 12).toISOString(),
    },
    {
      id: 'kn_006',
      nodeType: 'consensus',
      content: 'Database connection pooling should use minimum 10, maximum 50 connections with 30-second idle timeout.',
      confidence: 0.88,
      tier: 'slow',
      sourceType: 'consensus',
      debateId: 'debate_003',
      topics: ['database', 'performance', 'configuration'],
      createdAt: new Date(Date.now() - 1000 * 60 * 60 * 48).toISOString(),
      updatedAt: new Date(Date.now() - 1000 * 60 * 60 * 48).toISOString(),
    },
  ];
}

function getMockStats(): KnowledgeStats {
  return {
    totalNodes: 1247,
    nodesByType: {
      memory: 456,
      consensus: 234,
      fact: 312,
      evidence: 178,
      critique: 67,
    },
    nodesByTier: {
      fast: 123,
      medium: 456,
      slow: 523,
      glacial: 145,
    },
    nodesBySource: {
      continuum: 456,
      consensus: 234,
      fact: 312,
      evidence: 178,
      critique: 67,
    },
    totalRelationships: 3421,
  };
}

function getMockRelationships(nodeId: string): KnowledgeRelationship[] {
  return [
    {
      id: 'rel_001',
      sourceId: nodeId,
      targetId: 'kn_support_001',
      relationshipType: 'supports',
      strength: 0.85,
      createdAt: new Date().toISOString(),
    },
    {
      id: 'rel_002',
      sourceId: 'kn_derive_001',
      targetId: nodeId,
      relationshipType: 'derived_from',
      strength: 0.92,
      createdAt: new Date().toISOString(),
    },
    {
      id: 'rel_003',
      sourceId: nodeId,
      targetId: 'kn_elaborates_001',
      relationshipType: 'elaborates',
      strength: 0.78,
      createdAt: new Date().toISOString(),
    },
  ];
}
