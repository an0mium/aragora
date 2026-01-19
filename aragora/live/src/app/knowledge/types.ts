/**
 * Knowledge Mound Types and Constants
 *
 * Shared types, interfaces, constants, and mock data for the Knowledge page.
 */

// =============================================================================
// Interfaces
// =============================================================================

export interface KnowledgeNode {
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

export interface KnowledgeRelationship {
  id: string;
  sourceId: string;
  targetId: string;
  relationshipType: string;
  strength: number;
  createdAt: string;
}

export interface Contradiction {
  fact_id: string;
  content: string;
  contradiction_type: string;
  severity: string;
  explanation: string;
}

export interface VerificationResult {
  status: string;
  confidence: number;
  verified_by: string[];
  verification_notes: string;
}

export interface KnowledgeStats {
  totalNodes: number;
  nodesByType: Record<string, number>;
  nodesByTier: Record<string, number>;
  nodesBySource: Record<string, number>;
  totalRelationships: number;
}

export interface StaleItem {
  node_id: string;
  staleness_score: number;
  reasons: string[];
  last_validated_at: string | null;
  recommended_action: string;
}

// =============================================================================
// Constants
// =============================================================================

export const SOURCE_COLORS: Record<string, { bg: string; text: string }> = {
  continuum: { bg: 'bg-blue-900/30', text: 'text-blue-400' },
  consensus: { bg: 'bg-green-900/30', text: 'text-green-400' },
  fact: { bg: 'bg-yellow-900/30', text: 'text-yellow-400' },
  evidence: { bg: 'bg-purple-900/30', text: 'text-purple-400' },
  critique: { bg: 'bg-orange-900/30', text: 'text-orange-400' },
  document: { bg: 'bg-cyan-900/30', text: 'text-cyan-400' },
};

export const TIER_COLORS: Record<string, { bg: string; text: string }> = {
  fast: { bg: 'bg-red-900/30', text: 'text-red-400' },
  medium: { bg: 'bg-yellow-900/30', text: 'text-yellow-400' },
  slow: { bg: 'bg-blue-900/30', text: 'text-blue-400' },
  glacial: { bg: 'bg-purple-900/30', text: 'text-purple-400' },
};

export const NODE_TYPE_ICONS: Record<string, string> = {
  memory: '🧠',
  consensus: '🤝',
  fact: '📌',
  evidence: '📄',
  critique: '💬',
  claim: '💡',
  entity: '🏷️',
};

// =============================================================================
// Helper Functions
// =============================================================================

export function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) return 'text-green-400';
  if (confidence >= 0.5) return 'text-yellow-400';
  return 'text-red-400';
}

export function formatRelativeDate(dateStr: string): string {
  const date = new Date(dateStr);
  const diff = Date.now() - date.getTime();
  const hours = Math.floor(diff / (1000 * 60 * 60));
  if (hours < 1) return 'Just now';
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

// =============================================================================
// Mock Data
// =============================================================================

export function getMockNodes(): KnowledgeNode[] {
  return [
    {
      id: 'mock-1',
      nodeType: 'memory',
      content: 'Claude demonstrates strong performance in reasoning tasks',
      confidence: 0.92,
      tier: 'slow',
      sourceType: 'continuum',
      topics: ['claude', 'reasoning', 'performance'],
      createdAt: new Date(Date.now() - 3600000).toISOString(),
      updatedAt: new Date(Date.now() - 3600000).toISOString(),
    },
    {
      id: 'mock-2',
      nodeType: 'consensus',
      content: 'Multi-agent debates produce more nuanced answers than single-agent responses',
      confidence: 0.88,
      tier: 'slow',
      sourceType: 'consensus',
      debateId: 'debate-123',
      topics: ['multi-agent', 'debate', 'quality'],
      createdAt: new Date(Date.now() - 7200000).toISOString(),
      updatedAt: new Date(Date.now() - 3600000).toISOString(),
    },
    {
      id: 'mock-3',
      nodeType: 'fact',
      content: 'GPT-4 was trained using RLHF techniques',
      confidence: 0.99,
      tier: 'glacial',
      sourceType: 'fact',
      topics: ['gpt-4', 'training', 'rlhf'],
      createdAt: new Date(Date.now() - 86400000).toISOString(),
      updatedAt: new Date(Date.now() - 86400000).toISOString(),
    },
    {
      id: 'mock-4',
      nodeType: 'evidence',
      content: 'DeepSeek R1 shows improved mathematical reasoning capabilities',
      confidence: 0.85,
      tier: 'medium',
      sourceType: 'evidence',
      documentId: 'doc-456',
      topics: ['deepseek', 'r1', 'math', 'reasoning'],
      createdAt: new Date(Date.now() - 172800000).toISOString(),
      updatedAt: new Date(Date.now() - 86400000).toISOString(),
    },
    {
      id: 'mock-5',
      nodeType: 'critique',
      content: 'Current LLM benchmarks may not accurately reflect real-world task performance',
      confidence: 0.72,
      tier: 'medium',
      sourceType: 'critique',
      agentId: 'claude-3',
      topics: ['benchmarks', 'evaluation', 'llm'],
      createdAt: new Date(Date.now() - 259200000).toISOString(),
      updatedAt: new Date(Date.now() - 172800000).toISOString(),
    },
  ];
}

export function getMockStats(): KnowledgeStats {
  return {
    totalNodes: 1247,
    nodesByType: {
      memory: 523,
      consensus: 312,
      fact: 156,
      evidence: 189,
      critique: 67,
    },
    nodesByTier: {
      fast: 89,
      medium: 456,
      slow: 512,
      glacial: 190,
    },
    nodesBySource: {
      continuum: 523,
      consensus: 312,
      fact: 156,
      evidence: 189,
      critique: 67,
    },
    totalRelationships: 3456,
  };
}

export function getMockRelationships(nodeId: string): KnowledgeRelationship[] {
  return [
    {
      id: `rel-1-${nodeId}`,
      sourceId: nodeId,
      targetId: 'mock-2',
      relationshipType: 'supports',
      strength: 0.85,
      createdAt: new Date(Date.now() - 3600000).toISOString(),
    },
    {
      id: `rel-2-${nodeId}`,
      sourceId: 'mock-3',
      targetId: nodeId,
      relationshipType: 'references',
      strength: 0.72,
      createdAt: new Date(Date.now() - 7200000).toISOString(),
    },
  ];
}
