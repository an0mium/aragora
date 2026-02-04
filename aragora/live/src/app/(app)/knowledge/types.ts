/**
 * Knowledge Mound Types and Constants
 *
 * Shared types, interfaces, constants, and helpers for the Knowledge page.
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
