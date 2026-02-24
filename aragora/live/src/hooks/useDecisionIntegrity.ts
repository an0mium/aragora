'use client';

import { useMemo } from 'react';
import { useSWRFetch } from './useSWRFetch';

// ============================================================================
// Types
// ============================================================================

export interface DebateListResponse {
  debates?: Array<{
    id: string;
    status?: string;
    consensus_reached?: boolean;
    confidence?: number;
    agents?: string[];
    task?: string;
    question?: string;
    created_at?: string;
  }>;
  results?: Array<{
    id: string;
    status?: string;
    consensus_reached?: boolean;
    confidence?: number;
    agents?: string[];
    task?: string;
    question?: string;
    created_at?: string;
  }>;
  total?: number;
  count?: number;
}

export interface ConsensusMetrics {
  total_topics?: number;
  high_confidence_count?: number;
  avg_confidence?: number;
  total_dissents?: number;
  domains?: string[];
  by_strength?: Record<string, number>;
  by_domain?: Record<string, number>;
}

export interface ComplianceStatus {
  status?: string;
  frameworks?: Array<{
    name: string;
    status: string;
    score?: number;
    last_assessed?: string;
  }>;
  overall_score?: number;
  violations_count?: number;
  findings?: Array<{
    id?: string;
    severity: string;
    description: string;
    framework?: string;
    detected_at?: string;
  }>;
}

export interface MemoryStats {
  total_entries?: number;
  memory_pressure?: number;
  tiers?: Record<string, { count?: number; size_bytes?: number }>;
  hit_rate?: number;
  eviction_count?: number;
}

export interface ReceiptStats {
  total_receipts?: number;
  delivered?: number;
  pending?: number;
  failed?: number;
  delivery_rate?: number;
  recent?: Array<{
    id: string;
    debate_id?: string;
    status: string;
    created_at?: string;
    delivered_at?: string;
    channel?: string;
  }>;
}

export interface AuditEvent {
  id?: string;
  event_type: string;
  actor?: string;
  resource?: string;
  action: string;
  timestamp: string;
  details?: string;
  severity?: string;
}

export interface AuditEventsResponse {
  events?: AuditEvent[];
  total?: number;
}

export interface AgentRanking {
  agent_id?: string;
  name: string;
  elo: number;
  wins?: number;
  losses?: number;
  debates_participated?: number;
  win_rate?: number;
  domains?: string[];
}

export interface LeaderboardResponse {
  agents?: AgentRanking[];
  rankings?: AgentRanking[];
  leaderboard?: AgentRanking[];
}

export interface ConsensusSettled {
  topics?: Array<{
    topic: string;
    confidence: number;
    strength?: string;
    domain?: string;
    settled_at?: string;
    debate_count?: number;
  }>;
}

// ============================================================================
// Derived metrics
// ============================================================================

export interface IntegrityMetrics {
  activeDebates: number;
  consensusHealth: number;
  complianceScore: number;
  memoryPressure: number;
  receiptDeliveryRate: number;
  systemIntegrity: number;
}

function computeIntegrityMetrics(
  debates: DebateListResponse | null,
  consensus: ConsensusMetrics | null,
  compliance: ComplianceStatus | null,
  memory: MemoryStats | null,
  receipts: ReceiptStats | null,
): IntegrityMetrics {
  const debateList = debates?.debates ?? debates?.results ?? [];
  const activeDebates = debateList.filter(
    (d) => d.status === 'active' || d.status === 'running',
  ).length;

  const consensusHealth = consensus?.avg_confidence
    ? Math.round(consensus.avg_confidence * 100)
    : 0;

  const complianceScore = compliance?.overall_score
    ? Math.round(compliance.overall_score * 100)
    : compliance?.frameworks
      ? Math.round(
          (compliance.frameworks.filter((f) => f.status === 'compliant').length /
            Math.max(compliance.frameworks.length, 1)) *
            100,
        )
      : 0;

  const memoryPressure = memory?.memory_pressure
    ? Math.round(memory.memory_pressure * 100)
    : 0;

  const receiptDeliveryRate = receipts?.delivery_rate
    ? Math.round(receipts.delivery_rate * 100)
    : receipts?.total_receipts && receipts.delivered
      ? Math.round((receipts.delivered / receipts.total_receipts) * 100)
      : 0;

  // System integrity: weighted average of consensus health, compliance,
  // inverse memory pressure, and receipt delivery rate
  const weights = { consensus: 0.3, compliance: 0.3, memory: 0.2, receipts: 0.2 };
  const components: number[] = [];
  if (consensusHealth > 0) components.push(consensusHealth * weights.consensus);
  else components.push(0);
  if (complianceScore > 0) components.push(complianceScore * weights.compliance);
  else components.push(0);
  components.push((100 - memoryPressure) * weights.memory);
  if (receiptDeliveryRate > 0) components.push(receiptDeliveryRate * weights.receipts);
  else components.push(0);

  const totalWeight =
    (consensusHealth > 0 ? weights.consensus : 0) +
    (complianceScore > 0 ? weights.compliance : 0) +
    weights.memory +
    (receiptDeliveryRate > 0 ? weights.receipts : 0);

  const systemIntegrity =
    totalWeight > 0
      ? Math.round(components.reduce((a, b) => a + b, 0) / totalWeight)
      : 0;

  return {
    activeDebates,
    consensusHealth,
    complianceScore,
    memoryPressure,
    receiptDeliveryRate,
    systemIntegrity,
  };
}

// ============================================================================
// Hook
// ============================================================================

const REFRESH_INTERVAL = 30_000;

interface DecisionIntegrityOptions {
  /** Override the default refresh interval (30s) */
  refreshInterval?: number;
  /** Whether to enable fetching (default: true) */
  enabled?: boolean;
}

export function useDecisionIntegrity(options?: DecisionIntegrityOptions) {
  const { refreshInterval = REFRESH_INTERVAL, enabled = true } = options ?? {};

  const swrOpts = { refreshInterval, enabled };

  // Parallel SWR fetches -- each degrades independently on 404/error
  const debates = useSWRFetch<DebateListResponse>('/api/v1/debates?status=active', swrOpts);
  const consensus = useSWRFetch<ConsensusMetrics>('/api/v1/consensus/metrics', swrOpts);
  const compliance = useSWRFetch<ComplianceStatus>('/api/v1/compliance/status', swrOpts);
  const memory = useSWRFetch<MemoryStats>('/api/v1/memory/stats', swrOpts);
  const receipts = useSWRFetch<ReceiptStats>('/api/v1/receipts/stats', swrOpts);
  const audit = useSWRFetch<AuditEventsResponse>('/api/v1/audit/events?limit=20', swrOpts);
  const leaderboard = useSWRFetch<LeaderboardResponse>('/api/v1/leaderboard', {
    refreshInterval: 60_000,
    enabled,
  });
  const settled = useSWRFetch<ConsensusSettled>('/api/v1/consensus/settled?limit=10', swrOpts);

  const metrics = useMemo(
    () =>
      computeIntegrityMetrics(
        debates.data,
        consensus.data,
        compliance.data,
        memory.data,
        receipts.data,
      ),
    [debates.data, consensus.data, compliance.data, memory.data, receipts.data],
  );

  const isLoading =
    debates.isLoading ||
    consensus.isLoading ||
    compliance.isLoading ||
    memory.isLoading ||
    receipts.isLoading;

  return {
    // Raw data from each subsystem
    debates: debates.data,
    consensus: consensus.data,
    compliance: compliance.data,
    memory: memory.data,
    receipts: receipts.data,
    audit: audit.data,
    leaderboard: leaderboard.data,
    settled: settled.data,

    // Derived metrics
    metrics,

    // Loading / error states
    isLoading,
    errors: {
      debates: debates.error,
      consensus: consensus.error,
      compliance: compliance.error,
      memory: memory.error,
      receipts: receipts.error,
      audit: audit.error,
      leaderboard: leaderboard.error,
      settled: settled.error,
    },
  };
}
