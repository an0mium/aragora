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

interface ComplianceStatusApiResponse {
  data?: {
    status?: string;
    compliance_score?: number;
    frameworks?: {
      soc2_type2?: {
        status?: string;
        controls_assessed?: number;
        controls_compliant?: number;
      };
      gdpr?: {
        status?: string;
        data_export?: boolean;
        consent_tracking?: boolean;
        retention_policy?: boolean;
      };
      hipaa?: {
        status?: string;
        note?: string;
      };
    };
    last_audit?: string;
    generated_at?: string;
  };
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
  verified_count?: number;
  delivered?: number;
  pending?: number;
  failed?: number;
  delivery_rate?: number;
  by_verdict?: Record<string, number>;
  by_risk_level?: Record<string, number>;
  generated_at?: string;
  recent?: Array<{
    id: string;
    debate_id?: string;
    status: 'delivered' | 'pending' | 'failed';
    created_at?: string;
    delivered_at?: string;
    channel?: string;
  }>;
}

type RecentReceipt = NonNullable<ReceiptStats['recent']>[number];

interface ReceiptStatsApiResponse {
  total?: number;
  verified?: number;
  by_verdict?: Record<string, number>;
  by_risk_level?: Record<string, number>;
  generated_at?: string;
}

interface ReceiptDeliveryHistoryResponse {
  deliveries?: Array<{
    id?: string;
    receiptId?: string;
    receipt_id?: string;
    status?: string;
    deliveredAt?: string;
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

function normalizeDeliveryStatus(
  value: unknown,
): 'delivered' | 'pending' | 'failed' {
  const status = typeof value === 'string' ? value.toLowerCase() : '';
  if (status === 'success' || status === 'delivered') return 'delivered';
  if (status === 'failed' || status === 'error') return 'failed';
  return 'pending';
}

function normalizeReceiptStats(
  stats: ReceiptStatsApiResponse | null,
  history: ReceiptDeliveryHistoryResponse | null,
): ReceiptStats | null {
  if (!stats && !history) return null;

  const recent: RecentReceipt[] = (history?.deliveries ?? []).flatMap((delivery) => {
    const id = delivery.receiptId ?? delivery.receipt_id ?? delivery.id;
    if (!id) return [];

    const deliveredAt = delivery.deliveredAt ?? delivery.delivered_at;
    return [
      {
        id,
        status: normalizeDeliveryStatus(delivery.status),
        created_at: deliveredAt,
        delivered_at: deliveredAt,
        channel: delivery.channel,
      },
    ];
  });

  const delivered = recent.filter((delivery) => delivery.status === 'delivered').length;
  const pending = recent.filter((delivery) => delivery.status === 'pending').length;
  const failed = recent.filter((delivery) => delivery.status === 'failed').length;
  const deliveryRate =
    delivered + failed > 0 ? delivered / (delivered + failed) : undefined;

  return {
    total_receipts: stats?.total ?? recent.length,
    verified_count: stats?.verified,
    delivered,
    pending,
    failed,
    delivery_rate: deliveryRate,
    by_verdict: stats?.by_verdict ?? {},
    by_risk_level: stats?.by_risk_level ?? {},
    generated_at: stats?.generated_at,
    recent,
  };
}

function normalizeFrameworkStatus(
  value: string | undefined,
): 'compliant' | 'partial' | 'non_compliant' | 'not_assessed' {
  switch (value) {
    case 'compliant':
    case 'supported':
      return 'compliant';
    case 'in_progress':
    case 'partial':
      return 'partial';
    case 'non_compliant':
      return 'non_compliant';
    default:
      return 'not_assessed';
  }
}

function normalizeComplianceStatus(
  response: ComplianceStatusApiResponse | null,
): ComplianceStatus | null {
  const status = response?.data;
  if (!status) return null;

  const frameworks: NonNullable<ComplianceStatus['frameworks']> = [];
  const lastAssessed = status.last_audit ?? status.generated_at;

  if (status.frameworks?.soc2_type2) {
    const soc2 = status.frameworks.soc2_type2;
    const controlsAssessed = soc2.controls_assessed ?? 0;
    const controlsCompliant = soc2.controls_compliant ?? 0;
    frameworks.push({
      name: 'SOC 2 Type II',
      status: normalizeFrameworkStatus(soc2.status),
      score:
        controlsAssessed > 0
          ? controlsCompliant / controlsAssessed
          : undefined,
      last_assessed: lastAssessed,
    });
  }

  if (status.frameworks?.gdpr) {
    const gdpr = status.frameworks.gdpr;
    const controlsMet = [
      gdpr.data_export,
      gdpr.consent_tracking,
      gdpr.retention_policy,
    ].filter(Boolean).length;
    frameworks.push({
      name: 'GDPR',
      status: normalizeFrameworkStatus(gdpr.status),
      score: controlsMet / 3,
      last_assessed: lastAssessed,
    });
  }

  if (status.frameworks?.hipaa) {
    const hipaa = status.frameworks.hipaa;
    frameworks.push({
      name: 'HIPAA',
      status: normalizeFrameworkStatus(hipaa.status),
      last_assessed: lastAssessed,
    });
  }

  return {
    status: status.status,
    overall_score:
      typeof status.compliance_score === 'number'
        ? status.compliance_score / 100
        : undefined,
    frameworks,
    findings: [],
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
  const consensus = useSWRFetch<ConsensusMetrics>('/api/v1/consensus/stats', swrOpts);
  const complianceStatus = useSWRFetch<ComplianceStatusApiResponse>(
    '/api/v2/compliance/status',
    swrOpts,
  );
  const memory = useSWRFetch<MemoryStats>('/api/v1/memory/stats', swrOpts);
  const receiptStats = useSWRFetch<ReceiptStatsApiResponse>('/api/v2/receipts/stats', swrOpts);
  const receiptDeliveries = useSWRFetch<ReceiptDeliveryHistoryResponse>(
    '/api/v1/receipts/deliveries?limit=20',
    swrOpts,
  );
  const audit = useSWRFetch<AuditEventsResponse>('/api/v1/audit/events?limit=20', swrOpts);
  const leaderboard = useSWRFetch<LeaderboardResponse>('/api/v1/leaderboard', {
    refreshInterval: 60_000,
    enabled,
  });
  const settled = useSWRFetch<ConsensusSettled>('/api/v1/consensus/settled?limit=10', swrOpts);

  const receipts = useMemo(
    () => normalizeReceiptStats(receiptStats.data, receiptDeliveries.data),
    [receiptStats.data, receiptDeliveries.data],
  );
  const compliance = useMemo(
    () => normalizeComplianceStatus(complianceStatus.data),
    [complianceStatus.data],
  );

  const metrics = useMemo(
    () =>
      computeIntegrityMetrics(
        debates.data,
        consensus.data,
        compliance,
        memory.data,
        receipts,
      ),
    [debates.data, consensus.data, compliance, memory.data, receipts],
  );

  const isLoading =
    debates.isLoading ||
    consensus.isLoading ||
    complianceStatus.isLoading ||
    memory.isLoading ||
    receiptStats.isLoading ||
    receiptDeliveries.isLoading;

  return {
    // Raw data from each subsystem
    debates: debates.data,
    consensus: consensus.data,
    compliance,
    memory: memory.data,
    receipts,
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
      compliance: complianceStatus.error,
      memory: memory.error,
      receipts: receiptStats.error ?? receiptDeliveries.error,
      audit: audit.error,
      leaderboard: leaderboard.error,
      settled: settled.error,
    },
  };
}
