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

interface ComplianceFrameworkSummary {
  status?: string;
  controls_assessed?: number;
  controls_compliant?: number;
  data_export?: boolean;
  consent_tracking?: boolean;
  retention_policy?: boolean;
  note?: string;
}

interface ComplianceStatusApiPayload {
  status?: string;
  compliance_score?: number;
  frameworks?: Record<string, ComplianceFrameworkSummary>;
  controls_summary?: {
    total?: number;
    compliant?: number;
    non_compliant?: number;
  };
  last_audit?: string;
}

interface WrappedData<T> {
  data?: T | null;
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

interface AuditEventsApiResponse {
  data?: {
    entries?: Array<Record<string, unknown>>;
    total?: number;
  };
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

function unwrapData<T>(value: T | WrappedData<T> | null): T | null {
  if (!value) return null;
  if (typeof value === 'object' && 'data' in value) {
    return (value as WrappedData<T>).data ?? null;
  }
  return value as T;
}

function normalizeScore(value: number | undefined): number | undefined {
  if (value == null || Number.isNaN(value)) return undefined;
  return value > 1 ? value / 100 : value;
}

function formatFrameworkName(key: string): string {
  const knownLabels: Record<string, string> = {
    soc2_type2: 'SOC 2 Type 2',
    gdpr: 'GDPR',
    hipaa: 'HIPAA',
  };

  return (
    knownLabels[key] ??
    key
      .split('_')
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(' ')
  );
}

function normalizeComplianceStatus(
  value: ComplianceStatus | WrappedData<ComplianceStatusApiPayload> | null,
): ComplianceStatus | null {
  if (!value) return null;

  if (
    'overall_score' in value ||
    Array.isArray((value as ComplianceStatus).frameworks)
  ) {
    return value as ComplianceStatus;
  }

  const raw = unwrapData<ComplianceStatusApiPayload>(
    value as WrappedData<ComplianceStatusApiPayload> | null,
  );
  if (!raw) return null;

  const frameworks = Object.entries(raw.frameworks ?? {}).map(([key, framework]) => ({
    name: formatFrameworkName(key),
    status: framework.status ?? 'not_assessed',
    score:
      framework.controls_assessed && framework.controls_assessed > 0
        ? framework.controls_compliant != null
          ? framework.controls_compliant / framework.controls_assessed
          : undefined
        : undefined,
    last_assessed: raw.last_audit,
  }));

  return {
    status: raw.status,
    overall_score: normalizeScore(raw.compliance_score),
    violations_count: raw.controls_summary?.non_compliant ?? 0,
    frameworks,
    findings: [],
  };
}

function normalizeAuditEvents(
  value: AuditEventsResponse | AuditEventsApiResponse | null,
): AuditEventsResponse | null {
  if (!value) return null;
  if ('events' in value) return value;

  const payload = unwrapData<{ entries?: Array<Record<string, unknown>>; total?: number }>(
    value as AuditEventsApiResponse,
  );
  const entries = payload?.entries ?? [];

  return {
    events: entries.map((entry) => ({
      id: typeof entry.id === 'string' ? entry.id : undefined,
      event_type: typeof entry.event_type === 'string' ? entry.event_type : 'unknown',
      actor: typeof entry.actor === 'string' ? entry.actor : undefined,
      resource: typeof entry.resource === 'string' ? entry.resource : undefined,
      action:
        typeof entry.action === 'string'
          ? entry.action
          : typeof entry.event_type === 'string'
            ? entry.event_type
            : 'unknown',
      timestamp: typeof entry.timestamp === 'string' ? entry.timestamp : '',
      details:
        typeof entry.details === 'string'
          ? entry.details
          : entry.details != null
            ? JSON.stringify(entry.details)
            : undefined,
      severity: entry.outcome === 'failure' ? 'high' : 'info',
    })),
    total: payload?.total ?? entries.length,
  };
}

function normalizeLeaderboard(
  value: LeaderboardResponse | null,
): LeaderboardResponse | null {
  if (!value) return null;

  const rawEntries = value.agents ?? value.rankings ?? value.leaderboard ?? [];
  const normalizedEntries = rawEntries.map((entry) => ({
    ...entry,
    debates_participated:
      entry.debates_participated ??
      (typeof (entry as AgentRanking & { matches?: number }).matches === 'number'
        ? (entry as AgentRanking & { matches?: number }).matches
        : undefined),
  }));

  return {
    ...value,
    leaderboard: normalizedEntries,
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
  const debates = useSWRFetch<DebateListResponse>('/api/v2/debates?status=active', swrOpts);
  const consensus = useSWRFetch<ConsensusMetrics>('/api/v2/consensus/stats', swrOpts);
  const compliance = useSWRFetch<WrappedData<ComplianceStatusApiPayload>>(
    '/api/v2/compliance/status',
    swrOpts,
  );
  const memory = useSWRFetch<MemoryStats>('/api/v2/memory/stats', swrOpts);
  const receiptStats = useSWRFetch<ReceiptStatsApiResponse>('/api/v2/receipts/stats', swrOpts);
  const receiptDeliveries = useSWRFetch<ReceiptDeliveryHistoryResponse>(
    '/api/v2/receipts/deliveries?limit=20',
    swrOpts,
  );
  const audit = useSWRFetch<AuditEventsApiResponse>(
    '/api/v2/compliance/audit-events?limit=20',
    swrOpts,
  );
  const leaderboard = useSWRFetch<LeaderboardResponse>('/api/v2/agents/leaderboard', {
    refreshInterval: 60_000,
    enabled,
  });
  const settled = useSWRFetch<ConsensusSettled>('/api/v2/consensus/settled?limit=10', swrOpts);

  const receipts = useMemo(
    () => normalizeReceiptStats(receiptStats.data, receiptDeliveries.data),
    [receiptStats.data, receiptDeliveries.data],
  );
  const normalizedCompliance = useMemo(
    () => normalizeComplianceStatus(compliance.data),
    [compliance.data],
  );
  const normalizedAudit = useMemo(
    () => normalizeAuditEvents(audit.data),
    [audit.data],
  );
  const normalizedLeaderboard = useMemo(
    () => normalizeLeaderboard(leaderboard.data),
    [leaderboard.data],
  );

  const metrics = useMemo(
    () =>
      computeIntegrityMetrics(
        debates.data,
        consensus.data,
        normalizedCompliance,
        memory.data,
        receipts,
      ),
    [debates.data, consensus.data, normalizedCompliance, memory.data, receipts],
  );

  const isLoading =
    debates.isLoading ||
    consensus.isLoading ||
    compliance.isLoading ||
    memory.isLoading ||
    receiptStats.isLoading ||
    receiptDeliveries.isLoading;

  return {
    // Raw data from each subsystem
    debates: debates.data,
    consensus: consensus.data,
    compliance: normalizedCompliance,
    memory: memory.data,
    receipts,
    audit: normalizedAudit,
    leaderboard: normalizedLeaderboard,
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
      receipts: receiptStats.error ?? receiptDeliveries.error,
      audit: audit.error,
      leaderboard: leaderboard.error,
      settled: settled.error,
    },
  };
}
