/**
 * Aragora SDK Client Wrapper for Frontend
 *
 * Singleton SDK client that integrates with AuthContext for token management
 * and BackendSelector for API URL configuration.
 *
 * Usage:
 * ```typescript
 * import { getClient, useAragoraClient } from '@/lib/aragora-client';
 *
 * // In components (hooks):
 * const client = useAragoraClient();
 * const debates = await client.debates.list();
 *
 * // Outside components (with token):
 * const client = getClient(token);
 * const health = await client.health();
 * ```
 */

// =============================================================================
// Core Types (matching SDK types for future migration)
// =============================================================================

export interface ConsensusResult {
  reached: boolean;
  conclusion?: string;
  final_answer?: string;
  confidence: number;
  agreement?: number;
  supporting_agents: string[];
  dissenting_agents?: string[];
}

export interface DebateMessage {
  agent_id: string;
  content: string;
  round: number;
  message_type?: 'proposal' | 'critique' | 'revision' | 'synthesis';
  timestamp?: string;
}

export interface DebateRound {
  round_number: number;
  messages: DebateMessage[];
}

export interface Debate {
  id?: string;
  debate_id: string;
  task: string;
  status: string;
  agents: string[];
  rounds: DebateRound[];
  consensus?: ConsensusResult;
  created_at?: string;
  completed_at?: string;
  metadata?: Record<string, unknown>;
}

export interface DebateCreateRequest {
  task: string;
  agents?: string[];
  max_rounds?: number;
  consensus_threshold?: number;
  enable_voting?: boolean;
  context?: string;
}

export interface DebateCreateResponse {
  debate_id: string;
  status: string;
  task: string;
}

export interface AgentProfile {
  agent_id: string;
  name: string;
  provider: string;
  elo_rating?: number;
  wins?: number;
  losses?: number;
  draws?: number;
  specializations?: string[];
}

export interface LeaderboardEntry {
  agent_id: string;
  elo_rating: number;
  rank: number;
  wins?: number;
  losses?: number;
  win_rate?: number;
}

export interface Organization {
  id: string;
  name: string;
  slug: string;
  tier: string;
  owner_id: string;
  member_count: number;
  debates_used: number;
  debates_limit: number;
  settings: Record<string, unknown>;
  created_at: string;
}

export interface OrganizationMember {
  id: string;
  email: string;
  name: string;
  role: 'member' | 'admin' | 'owner';
  is_active: boolean;
  created_at: string;
  last_login_at: string | null;
}

// BillingUsage, BillingPlan, BillingSubscription interfaces defined below with billing types

export interface AnalyticsOverview {
  total_debates: number;
  active_debates: number;
  completed_debates: number;
  failed_debates: number;
  avg_debate_duration_seconds: number;
  consensus_rate: number;
  period_days: number;
}

export interface AnalyticsResponse {
  overview: AnalyticsOverview;
  top_agents: Array<{
    agent_id: string;
    debates_participated: number;
    wins: number;
    losses: number;
    draws: number;
    avg_contribution_score: number;
  }>;
  debates_by_day: Array<{ date: string; count: number }>;
}

// =============================================================================
// Types
// =============================================================================

export interface AragoraClientConfig {
  baseUrl: string;
  apiKey?: string;
  timeout?: number;
  headers?: Record<string, string>;
}

export interface RequestOptions {
  timeout?: number;
  headers?: Record<string, string>;
  signal?: AbortSignal;
}

// =============================================================================
// Error Class
// =============================================================================

export class AragoraError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details?: Record<string, unknown>;

  constructor(
    message: string,
    code: string,
    status: number,
    details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'AragoraError';
    this.code = code;
    this.status = status;
    this.details = details;
  }

  /** Create a user-friendly error message */
  toUserMessage(): string {
    switch (this.code) {
      case 'TIMEOUT':
        return 'Request timed out. Please try again or check your network connection.';
      case 'NETWORK_ERROR':
        return 'Network error. Please check your internet connection and try again.';
      case 'RATE_LIMITED':
        return 'Too many requests. Please wait a moment before trying again.';
      case 'UNAUTHORIZED':
        return 'Authentication failed. Please sign in again.';
      case 'FORBIDDEN':
        return 'Access denied. You do not have permission to perform this action.';
      case 'NOT_FOUND':
        return 'The requested resource was not found.';
      default:
        return this.message;
    }
  }
}

// =============================================================================
// HTTP Client
// =============================================================================

class HttpClient {
  private _baseUrl: string;
  private _apiKey?: string;
  private timeout: number;
  private defaultHeaders: Record<string, string>;

  get baseUrl(): string {
    return this._baseUrl;
  }

  get apiKey(): string | undefined {
    return this._apiKey;
  }

  constructor(config: AragoraClientConfig) {
    this._baseUrl = config.baseUrl.replace(/\/$/, '');
    this._apiKey = config.apiKey;
    this.timeout = config.timeout ?? 30000;
    this.defaultHeaders = {
      'Content-Type': 'application/json',
      ...config.headers,
    };

    if (this._apiKey) {
      this.defaultHeaders['Authorization'] = `Bearer ${this._apiKey}`;
    }
  }

  private async request<T>(
    method: string,
    path: string,
    data?: unknown,
    options?: RequestOptions
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      options?.timeout ?? this.timeout
    );

    try {
      const response = await fetch(url, {
        method,
        headers: {
          ...this.defaultHeaders,
          ...options?.headers,
        },
        body: data ? JSON.stringify(data) : undefined,
        signal: options?.signal ?? controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new AragoraError(
          errorData.error || `HTTP ${response.status}`,
          errorData.code || 'HTTP_ERROR',
          response.status,
          errorData
        );
      }

      return response.json();
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof AragoraError) {
        throw error;
      }

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new AragoraError('Request timed out', 'TIMEOUT', 408);
        }
        throw new AragoraError(error.message, 'NETWORK_ERROR', 0);
      }

      throw new AragoraError('Unknown error', 'UNKNOWN_ERROR', 0);
    }
  }

  async get<T>(path: string, options?: RequestOptions): Promise<T> {
    return this.request<T>('GET', path, undefined, options);
  }

  async post<T>(path: string, data?: unknown, options?: RequestOptions): Promise<T> {
    return this.request<T>('POST', path, data, options);
  }

  async put<T>(path: string, data?: unknown, options?: RequestOptions): Promise<T> {
    return this.request<T>('PUT', path, data, options);
  }

  async delete<T>(path: string, options?: RequestOptions): Promise<T> {
    return this.request<T>('DELETE', path, undefined, options);
  }
}

// =============================================================================
// API Classes (matching SDK structure)
// =============================================================================

class DebatesAPI {
  constructor(private http: HttpClient) {}

  async list(options?: { limit?: number; offset?: number; status?: string }) {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));
    if (options?.status) params.set('status', options.status);

    const query = params.toString();
    const path = query ? `/api/debates?${query}` : '/api/debates';
    return this.http.get<{ debates: unknown[] }>(path);
  }

  async get(debateId: string) {
    return this.http.get<unknown>(`/api/debates/${debateId}`);
  }

  async create(request: { task: string; agents?: string[]; max_rounds?: number }) {
    return this.http.post<{ debate_id: string }>('/api/debates', request);
  }
}

class AgentsAPI {
  constructor(private http: HttpClient) {}

  async list() {
    return this.http.get<{ agents: unknown[] }>('/api/agents');
  }

  async get(agentId: string) {
    return this.http.get<unknown>(`/api/agents/${agentId}`);
  }
}

class LeaderboardAPI {
  constructor(private http: HttpClient) {}

  async get(options?: { limit?: number }) {
    const params = options?.limit ? `?limit=${options.limit}` : '';
    return this.http.get<{ entries: unknown[] }>(`/api/leaderboard${params}`);
  }
}

class OrganizationsAPI {
  constructor(private http: HttpClient) {}

  async get(orgId: string) {
    return this.http.get<{ organization: unknown }>(`/api/org/${orgId}`);
  }

  async members(orgId: string) {
    return this.http.get<{ members: unknown[] }>(`/api/org/${orgId}/members`);
  }
}

// Billing Types
export interface BillingPlan {
  id: string;
  name: string;
  price_monthly_cents: number;
  price_monthly: string;
  features: {
    debates_per_month: number;
    users_per_org: number;
    api_access: boolean;
    all_agents: boolean;
    custom_agents: boolean;
    sso_enabled: boolean;
    audit_logs: boolean;
    priority_support: boolean;
  };
}

export interface BillingUsage {
  debates_used: number;
  debates_limit: number;
  debates_remaining: number;
  tokens_used: number;
  tokens_in: number;
  tokens_out: number;
  estimated_cost_usd: number;
  cost_breakdown?: {
    input_cost: number;
    output_cost: number;
    total: number;
  };
  cost_by_provider?: Record<string, string>;
  period_start: string | null;
  period_end: string | null;
}

export interface BillingSubscription {
  tier: string;
  status: string;
  is_active: boolean;
  organization?: {
    id: string;
    name: string;
  };
  limits?: {
    debates_per_month: number;
    users_per_org: number;
    api_access: boolean;
    all_agents: boolean;
    custom_agents: boolean;
    sso_enabled: boolean;
    audit_logs: boolean;
    priority_support: boolean;
    price_monthly_cents: number;
  };
  current_period_end?: string;
  cancel_at_period_end?: boolean;
  trial_start?: string;
  trial_end?: string;
  is_trialing?: boolean;
  payment_failed?: boolean;
}

export interface BillingInvoice {
  id: string;
  number: string;
  status: string;
  amount_due: number;
  amount_paid: number;
  currency: string;
  created: string;
  period_start: string | null;
  period_end: string | null;
  hosted_invoice_url: string;
  invoice_pdf: string;
}

export interface UsageForecast {
  current_usage: {
    debates: number;
    debates_limit: number;
  };
  projection: {
    debates_end_of_cycle: number;
    debates_per_day: number;
    tokens_per_day: number;
    cost_end_of_cycle_usd: number;
  };
  days_remaining: number;
  days_elapsed: number;
  will_hit_limit: boolean;
  debates_overage: number;
  tier_recommendation?: {
    recommended_tier: string;
    debates_limit: number;
    price_monthly: string;
  };
}

class BillingAPI {
  constructor(private http: HttpClient) {}

  async usage(): Promise<{ usage: BillingUsage }> {
    return this.http.get<{ usage: BillingUsage }>('/api/billing/usage');
  }

  async subscription(): Promise<{ subscription: BillingSubscription }> {
    return this.http.get<{ subscription: BillingSubscription }>('/api/billing/subscription');
  }

  async plans(): Promise<{ plans: BillingPlan[] }> {
    return this.http.get<{ plans: BillingPlan[] }>('/api/billing/plans');
  }

  async invoices(limit = 10): Promise<{ invoices: BillingInvoice[] }> {
    return this.http.get<{ invoices: BillingInvoice[] }>(`/api/billing/invoices?limit=${limit}`);
  }

  async forecast(): Promise<{ forecast: UsageForecast }> {
    return this.http.get<{ forecast: UsageForecast }>('/api/billing/usage/forecast');
  }

  async createCheckout(tier: string, successUrl: string, cancelUrl: string): Promise<{
    checkout: { id: string; url: string };
  }> {
    return this.http.post('/api/billing/checkout', {
      tier,
      success_url: successUrl,
      cancel_url: cancelUrl,
    });
  }

  async createPortal(returnUrl: string): Promise<{ portal: { url: string } }> {
    return this.http.post('/api/billing/portal', {
      return_url: returnUrl,
    });
  }

  async cancelSubscription(): Promise<{ message: string; subscription: unknown }> {
    return this.http.post('/api/billing/cancel', {});
  }

  async resumeSubscription(): Promise<{ message: string; subscription: unknown }> {
    return this.http.post('/api/billing/resume', {});
  }

  async exportUsage(startDate?: string, endDate?: string): Promise<Blob> {
    const params = new URLSearchParams();
    if (startDate) params.set('start', startDate);
    if (endDate) params.set('end', endDate);
    const query = params.toString();
    const path = query ? `/api/billing/usage/export?${query}` : '/api/billing/usage/export';

    // This returns CSV data, so we need raw response
    const response = await fetch(this.http.baseUrl + path, {
      headers: this.http.apiKey
        ? { Authorization: `Bearer ${this.http.apiKey}` }
        : {},
    });
    return response.blob();
  }
}

class AnalyticsAPI {
  constructor(private http: HttpClient) {}

  async overview(days = 30) {
    return this.http.get<unknown>(`/api/analytics?days=${days}`);
  }
}

// =============================================================================
// Admin API (not in SDK, specific to admin console)
// =============================================================================

interface TierRevenue {
  count: number;
  price_cents: number;
  mrr_cents: number;
}

interface RevenueResponse {
  revenue: {
    mrr_cents: number;
    mrr_dollars: number;
    arr_dollars: number;
    tier_breakdown: Record<string, TierRevenue>;
    total_organizations: number;
    paying_organizations: number;
  };
}

interface AdminStatsResponse {
  stats: {
    total_users: number;
    active_users: number;
    total_organizations: number;
    tier_distribution: Record<string, number>;
    total_debates_this_month: number;
    users_active_24h: number;
    new_users_7d: number;
    new_orgs_7d: number;
  };
}

interface AdminUsersResponse {
  users: Array<{
    id: string;
    email: string;
    name: string;
    role: string;
    org_id: string | null;
    is_active: boolean;
    created_at: string;
    last_login_at: string | null;
  }>;
  total: number;
  limit: number;
  offset: number;
}

interface AdminOrganizationsResponse {
  organizations: Array<{
    id: string;
    name: string;
    slug: string;
    tier: string;
    owner_id: string;
    member_count: number;
    debates_used: number;
    debates_limit: number;
    created_at: string;
  }>;
  total: number;
  limit: number;
  offset: number;
}

interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  uptime_seconds: number;
  version: string;
  components: {
    database: { status: string; latency_ms?: number };
    agents: { status: string; available: number; total: number };
    memory: { status: string; usage_mb?: number };
    websocket: { status: string; connections: number };
  };
  timestamp: string;
}

interface CircuitBreakerState {
  agent: string;
  state: 'closed' | 'open' | 'half_open';
  failures: number;
  last_failure?: string;
  last_success?: string;
}

interface RecentError {
  id: string;
  timestamp: string;
  level: string;
  message: string;
  endpoint?: string;
  user_id?: string;
}

interface RateLimitState {
  endpoint: string;
  limit: number;
  remaining: number;
  reset_at: string;
}

class AdminAPI {
  constructor(private http: HttpClient) {}

  async revenue(): Promise<RevenueResponse> {
    return this.http.get<RevenueResponse>('/api/admin/revenue');
  }

  async stats(): Promise<AdminStatsResponse> {
    return this.http.get<AdminStatsResponse>('/api/admin/stats');
  }

  async users(options?: { limit?: number; offset?: number; search?: string }): Promise<AdminUsersResponse> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));
    if (options?.search) params.set('search', options.search);

    const query = params.toString();
    const path = query ? `/api/admin/users?${query}` : '/api/admin/users';
    return this.http.get<AdminUsersResponse>(path);
  }

  async organizations(options?: { limit?: number; offset?: number; tier?: string }): Promise<AdminOrganizationsResponse> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));
    if (options?.tier) params.set('tier', options.tier);

    const query = params.toString();
    const path = query ? `/api/admin/organizations?${query}` : '/api/admin/organizations';
    return this.http.get<AdminOrganizationsResponse>(path);
  }
}

// =============================================================================
// Training API (for ML training data exports)
// =============================================================================

interface TrainingExportOptions {
  min_confidence?: number;
  min_success_rate?: number;
  limit?: number;
  offset?: number;
  include_critiques?: boolean;
  include_patterns?: boolean;
  include_debates?: boolean;
  format?: 'json' | 'jsonl';
}

interface DPOExportOptions {
  min_confidence_diff?: number;
  limit?: number;
  offset?: number;
  format?: 'json' | 'jsonl';
}

interface GauntletExportOptions {
  persona?: 'gdpr' | 'hipaa' | 'ai_act' | 'all';
  min_severity?: number;
  limit?: number;
  offset?: number;
  format?: 'json' | 'jsonl';
}

interface TrainingStatsResponse {
  stats: {
    sft_available: number;
    dpo_available: number;
    gauntlet_available: number;
    total_debates: number;
    debates_with_consensus: number;
    last_export?: string;
  };
}

interface TrainingFormatsResponse {
  formats: {
    sft: { schema: object; description: string };
    dpo: { schema: object; description: string };
    gauntlet: { schema: object; description: string };
  };
}

interface TrainingExportResponse {
  data: unknown[];
  total: number;
  format: string;
  exported_at: string;
}

// =============================================================================
// Evidence Types
// =============================================================================

export interface EvidenceSnippet {
  id: string;
  source: string;
  title: string;
  snippet: string;
  url?: string;
  reliability_score: number;
  freshness_score?: number;
  quality_score?: number;
  metadata?: Record<string, unknown>;
  collected_at?: string;
}

export interface EvidenceSearchOptions {
  query: string;
  limit?: number;
  source?: string;
  min_reliability?: number;
  context?: {
    topic?: string;
    keywords?: string[];
    required_topics?: string[];
    preferred_sources?: string[];
    blocked_sources?: string[];
    max_age_days?: number;
    min_word_count?: number;
    require_citations?: boolean;
  };
}

export interface EvidenceCollectOptions {
  task: string;
  connectors?: string[];
  debate_id?: string;
  round?: number;
}

export interface EvidenceListOptions {
  limit?: number;
  offset?: number;
  source?: string;
  min_reliability?: number;
}

export interface EvidenceStatistics {
  total_evidence: number;
  by_source: Record<string, number>;
  average_reliability: number;
  average_quality?: number;
  last_collected?: string;
}

interface EvidenceListResponse {
  evidence: EvidenceSnippet[];
  total: number;
  limit: number;
  offset: number;
}

interface EvidenceSearchResponse {
  query: string;
  results: EvidenceSnippet[];
  count: number;
}

interface EvidenceCollectResponse {
  task: string;
  keywords: string[];
  snippets: EvidenceSnippet[];
  count: number;
  total_searched: number;
  average_reliability: number;
  average_freshness: number;
  saved_ids: string[];
  debate_id?: string;
}

// =============================================================================
// Evidence API
// =============================================================================

class EvidenceAPI {
  constructor(private http: HttpClient) {}

  async list(options?: EvidenceListOptions): Promise<EvidenceListResponse> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));
    if (options?.source) params.set('source', options.source);
    if (options?.min_reliability !== undefined) params.set('min_reliability', String(options.min_reliability));

    const query = params.toString();
    const path = query ? `/api/evidence?${query}` : '/api/evidence';
    return this.http.get<EvidenceListResponse>(path);
  }

  async get(id: string): Promise<{ evidence: EvidenceSnippet }> {
    return this.http.get<{ evidence: EvidenceSnippet }>(`/api/evidence/${id}`);
  }

  async search(options: EvidenceSearchOptions): Promise<EvidenceSearchResponse> {
    return this.http.post<EvidenceSearchResponse>('/api/evidence/search', options);
  }

  async collect(options: EvidenceCollectOptions): Promise<EvidenceCollectResponse> {
    return this.http.post<EvidenceCollectResponse>('/api/evidence/collect', options);
  }

  async getDebateEvidence(debateId: string, round?: number): Promise<{
    debate_id: string;
    round: number | null;
    evidence: EvidenceSnippet[];
    count: number;
  }> {
    const params = round !== undefined ? `?round=${round}` : '';
    return this.http.get(`/api/evidence/debate/${debateId}${params}`);
  }

  async associateWithDebate(debateId: string, evidenceIds: string[], round?: number): Promise<{
    debate_id: string;
    associated: string[];
    count: number;
  }> {
    return this.http.post(`/api/evidence/debate/${debateId}`, {
      evidence_ids: evidenceIds,
      round,
    });
  }

  async delete(id: string): Promise<{ deleted: boolean; evidence_id: string }> {
    return this.http.delete<{ deleted: boolean; evidence_id: string }>(`/api/evidence/${id}`);
  }

  async statistics(): Promise<{ statistics: EvidenceStatistics }> {
    return this.http.get<{ statistics: EvidenceStatistics }>('/api/evidence/statistics');
  }
}

// =============================================================================
// Training API
// =============================================================================

class TrainingAPI {
  constructor(private http: HttpClient) {}

  async stats(): Promise<TrainingStatsResponse> {
    return this.http.get<TrainingStatsResponse>('/api/training/stats');
  }

  async formats(): Promise<TrainingFormatsResponse> {
    return this.http.get<TrainingFormatsResponse>('/api/training/formats');
  }

  async exportSFT(options?: TrainingExportOptions): Promise<TrainingExportResponse> {
    const params = new URLSearchParams();
    if (options?.min_confidence !== undefined) params.set('min_confidence', String(options.min_confidence));
    if (options?.min_success_rate !== undefined) params.set('min_success_rate', String(options.min_success_rate));
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));
    if (options?.include_critiques !== undefined) params.set('include_critiques', String(options.include_critiques));
    if (options?.include_patterns !== undefined) params.set('include_patterns', String(options.include_patterns));
    if (options?.include_debates !== undefined) params.set('include_debates', String(options.include_debates));
    if (options?.format) params.set('format', options.format);

    const query = params.toString();
    const path = query ? `/api/training/export/sft?${query}` : '/api/training/export/sft';
    return this.http.get<TrainingExportResponse>(path);
  }

  async exportDPO(options?: DPOExportOptions): Promise<TrainingExportResponse> {
    const params = new URLSearchParams();
    if (options?.min_confidence_diff !== undefined) params.set('min_confidence_diff', String(options.min_confidence_diff));
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));
    if (options?.format) params.set('format', options.format);

    const query = params.toString();
    const path = query ? `/api/training/export/dpo?${query}` : '/api/training/export/dpo';
    return this.http.get<TrainingExportResponse>(path);
  }

  async exportGauntlet(options?: GauntletExportOptions): Promise<TrainingExportResponse> {
    const params = new URLSearchParams();
    if (options?.persona) params.set('persona', options.persona);
    if (options?.min_severity !== undefined) params.set('min_severity', String(options.min_severity));
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.offset) params.set('offset', String(options.offset));
    if (options?.format) params.set('format', options.format);

    const query = params.toString();
    const path = query ? `/api/training/export/gauntlet?${query}` : '/api/training/export/gauntlet';
    return this.http.get<TrainingExportResponse>(path);
  }
}

class SystemAPI {
  constructor(private http: HttpClient) {}

  async health(): Promise<HealthStatus> {
    return this.http.get<HealthStatus>('/api/health');
  }

  async circuitBreakers(): Promise<{ breakers: CircuitBreakerState[] }> {
    return this.http.get<{ breakers: CircuitBreakerState[] }>('/api/system/circuit-breakers');
  }

  async errors(limit = 20): Promise<{ errors: RecentError[] }> {
    return this.http.get<{ errors: RecentError[] }>(`/api/system/errors?limit=${limit}`);
  }

  async rateLimits(): Promise<{ limits: RateLimitState[] }> {
    return this.http.get<{ limits: RateLimitState[] }>('/api/system/rate-limits');
  }
}

// =============================================================================
// Tournaments API
// =============================================================================

export interface Tournament {
  id: string;
  name: string;
  topic: string;
  status: 'pending' | 'in_progress' | 'completed';
  bracket_type: 'single_elimination' | 'double_elimination' | 'round_robin';
  participants: string[];
  matches: TournamentMatch[];
  winner?: string;
  created_at: string;
  completed_at?: string;
}

export interface TournamentMatch {
  id: string;
  round: number;
  participant1: string;
  participant2: string;
  winner?: string;
  debate_id?: string;
  status: 'pending' | 'in_progress' | 'completed';
}

export interface TournamentStanding {
  agent_id: string;
  wins: number;
  losses: number;
  points: number;
  rank: number;
}

class TournamentsAPI {
  constructor(private http: HttpClient) {}

  async list(options?: { limit?: number; status?: string }): Promise<{ tournaments: Tournament[] }> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.status) params.set('status', options.status);
    const query = params.toString();
    return this.http.get(`/api/tournaments${query ? `?${query}` : ''}`);
  }

  async get(id: string): Promise<{ tournament: Tournament }> {
    return this.http.get(`/api/tournaments/${id}`);
  }

  async create(data: {
    name: string;
    topic: string;
    participants: string[];
    bracket_type?: string;
  }): Promise<{ tournament: Tournament }> {
    return this.http.post('/api/tournaments/create', data);
  }

  async bracket(id: string): Promise<{ bracket: TournamentMatch[] }> {
    return this.http.get(`/api/tournaments/${id}/bracket`);
  }

  async standings(id: string): Promise<{ standings: TournamentStanding[] }> {
    return this.http.get(`/api/tournaments/${id}/standings`);
  }

  async matches(id: string): Promise<{ matches: TournamentMatch[] }> {
    return this.http.get(`/api/tournaments/${id}/matches`);
  }

  async advance(id: string): Promise<{ tournament: Tournament }> {
    return this.http.post(`/api/tournaments/${id}/advance`, {});
  }

  async results(): Promise<{ results: Tournament[] }> {
    return this.http.get('/api/tournaments/results');
  }
}

// =============================================================================
// Pulse API (Trending Topics)
// =============================================================================

export interface TrendingTopic {
  topic: string;
  score: number;
  category: string;
  source: string;
  timestamp: string;
}

export interface PulseStats {
  total_topics: number;
  categories: Record<string, number>;
  last_updated: string;
}

class PulseAPI {
  constructor(private http: HttpClient) {}

  async trending(options?: { limit?: number; category?: string }): Promise<{ topics: TrendingTopic[] }> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.category) params.set('category', options.category);
    const query = params.toString();
    return this.http.get(`/api/pulse/trending${query ? `?${query}` : ''}`);
  }

  async categories(): Promise<{ categories: string[] }> {
    return this.http.get('/api/pulse/categories');
  }

  async suggest(topic: string): Promise<{ suggestions: string[] }> {
    return this.http.post('/api/pulse/suggest', { topic });
  }

  async stats(): Promise<{ stats: PulseStats }> {
    return this.http.get('/api/pulse/stats');
  }

  async analytics(): Promise<{ analytics: unknown }> {
    return this.http.get('/api/pulse/analytics');
  }

  async debateTopic(topicId: string): Promise<{ debate_id: string }> {
    return this.http.post('/api/pulse/debate-topic', { topic_id: topicId });
  }
}

// =============================================================================
// Gallery API (Public Debates Showcase)
// =============================================================================

export interface GalleryEntry {
  id: string;
  debate_id: string;
  title: string;
  summary: string;
  agents: string[];
  consensus_reached: boolean;
  featured: boolean;
  views: number;
  created_at: string;
}

class GalleryAPI {
  constructor(private http: HttpClient) {}

  async list(options?: { limit?: number; featured?: boolean }): Promise<{ entries: GalleryEntry[] }> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', String(options.limit));
    if (options?.featured !== undefined) params.set('featured', String(options.featured));
    const query = params.toString();
    return this.http.get(`/api/gallery${query ? `?${query}` : ''}`);
  }

  async get(debateId: string): Promise<{ entry: GalleryEntry }> {
    return this.http.get(`/api/gallery/${debateId}`);
  }

  async embed(debateId: string): Promise<{ embed_url: string; embed_html: string }> {
    return this.http.get(`/api/gallery/${debateId}/embed`);
  }
}

// =============================================================================
// Moments API (Notable Debate Moments)
// =============================================================================

export interface DebateMoment {
  id: string;
  debate_id: string;
  type: 'flip' | 'breakthrough' | 'consensus' | 'disagreement' | 'insight';
  agent_id: string;
  content: string;
  round: number;
  score: number;
  timestamp: string;
}

class MomentsAPI {
  constructor(private http: HttpClient) {}

  async recent(limit = 20): Promise<{ moments: DebateMoment[] }> {
    return this.http.get(`/api/moments/recent?limit=${limit}`);
  }

  async trending(): Promise<{ moments: DebateMoment[] }> {
    return this.http.get('/api/moments/trending');
  }

  async byType(type: string, limit = 20): Promise<{ moments: DebateMoment[] }> {
    return this.http.get(`/api/moments/by-type/${type}?limit=${limit}`);
  }

  async timeline(debateId: string): Promise<{ moments: DebateMoment[] }> {
    return this.http.get(`/api/moments/timeline?debate_id=${debateId}`);
  }

  async summary(): Promise<{ summary: Record<string, number> }> {
    return this.http.get('/api/moments/summary');
  }
}

// =============================================================================
// Agent Detail API (Extended Agent Info)
// =============================================================================

export interface AgentHistory {
  debate_id: string;
  task: string;
  outcome: 'win' | 'loss' | 'draw';
  elo_change: number;
  date: string;
}

export interface AgentNetwork {
  allies: Array<{ agent_id: string; synergy_score: number; matches: number }>;
  rivals: Array<{ agent_id: string; rivalry_score: number; matches: number }>;
}

export interface AgentPerformance {
  agent_id: string;
  total_debates: number;
  wins: number;
  losses: number;
  draws: number;
  win_rate: number;
  avg_elo_gain: number;
  domains: Record<string, { wins: number; losses: number }>;
}

class AgentDetailAPI {
  constructor(private http: HttpClient) {}

  async history(agentId: string, limit = 20): Promise<{ history: AgentHistory[] }> {
    return this.http.get(`/api/agent/${agentId}/history?limit=${limit}`);
  }

  async network(agentId: string): Promise<AgentNetwork> {
    return this.http.get(`/api/agent/${agentId}/network`);
  }

  async performance(agentId: string): Promise<{ performance: AgentPerformance }> {
    return this.http.get(`/api/agent/${agentId}/performance`);
  }

  async profile(agentId: string): Promise<{ profile: unknown }> {
    return this.http.get(`/api/agent/${agentId}/profile`);
  }

  async consistency(agentId: string): Promise<{ consistency: unknown }> {
    return this.http.get(`/api/agent/${agentId}/consistency`);
  }

  async calibration(agentId: string): Promise<{ calibration: unknown }> {
    return this.http.get(`/api/agent/${agentId}/calibration`);
  }

  async domains(agentId: string): Promise<{ domains: Record<string, unknown> }> {
    return this.http.get(`/api/agent/${agentId}/domains`);
  }

  async headToHead(agentId: string, opponentId: string): Promise<{ stats: unknown }> {
    return this.http.get(`/api/agent/${agentId}/head-to-head/${opponentId}`);
  }

  async compare(agents: string[]): Promise<{ comparison: unknown }> {
    const params = agents.map(a => `agents=${a}`).join('&');
    return this.http.get(`/api/agent/compare?${params}`);
  }
}

// =============================================================================
// Nomic Admin API (Loop Management)
// =============================================================================

export interface NomicStatus {
  running: boolean;
  current_phase: string | null;
  cycle_id: string | null;
  state_machine: Record<string, unknown> | null;
  metrics: Record<string, unknown> | null;
  circuit_breakers: { open: string[]; details: Record<string, unknown> } | null;
  last_checkpoint: string | null;
  stuck_detection: { is_stuck: boolean; stuck_duration_seconds: number };
  errors: string[];
}

export interface NomicCircuitBreakers {
  circuit_breakers: Record<string, unknown>;
  open_circuits: string[];
  total_count: number;
}

class NomicAdminAPI {
  constructor(private http: HttpClient) {}

  async status(): Promise<NomicStatus> {
    return this.http.get('/api/admin/nomic/status');
  }

  async circuitBreakers(): Promise<NomicCircuitBreakers> {
    return this.http.get('/api/admin/nomic/circuit-breakers');
  }

  async reset(options: {
    target_phase: string;
    clear_errors?: boolean;
    reason?: string;
  }): Promise<{ success: boolean; previous_phase: string; new_phase: string }> {
    return this.http.post('/api/admin/nomic/reset', options);
  }

  async pause(reason?: string): Promise<{ success: boolean; status: string }> {
    return this.http.post('/api/admin/nomic/pause', { reason });
  }

  async resume(targetPhase?: string): Promise<{ success: boolean; phase: string }> {
    return this.http.post('/api/admin/nomic/resume', { target_phase: targetPhase });
  }

  async resetCircuitBreakers(): Promise<{ success: boolean; previously_open: string[] }> {
    return this.http.post('/api/admin/nomic/circuit-breakers/reset', {});
  }
}

// =============================================================================
// Genesis API (Genetic Evolution)
// =============================================================================

export interface GenesisStats {
  total_genomes: number;
  active_genomes: number;
  total_debates: number;
  average_fitness: number;
  top_fitness: number;
}

export interface GenesisEvent {
  event_type: string;
  genome_id: string;
  timestamp: string;
  details: Record<string, unknown>;
}

export interface Genome {
  genome_id: string;
  fitness: number;
  generation: number;
  parent_ids?: string[];
  traits: Record<string, unknown>;
  created_at: string;
  debates_count: number;
}

export interface GenesisLineage {
  genome_id: string;
  ancestors: Genome[];
  descendants: Genome[];
  depth: number;
}

export interface GenesisTree {
  debate_id: string;
  nodes: Array<{
    id: string;
    genome_id: string;
    parent_id?: string;
    fitness: number;
  }>;
}

class GenesisAPI {
  constructor(private http: HttpClient) {}

  async stats(): Promise<{ stats: GenesisStats }> {
    return this.http.get('/api/genesis/stats');
  }

  async events(limit = 20): Promise<{ events: GenesisEvent[] }> {
    return this.http.get(`/api/genesis/events?limit=${limit}`);
  }

  async genomes(params?: { limit?: number; offset?: number }): Promise<{ genomes: Genome[] }> {
    const query = new URLSearchParams();
    if (params?.limit) query.set('limit', params.limit.toString());
    if (params?.offset) query.set('offset', params.offset.toString());
    return this.http.get(`/api/genesis/genomes?${query}`);
  }

  async topGenomes(limit = 10): Promise<{ genomes: Genome[] }> {
    return this.http.get(`/api/genesis/genomes/top?limit=${limit}`);
  }

  async population(): Promise<{ population: Genome[]; generation: number }> {
    return this.http.get('/api/genesis/population');
  }

  async genome(genomeId: string): Promise<{ genome: Genome }> {
    return this.http.get(`/api/genesis/genomes/${genomeId}`);
  }

  async lineage(genomeId: string): Promise<{ lineage: GenesisLineage }> {
    return this.http.get(`/api/genesis/lineage/${genomeId}`);
  }

  async tree(debateId: string): Promise<{ tree: GenesisTree }> {
    return this.http.get(`/api/genesis/tree/${debateId}`);
  }
}

// =============================================================================
// Gauntlet API (Stress Testing)
// =============================================================================

export interface GauntletPersona {
  id: string;
  name: string;
  description: string;
  traits: string[];
  difficulty: 'easy' | 'medium' | 'hard' | 'extreme';
}

export interface GauntletRunRequest {
  decision: string;
  personas?: string[];
  rounds?: number;
  stress_level?: number;
}

export interface GauntletResult {
  gauntlet_id: string;
  decision: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  personas_used: string[];
  rounds_completed: number;
  risk_score: number;
  vulnerabilities: Array<{
    category: string;
    severity: string;
    description: string;
  }>;
  recommendation: string;
  created_at: string;
  completed_at?: string;
}

export interface GauntletReceipt {
  gauntlet_id: string;
  decision: string;
  verdict: 'approved' | 'rejected' | 'needs_review';
  confidence: number;
  risk_factors: Array<{
    factor: string;
    weight: number;
    assessment: string;
  }>;
  signatures: string[];
}

export interface GauntletHeatmap {
  gauntlet_id: string;
  categories: string[];
  data: number[][];
  max_risk: number;
}

export interface GauntletComparison {
  gauntlet_a: GauntletResult;
  gauntlet_b: GauntletResult;
  differences: Array<{
    aspect: string;
    a_value: unknown;
    b_value: unknown;
  }>;
  recommendation: string;
}

class GauntletAPI {
  constructor(private http: HttpClient) {}

  async run(request: GauntletRunRequest): Promise<{ gauntlet_id: string; status: string }> {
    return this.http.post('/api/gauntlet/run', request);
  }

  async personas(): Promise<{ personas: GauntletPersona[] }> {
    return this.http.get('/api/gauntlet/personas');
  }

  async results(params?: { limit?: number; offset?: number }): Promise<{ results: GauntletResult[] }> {
    const query = new URLSearchParams();
    if (params?.limit) query.set('limit', params.limit.toString());
    if (params?.offset) query.set('offset', params.offset.toString());
    return this.http.get(`/api/gauntlet/results?${query}`);
  }

  async get(gauntletId: string): Promise<{ gauntlet: GauntletResult }> {
    return this.http.get(`/api/gauntlet/${gauntletId}`);
  }

  async receipt(gauntletId: string): Promise<{ receipt: GauntletReceipt }> {
    return this.http.get(`/api/gauntlet/${gauntletId}/receipt`);
  }

  async heatmap(gauntletId: string): Promise<{ heatmap: GauntletHeatmap }> {
    return this.http.get(`/api/gauntlet/${gauntletId}/heatmap`);
  }

  async compare(gauntletIdA: string, gauntletIdB: string): Promise<{ comparison: GauntletComparison }> {
    return this.http.get(`/api/gauntlet/${gauntletIdA}/compare/${gauntletIdB}`);
  }

  async delete(gauntletId: string): Promise<{ success: boolean }> {
    return this.http.delete(`/api/gauntlet/${gauntletId}`);
  }
}

// =============================================================================
// Main Client
// =============================================================================

export class AragoraClient {
  private http: HttpClient;

  readonly debates: DebatesAPI;
  readonly agents: AgentsAPI;
  readonly leaderboard: LeaderboardAPI;
  readonly organizations: OrganizationsAPI;
  readonly billing: BillingAPI;
  readonly analytics: AnalyticsAPI;
  readonly admin: AdminAPI;
  readonly system: SystemAPI;
  readonly training: TrainingAPI;
  readonly evidence: EvidenceAPI;
  // New APIs
  readonly tournaments: TournamentsAPI;
  readonly pulse: PulseAPI;
  readonly gallery: GalleryAPI;
  readonly moments: MomentsAPI;
  readonly agentDetail: AgentDetailAPI;
  readonly nomicAdmin: NomicAdminAPI;
  readonly genesis: GenesisAPI;
  readonly gauntlet: GauntletAPI;

  constructor(config: AragoraClientConfig) {
    this.http = new HttpClient(config);

    this.debates = new DebatesAPI(this.http);
    this.agents = new AgentsAPI(this.http);
    this.leaderboard = new LeaderboardAPI(this.http);
    this.organizations = new OrganizationsAPI(this.http);
    this.billing = new BillingAPI(this.http);
    this.analytics = new AnalyticsAPI(this.http);
    this.admin = new AdminAPI(this.http);
    this.system = new SystemAPI(this.http);
    this.training = new TrainingAPI(this.http);
    this.evidence = new EvidenceAPI(this.http);
    // New APIs
    this.tournaments = new TournamentsAPI(this.http);
    this.pulse = new PulseAPI(this.http);
    this.gallery = new GalleryAPI(this.http);
    this.moments = new MomentsAPI(this.http);
    this.agentDetail = new AgentDetailAPI(this.http);
    this.nomicAdmin = new NomicAdminAPI(this.http);
    this.genesis = new GenesisAPI(this.http);
    this.gauntlet = new GauntletAPI(this.http);
  }

  async health(): Promise<HealthStatus> {
    return this.system.health();
  }
}

// =============================================================================
// Export types for admin APIs
// =============================================================================

// Re-export internal types (not already exported with `export interface`)
export type {
  TierRevenue,
  RevenueResponse,
  AdminStatsResponse,
  AdminUsersResponse,
  AdminOrganizationsResponse,
  HealthStatus,
  CircuitBreakerState,
  RecentError,
  RateLimitState,
  TrainingExportOptions,
  DPOExportOptions,
  GauntletExportOptions,
  TrainingStatsResponse,
  TrainingFormatsResponse,
  TrainingExportResponse,
};

// =============================================================================
// Singleton and React Hook Integration
// =============================================================================

let clientInstance: AragoraClient | null = null;
let currentConfig: { baseUrl: string; apiKey?: string } | null = null;

/**
 * Get or create an AragoraClient instance.
 *
 * This creates a singleton client configured with the provided base URL and token.
 * If the configuration changes, a new client is created.
 *
 * @param token - Optional auth token (Bearer token)
 * @param baseUrl - API base URL (defaults to production)
 */
export function getClient(token?: string, baseUrl = 'https://api.aragora.ai'): AragoraClient {
  const newConfig = { baseUrl, apiKey: token };

  // Check if we need to create a new client
  if (
    !clientInstance ||
    !currentConfig ||
    currentConfig.baseUrl !== newConfig.baseUrl ||
    currentConfig.apiKey !== newConfig.apiKey
  ) {
    clientInstance = new AragoraClient(newConfig);
    currentConfig = newConfig;
  }

  return clientInstance;
}

/**
 * Clear the cached client instance.
 * Call this on logout to ensure fresh client on next login.
 */
export function clearClient(): void {
  clientInstance = null;
  currentConfig = null;
}
