/**
 * Dashboard Namespace API
 *
 * Provides endpoints for dashboard views, KPIs, analytics, and monitoring.
 * Includes debate tracking, team performance, email analytics, and activity feeds.
 */
import type { PaginationParams } from '../types';

// =============================================================================
// Types
// =============================================================================

/** Debate entry displayed on dashboard */
export interface DashboardDebateEntry {
  debate_id: string;
  task?: string;
  status?: 'pending' | 'active' | 'completed' | 'failed';
  consensus_reached?: boolean;
  agent_count?: number;
  round_count?: number;
  created_at?: string;
  updated_at?: string;
  duration_ms?: number;
}

/** Dashboard stat card */
export interface StatCard {
  id: string;
  label: string;
  value: number | string;
  change?: number;
  change_period?: string;
  trend?: 'up' | 'down' | 'stable';
  icon?: string;
  color?: string;
}

/** Dashboard overview summary */
export interface DashboardOverview {
  stats: StatCard[];
  recent_debates: DashboardDebateEntry[];
  active_debates: number;
  total_debates_today: number;
  consensus_rate: number;
  avg_debate_duration_ms: number;
  system_health: 'healthy' | 'degraded' | 'down';
  last_updated: string;
}

/** Dashboard statistics */
export interface DashboardStats {
  debates: {
    total: number;
    today: number;
    this_week: number;
    this_month: number;
    by_status: Record<string, number>;
  };
  agents: {
    total: number;
    active: number;
    by_provider: Record<string, number>;
  };
  performance: {
    avg_response_time_ms: number;
    success_rate: number;
    consensus_rate: number;
    error_rate: number;
  };
  usage: {
    api_calls_today: number;
    tokens_used_today: number;
    storage_used_bytes: number;
  };
}

/** Team performance metrics */
export interface TeamPerformance {
  team_id: string;
  team_name: string;
  member_count: number;
  debates_participated: number;
  avg_response_time_ms: number;
  consensus_contribution_rate: number;
  quality_score: number;
  elo_rating?: number;
  rank?: number;
}

/** Top email sender analytics */
export interface TopSender {
  email_address: string;
  display_name?: string;
  message_count: number;
  avg_importance_score: number;
  response_rate: number;
  avg_response_time_hours?: number;
  domain: string;
  last_message_at: string;
}

/** Activity item in timeline */
export interface ActivityItem {
  id: string;
  type: 'debate_started' | 'debate_completed' | 'consensus_reached' | 'agent_joined' | 'agent_left' | 'error' | 'system' | 'user_action';
  title: string;
  description?: string;
  actor?: string;
  metadata?: Record<string, unknown>;
  timestamp: string;
  debate_id?: string;
}

/** Email label information */
export interface LabelInfo {
  id: string;
  name: string;
  message_count: number;
  unread_count: number;
  color?: string;
  type: 'system' | 'user';
}

/** Urgent email item */
export interface UrgentEmail {
  id: string;
  subject: string;
  from_address: string;
  from_name?: string;
  snippet: string;
  received_at: string;
  importance_score: number;
  importance_reason: string;
  requires_action: boolean;
  action_type?: 'reply' | 'review' | 'approve' | 'forward';
  deadline?: string;
  thread_id: string;
  label_ids: string[];
}

/** Pending action item */
export interface PendingAction {
  id: string;
  type: 'approval' | 'review' | 'response' | 'task' | 'decision';
  title: string;
  description?: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  due_date?: string;
  source_type: 'email' | 'debate' | 'workflow' | 'manual';
  source_id?: string;
  created_at: string;
  assignee?: string;
}

/** Inbox summary statistics */
export interface InboxSummary {
  total_messages: number;
  unread_messages: number;
  urgent_count: number;
  today_count: number;
  by_label: LabelInfo[];
  by_importance: {
    high: number;
    medium: number;
    low: number;
  };
  response_rate: number;
  avg_response_time_hours: number;
}

/** Quick action definition */
export interface QuickAction {
  id: string;
  name: string;
  description: string;
  icon?: string;
  category: 'email' | 'debate' | 'workflow' | 'system';
  requires_confirmation: boolean;
  parameters?: Array<{
    name: string;
    type: 'string' | 'number' | 'boolean' | 'select';
    required: boolean;
    options?: string[];
    default?: unknown;
  }>;
}

/** Quick action execution result */
export interface QuickActionResult {
  success: boolean;
  action_id: string;
  result?: unknown;
  message?: string;
  error?: string;
  executed_at: string;
}

/** List debates parameters */
export interface ListDebatesParams extends PaginationParams {
  status?: 'pending' | 'active' | 'completed' | 'failed';
  start_date?: string;
  end_date?: string;
}

/** List activity parameters */
export interface ListActivityParams extends PaginationParams {
  type?: ActivityItem['type'];
  debate_id?: string;
  start_date?: string;
  end_date?: string;
}

/** Team performance filter parameters */
export interface TeamPerformanceParams extends PaginationParams {
  sort_by?: 'quality_score' | 'elo_rating' | 'debates_participated';
  sort_order?: 'asc' | 'desc';
  min_debates?: number;
}

/** Top senders filter parameters */
export interface TopSendersParams extends PaginationParams {
  domain?: string;
  min_messages?: number;
  sort_by?: 'message_count' | 'importance_score' | 'response_rate';
}

/** Urgent items filter parameters */
export interface UrgentItemsParams extends PaginationParams {
  action_type?: UrgentEmail['action_type'];
  min_importance?: number;
  include_deadline_passed?: boolean;
}

// =============================================================================
// Client Interface
// =============================================================================

interface DashboardClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; body?: unknown }
  ): Promise<T>;
}

// =============================================================================
// Dashboard API Class
// =============================================================================

/**
 * Dashboard API provides endpoints for dashboard views and analytics.
 *
 * @example
 * ```typescript
 * const client = new AragoraClient({ apiKey: 'key' });
 * const dashboard = client.dashboard;
 *
 * // Get dashboard overview
 * const overview = await dashboard.getOverview();
 *
 * // Get team performance
 * const teams = await dashboard.getTeamPerformance({ limit: 10 });
 *
 * // Get urgent items
 * const urgent = await dashboard.getUrgentItems({ min_importance: 0.8 });
 * ```
 */
export class DashboardAPI {
  constructor(private client: DashboardClientInterface) {}

  // ---------------------------------------------------------------------------
  // Debates
  // ---------------------------------------------------------------------------

  /**
   * List recent debates for dashboard display.
   *
   * @param params - Pagination and filter options
   * @returns List of debates with total count
   */
  async listDebates(
    params?: ListDebatesParams
  ): Promise<{ debates: DashboardDebateEntry[]; total: number }> {
    return this.client.request('GET', '/api/v1/dashboard/debates', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get a specific debate's summary.
   *
   * @param debateId - The debate ID
   * @returns Debate summary
   */
  async getDebate(debateId: string): Promise<DashboardDebateEntry> {
    return this.client.request('GET', `/api/v1/dashboard/debates/${debateId}`);
  }

  // ---------------------------------------------------------------------------
  // Overview and Stats
  // ---------------------------------------------------------------------------

  /**
   * Get dashboard overview with KPIs and summary stats.
   *
   * @returns Dashboard overview data
   */
  async getOverview(): Promise<DashboardOverview> {
    return this.client.request('GET', '/api/v1/dashboard/overview');
  }

  /**
   * Get detailed dashboard statistics.
   *
   * @returns Comprehensive statistics
   */
  async getStats(): Promise<DashboardStats> {
    return this.client.request('GET', '/api/v1/dashboard/stats');
  }

  /**
   * Get stat cards for dashboard widgets.
   *
   * @returns Array of stat cards
   */
  async getStatCards(): Promise<{ cards: StatCard[] }> {
    return this.client.request('GET', '/api/v1/dashboard/stat-cards');
  }

  // ---------------------------------------------------------------------------
  // Team Performance
  // ---------------------------------------------------------------------------

  /**
   * Get team performance metrics.
   *
   * @param params - Filter and sort options
   * @returns Team performance data
   */
  async getTeamPerformance(
    params?: TeamPerformanceParams
  ): Promise<{ teams: TeamPerformance[]; total: number }> {
    return this.client.request('GET', '/api/v1/dashboard/team-performance', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get performance metrics for a specific team.
   *
   * @param teamId - The team ID
   * @returns Team performance data
   */
  async getTeamById(teamId: string): Promise<TeamPerformance> {
    return this.client.request('GET', `/api/v1/dashboard/team-performance/${teamId}`);
  }

  // ---------------------------------------------------------------------------
  // Email Analytics
  // ---------------------------------------------------------------------------

  /**
   * Get top email senders by volume and importance.
   *
   * @param params - Filter and sort options
   * @returns Top senders data
   */
  async getTopSenders(
    params?: TopSendersParams
  ): Promise<{ senders: TopSender[]; total: number }> {
    return this.client.request('GET', '/api/v1/dashboard/top-senders', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get inbox summary statistics.
   *
   * @returns Inbox summary data
   */
  async getInboxSummary(): Promise<InboxSummary> {
    return this.client.request('GET', '/api/v1/dashboard/inbox-summary');
  }

  /**
   * Get email labels with counts.
   *
   * @returns Label information
   */
  async getLabels(): Promise<{ labels: LabelInfo[] }> {
    return this.client.request('GET', '/api/v1/dashboard/labels');
  }

  // ---------------------------------------------------------------------------
  // Activity Feed
  // ---------------------------------------------------------------------------

  /**
   * Get activity timeline.
   *
   * @param params - Filter and pagination options
   * @returns Activity items
   */
  async getActivity(
    params?: ListActivityParams
  ): Promise<{ activity: ActivityItem[]; total: number }> {
    return this.client.request('GET', '/api/v1/dashboard/activity', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get recent activity (convenience method).
   *
   * @param limit - Maximum items to return (default: 20)
   * @returns Recent activity items
   */
  async getRecentActivity(limit: number = 20): Promise<{ activity: ActivityItem[] }> {
    return this.client.request('GET', '/api/v1/dashboard/activity', {
      params: { limit },
    });
  }

  // ---------------------------------------------------------------------------
  // Urgent Items
  // ---------------------------------------------------------------------------

  /**
   * Get urgent emails requiring attention.
   *
   * @param params - Filter options
   * @returns Urgent email items
   */
  async getUrgentItems(
    params?: UrgentItemsParams
  ): Promise<{ items: UrgentEmail[]; total: number }> {
    return this.client.request('GET', '/api/v1/dashboard/urgent', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Get pending actions requiring user attention.
   *
   * @param params - Pagination options
   * @returns Pending action items
   */
  async getPendingActions(
    params?: PaginationParams
  ): Promise<{ actions: PendingAction[]; total: number }> {
    return this.client.request('GET', '/api/v1/dashboard/pending-actions', {
      params: params as Record<string, unknown>,
    });
  }

  /**
   * Mark an urgent item as handled.
   *
   * @param itemId - The urgent item ID
   * @returns Success status
   */
  async dismissUrgentItem(itemId: string): Promise<{ success: boolean }> {
    return this.client.request('POST', `/api/v1/dashboard/urgent/${itemId}/dismiss`);
  }

  /**
   * Complete a pending action.
   *
   * @param actionId - The action ID
   * @param result - Optional result data
   * @returns Success status
   */
  async completeAction(
    actionId: string,
    result?: Record<string, unknown>
  ): Promise<{ success: boolean }> {
    return this.client.request('POST', `/api/v1/dashboard/pending-actions/${actionId}/complete`, {
      body: result,
    });
  }

  // ---------------------------------------------------------------------------
  // Quick Actions
  // ---------------------------------------------------------------------------

  /**
   * Get available quick actions.
   *
   * @returns Quick action definitions
   */
  async getQuickActions(): Promise<{ actions: QuickAction[] }> {
    return this.client.request('GET', '/api/v1/dashboard/quick-actions');
  }

  /**
   * Execute a quick action.
   *
   * @param actionId - The action ID
   * @param params - Action parameters
   * @returns Execution result
   */
  async executeQuickAction(
    actionId: string,
    params?: Record<string, unknown>
  ): Promise<QuickActionResult> {
    return this.client.request('POST', `/api/v1/dashboard/quick-actions/${actionId}`, {
      body: params,
    });
  }

  // ---------------------------------------------------------------------------
  // Search
  // ---------------------------------------------------------------------------

  /**
   * Search across dashboard data.
   *
   * @param query - Search query
   * @param params - Optional filters
   * @returns Search results
   */
  async search(
    query: string,
    params?: { types?: Array<'debate' | 'email' | 'action'>; limit?: number }
  ): Promise<{
    results: Array<{
      type: 'debate' | 'email' | 'action';
      id: string;
      title: string;
      snippet: string;
      score: number;
    }>;
    total: number;
  }> {
    return this.client.request('GET', '/api/v1/dashboard/search', {
      params: { query, ...params } as Record<string, unknown>,
    });
  }

  // ---------------------------------------------------------------------------
  // Export
  // ---------------------------------------------------------------------------

  /**
   * Export dashboard data.
   *
   * @param format - Export format
   * @param options - Export options
   * @returns Export data or download URL
   */
  async exportData(
    format: 'json' | 'csv' | 'xlsx',
    options?: {
      include?: Array<'debates' | 'stats' | 'activity' | 'emails'>;
      start_date?: string;
      end_date?: string;
    }
  ): Promise<{ url: string; expires_at: string }> {
    return this.client.request('POST', '/api/v1/dashboard/export', {
      body: { format, ...options },
    });
  }
}

export default DashboardAPI;
