/**
 * Dashboard Namespace API
 *
 * Provides REST APIs for the main dashboard:
 * - Overview stats and metrics
 * - Quick actions
 * - Recent activity
 * - Inbox summary
 */

/**
 * Dashboard stat card.
 */
export interface StatCard {
  id: string;
  title: string;
  value: string;
  change: string;
  change_type: 'increase' | 'decrease' | 'neutral';
  icon: string;
}

/**
 * Dashboard overview response.
 */
export interface DashboardOverview {
  user_id: string;
  generated_at: string;
  inbox: {
    total_unread: number;
    high_priority: number;
    needs_response: number;
    snoozed: number;
    assigned_to_me: number;
  };
  today: {
    emails_received: number;
    emails_sent: number;
    emails_archived: number;
    meetings_scheduled: number;
    action_items_completed: number;
    action_items_created: number;
  };
  team: {
    active_members: number;
    open_tickets: number;
    avg_response_time_mins: number;
    resolved_today: number;
  };
  ai: {
    emails_categorized: number;
    auto_responses_suggested: number;
    priority_predictions: number;
    debates_run: number;
  };
  cards: StatCard[];
}

/**
 * Team member performance.
 */
export interface TeamPerformance {
  name: string;
  resolved: number;
  avg_response: number;
}

/**
 * Top sender info.
 */
export interface TopSender {
  email: string;
  count: number;
  priority: 'high' | 'medium' | 'low';
}

/**
 * Dashboard stats response.
 */
export interface DashboardStats {
  period: 'day' | 'week' | 'month';
  generated_at: string;
  email_volume: {
    labels: string[];
    received: number[];
    sent: number[];
    archived: number[];
  };
  response_time: {
    labels: string[];
    values: number[];
  };
  priority_distribution: {
    labels: string[];
    values: number[];
  };
  categories: {
    labels: string[];
    values: number[];
  };
  team_performance: TeamPerformance[];
  top_senders: TopSender[];
  summary: {
    total_emails: number;
    avg_daily_emails: number;
    response_rate: number;
    avg_response_time_mins: number;
    ai_accuracy: number;
  };
}

/**
 * Activity item.
 */
export interface ActivityItem {
  id: string;
  type: string;
  title: string;
  description: string;
  timestamp: string;
  priority: 'high' | 'medium' | 'low';
  icon: string;
}

/**
 * Label info.
 */
export interface LabelInfo {
  name: string;
  count: number;
  color: string;
}

/**
 * Urgent email preview.
 */
export interface UrgentEmail {
  id: string;
  subject: string;
  from: string;
  received_at: string;
  snippet: string;
}

/**
 * Pending action item.
 */
export interface PendingAction {
  id: string;
  title: string;
  deadline: string;
  from_email: string;
}

/**
 * Inbox summary response.
 */
export interface InboxSummary {
  generated_at: string;
  counts: {
    unread: number;
    starred: number;
    snoozed: number;
    drafts: number;
    trash: number;
  };
  by_priority: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
  by_category: {
    inbox: number;
    updates: number;
    promotions: number;
    social: number;
    forums: number;
  };
  top_labels: LabelInfo[];
  urgent_emails: UrgentEmail[];
  pending_actions: PendingAction[];
}

/**
 * Quick action definition.
 */
export interface QuickAction {
  id: string;
  name: string;
  description: string;
  icon: string;
  available: boolean;
  estimated_count: number | null;
}

/**
 * Quick action result.
 */
export interface QuickActionResult {
  action_id: string;
  executed: boolean;
  timestamp: string;
  affected_count?: number;
  message: string;
  [key: string]: unknown;
}

/**
 * Dashboard API for overview and quick actions.
 */
export class DashboardAPI {
  private baseUrl: string;
  private headers: HeadersInit;

  constructor(baseUrl: string, apiKey: string) {
    this.baseUrl = baseUrl;
    this.headers = {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    };
  }

  /**
   * Get dashboard overview.
   *
   * @param refresh - Force refresh cache
   */
  async getOverview(refresh = false): Promise<DashboardOverview> {
    const url = `${this.baseUrl}/api/v1/dashboard${refresh ? '?refresh=true' : ''}`;
    const response = await fetch(url, {
      method: 'GET',
      headers: this.headers,
    });
    if (!response.ok) throw new Error(`Failed to get dashboard: ${response.statusText}`);
    return response.json();
  }

  /**
   * Get detailed statistics.
   *
   * @param period - Time period (day, week, month)
   */
  async getStats(period: 'day' | 'week' | 'month' = 'week'): Promise<DashboardStats> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/dashboard/stats?period=${period}`,
      {
        method: 'GET',
        headers: this.headers,
      }
    );
    if (!response.ok) throw new Error(`Failed to get stats: ${response.statusText}`);
    return response.json();
  }

  /**
   * Get recent activity.
   *
   * @param options - Pagination and filter options
   */
  async getActivity(options?: {
    limit?: number;
    offset?: number;
    type?: string;
  }): Promise<{
    activities: ActivityItem[];
    total: number;
    limit: number;
    offset: number;
    has_more: boolean;
  }> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', options.limit.toString());
    if (options?.offset) params.set('offset', options.offset.toString());
    if (options?.type) params.set('type', options.type);

    const url = `${this.baseUrl}/api/v1/dashboard/activity${params.toString() ? `?${params}` : ''}`;
    const response = await fetch(url, {
      method: 'GET',
      headers: this.headers,
    });
    if (!response.ok) throw new Error(`Failed to get activity: ${response.statusText}`);
    return response.json();
  }

  /**
   * Get inbox summary for dashboard.
   */
  async getInboxSummary(): Promise<InboxSummary> {
    const response = await fetch(`${this.baseUrl}/api/v1/dashboard/inbox-summary`, {
      method: 'GET',
      headers: this.headers,
    });
    if (!response.ok)
      throw new Error(`Failed to get inbox summary: ${response.statusText}`);
    return response.json();
  }

  /**
   * Get available quick actions.
   */
  async getQuickActions(): Promise<{ actions: QuickAction[]; count: number }> {
    const response = await fetch(`${this.baseUrl}/api/v1/dashboard/quick-actions`, {
      method: 'GET',
      headers: this.headers,
    });
    if (!response.ok)
      throw new Error(`Failed to get quick actions: ${response.statusText}`);
    return response.json();
  }

  /**
   * Execute a quick action.
   *
   * @param actionId - Action ID to execute
   * @param options - Action-specific options
   */
  async executeQuickAction(
    actionId: string,
    options?: {
      confirm?: boolean;
      options?: Record<string, unknown>;
    }
  ): Promise<QuickActionResult> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/dashboard/quick-actions/${actionId}`,
      {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify({
          confirm: options?.confirm ?? true,
          options: options?.options,
        }),
      }
    );
    if (!response.ok)
      throw new Error(`Failed to execute action: ${response.statusText}`);
    return response.json();
  }
}
