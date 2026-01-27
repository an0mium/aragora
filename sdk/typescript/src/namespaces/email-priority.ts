/**
 * Email Priority Namespace API
 *
 * Provides email prioritization and inbox management:
 * - Email priority scoring with ML/LLM tiers
 * - Inbox ranking and categorization
 * - User feedback for learning
 * - VIP sender management
 * - Gmail OAuth integration
 * - Cross-channel context boosts
 */

/**
 * Priority scoring tier.
 */
export type ScoringTier = 'tier_1_rules' | 'tier_2_ml' | 'tier_3_llm';

/**
 * User action types for feedback.
 */
export type UserAction =
  | 'read'
  | 'archived'
  | 'deleted'
  | 'replied'
  | 'starred'
  | 'important'
  | 'snoozed';

/**
 * Email category.
 */
export type EmailCategory =
  | 'invoices'
  | 'receipts'
  | 'newsletters'
  | 'promotions'
  | 'personal'
  | 'work'
  | 'other';

/**
 * Gmail OAuth scopes.
 */
export type GmailScopes = 'readonly' | 'full';

/**
 * Email message for prioritization.
 */
export interface EmailMessage {
  id: string;
  thread_id?: string;
  subject: string;
  from_address: string;
  to_addresses?: string[];
  cc_addresses?: string[];
  bcc_addresses?: string[];
  date?: string;
  body_text?: string;
  body_html?: string;
  snippet?: string;
  labels?: string[];
  headers?: Record<string, string>;
  is_read?: boolean;
  is_starred?: boolean;
  is_important?: boolean;
  has_attachments?: boolean;
}

/**
 * Priority factor contributing to score.
 */
export interface PriorityFactor {
  name: string;
  weight: number;
  value: number;
  reason: string;
}

/**
 * Priority scoring result.
 */
export interface PriorityResult {
  email_id: string;
  priority_score: number;
  confidence: number;
  tier_used?: ScoringTier;
  rationale: string;
  factors?: PriorityFactor[];
}

/**
 * Category result.
 */
export interface CategoryResult {
  email_id: string;
  category: EmailCategory;
  confidence: number;
  sub_category?: string;
}

/**
 * Category statistics.
 */
export interface CategoryStats {
  total: number;
  by_category: Record<EmailCategory, number>;
  avg_confidence: number;
}

/**
 * Email configuration.
 */
export interface EmailConfig {
  vip_domains: string[];
  vip_addresses: string[];
  internal_domains: string[];
  auto_archive_senders: string[];
  tier_1_confidence_threshold: number;
  tier_2_confidence_threshold: number;
  enable_slack_signals: boolean;
  enable_calendar_signals: boolean;
  enable_drive_signals: boolean;
}

/**
 * Feedback item for batch recording.
 */
export interface FeedbackItem {
  email_id: string;
  action: UserAction;
  email?: EmailMessage;
  response_time_minutes?: number;
}

/**
 * Inbox item with email and priority.
 */
export interface InboxItem {
  email: EmailMessage;
  priority: PriorityResult;
}

/**
 * Inbox fetch parameters.
 */
export interface InboxParams {
  limit?: number;
  offset?: number;
  labels?: string[];
  unread_only?: boolean;
}

/**
 * Cross-channel context.
 */
export interface CrossChannelContext {
  email_address: string;
  slack_activity?: {
    recent_messages: number;
    active_channels: string[];
    last_active: string;
  };
  calendar_events?: {
    upcoming_meetings: number;
    shared_meetings: number;
  };
  drive_activity?: {
    shared_documents: number;
    recent_collaborations: number;
  };
}

/**
 * Context-based priority boost.
 */
export interface ContextBoost {
  email_id: string;
  total_boost: number;
  slack_activity_boost: number;
  drive_relevance_boost: number;
  calendar_urgency_boost: number;
  slack_reason?: string;
  drive_reason?: string;
  calendar_reason?: string;
  related_slack_channels?: string[];
  related_drive_files?: string[];
  related_meetings?: string[];
}

/**
 * Client interface for email operations.
 */
interface EmailPriorityClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Email Priority API namespace.
 *
 * Provides methods for email prioritization and inbox management:
 * - Score and rank emails by priority
 * - Categorize emails into smart folders
 * - Record user feedback for ML learning
 * - Manage VIP senders
 * - Connect Gmail via OAuth
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Score a single email
 * const result = await client.emailPriority.prioritize(email);
 *
 * // Rank entire inbox
 * const ranked = await client.emailPriority.rankInbox(emails);
 *
 * // Record user action for learning
 * await client.emailPriority.recordFeedback('msg_123', 'replied');
 * ```
 */
export class EmailPriorityAPI {
  constructor(private client: EmailPriorityClientInterface) {}

  // =========================================================================
  // Priority Scoring
  // =========================================================================

  /**
   * Score a single email for priority.
   */
  async prioritize(
    email: EmailMessage,
    forceTier?: ScoringTier
  ): Promise<{ success: boolean; result: PriorityResult }> {
    return this.client.request('POST', '/api/v1/email/prioritize', {
      json: { email, force_tier: forceTier },
    });
  }

  /**
   * Rank multiple emails by priority.
   */
  async rankInbox(
    emails: EmailMessage[],
    limit?: number
  ): Promise<{ success: boolean; results: PriorityResult[]; total: number }> {
    return this.client.request('POST', '/api/v1/email/rank-inbox', {
      json: { emails, limit },
    });
  }

  // =========================================================================
  // Feedback and Learning
  // =========================================================================

  /**
   * Record user action for ML learning.
   */
  async recordFeedback(
    emailId: string,
    action: UserAction,
    email?: EmailMessage
  ): Promise<{ success: boolean }> {
    return this.client.request('POST', '/api/v1/email/feedback', {
      json: { email_id: emailId, action, email },
    });
  }

  /**
   * Record batch of user actions.
   */
  async recordFeedbackBatch(
    items: FeedbackItem[]
  ): Promise<{ recorded: number; errors: number }> {
    return this.client.request('POST', '/api/v1/email/feedback/batch', {
      json: { items },
    });
  }

  // =========================================================================
  // Categorization
  // =========================================================================

  /**
   * Categorize an email to a smart folder.
   */
  async categorize(
    email: EmailMessage
  ): Promise<{ success: boolean; result: CategoryResult }> {
    return this.client.request('POST', '/api/v1/email/categorize', {
      json: { email },
    });
  }

  /**
   * Categorize multiple emails.
   */
  async categorizeBatch(
    emails: EmailMessage[],
    concurrency?: number
  ): Promise<{ results: CategoryResult[]; stats: CategoryStats }> {
    return this.client.request('POST', '/api/v1/email/categorize/batch', {
      json: { emails, concurrency },
    });
  }

  /**
   * Apply Gmail label based on category.
   */
  async applyLabel(
    emailId: string,
    category: EmailCategory
  ): Promise<{ success: boolean; label_applied: string }> {
    return this.client.request('POST', '/api/v1/email/categorize/apply-label', {
      json: { email_id: emailId, category },
    });
  }

  // =========================================================================
  // Inbox Management
  // =========================================================================

  /**
   * Fetch and rank inbox emails.
   */
  async fetchInbox(
    params?: InboxParams
  ): Promise<{ success: boolean; inbox: InboxItem[]; total: number; fetched_at: string }> {
    return this.client.request('GET', '/api/v1/email/inbox', {
      params: params as Record<string, unknown>,
    });
  }

  // =========================================================================
  // Configuration
  // =========================================================================

  /**
   * Get email prioritization configuration.
   */
  async getConfig(): Promise<{ success: boolean; config: EmailConfig }> {
    return this.client.request('GET', '/api/v1/email/config');
  }

  /**
   * Update email prioritization configuration.
   */
  async updateConfig(
    updates: Partial<EmailConfig>
  ): Promise<{ success: boolean; config: EmailConfig }> {
    return this.client.request('PUT', '/api/v1/email/config', {
      json: updates,
    });
  }

  // =========================================================================
  // VIP Management
  // =========================================================================

  /**
   * Add a VIP email address or domain.
   */
  async addVip(options: {
    email?: string;
    domain?: string;
  }): Promise<{ success: boolean; message: string }> {
    return this.client.request('POST', '/api/v1/email/vip', {
      json: options,
    });
  }

  /**
   * Remove a VIP email address or domain.
   */
  async removeVip(options: {
    email?: string;
    domain?: string;
  }): Promise<{ success: boolean; message: string }> {
    return this.client.request('DELETE', '/api/v1/email/vip', {
      json: options,
    });
  }

  // =========================================================================
  // Gmail OAuth
  // =========================================================================

  /**
   * Get Gmail OAuth authorization URL.
   */
  async getGmailOAuthUrl(
    redirectUri: string,
    state?: string,
    scopes?: GmailScopes
  ): Promise<{ success: boolean; oauth_url: string; scopes: GmailScopes }> {
    return this.client.request('POST', '/api/v1/email/gmail/oauth/url', {
      json: { redirect_uri: redirectUri, state, scopes },
    });
  }

  /**
   * Handle Gmail OAuth callback.
   */
  async handleGmailOAuthCallback(
    code: string,
    redirectUri: string
  ): Promise<{ success: boolean; email?: string; message: string }> {
    return this.client.request('POST', '/api/v1/email/gmail/oauth/callback', {
      json: { code, redirect_uri: redirectUri },
    });
  }

  /**
   * Check Gmail connection status.
   */
  async getGmailStatus(): Promise<{
    authenticated: boolean;
    email?: string;
    scopes?: string[];
    expires_at?: string;
  }> {
    return this.client.request('GET', '/api/v1/email/gmail/status');
  }

  // =========================================================================
  // Cross-Channel Context
  // =========================================================================

  /**
   * Get cross-channel context for an email address.
   */
  async getContext(emailAddress: string): Promise<{ context: CrossChannelContext }> {
    return this.client.request('GET', `/api/v1/email/context/${encodeURIComponent(emailAddress)}`);
  }

  /**
   * Get context-based priority boosts for an email.
   */
  async getContextBoost(email: EmailMessage): Promise<{ success: boolean; boost: ContextBoost }> {
    return this.client.request('POST', '/api/v1/email/context/boost', {
      json: { email },
    });
  }
}
