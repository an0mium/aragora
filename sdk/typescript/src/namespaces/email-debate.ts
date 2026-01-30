/**
 * Email Debate Namespace API
 *
 * Provides endpoints for multi-agent email prioritization and triage.
 */

import type { AragoraClient } from '../client';

/**
 * Email input for prioritization
 */
export interface EmailInput {
  subject?: string;
  body?: string;
  sender: string;
  received_at?: string;
  message_id?: string;
  user_id?: string;
  thread_id?: string;
  labels?: string[];
  attachments?: Array<{
    filename: string;
    mime_type: string;
    size_bytes: number;
  }>;
}

/**
 * Priority level for emails
 */
export type PriorityLevel = 'critical' | 'high' | 'medium' | 'low' | 'none';

/**
 * Email prioritization result
 */
export interface PrioritizationResult {
  message_id?: string;
  priority: PriorityLevel;
  score: number;
  confidence: number;
  reasoning: string;
  suggested_actions: string[];
  category?: string;
  response_urgency?: 'immediate' | 'same_day' | 'this_week' | 'when_possible';
  agents_consulted: string[];
  consensus_reached: boolean;
  debate_id?: string;
}

/**
 * Batch prioritization result
 */
export interface BatchPrioritizationResult {
  results: PrioritizationResult[];
  total_processed: number;
  processing_time_ms: number;
  summary: {
    critical: number;
    high: number;
    medium: number;
    low: number;
    none: number;
  };
}

/**
 * Email category for triage
 */
export type EmailCategory =
  | 'action_required'
  | 'fyi'
  | 'meeting'
  | 'follow_up'
  | 'newsletter'
  | 'spam'
  | 'personal'
  | 'finance'
  | 'legal'
  | 'other';

/**
 * Triage result for an email
 */
export interface TriageResult {
  message_id?: string;
  category: EmailCategory;
  priority: PriorityLevel;
  suggested_folder?: string;
  suggested_labels?: string[];
  auto_reply_suggested: boolean;
  auto_reply_draft?: string;
  delegate_to?: string;
  snooze_until?: string;
  reasoning: string;
}

/**
 * Inbox triage response
 */
export interface InboxTriageResponse {
  results: TriageResult[];
  total_triaged: number;
  processing_time_ms: number;
  summary: {
    by_category: Record<EmailCategory, number>;
    by_priority: Record<PriorityLevel, number>;
    auto_replies_suggested: number;
    delegations_suggested: number;
  };
}

/**
 * Email Debate namespace for AI-powered email prioritization.
 *
 * @example
 * ```typescript
 * // Prioritize a single email
 * const result = await client.emailDebate.prioritize({
 *   subject: 'Q4 Budget Review - Action Required',
 *   body: 'Please review the attached budget proposal...',
 *   sender: 'cfo@company.com'
 * });
 * console.log(`Priority: ${result.priority}, Score: ${result.score}`);
 *
 * // Batch prioritize inbox
 * const batch = await client.emailDebate.prioritizeBatch([
 *   { subject: 'Meeting tomorrow', sender: 'bob@example.com' },
 *   { subject: 'Invoice #1234', sender: 'billing@vendor.com' }
 * ]);
 * console.log(`Processed ${batch.total_processed} emails`);
 *
 * // Full inbox triage
 * const triage = await client.emailDebate.triageInbox(emails);
 * triage.results.forEach(r => {
 *   console.log(`${r.category}: ${r.priority} - ${r.suggested_folder}`);
 * });
 * ```
 */
export class EmailDebateNamespace {
  constructor(private client: AragoraClient) {}

  /**
   * Prioritize a single email using multi-agent debate.
   *
   * Agents analyze the email content, sender reputation, and context
   * to determine priority and suggest actions.
   *
   * @param email - Email to prioritize
   * @param options.userId - User ID for personalized prioritization
   */
  async prioritize(
    email: EmailInput,
    options?: { userId?: string }
  ): Promise<PrioritizationResult> {
    const body = { ...email };
    if (options?.userId) {
      body.user_id = options.userId;
    }
    return this.client.request<PrioritizationResult>('POST', '/api/v1/email/prioritize', body);
  }

  /**
   * Prioritize multiple emails in batch.
   *
   * More efficient than calling prioritize() multiple times.
   *
   * @param emails - Array of emails to prioritize
   * @param options.userId - User ID for personalized prioritization
   * @param options.parallel - Process emails in parallel (default: true)
   */
  async prioritizeBatch(
    emails: EmailInput[],
    options?: { userId?: string; parallel?: boolean }
  ): Promise<BatchPrioritizationResult> {
    const body = {
      emails,
      user_id: options?.userId,
      parallel: options?.parallel ?? true,
    };
    return this.client.request<BatchPrioritizationResult>(
      'POST',
      '/api/v1/email/prioritize/batch',
      body
    );
  }

  /**
   * Triage inbox with full categorization.
   *
   * Provides comprehensive email triage including:
   * - Category assignment (action_required, fyi, meeting, etc.)
   * - Priority scoring
   * - Folder suggestions
   * - Auto-reply drafts
   * - Delegation recommendations
   *
   * @param emails - Emails to triage
   * @param options.userId - User ID for personalized triage
   * @param options.includeAutoReplies - Generate auto-reply drafts
   */
  async triageInbox(
    emails: EmailInput[],
    options?: { userId?: string; includeAutoReplies?: boolean }
  ): Promise<InboxTriageResponse> {
    const body = {
      emails,
      user_id: options?.userId,
      include_auto_replies: options?.includeAutoReplies ?? false,
    };
    return this.client.request<InboxTriageResponse>('POST', '/api/v1/email/triage', body);
  }

  /**
   * Get prioritization history for a user.
   *
   * @param options.userId - User ID
   * @param options.limit - Maximum results (default: 50)
   * @param options.since - Filter to emails since this timestamp
   */
  async getHistory(options: {
    userId: string;
    limit?: number;
    since?: string;
  }): Promise<PrioritizationResult[]> {
    const params = new URLSearchParams();
    params.set('user_id', options.userId);
    if (options.limit) params.set('limit', options.limit.toString());
    if (options.since) params.set('since', options.since);

    return this.client.request<PrioritizationResult[]>(
      'GET',
      `/api/v1/email/prioritize/history?${params.toString()}`
    );
  }
}
