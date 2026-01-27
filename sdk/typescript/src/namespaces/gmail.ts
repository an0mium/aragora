/**
 * Gmail Namespace API
 *
 * Provides a namespaced interface for Gmail connector integration.
 * Enables email-triggered debates, triage rules, and email processing.
 */

import type { PaginationParams } from '../types';

// =============================================================================
// Type Definitions
// =============================================================================

/**
 * Gmail connection status.
 */
export interface GmailConnection {
  id: string;
  email: string;
  status: 'connected' | 'disconnected' | 'error';
  scopes: string[];
  connected_at?: string;
  last_synced_at?: string;
}

/**
 * Email triage rule.
 */
export interface EmailTriageRule {
  id: string;
  name: string;
  enabled: boolean;
  conditions: Record<string, unknown>;
  actions: string[];
  priority: number;
}

/**
 * Email debate configuration.
 */
export interface EmailDebateConfig {
  id: string;
  name: string;
  enabled: boolean;
  trigger_conditions: Record<string, unknown>;
  debate_template?: string;
  agents: string[];
  auto_reply: boolean;
}

/**
 * Processed email record.
 */
export interface ProcessedEmail {
  id: string;
  message_id: string;
  subject: string;
  sender: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  debate_id?: string;
  processed_at?: string;
  summary?: string;
}

/**
 * Gmail processing statistics.
 */
export interface GmailStats {
  total_processed: number;
  debates_triggered: number;
  auto_replies_sent: number;
  errors: number;
  avg_processing_time_ms: number;
}

/**
 * Processed emails list response.
 */
export interface ProcessedEmailList {
  emails: ProcessedEmail[];
  total: number;
}

/**
 * OAuth initiation response.
 */
export interface OAuthInitResponse {
  auth_url: string;
  state: string;
}

/**
 * Sync result response.
 */
export interface SyncResult {
  synced: boolean;
  messages_processed?: number;
  last_synced_at?: string;
}

/**
 * Create triage rule request.
 */
export interface CreateTriageRuleRequest {
  name: string;
  conditions: Record<string, unknown>;
  actions: string[];
  priority?: number;
}

/**
 * Update triage rule request.
 */
export interface UpdateTriageRuleRequest {
  name?: string;
  conditions?: Record<string, unknown>;
  actions?: string[];
  enabled?: boolean;
  priority?: number;
}

/**
 * Create debate config request.
 */
export interface CreateDebateConfigRequest {
  name: string;
  trigger_conditions: Record<string, unknown>;
  agents?: string[];
  debate_template?: string;
  auto_reply?: boolean;
}

// =============================================================================
// Client Interface
// =============================================================================

interface GmailClientInterface {
  getGmailConnection(): Promise<GmailConnection | null>;
  initiateGmailConnection(redirectUri?: string): Promise<OAuthInitResponse>;
  completeGmailConnection(code: string, state: string): Promise<GmailConnection>;
  disconnectGmail(): Promise<{ disconnected: boolean }>;
  syncGmail(): Promise<SyncResult>;
  listTriageRules(): Promise<{ rules: EmailTriageRule[] }>;
  createTriageRule(body: CreateTriageRuleRequest): Promise<EmailTriageRule>;
  updateTriageRule(ruleId: string, body: UpdateTriageRuleRequest): Promise<EmailTriageRule>;
  deleteTriageRule(ruleId: string): Promise<{ deleted: boolean }>;
  listDebateConfigs(): Promise<{ configs: EmailDebateConfig[] }>;
  createDebateConfig(body: CreateDebateConfigRequest): Promise<EmailDebateConfig>;
  listProcessedEmails(params?: PaginationParams & { status?: string }): Promise<ProcessedEmailList>;
  getGmailStats(): Promise<GmailStats>;
}

// =============================================================================
// Gmail API Class
// =============================================================================

/**
 * Gmail API namespace.
 *
 * Provides methods for Gmail integration:
 * - Account connection via OAuth
 * - Email triage rules
 * - Debate-triggering configurations
 * - Processed email history
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Connect Gmail
 * const { auth_url, state } = await client.gmail.initiateConnection();
 * // Redirect user to auth_url...
 *
 * // After OAuth callback
 * const connection = await client.gmail.completeConnection(code, state);
 * console.log(`Connected: ${connection.email}`);
 *
 * // Create a triage rule
 * const rule = await client.gmail.createTriageRule({
 *   name: 'Urgent emails',
 *   conditions: { subject_contains: 'URGENT' },
 *   actions: ['trigger_debate', 'notify']
 * });
 *
 * // Check processing stats
 * const stats = await client.gmail.getStats();
 * console.log(`Processed ${stats.total_processed} emails`);
 * ```
 */
export class GmailAPI {
  constructor(private client: GmailClientInterface) {}

  // ===========================================================================
  // Connection Management
  // ===========================================================================

  /**
   * Get current Gmail connection status.
   *
   * @returns Connection object if connected, null otherwise.
   */
  async getConnection(): Promise<GmailConnection | null> {
    return this.client.getGmailConnection();
  }

  /**
   * Initiate Gmail OAuth connection.
   *
   * @param redirectUri - Optional redirect URI after OAuth.
   * @returns OAuth authorization URL and state.
   *
   * @example
   * ```typescript
   * const { auth_url, state } = await client.gmail.initiateConnection();
   * // Store state in session, redirect user to auth_url
   * window.location.href = auth_url;
   * ```
   */
  async initiateConnection(redirectUri?: string): Promise<OAuthInitResponse> {
    return this.client.initiateGmailConnection(redirectUri);
  }

  /**
   * Complete Gmail OAuth connection.
   *
   * @param code - OAuth authorization code from callback.
   * @param state - OAuth state parameter from callback.
   * @returns Connected Gmail account details.
   */
  async completeConnection(code: string, state: string): Promise<GmailConnection> {
    return this.client.completeGmailConnection(code, state);
  }

  /**
   * Disconnect Gmail account.
   */
  async disconnect(): Promise<{ disconnected: boolean }> {
    return this.client.disconnectGmail();
  }

  /**
   * Trigger manual email sync.
   */
  async sync(): Promise<SyncResult> {
    return this.client.syncGmail();
  }

  // ===========================================================================
  // Triage Rules
  // ===========================================================================

  /**
   * List email triage rules.
   */
  async listTriageRules(): Promise<EmailTriageRule[]> {
    const response = await this.client.listTriageRules();
    return response.rules;
  }

  /**
   * Create an email triage rule.
   *
   * @example
   * ```typescript
   * const rule = await client.gmail.createTriageRule({
   *   name: 'VIP Customers',
   *   conditions: { from_domain: 'bigcustomer.com' },
   *   actions: ['priority_high', 'trigger_debate'],
   *   priority: 10
   * });
   * ```
   */
  async createTriageRule(request: CreateTriageRuleRequest): Promise<EmailTriageRule> {
    return this.client.createTriageRule(request);
  }

  /**
   * Update an email triage rule.
   */
  async updateTriageRule(ruleId: string, request: UpdateTriageRuleRequest): Promise<EmailTriageRule> {
    return this.client.updateTriageRule(ruleId, request);
  }

  /**
   * Delete an email triage rule.
   */
  async deleteTriageRule(ruleId: string): Promise<{ deleted: boolean }> {
    return this.client.deleteTriageRule(ruleId);
  }

  // ===========================================================================
  // Debate Configurations
  // ===========================================================================

  /**
   * List email debate configurations.
   */
  async listDebateConfigs(): Promise<EmailDebateConfig[]> {
    const response = await this.client.listDebateConfigs();
    return response.configs;
  }

  /**
   * Create an email debate configuration.
   *
   * @example
   * ```typescript
   * const config = await client.gmail.createDebateConfig({
   *   name: 'Contract Review',
   *   trigger_conditions: { subject_contains: 'contract' },
   *   agents: ['claude', 'gpt-4'],
   *   auto_reply: true
   * });
   * ```
   */
  async createDebateConfig(request: CreateDebateConfigRequest): Promise<EmailDebateConfig> {
    return this.client.createDebateConfig(request);
  }

  // ===========================================================================
  // Processed Emails
  // ===========================================================================

  /**
   * List processed emails with pagination.
   *
   * @param params.status - Filter by processing status.
   * @param params.limit - Maximum number of results.
   * @param params.offset - Offset for pagination.
   */
  async listProcessedEmails(params?: PaginationParams & { status?: string }): Promise<ProcessedEmailList> {
    return this.client.listProcessedEmails(params);
  }

  /**
   * Get Gmail processing statistics.
   */
  async getStats(): Promise<GmailStats> {
    return this.client.getGmailStats();
  }
}
