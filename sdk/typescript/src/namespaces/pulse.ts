/**
 * Pulse Namespace API
 *
 * Provides access to trending topics from HackerNews, Reddit, Twitter, and other sources.
 * Use Pulse to discover timely debate topics and analyze sentiment across platforms.
 */

/**
 * Supported pulse sources
 */
export type PulseSource = 'hackernews' | 'reddit' | 'twitter' | 'all';

/**
 * Trending topic from pulse sources
 */
export interface TrendingTopic {
  id: string;
  title: string;
  url?: string;
  source: PulseSource;
  score: number;
  comments_count?: number;
  posted_at: string;
  fetched_at: string;
  category?: string;
  tags?: string[];
  sentiment?: 'positive' | 'negative' | 'neutral' | 'mixed';
}

/**
 * Debate suggestion based on trending topic
 */
export interface DebateSuggestion {
  topic: TrendingTopic;
  suggested_task: string;
  suggested_agents: string[];
  relevance_score: number;
  controversy_score: number;
  timeliness_score: number;
}

/**
 * Pulse analytics for debate outcomes on trending topics
 */
export interface PulseAnalytics {
  total_debates: number;
  consensus_rate: number;
  average_rounds: number;
  top_sources: Array<{ source: PulseSource; count: number; consensus_rate: number }>;
  trending_categories: Array<{ category: string; count: number }>;
  period: string;
}

/**
 * Pulse scheduler status
 */
export interface SchedulerStatus {
  running: boolean;
  paused: boolean;
  last_fetch?: string;
  next_fetch?: string;
  fetch_interval_minutes: number;
  topics_fetched: number;
  debates_started: number;
  errors_count: number;
}

/**
 * Scheduler configuration
 */
export interface SchedulerConfig {
  fetch_interval_minutes?: number;
  auto_debate?: boolean;
  min_score?: number;
  max_debates_per_hour?: number;
  sources?: PulseSource[];
  categories?: string[];
}

/**
 * Scheduled debate history entry
 */
export interface ScheduledDebate {
  id: string;
  topic_id: string;
  topic_title: string;
  source: PulseSource;
  debate_id?: string;
  started_at: string;
  completed_at?: string;
  result?: 'consensus' | 'no_consensus' | 'error';
}

/**
 * Interface for the internal client used by PulseAPI.
 */
interface PulseClientInterface {
  request<T>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Pulse API namespace.
 *
 * Provides methods for discovering and debating trending topics:
 * - Fetch trending topics from multiple sources
 * - Get debate suggestions based on current trends
 * - Manage the pulse scheduler for automated debates
 * - Analyze debate outcomes on trending topics
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai', apiKey: 'your-key' });
 *
 * // Get trending topics
 * const { topics } = await client.pulse.getTrending({ source: 'hackernews', limit: 10 });
 *
 * // Get a debate suggestion
 * const suggestion = await client.pulse.suggest();
 *
 * // Start a debate on a trending topic
 * const debate = await client.pulse.startDebate(suggestion.topic.id);
 * ```
 */
export class PulseAPI {
  constructor(private client: PulseClientInterface) {}

  /**
   * Get trending topics from pulse sources.
   */
  async getTrending(options?: {
    source?: PulseSource;
    category?: string;
    limit?: number;
    min_score?: number;
  }): Promise<{ topics: TrendingTopic[]; fetched_at: string }> {
    return this.client.request('GET', '/api/pulse/trending', { params: options });
  }

  /**
   * Get a debate suggestion based on current trending topics.
   */
  async suggest(options?: {
    source?: PulseSource;
    category?: string;
    prefer_controversial?: boolean;
  }): Promise<DebateSuggestion> {
    return this.client.request('GET', '/api/pulse/suggest', { params: options });
  }

  /**
   * Get analytics on debates started from trending topics.
   */
  async getAnalytics(options?: {
    source?: PulseSource;
    period?: '7d' | '30d' | '90d';
  }): Promise<PulseAnalytics> {
    return this.client.request('GET', '/api/pulse/analytics', { params: options });
  }

  /**
   * Start a debate on a trending topic.
   */
  async startDebate(
    topicId: string,
    options?: {
      agents?: string[];
      protocol?: Record<string, unknown>;
    }
  ): Promise<{ debate_id: string; task: string }> {
    return this.client.request('POST', '/api/pulse/debate-topic', {
      json: { topic_id: topicId, ...options },
    });
  }

  // =========================================================================
  // Scheduler Management
  // =========================================================================

  /**
   * Get the pulse scheduler status.
   */
  async getSchedulerStatus(): Promise<SchedulerStatus> {
    return this.client.request('GET', '/api/pulse/scheduler/status');
  }

  /**
   * Start the pulse scheduler.
   */
  async startScheduler(): Promise<{ success: boolean; message: string }> {
    return this.client.request('POST', '/api/pulse/scheduler/start');
  }

  /**
   * Stop the pulse scheduler.
   */
  async stopScheduler(): Promise<{ success: boolean; message: string }> {
    return this.client.request('POST', '/api/pulse/scheduler/stop');
  }

  /**
   * Pause the pulse scheduler (keeps state but stops fetching).
   */
  async pauseScheduler(): Promise<{ success: boolean; message: string }> {
    return this.client.request('POST', '/api/pulse/scheduler/pause');
  }

  /**
   * Resume a paused pulse scheduler.
   */
  async resumeScheduler(): Promise<{ success: boolean; message: string }> {
    return this.client.request('POST', '/api/pulse/scheduler/resume');
  }

  /**
   * Update the pulse scheduler configuration.
   */
  async updateSchedulerConfig(config: SchedulerConfig): Promise<{ success: boolean; config: SchedulerConfig }> {
    return this.client.request('PATCH', '/api/pulse/scheduler/config', {
      json: config as unknown as Record<string, unknown>,
    });
  }

  /**
   * Get the history of scheduled debates.
   */
  async getSchedulerHistory(options?: {
    platform?: PulseSource;
    limit?: number;
    offset?: number;
  }): Promise<{ debates: ScheduledDebate[]; total: number }> {
    return this.client.request('GET', '/api/pulse/scheduler/history', { params: options });
  }
}
