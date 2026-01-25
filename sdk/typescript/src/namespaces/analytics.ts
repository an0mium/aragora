/**
 * Analytics Namespace API
 *
 * Provides a namespaced interface for analytics and metrics operations.
 * This wraps the flat client methods for a more intuitive API.
 */

import type {
  DisagreementAnalytics,
  RoleRotationAnalytics,
  EarlyStopAnalytics,
  ConsensusQualityAnalytics,
  RankingStats,
  MemoryStats,
  CrossPollinationMetrics,
  LearningEfficiencyMetrics,
  VotingAccuracyMetrics,
  CalibrationMetrics,
} from '../types';

/**
 * Date range options for analytics queries.
 */
export interface AnalyticsDateRange {
  /** Start date (ISO string) */
  start_date?: string;
  /** End date (ISO string) */
  end_date?: string;
  /** Number of days to look back */
  days?: number;
}

/**
 * Interface for the internal client methods used by AnalyticsAPI.
 */
interface AnalyticsClientInterface {
  getDisagreementAnalytics(options?: AnalyticsDateRange): Promise<DisagreementAnalytics>;
  getRoleRotationAnalytics(options?: AnalyticsDateRange): Promise<RoleRotationAnalytics>;
  getEarlyStopAnalytics(options?: AnalyticsDateRange): Promise<EarlyStopAnalytics>;
  getConsensusQualityAnalytics(options?: AnalyticsDateRange): Promise<ConsensusQualityAnalytics>;
  getRankingStats(options?: AnalyticsDateRange): Promise<RankingStats>;
  getMemoryStats(): Promise<MemoryStats>;
  getCrossPollinationMetrics(options?: AnalyticsDateRange): Promise<CrossPollinationMetrics>;
  getLearningEfficiencyMetrics(options?: AnalyticsDateRange): Promise<LearningEfficiencyMetrics>;
  getVotingAccuracyMetrics(options?: AnalyticsDateRange): Promise<VotingAccuracyMetrics>;
  getCalibrationMetrics(options?: AnalyticsDateRange): Promise<CalibrationMetrics>;
}

/**
 * Analytics API namespace.
 *
 * Provides methods for retrieving analytics and metrics:
 * - Disagreement patterns between agents
 * - Role rotation statistics
 * - Early stop analysis
 * - Consensus quality metrics
 * - Ranking and memory statistics
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Get disagreement analytics for the last 7 days
 * const disagreements = await client.analytics.disagreements({ days: 7 });
 *
 * // Get consensus quality metrics
 * const quality = await client.analytics.consensusQuality();
 *
 * // Get memory statistics
 * const memStats = await client.analytics.memoryStats();
 * ```
 */
export class AnalyticsAPI {
  constructor(private client: AnalyticsClientInterface) {}

  /**
   * Get disagreement analytics showing patterns of agent disagreements.
   */
  async disagreements(options?: AnalyticsDateRange): Promise<DisagreementAnalytics> {
    return this.client.getDisagreementAnalytics(options);
  }

  /**
   * Get role rotation analytics showing how agents switch roles.
   */
  async roleRotation(options?: AnalyticsDateRange): Promise<RoleRotationAnalytics> {
    return this.client.getRoleRotationAnalytics(options);
  }

  /**
   * Get early stop analytics showing debates that ended early.
   */
  async earlyStops(options?: AnalyticsDateRange): Promise<EarlyStopAnalytics> {
    return this.client.getEarlyStopAnalytics(options);
  }

  /**
   * Get consensus quality analytics.
   */
  async consensusQuality(options?: AnalyticsDateRange): Promise<ConsensusQualityAnalytics> {
    return this.client.getConsensusQualityAnalytics(options);
  }

  /**
   * Get ranking statistics for agents.
   */
  async rankingStats(options?: AnalyticsDateRange): Promise<RankingStats> {
    return this.client.getRankingStats(options);
  }

  /**
   * Get memory system statistics.
   */
  async memoryStats(): Promise<MemoryStats> {
    return this.client.getMemoryStats();
  }

  /**
   * Get cross-pollination metrics showing knowledge transfer between debates.
   */
  async crossPollination(options?: AnalyticsDateRange): Promise<CrossPollinationMetrics> {
    return this.client.getCrossPollinationMetrics(options);
  }

  /**
   * Get learning efficiency metrics.
   */
  async learningEfficiency(options?: AnalyticsDateRange): Promise<LearningEfficiencyMetrics> {
    return this.client.getLearningEfficiencyMetrics(options);
  }

  /**
   * Get voting accuracy metrics.
   */
  async votingAccuracy(options?: AnalyticsDateRange): Promise<VotingAccuracyMetrics> {
    return this.client.getVotingAccuracyMetrics(options);
  }

  /**
   * Get calibration metrics showing agent confidence calibration.
   */
  async calibration(options?: AnalyticsDateRange): Promise<CalibrationMetrics> {
    return this.client.getCalibrationMetrics(options);
  }
}
