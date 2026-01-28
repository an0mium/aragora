/**
 * Feedback Namespace API
 *
 * Provides endpoints for collecting user feedback including:
 * - NPS (Net Promoter Score) surveys
 * - Feature feedback
 * - Bug reports
 * - General suggestions
 */

/**
 * Types of feedback.
 */
export type FeedbackType =
  | 'nps'
  | 'feature_request'
  | 'bug_report'
  | 'general'
  | 'debate_quality';

/**
 * NPS submission request.
 */
export interface NPSSubmission {
  /** Score from 0-10 */
  score: number;
  /** Optional comment explaining the score */
  comment?: string;
  /** Optional metadata context */
  context?: Record<string, unknown>;
}

/**
 * General feedback submission.
 */
export interface FeedbackSubmission {
  /** Type of feedback */
  type: FeedbackType;
  /** Required comment/description */
  comment: string;
  /** Optional rating score */
  score?: number;
  /** Optional metadata context */
  context?: Record<string, unknown>;
}

/**
 * Feedback submission response.
 */
export interface FeedbackResponse {
  success: boolean;
  feedback_id: string;
  message: string;
}

/**
 * NPS summary statistics.
 */
export interface NPSSummary {
  /** Overall NPS score (-100 to 100) */
  nps_score: number;
  /** Total number of responses */
  total_responses: number;
  /** Count of promoters (9-10 scores) */
  promoters: number;
  /** Count of passives (7-8 scores) */
  passives: number;
  /** Count of detractors (0-6 scores) */
  detractors: number;
  /** Time period in days */
  period_days: number;
}

/**
 * Feedback prompt configuration.
 */
export interface FeedbackPrompt {
  type: string;
  question: string;
  scale?: {
    min: number;
    max: number;
    labels: Record<string, string>;
  };
  follow_up?: string;
}

/**
 * Client interface for making HTTP requests.
 */
interface FeedbackClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Feedback API for collecting user feedback.
 */
export class FeedbackAPI {
  constructor(private client: FeedbackClientInterface) {}

  /**
   * Submit NPS feedback.
   *
   * @param submission - NPS score and optional comment
   */
  async submitNPS(submission: NPSSubmission): Promise<FeedbackResponse> {
    if (submission.score < 0 || submission.score > 10) {
      throw new Error('NPS score must be between 0 and 10');
    }

    return this.client.request('POST', '/api/v1/feedback/nps', {
      json: submission as unknown as Record<string, unknown>,
    });
  }

  /**
   * Submit general feedback.
   *
   * @param submission - Feedback type and comment
   */
  async submitFeedback(submission: FeedbackSubmission): Promise<FeedbackResponse> {
    if (!submission.comment) {
      throw new Error('Comment is required for feedback submission');
    }

    return this.client.request('POST', '/api/v1/feedback/general', {
      json: submission as unknown as Record<string, unknown>,
    });
  }

  /**
   * Submit a feature request.
   *
   * @param description - Description of the feature
   * @param context - Optional context metadata
   */
  async submitFeatureRequest(
    description: string,
    context?: Record<string, unknown>
  ): Promise<FeedbackResponse> {
    return this.submitFeedback({
      type: 'feature_request',
      comment: description,
      context,
    });
  }

  /**
   * Submit a bug report.
   *
   * @param description - Description of the bug
   * @param context - Optional context (steps to reproduce, etc.)
   */
  async submitBugReport(
    description: string,
    context?: Record<string, unknown>
  ): Promise<FeedbackResponse> {
    return this.submitFeedback({
      type: 'bug_report',
      comment: description,
      context,
    });
  }

  /**
   * Submit debate quality feedback.
   *
   * @param debateId - ID of the debate
   * @param comment - Quality feedback comment
   * @param score - Optional quality score
   */
  async submitDebateQualityFeedback(
    debateId: string,
    comment: string,
    score?: number
  ): Promise<FeedbackResponse> {
    return this.submitFeedback({
      type: 'debate_quality',
      comment,
      score,
      context: { debate_id: debateId },
    });
  }

  /**
   * Get NPS summary statistics (admin only).
   *
   * @param days - Number of days to include (default 30)
   */
  async getNPSSummary(days: number = 30): Promise<NPSSummary> {
    return this.client.request('GET', '/api/v1/feedback/nps/summary', {
      params: { days },
    });
  }

  /**
   * Get active feedback prompts for the current user.
   */
  async getPrompts(): Promise<{ prompts: FeedbackPrompt[] }> {
    return this.client.request('GET', '/api/v1/feedback/prompts');
  }
}
