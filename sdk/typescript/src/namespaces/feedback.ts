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
 * Feedback API for collecting user feedback.
 */
export class FeedbackAPI {
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
   * Submit NPS feedback.
   *
   * @param submission - NPS score and optional comment
   */
  async submitNPS(submission: NPSSubmission): Promise<FeedbackResponse> {
    if (submission.score < 0 || submission.score > 10) {
      throw new Error('NPS score must be between 0 and 10');
    }

    const response = await fetch(`${this.baseUrl}/api/v1/feedback/nps`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(submission),
    });
    if (!response.ok) throw new Error(`Failed to submit NPS: ${response.statusText}`);
    return response.json();
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

    const response = await fetch(`${this.baseUrl}/api/v1/feedback/general`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(submission),
    });
    if (!response.ok) throw new Error(`Failed to submit feedback: ${response.statusText}`);
    return response.json();
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
    const response = await fetch(
      `${this.baseUrl}/api/v1/feedback/nps/summary?days=${days}`,
      {
        method: 'GET',
        headers: this.headers,
      }
    );
    if (!response.ok) throw new Error(`Failed to get NPS summary: ${response.statusText}`);
    return response.json();
  }

  /**
   * Get active feedback prompts for the current user.
   */
  async getPrompts(): Promise<{ prompts: FeedbackPrompt[] }> {
    const response = await fetch(`${this.baseUrl}/api/v1/feedback/prompts`, {
      method: 'GET',
      headers: this.headers,
    });
    if (!response.ok) throw new Error(`Failed to get prompts: ${response.statusText}`);
    return response.json();
  }
}
