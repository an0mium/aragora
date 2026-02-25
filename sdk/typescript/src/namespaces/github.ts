/**
 * GitHub Namespace API
 *
 * Provides endpoints for GitHub integration including
 * pull request reviews, issue management, and repository operations.
 */

import type { AragoraClient } from '../client';

/** Pull request details */
export interface PullRequest {
  id: string;
  number: number;
  title: string;
  description: string;
  author: string;
  repo: string;
  base_branch: string;
  head_branch: string;
  status: string;
  created_at: string;
  updated_at: string;
}

/** PR review result */
export interface PRReviewResult {
  id: string;
  pr_number: number;
  repo: string;
  verdict: 'approve' | 'request_changes' | 'comment';
  summary: string;
  findings: PRFinding[];
  created_at: string;
}

/** Individual finding from a PR review */
export interface PRFinding {
  file: string;
  line: number;
  severity: 'info' | 'warning' | 'error';
  message: string;
  suggestion?: string;
}

/** Request to trigger a PR review */
export interface TriggerReviewRequest {
  repo: string;
  pr_number: number;
  review_type?: 'security' | 'quality' | 'full';
}

/**
 * GitHub namespace for repository and PR integration.
 *
 * @example
 * ```typescript
 * const review = await client.github.triggerReview({
 *   repo: 'org/repo',
 *   pr_number: 42,
 * });
 * console.log(review.verdict);
 * ```
 */
export class GitHubNamespace {
  constructor(private client: AragoraClient) {}

  /** Trigger a PR review. */
  async triggerReview(request: TriggerReviewRequest): Promise<PRReviewResult> {
    return this.client.request<PRReviewResult>(
      'POST',
      '/api/v1/github/pr/review',
      { body: request }
    );
  }

  /** Trigger a PR review using the compatibility route. */
  async triggerPRReview(request: {
    repository: string;
    pr_number: number;
    review_type?: string;
    workspace_id?: string;
  }): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/github/pr/review',
      { body: request }
    );
  }

  /** Fetch audit issues for a finding session. */
  async getAuditIssues(params?: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'GET',
      '/api/v1/github/audit/issues',
      { params: params as Record<string, unknown> }
    );
  }

  /** Bulk create audit issues from findings. */
  async createAuditIssuesBulk(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/github/audit/issues/bulk',
      { body }
    );
  }

  /** Create a PR with automated fixes from audit findings. */
  async createAuditPR(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request<Record<string, unknown>>(
      'POST',
      '/api/v1/github/audit/pr',
      { body }
    );
  }

  /** Get PR details. */
  async getPullRequest(repo: string, prNumber: number): Promise<PullRequest> {
    return this.client.request<PullRequest>(
      'GET',
      '/api/v1/github/pr/review',
      { params: { repo, pr_number: prNumber } }
    );
  }

  /** Get a review result. */
  async getReview(reviewId: string): Promise<PRReviewResult> {
    return this.client.request<PRReviewResult>(
      'GET',
      '/api/v1/github/pr/review',
      { params: { review_id: reviewId } }
    );
  }

  /** Submit a review comment. */
  async submitReview(
    reviewId: string,
    body: { verdict: string; comment?: string }
  ): Promise<{ success: boolean }> {
    return this.client.request<{ success: boolean }>(
      'POST',
      '/api/v1/github/pr/review',
      { body: { review_id: reviewId, ...body } }
    );
  }
}
