/**
 * Code Review Namespace API
 *
 * Provides multi-agent code review capabilities including:
 * - Code snippet review
 * - Diff/patch review
 * - GitHub PR review
 * - Security scanning
 */

/**
 * Review finding severity levels.
 */
export type FindingSeverity = 'critical' | 'high' | 'medium' | 'low' | 'info';

/**
 * Review finding categories.
 */
export type FindingCategory =
  | 'security'
  | 'performance'
  | 'maintainability'
  | 'test_coverage'
  | 'style'
  | 'logic';

/**
 * A single review finding.
 */
export interface ReviewFinding {
  id: string;
  category: FindingCategory;
  severity: FindingSeverity;
  title: string;
  description: string;
  line_start?: number;
  line_end?: number;
  file_path?: string;
  suggestion?: string;
  code_snippet?: string;
}

/**
 * Code review result.
 */
export interface ReviewResult {
  id: string;
  findings: ReviewFinding[];
  summary: string;
  overall_score?: number;
  review_types: string[];
  language?: string;
  created_at: string;
  stored_at?: string;
}

/**
 * Code review request.
 */
export interface CodeReviewRequest {
  /** Code to review */
  code: string;
  /** Programming language (auto-detected if not provided) */
  language?: string;
  /** File path for context */
  file_path?: string;
  /** Types of review to perform */
  review_types?: FindingCategory[];
  /** Additional context */
  context?: string;
}

/**
 * Diff review request.
 */
export interface DiffReviewRequest {
  /** Unified diff format */
  diff: string;
  /** Base branch name */
  base_branch?: string;
  /** Head branch name */
  head_branch?: string;
  /** Types of review to perform */
  review_types?: FindingCategory[];
  /** Additional context */
  context?: string;
}

/**
 * PR review request.
 */
export interface PRReviewRequest {
  /** GitHub PR URL */
  pr_url: string;
  /** Types of review to perform */
  review_types?: FindingCategory[];
  /** Post comments to PR (requires GitHub token) */
  post_comments?: boolean;
}

/**
 * Security scan result.
 */
export interface SecurityScanResult {
  findings: ReviewFinding[];
  total: number;
  severity_summary: {
    critical: number;
    high: number;
    medium: number;
    low: number;
  };
}

/**
 * Review history response.
 */
export interface ReviewHistoryResponse {
  reviews: ReviewResult[];
  total: number;
  limit: number;
  offset: number;
}

/**
 * Client interface for making HTTP requests.
 */
interface CodeReviewClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Code Review API for multi-agent code analysis.
 */
export class CodeReviewAPI {
  constructor(private client: CodeReviewClientInterface) {}

  /**
   * Review a code snippet.
   *
   * @param request - Code review request
   */
  async reviewCode(
    request: CodeReviewRequest
  ): Promise<{ result: ReviewResult; result_id: string; message: string }> {
    return this.client.request('POST', '/api/v1/code-review/review', {
      json: request as unknown as Record<string, unknown>,
    });
  }

  /**
   * Review a diff/patch.
   *
   * @param request - Diff review request
   */
  async reviewDiff(
    request: DiffReviewRequest
  ): Promise<{ result: ReviewResult; result_id: string; message: string }> {
    return this.client.request('POST', '/api/v1/code-review/diff', {
      json: request as unknown as Record<string, unknown>,
    });
  }

  /**
   * Review a GitHub pull request.
   *
   * @param request - PR review request
   */
  async reviewPR(
    request: PRReviewRequest
  ): Promise<{ result: ReviewResult; result_id: string; message: string }> {
    // Validate PR URL
    if (!request.pr_url.includes('github.com') || !request.pr_url.includes('/pull/')) {
      throw new Error(
        'Invalid PR URL. Expected format: https://github.com/owner/repo/pull/123'
      );
    }

    return this.client.request('POST', '/api/v1/code-review/pr', {
      json: request as unknown as Record<string, unknown>,
    });
  }

  /**
   * Get a review result by ID.
   *
   * @param resultId - Review result ID
   */
  async getResult(resultId: string): Promise<{ result: ReviewResult }> {
    return this.client.request('GET', `/api/v1/code-review/results/${resultId}`);
  }

  /**
   * Get review history.
   *
   * @param options - Pagination options
   */
  async getHistory(options?: {
    limit?: number;
    offset?: number;
  }): Promise<ReviewHistoryResponse> {
    const params: Record<string, unknown> = {};
    if (options?.limit) params.limit = options.limit;
    if (options?.offset) params.offset = options.offset;

    return this.client.request('GET', '/api/v1/code-review/history', {
      params: Object.keys(params).length > 0 ? params : undefined,
    });
  }

  /**
   * Quick security-focused code scan.
   *
   * @param code - Code to scan
   * @param language - Programming language (optional)
   */
  async securityScan(code: string, language?: string): Promise<SecurityScanResult> {
    return this.client.request('POST', '/api/v1/code-review/security-scan', {
      json: { code, language },
    });
  }
}
