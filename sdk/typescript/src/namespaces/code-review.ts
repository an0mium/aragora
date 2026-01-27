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
 * Code Review API for multi-agent code analysis.
 */
export class CodeReviewAPI {
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
   * Review a code snippet.
   *
   * @param request - Code review request
   */
  async reviewCode(
    request: CodeReviewRequest
  ): Promise<{ result: ReviewResult; result_id: string; message: string }> {
    const response = await fetch(`${this.baseUrl}/api/v1/code-review/review`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(request),
    });
    if (!response.ok) throw new Error(`Code review failed: ${response.statusText}`);
    return response.json();
  }

  /**
   * Review a diff/patch.
   *
   * @param request - Diff review request
   */
  async reviewDiff(
    request: DiffReviewRequest
  ): Promise<{ result: ReviewResult; result_id: string; message: string }> {
    const response = await fetch(`${this.baseUrl}/api/v1/code-review/diff`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(request),
    });
    if (!response.ok) throw new Error(`Diff review failed: ${response.statusText}`);
    return response.json();
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

    const response = await fetch(`${this.baseUrl}/api/v1/code-review/pr`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(request),
    });
    if (!response.ok) throw new Error(`PR review failed: ${response.statusText}`);
    return response.json();
  }

  /**
   * Get a review result by ID.
   *
   * @param resultId - Review result ID
   */
  async getResult(resultId: string): Promise<{ result: ReviewResult }> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/code-review/results/${resultId}`,
      {
        method: 'GET',
        headers: this.headers,
      }
    );
    if (!response.ok) throw new Error(`Failed to get result: ${response.statusText}`);
    return response.json();
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
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', options.limit.toString());
    if (options?.offset) params.set('offset', options.offset.toString());

    const url = `${this.baseUrl}/api/v1/code-review/history${params.toString() ? `?${params}` : ''}`;
    const response = await fetch(url, {
      method: 'GET',
      headers: this.headers,
    });
    if (!response.ok) throw new Error(`Failed to get history: ${response.statusText}`);
    return response.json();
  }

  /**
   * Quick security-focused code scan.
   *
   * @param code - Code to scan
   * @param language - Programming language (optional)
   */
  async securityScan(code: string, language?: string): Promise<SecurityScanResult> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/code-review/security-scan`,
      {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify({ code, language }),
      }
    );
    if (!response.ok) throw new Error(`Security scan failed: ${response.statusText}`);
    return response.json();
  }
}
