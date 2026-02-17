/**
 * Codebase Analysis Namespace API
 *
 * Provides a namespaced interface for codebase analysis operations including
 * security scanning, dependency analysis, metrics, and code intelligence.
 */

import type { AragoraClient } from '../client';

/**
 * Codebase Analysis API namespace.
 *
 * Provides comprehensive codebase analysis capabilities:
 * - Security scanning (vulnerabilities, secrets, SAST)
 * - Dependency analysis and license checking
 * - Code metrics and quality analysis
 * - Code intelligence (symbols, call graphs, dead code)
 * - Impact analysis for changes
 * - Full codebase audits
 */
export class CodebaseAPI {
  constructor(private client: AragoraClient) {}

  // ===========================================================================
  // Top-level Scanning & Analysis
  // ===========================================================================

  /**
   * Analyze codebase.
   * @route GET /api/v1/codebase/analyze
   */
  async analyze(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/codebase/analyze') as Promise<Record<string, unknown>>;
  }

  /**
   * Analyze dependencies.
   * @route POST /api/v1/codebase/analyze-dependencies
   */
  async analyzeDependencies(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/codebase/analyze-dependencies', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get codebase audit results.
   * @route GET /api/v1/codebase/audit
   */
  async getAudit(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/codebase/audit') as Promise<Record<string, unknown>>;
  }

  /**
   * Get codebase bugs.
   * @route GET /api/v1/codebase/bugs
   */
  async getBugs(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/codebase/bugs') as Promise<Record<string, unknown>>;
  }

  /**
   * Get call graph.
   * @route GET /api/v1/codebase/callgraph
   */
  async getCallgraph(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/codebase/callgraph') as Promise<Record<string, unknown>>;
  }

  /**
   * Check licenses.
   * @route POST /api/v1/codebase/check-licenses
   */
  async checkLicenses(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/codebase/check-licenses', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Clear analysis cache.
   * @route POST /api/v1/codebase/clear-cache
   */
  async clearCache(): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/codebase/clear-cache') as Promise<Record<string, unknown>>;
  }

  /**
   * Get codebase dashboard.
   * @route GET /api/v1/codebase/dashboard
   */
  async getDashboard(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/codebase/dashboard') as Promise<Record<string, unknown>>;
  }

  /**
   * Get dead code report.
   * @route GET /api/v1/codebase/deadcode
   */
  async getDeadcode(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/codebase/deadcode') as Promise<Record<string, unknown>>;
  }

  /**
   * Get codebase demo.
   * @route GET /api/v1/codebase/demo
   */
  async getDemo(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/codebase/demo') as Promise<Record<string, unknown>>;
  }

  /**
   * Get dependencies.
   * @route GET /api/v1/codebase/dependencies
   */
  async getDependencies(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/codebase/dependencies') as Promise<Record<string, unknown>>;
  }

  /**
   * Get findings.
   * @route GET /api/v1/codebase/findings
   */
  async getFindings(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/codebase/findings') as Promise<Record<string, unknown>>;
  }

  /**
   * Create issue for a finding.
   * @route GET /api/v1/codebase/findings/{finding_id}/create-issue
   */
  async createFindingIssue(findingId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/findings/${encodeURIComponent(findingId)}/create-issue`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Dismiss a finding.
   * @route GET /api/v1/codebase/findings/{finding_id}/dismiss
   */
  async dismissFinding(findingId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/findings/${encodeURIComponent(findingId)}/dismiss`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get impact analysis.
   * @route GET /api/v1/codebase/impact
   */
  async getImpact(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/codebase/impact') as Promise<Record<string, unknown>>;
  }

  /**
   * Get codebase metrics.
   * @route GET /api/v1/codebase/metrics
   */
  async getMetrics(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/codebase/metrics') as Promise<Record<string, unknown>>;
  }

  /**
   * Get SAST results.
   * @route GET /api/v1/codebase/sast
   */
  async getSast(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/codebase/sast') as Promise<Record<string, unknown>>;
  }

  /**
   * Generate SBOM.
   * @route POST /api/v1/codebase/sbom
   */
  async generateSbom(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/codebase/sbom', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get scan results.
   * @route GET /api/v1/codebase/scan
   */
  async getScan(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/codebase/scan') as Promise<Record<string, unknown>>;
  }

  /**
   * Scan for vulnerabilities.
   * @route POST /api/v1/codebase/scan-vulnerabilities
   */
  async scanVulnerabilities(body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request('POST', '/api/v1/codebase/scan-vulnerabilities', {
      body,
    }) as Promise<Record<string, unknown>>;
  }

  /**
   * Get scan by ID.
   * @route GET /api/v1/codebase/scan/{scan_id}
   */
  async getScanById(scanId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/scan/${encodeURIComponent(scanId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * List scans.
   * @route GET /api/v1/codebase/scans
   */
  async listScans(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/codebase/scans') as Promise<Record<string, unknown>>;
  }

  /**
   * Get secrets scan results.
   * @route GET /api/v1/codebase/secrets
   */
  async getSecrets(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/codebase/secrets') as Promise<Record<string, unknown>>;
  }

  /**
   * Get symbols.
   * @route GET /api/v1/codebase/symbols
   */
  async getSymbols(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/codebase/symbols') as Promise<Record<string, unknown>>;
  }

  /**
   * Get codebase understanding.
   * @route GET /api/v1/codebase/understand
   */
  async getUnderstanding(): Promise<Record<string, unknown>> {
    return this.client.request('GET', '/api/v1/codebase/understand') as Promise<Record<string, unknown>>;
  }

  // ===========================================================================
  // Per-repo Operations
  // ===========================================================================

  /**
   * Analyze a specific repo.
   * @route POST /api/v1/codebase/{repo}/analyze
   */
  async analyzeRepo(repo: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/v1/codebase/${encodeURIComponent(repo)}/analyze`,
      { body }
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Start a codebase audit for a repo.
   * @route POST /api/v1/codebase/{repo}/audit
   */
  async startRepoAudit(repo: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/v1/codebase/${encodeURIComponent(repo)}/audit`,
      { body }
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get repo audit by ID.
   * @route GET /api/v1/codebase/{repo}/audit/{audit_id}
   */
  async getRepoAudit(repo: string, auditId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/audit/${encodeURIComponent(auditId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get repo call graph.
   * @route GET /api/v1/codebase/{repo}/callgraph
   */
  async getRepoCallgraph(repo: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/callgraph`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get repo dead code.
   * @route GET /api/v1/codebase/{repo}/deadcode
   */
  async getRepoDeadcode(repo: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/deadcode`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get repo duplicates.
   * @route GET /api/v1/codebase/{repo}/duplicates
   */
  async getRepoDuplicates(repo: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/duplicates`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get repo hotspots.
   * @route GET /api/v1/codebase/{repo}/hotspots
   */
  async getRepoHotspots(repo: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/hotspots`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Analyze impact for a repo.
   * @route POST /api/v1/codebase/{repo}/impact
   */
  async analyzeRepoImpact(repo: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/v1/codebase/${encodeURIComponent(repo)}/impact`,
      { body }
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get repo metrics.
   * @route GET /api/v1/codebase/{repo}/metrics
   */
  async getRepoMetrics(repo: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/metrics`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Run metrics analysis for a repo.
   * @route POST /api/v1/codebase/{repo}/metrics/analyze
   */
  async analyzeRepoMetrics(repo: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/v1/codebase/${encodeURIComponent(repo)}/metrics/analyze`,
      { body }
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get file metrics for a repo.
   * @route GET /api/v1/codebase/{repo}/metrics/file/{file_path}
   */
  async getRepoFileMetrics(repo: string, filePath: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/metrics/file/${encodeURIComponent(filePath)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get metrics history for a repo.
   * @route GET /api/v1/codebase/{repo}/metrics/history
   */
  async getRepoMetricsHistory(repo: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/metrics/history`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get metrics by analysis ID.
   * @route GET /api/v1/codebase/{repo}/metrics/{analysis_id}
   */
  async getRepoMetricsById(repo: string, analysisId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/metrics/${encodeURIComponent(analysisId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get SAST findings for a repo.
   * @route GET /api/v1/codebase/{repo}/sast/findings
   */
  async getRepoSastFindings(repo: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/sast/findings`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get OWASP summary for a repo.
   * @route GET /api/v1/codebase/{repo}/sast/owasp-summary
   */
  async getRepoOwaspSummary(repo: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/sast/owasp-summary`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Start a scan for a repo.
   * @route POST /api/v1/codebase/{repo}/scan
   */
  async startRepoScan(repo: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scan`,
      { body }
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get latest scan for a repo.
   * @route GET /api/v1/codebase/{repo}/scan/latest
   */
  async getRepoLatestScan(repo: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scan/latest`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Start SAST scan for a repo.
   * @route POST /api/v1/codebase/{repo}/scan/sast
   */
  async startRepoSastScan(repo: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scan/sast`,
      { body }
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get SAST scan by ID.
   * @route GET /api/v1/codebase/{repo}/scan/sast/{scan_id}
   */
  async getRepoSastScan(repo: string, scanId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scan/sast/${encodeURIComponent(scanId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Start secrets scan for a repo.
   * @route POST /api/v1/codebase/{repo}/scan/secrets
   */
  async startRepoSecretsScan(repo: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scan/secrets`,
      { body }
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get latest secrets scan for a repo.
   * @route GET /api/v1/codebase/{repo}/scan/secrets/latest
   */
  async getRepoLatestSecretsScan(repo: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scan/secrets/latest`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get secrets scan by ID.
   * @route GET /api/v1/codebase/{repo}/scan/secrets/{scan_id}
   */
  async getRepoSecretsScan(repo: string, scanId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scan/secrets/${encodeURIComponent(scanId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get scan by ID for a repo.
   * @route GET /api/v1/codebase/{repo}/scan/{scan_id}
   */
  async getRepoScan(repo: string, scanId: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scan/${encodeURIComponent(scanId)}`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * List scans for a repo.
   * @route GET /api/v1/codebase/{repo}/scans
   */
  async listRepoScans(repo: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scans`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * List secrets scans for a repo.
   * @route GET /api/v1/codebase/{repo}/scans/secrets
   */
  async listRepoSecretsScans(repo: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/scans/secrets`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get repo secrets.
   * @route GET /api/v1/codebase/{repo}/secrets
   */
  async getRepoSecrets(repo: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/secrets`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get repo symbols.
   * @route GET /api/v1/codebase/{repo}/symbols
   */
  async getRepoSymbols(repo: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/symbols`
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Understand a repo.
   * @route POST /api/v1/codebase/{repo}/understand
   */
  async understandRepo(repo: string, body: Record<string, unknown>): Promise<Record<string, unknown>> {
    return this.client.request(
      'POST',
      `/api/v1/codebase/${encodeURIComponent(repo)}/understand`,
      { body }
    ) as Promise<Record<string, unknown>>;
  }

  /**
   * Get repo vulnerabilities.
   * @route GET /api/v1/codebase/{repo}/vulnerabilities
   */
  async getRepoVulnerabilities(repo: string): Promise<Record<string, unknown>> {
    return this.client.request(
      'GET',
      `/api/v1/codebase/${encodeURIComponent(repo)}/vulnerabilities`
    ) as Promise<Record<string, unknown>>;
  }
}
