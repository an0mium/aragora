/**
 * Codebase Analysis Namespace API
 *
 * Provides a namespaced interface for codebase analysis operations including
 * security scanning, dependency analysis, metrics, and code intelligence.
 */

import type {
  CodebaseScan,
  CodebaseMetrics,
  CodebaseSymbols,
  CodebaseCallgraph,
  CodebaseDeadcode,
  CodebaseAudit,
  CodebaseUnderstanding,
  CodebaseImpact,
  VulnerabilityScan,
  DependencyAnalysis,
  LicenseCheck,
  Sbom,
  SecretsScan,
  SastScan,
  PaginationParams,
} from '../types';

/**
 * Interface for the internal client methods used by CodebaseAPI.
 */
interface CodebaseClientInterface {
  // Scanning
  startCodebaseScan(repo: string, body: {
    repo_path: string;
    branch?: string;
    commit_sha?: string;
    workspace_id?: string;
  }): Promise<CodebaseScan>;
  getLatestCodebaseScan(repo: string): Promise<CodebaseScan>;
  getCodebaseScan(repo: string, scanId: string): Promise<CodebaseScan>;
  listCodebaseScans(repo: string, params?: {
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ scans: CodebaseScan[]; total: number }>;

  // Vulnerabilities
  listCodebaseVulnerabilities(repo: string, params?: {
    severity?: string;
    package?: string;
    ecosystem?: string;
    limit?: number;
    offset?: number;
  }): Promise<VulnerabilityScan>;

  // Dependencies
  analyzeDependencies(body: { repo_path: string; depth?: number }): Promise<DependencyAnalysis>;
  scanVulnerabilities(body: { repo_path: string; fail_on_severity?: string }): Promise<VulnerabilityScan>;

  // Licenses & SBOM
  checkCodebaseLicenses(body: {
    repo_path: string;
    allowed_licenses?: string[];
  }): Promise<LicenseCheck>;
  generateCodebaseSbom(body: {
    repo_path: string;
    format?: 'spdx' | 'cyclonedx';
  }): Promise<Sbom>;

  // Secrets & SAST
  startSecretsScan(repo: string, body: {
    repo_path: string;
    scan_history?: boolean;
  }): Promise<SecretsScan>;
  startSastScan(repo: string, body: {
    repo_path: string;
    rule_sets?: string[];
    workspace_id?: string;
  }): Promise<SastScan>;

  // Metrics
  runCodebaseMetricsAnalysis(repo: string, body: {
    repo_path: string;
    include_coverage?: boolean;
  }): Promise<CodebaseMetrics>;
  getLatestCodebaseMetrics(repo: string): Promise<CodebaseMetrics>;
  getCodebaseMetrics(repo: string, analysisId: string): Promise<CodebaseMetrics>;
  listCodebaseMetricsHistory(repo: string, params?: PaginationParams): Promise<{
    metrics: CodebaseMetrics[];
    total: number;
  }>;
  getCodebaseHotspots(repo: string, params?: PaginationParams): Promise<{
    hotspots: Array<{ file_path: string; complexity: number; churn: number; risk_score: number }>;
    total: number;
  }>;
  getCodebaseDuplicates(repo: string, params?: PaginationParams): Promise<{
    duplicates: Array<{ files: string[]; lines: number; tokens: number }>;
    total: number;
    total_duplicated_lines: number;
  }>;
  getCodebaseFileMetrics(repo: string, filePath: string): Promise<{
    file_path: string;
    lines: number;
    complexity: number;
    maintainability: number;
  }>;

  // Code Intelligence
  analyzeCodebase(repo: string, body: {
    repo_path: string;
    depth?: number;
  }): Promise<CodebaseUnderstanding>;
  getCodebaseSymbols(repo: string, params?: {
    type?: string;
    limit?: number;
    offset?: number;
  }): Promise<CodebaseSymbols>;
  getCodebaseCallgraph(repo: string, params?: {
    entry_point?: string;
    depth?: number;
  }): Promise<CodebaseCallgraph>;
  getCodebaseDeadcode(repo: string, params?: {
    confidence_threshold?: number;
    limit?: number;
  }): Promise<CodebaseDeadcode>;
  analyzeCodebaseImpact(repo: string, body: {
    file_path: string;
    change_type?: string;
  }): Promise<CodebaseImpact>;
  understandCodebase(repo: string, body: {
    query?: string;
    include_architecture?: boolean;
  }): Promise<CodebaseUnderstanding>;

  // Audits
  startCodebaseAudit(repo: string, body: {
    repo_path: string;
    categories?: string[];
  }): Promise<CodebaseAudit>;
  getCodebaseAudit(repo: string, auditId: string): Promise<CodebaseAudit>;

  // Cache
  clearCodebaseCache(): Promise<{ success: boolean }>;
}

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
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Run a security scan
 * const scan = await client.codebase.startScan('my-repo', {
 *   repo_path: '/path/to/repo',
 *   branch: 'main'
 * });
 *
 * // Check for vulnerabilities
 * const vulns = await client.codebase.listVulnerabilities('my-repo', {
 *   severity: 'critical'
 * });
 *
 * // Analyze code metrics
 * const metrics = await client.codebase.analyzeMetrics('my-repo', {
 *   repo_path: '/path/to/repo'
 * });
 *
 * // Get dead code report
 * const deadcode = await client.codebase.getDeadcode('my-repo');
 *
 * // Generate SBOM
 * const sbom = await client.codebase.generateSbom({
 *   repo_path: '/path/to/repo',
 *   format: 'cyclonedx'
 * });
 * ```
 */
export class CodebaseAPI {
  constructor(private client: CodebaseClientInterface) {}

  // ===========================================================================
  // Security Scanning
  // ===========================================================================

  /**
   * Start a security scan for a repository.
   */
  async startScan(repo: string, body: {
    repo_path: string;
    branch?: string;
    commit_sha?: string;
    workspace_id?: string;
  }): Promise<CodebaseScan> {
    return this.client.startCodebaseScan(repo, body);
  }

  /**
   * Get the latest scan for a repository.
   */
  async getLatestScan(repo: string): Promise<CodebaseScan> {
    return this.client.getLatestCodebaseScan(repo);
  }

  /**
   * Get a specific scan by ID.
   */
  async getScan(repo: string, scanId: string): Promise<CodebaseScan> {
    return this.client.getCodebaseScan(repo, scanId);
  }

  /**
   * List all scans for a repository.
   */
  async listScans(repo: string, params?: {
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ scans: CodebaseScan[]; total: number }> {
    return this.client.listCodebaseScans(repo, params);
  }

  // ===========================================================================
  // Vulnerability Analysis
  // ===========================================================================

  /**
   * List vulnerabilities from the latest scan.
   */
  async listVulnerabilities(repo: string, params?: {
    severity?: string;
    package?: string;
    ecosystem?: string;
    limit?: number;
    offset?: number;
  }): Promise<VulnerabilityScan> {
    return this.client.listCodebaseVulnerabilities(repo, params);
  }

  /**
   * Scan for vulnerabilities in dependencies.
   */
  async scanVulnerabilities(body: {
    repo_path: string;
    fail_on_severity?: string;
  }): Promise<VulnerabilityScan> {
    return this.client.scanVulnerabilities(body);
  }

  // ===========================================================================
  // Dependency Analysis
  // ===========================================================================

  /**
   * Analyze dependencies for a repository.
   */
  async analyzeDependencies(body: {
    repo_path: string;
    depth?: number;
  }): Promise<DependencyAnalysis> {
    return this.client.analyzeDependencies(body);
  }

  /**
   * Check license compatibility.
   */
  async checkLicenses(body: {
    repo_path: string;
    allowed_licenses?: string[];
  }): Promise<LicenseCheck> {
    return this.client.checkCodebaseLicenses(body);
  }

  /**
   * Generate a Software Bill of Materials (SBOM).
   */
  async generateSbom(body: {
    repo_path: string;
    format?: 'spdx' | 'cyclonedx';
  }): Promise<Sbom> {
    return this.client.generateCodebaseSbom(body);
  }

  // ===========================================================================
  // Secrets & SAST
  // ===========================================================================

  /**
   * Scan for secrets (API keys, passwords, etc.) in code.
   */
  async scanSecrets(repo: string, body: {
    repo_path: string;
    scan_history?: boolean;
  }): Promise<SecretsScan> {
    return this.client.startSecretsScan(repo, body);
  }

  /**
   * Run Static Application Security Testing (SAST).
   */
  async runSast(repo: string, body: {
    repo_path: string;
    rule_sets?: string[];
    workspace_id?: string;
  }): Promise<SastScan> {
    return this.client.startSastScan(repo, body);
  }

  // ===========================================================================
  // Code Metrics
  // ===========================================================================

  /**
   * Run metrics analysis for a repository.
   */
  async analyzeMetrics(repo: string, body: {
    repo_path: string;
    include_coverage?: boolean;
  }): Promise<CodebaseMetrics> {
    return this.client.runCodebaseMetricsAnalysis(repo, body);
  }

  /**
   * Get the latest metrics for a repository.
   */
  async getLatestMetrics(repo: string): Promise<CodebaseMetrics> {
    return this.client.getLatestCodebaseMetrics(repo);
  }

  /**
   * Get metrics by analysis ID.
   */
  async getMetrics(repo: string, analysisId: string): Promise<CodebaseMetrics> {
    return this.client.getCodebaseMetrics(repo, analysisId);
  }

  /**
   * Get metrics history for a repository.
   */
  async listMetricsHistory(repo: string, params?: PaginationParams): Promise<{
    metrics: CodebaseMetrics[];
    total: number;
  }> {
    return this.client.listCodebaseMetricsHistory(repo, params);
  }

  /**
   * Get complexity hotspots in the codebase.
   */
  async getHotspots(repo: string, params?: PaginationParams): Promise<{
    hotspots: Array<{
      file_path: string;
      complexity: number;
      churn: number;
      risk_score: number;
    }>;
    total: number;
  }> {
    return this.client.getCodebaseHotspots(repo, params);
  }

  /**
   * Get code duplication report.
   */
  async getDuplicates(repo: string, params?: PaginationParams): Promise<{
    duplicates: Array<{ files: string[]; lines: number; tokens: number }>;
    total: number;
    total_duplicated_lines: number;
  }> {
    return this.client.getCodebaseDuplicates(repo, params);
  }

  /**
   * Get metrics for a specific file.
   */
  async getFileMetrics(repo: string, filePath: string): Promise<{
    file_path: string;
    lines: number;
    complexity: number;
    maintainability: number;
  }> {
    return this.client.getCodebaseFileMetrics(repo, filePath);
  }

  // ===========================================================================
  // Code Intelligence
  // ===========================================================================

  /**
   * Run full codebase analysis.
   */
  async analyze(repo: string, body: {
    repo_path: string;
    depth?: number;
  }): Promise<CodebaseUnderstanding> {
    return this.client.analyzeCodebase(repo, body);
  }

  /**
   * Get symbols (classes, functions, etc.) from the codebase.
   */
  async getSymbols(repo: string, params?: {
    type?: string;
    limit?: number;
    offset?: number;
  }): Promise<CodebaseSymbols> {
    return this.client.getCodebaseSymbols(repo, params);
  }

  /**
   * Get the call graph for the codebase.
   */
  async getCallgraph(repo: string, params?: {
    entry_point?: string;
    depth?: number;
  }): Promise<CodebaseCallgraph> {
    return this.client.getCodebaseCallgraph(repo, params);
  }

  /**
   * Get dead code report.
   */
  async getDeadcode(repo: string, params?: {
    confidence_threshold?: number;
    limit?: number;
  }): Promise<CodebaseDeadcode> {
    return this.client.getCodebaseDeadcode(repo, params);
  }

  /**
   * Analyze impact of changes to a file.
   */
  async analyzeImpact(repo: string, body: {
    file_path: string;
    change_type?: string;
  }): Promise<CodebaseImpact> {
    return this.client.analyzeCodebaseImpact(repo, body);
  }

  /**
   * Get AI-powered understanding of the codebase.
   */
  async understand(repo: string, body: {
    query?: string;
    include_architecture?: boolean;
  }): Promise<CodebaseUnderstanding> {
    return this.client.understandCodebase(repo, body);
  }

  // ===========================================================================
  // Audits
  // ===========================================================================

  /**
   * Start a comprehensive codebase audit.
   */
  async startAudit(repo: string, body: {
    repo_path: string;
    categories?: string[];
  }): Promise<CodebaseAudit> {
    return this.client.startCodebaseAudit(repo, body);
  }

  /**
   * Get audit results by ID.
   */
  async getAudit(repo: string, auditId: string): Promise<CodebaseAudit> {
    return this.client.getCodebaseAudit(repo, auditId);
  }

  // ===========================================================================
  // Utilities
  // ===========================================================================

  /**
   * Clear analysis cache.
   */
  async clearCache(): Promise<{ success: boolean }> {
    return this.client.clearCodebaseCache();
  }
}
