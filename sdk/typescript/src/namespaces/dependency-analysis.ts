/**
 * Dependency Analysis Namespace API
 *
 * Provides a namespaced interface for codebase dependency analysis operations.
 * This wraps the flat client methods for SBOM generation, vulnerability scanning,
 * and license compliance checking.
 */

/**
 * Dependency analysis request.
 */
export interface AnalyzeDependenciesRequest {
  repository_url?: string;
  local_path?: string;
  package_managers?: ('npm' | 'pip' | 'cargo' | 'go' | 'maven' | 'gradle')[];
  include_dev_dependencies?: boolean;
  include_transitive?: boolean;
  max_depth?: number;
}

/**
 * Dependency information.
 */
export interface Dependency {
  name: string;
  version: string;
  package_manager: string;
  is_direct: boolean;
  is_dev: boolean;
  license: string | null;
  repository_url: string | null;
  dependencies: string[];
}

/**
 * Dependency analysis result.
 */
export interface DependencyAnalysisResult {
  total_dependencies: number;
  direct_dependencies: number;
  transitive_dependencies: number;
  dev_dependencies: number;
  dependencies: Dependency[];
  dependency_tree: Record<string, string[]>;
  analysis_time_ms: number;
}

/**
 * SBOM generation request.
 */
export interface GenerateSBOMRequest {
  repository_url?: string;
  local_path?: string;
  format?: 'spdx' | 'cyclonedx' | 'json';
  include_checksums?: boolean;
  include_licenses?: boolean;
}

/**
 * SBOM result.
 */
export interface SBOMResult {
  format: string;
  version: string;
  created_at: string;
  tool: string;
  components: SBOMComponent[];
  relationships: SBOMRelationship[];
}

/**
 * SBOM component.
 */
export interface SBOMComponent {
  type: 'library' | 'framework' | 'application' | 'file';
  name: string;
  version: string;
  purl?: string;
  licenses: string[];
  checksums?: Record<string, string>;
  external_references?: { type: string; url: string }[];
}

/**
 * SBOM relationship.
 */
export interface SBOMRelationship {
  source: string;
  target: string;
  type: 'depends_on' | 'dev_depends_on' | 'optional_depends_on';
}

/**
 * Vulnerability scan request.
 */
export interface ScanVulnerabilitiesRequest {
  repository_url?: string;
  local_path?: string;
  sbom?: SBOMResult;
  severity_threshold?: 'low' | 'medium' | 'high' | 'critical';
  ignore_cves?: string[];
}

/**
 * Vulnerability information.
 */
export interface Vulnerability {
  cve_id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  cvss_score: number;
  title: string;
  description: string;
  affected_package: string;
  affected_versions: string;
  fixed_version: string | null;
  references: string[];
  published_at: string;
  exploitability: 'none' | 'poc' | 'active';
}

/**
 * Vulnerability scan result.
 */
export interface VulnerabilityScanResult {
  total_vulnerabilities: number;
  critical: number;
  high: number;
  medium: number;
  low: number;
  vulnerabilities: Vulnerability[];
  scan_time_ms: number;
  databases_checked: string[];
}

/**
 * License check request.
 */
export interface CheckLicensesRequest {
  repository_url?: string;
  local_path?: string;
  allowed_licenses?: string[];
  denied_licenses?: string[];
  policy?: 'permissive' | 'copyleft' | 'commercial' | 'custom';
}

/**
 * License information.
 */
export interface LicenseInfo {
  package: string;
  version: string;
  license: string;
  license_url: string | null;
  is_osi_approved: boolean;
  is_copyleft: boolean;
  compatibility: 'compatible' | 'incompatible' | 'unknown';
  issues: string[];
}

/**
 * License check result.
 */
export interface LicenseCheckResult {
  compliant: boolean;
  total_packages: number;
  compatible: number;
  incompatible: number;
  unknown: number;
  licenses: LicenseInfo[];
  policy_used: string;
  issues: string[];
}

/**
 * Interface for the internal client methods used by DependencyAnalysisAPI.
 */
interface DependencyAnalysisClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: unknown }
  ): Promise<T>;
}

/**
 * Dependency Analysis API namespace.
 *
 * Provides methods for codebase dependency analysis:
 * - Analyzing project dependencies
 * - Generating SBOM (Software Bill of Materials)
 * - Scanning for vulnerabilities (CVEs)
 * - Checking license compliance
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Analyze dependencies
 * const deps = await client.dependencyAnalysis.analyze({
 *   repository_url: 'https://github.com/org/repo',
 *   include_transitive: true,
 * });
 *
 * // Generate SBOM
 * const sbom = await client.dependencyAnalysis.generateSBOM({
 *   repository_url: 'https://github.com/org/repo',
 *   format: 'cyclonedx',
 * });
 *
 * // Scan for vulnerabilities
 * const vulns = await client.dependencyAnalysis.scanVulnerabilities({
 *   sbom,
 *   severity_threshold: 'medium',
 * });
 *
 * // Check licenses
 * const licenses = await client.dependencyAnalysis.checkLicenses({
 *   repository_url: 'https://github.com/org/repo',
 *   policy: 'permissive',
 * });
 * ```
 */
export class DependencyAnalysisAPI {
  constructor(private client: DependencyAnalysisClientInterface) {}

  /**
   * Analyze project dependencies.
   */
  async analyze(request: AnalyzeDependenciesRequest): Promise<DependencyAnalysisResult> {
    return this.client.request<DependencyAnalysisResult>('POST', '/api/v1/codebase/analyze-dependencies', {
      json: request,
    });
  }

  /**
   * Generate Software Bill of Materials (SBOM).
   */
  async generateSBOM(request: GenerateSBOMRequest): Promise<SBOMResult> {
    return this.client.request<SBOMResult>('POST', '/api/v1/codebase/sbom', {
      json: request,
    });
  }

  /**
   * Scan for known vulnerabilities (CVEs).
   */
  async scanVulnerabilities(request: ScanVulnerabilitiesRequest): Promise<VulnerabilityScanResult> {
    return this.client.request<VulnerabilityScanResult>('POST', '/api/v1/codebase/scan-vulnerabilities', {
      json: request,
    });
  }

  /**
   * Check license compliance.
   */
  async checkLicenses(request: CheckLicensesRequest): Promise<LicenseCheckResult> {
    return this.client.request<LicenseCheckResult>('POST', '/api/v1/codebase/check-licenses', {
      json: request,
    });
  }

  /**
   * Clear analysis cache.
   */
  async clearCache(repositoryUrl?: string): Promise<{ cleared: boolean }> {
    return this.client.request<{ cleared: boolean }>('POST', '/api/v1/codebase/clear-cache', {
      json: { repository_url: repositoryUrl },
    });
  }

  /**
   * Run a full security audit (dependencies + vulnerabilities + licenses).
   * Convenience method that combines multiple checks.
   */
  async fullAudit(
    request: AnalyzeDependenciesRequest & { license_policy?: 'permissive' | 'copyleft' | 'commercial' }
  ): Promise<{
    dependencies: DependencyAnalysisResult;
    sbom: SBOMResult;
    vulnerabilities: VulnerabilityScanResult;
    licenses: LicenseCheckResult;
    summary: {
      total_dependencies: number;
      total_vulnerabilities: number;
      license_compliant: boolean;
      risk_level: 'low' | 'medium' | 'high' | 'critical';
    };
  }> {
    // Run dependency analysis first
    const dependencies = await this.analyze(request);

    // Generate SBOM
    const sbom = await this.generateSBOM({
      repository_url: request.repository_url,
      local_path: request.local_path,
      format: 'cyclonedx',
      include_licenses: true,
    });

    // Scan for vulnerabilities using the SBOM
    const vulnerabilities = await this.scanVulnerabilities({
      sbom,
      severity_threshold: 'low',
    });

    // Check licenses
    const licenses = await this.checkLicenses({
      repository_url: request.repository_url,
      local_path: request.local_path,
      policy: request.license_policy || 'permissive',
    });

    // Determine overall risk level
    let riskLevel: 'low' | 'medium' | 'high' | 'critical' = 'low';
    if (vulnerabilities.critical > 0) {
      riskLevel = 'critical';
    } else if (vulnerabilities.high > 0 || !licenses.compliant) {
      riskLevel = 'high';
    } else if (vulnerabilities.medium > 0) {
      riskLevel = 'medium';
    }

    return {
      dependencies,
      sbom,
      vulnerabilities,
      licenses,
      summary: {
        total_dependencies: dependencies.total_dependencies,
        total_vulnerabilities: vulnerabilities.total_vulnerabilities,
        license_compliant: licenses.compliant,
        risk_level: riskLevel,
      },
    };
  }
}

export default DependencyAnalysisAPI;
