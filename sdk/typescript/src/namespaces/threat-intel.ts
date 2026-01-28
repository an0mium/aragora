/**
 * Threat Intelligence Namespace API
 *
 * Provides a namespaced interface for threat intelligence operations.
 * Supports URL scanning, IP reputation, file hash lookup, and email scanning.
 */

// =============================================================================
// Types
// =============================================================================

/** Threat severity levels */
export type ThreatSeverity = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE';

/** Threat types */
export type ThreatType =
  | 'malware'
  | 'phishing'
  | 'spam'
  | 'suspicious'
  | 'ransomware'
  | 'trojan'
  | 'botnet'
  | 'none';

/** Hash types */
export type HashType = 'md5' | 'sha1' | 'sha256';

/** URL check result */
export interface URLCheckResult {
  target: string;
  is_malicious: boolean;
  threat_type: ThreatType;
  severity: ThreatSeverity;
  confidence: number;
  virustotal?: {
    positives: number;
    total: number;
    scan_date: string;
    permalink: string;
  };
  phishtank?: {
    is_phish: boolean;
    phish_id: string | null;
    verified: boolean;
  };
  cached: boolean;
}

/** Batch URL check summary */
export interface URLBatchSummary {
  total: number;
  malicious: number;
  suspicious: number;
  clean: number;
}

/** IP reputation result */
export interface IPReputationResult {
  ip_address: string;
  is_malicious: boolean;
  abuse_score: number;
  total_reports: number;
  country_code: string | null;
  isp: string | null;
  domain: string | null;
  usage_type: string | null;
  last_reported: string | null;
  categories: string[];
  cached: boolean;
}

/** IP batch check summary */
export interface IPBatchSummary {
  total: number;
  malicious: number;
  clean: number;
}

/** File hash result */
export interface HashCheckResult {
  hash_value: string;
  hash_type: HashType;
  is_malware: boolean;
  threat_type: ThreatType;
  detection_ratio: string;
  positives: number;
  total_scanners: number;
  scan_date: string | null;
  file_name: string | null;
  file_size: number | null;
  file_type: string | null;
  cached: boolean;
}

/** Hash batch check summary */
export interface HashBatchSummary {
  total: number;
  malware: number;
  clean: number;
}

/** Email scan result */
export interface EmailScanResult {
  urls: URLCheckResult[];
  ips: IPReputationResult[];
  overall_threat_score: number;
  is_suspicious: boolean;
  threat_summary: string[];
}

/** Service status */
export interface ThreatIntelStatus {
  virustotal: {
    enabled: boolean;
    has_key: boolean;
    rate_limit: number;
  };
  abuseipdb: {
    enabled: boolean;
    has_key: boolean;
    rate_limit: number;
  };
  phishtank: {
    enabled: boolean;
    has_key: boolean;
    rate_limit: number;
  };
  caching: boolean;
  cache_ttl_hours: number;
}

/** URL check request */
export interface CheckURLRequest {
  url: string;
  check_virustotal?: boolean;
  check_phishtank?: boolean;
}

/** Batch URL check request */
export interface CheckURLsBatchRequest {
  urls: string[];
  max_concurrent?: number;
}

/** Batch IP check request */
export interface CheckIPsBatchRequest {
  ips: string[];
}

/** Batch hash check request */
export interface CheckHashesBatchRequest {
  hashes: string[];
}

/** Email scan request */
export interface ScanEmailRequest {
  body: string;
  headers?: Record<string, string>;
}

// =============================================================================
// Threat Intel API
// =============================================================================

/**
 * Client interface for threat intel operations.
 */
interface ThreatIntelClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Threat Intelligence namespace API for security scanning.
 *
 * Provides comprehensive threat detection:
 * - URL scanning against VirusTotal and PhishTank
 * - IP reputation checking via AbuseIPDB
 * - File hash lookup in VirusTotal
 * - Email content analysis
 */
export class ThreatIntelAPI {
  constructor(private client: ThreatIntelClientInterface) {}

  // ===========================================================================
  // URL Scanning
  // ===========================================================================

  /**
   * Check a URL for threats.
   *
   * Scans against VirusTotal and PhishTank for malware and phishing detection.
   *
   * @param request - URL and scan options
   * @returns Threat analysis result
   */
  async checkURL(request: CheckURLRequest): Promise<URLCheckResult> {
    const response = await this.client.request<URLCheckResult | { data: URLCheckResult }>('POST', '/threat/url', {
      json: request as unknown as Record<string, unknown>,
    });
    return (response as { data: URLCheckResult }).data ?? (response as URLCheckResult);
  }

  /**
   * Batch check multiple URLs for threats.
   *
   * @param request - URLs to check and concurrency options
   * @returns Results and summary
   */
  async checkURLsBatch(request: CheckURLsBatchRequest): Promise<{
    results: URLCheckResult[];
    summary: URLBatchSummary;
  }> {
    type ResponseType = { results: URLCheckResult[]; summary: URLBatchSummary };
    const response = await this.client.request<ResponseType | { data: ResponseType }>('POST', '/threat/urls', {
      json: request as unknown as Record<string, unknown>,
    });
    return (response as { data: ResponseType }).data ?? (response as ResponseType);
  }

  // ===========================================================================
  // IP Reputation
  // ===========================================================================

  /**
   * Check IP address reputation.
   *
   * Gets reputation data from AbuseIPDB including abuse score and reports.
   *
   * @param ipAddress - IP address to check
   * @returns Reputation result
   */
  async checkIP(ipAddress: string): Promise<IPReputationResult> {
    const response = await this.client.request<IPReputationResult | { data: IPReputationResult }>('GET', `/threat/ip/${encodeURIComponent(ipAddress)}`);
    return (response as { data: IPReputationResult }).data ?? (response as IPReputationResult);
  }

  /**
   * Batch check multiple IP addresses.
   *
   * @param request - IPs to check
   * @returns Results and summary
   */
  async checkIPsBatch(request: CheckIPsBatchRequest): Promise<{
    results: IPReputationResult[];
    summary: IPBatchSummary;
  }> {
    type ResponseType = { results: IPReputationResult[]; summary: IPBatchSummary };
    const response = await this.client.request<ResponseType | { data: ResponseType }>('POST', '/threat/ips', {
      json: request as unknown as Record<string, unknown>,
    });
    return (response as { data: ResponseType }).data ?? (response as ResponseType);
  }

  // ===========================================================================
  // File Hash Lookup
  // ===========================================================================

  /**
   * Check file hash for malware.
   *
   * Looks up MD5, SHA1, or SHA256 hash in VirusTotal.
   *
   * @param hashValue - File hash to check
   * @returns Hash analysis result
   */
  async checkHash(hashValue: string): Promise<HashCheckResult> {
    const response = await this.client.request<HashCheckResult | { data: HashCheckResult }>('GET', `/threat/hash/${encodeURIComponent(hashValue)}`);
    return (response as { data: HashCheckResult }).data ?? (response as HashCheckResult);
  }

  /**
   * Batch check multiple file hashes.
   *
   * @param request - Hashes to check
   * @returns Results and summary
   */
  async checkHashesBatch(request: CheckHashesBatchRequest): Promise<{
    results: HashCheckResult[];
    summary: HashBatchSummary;
  }> {
    type ResponseType = { results: HashCheckResult[]; summary: HashBatchSummary };
    const response = await this.client.request<ResponseType | { data: ResponseType }>('POST', '/threat/hashes', {
      json: request as unknown as Record<string, unknown>,
    });
    return (response as { data: ResponseType }).data ?? (response as ResponseType);
  }

  // ===========================================================================
  // Email Scanning
  // ===========================================================================

  /**
   * Scan email content for threats.
   *
   * Extracts and checks URLs and IPs from email body and headers.
   *
   * @param request - Email body and optional headers
   * @returns Comprehensive email analysis
   */
  async scanEmail(request: ScanEmailRequest): Promise<EmailScanResult> {
    const response = await this.client.request<EmailScanResult | { data: EmailScanResult }>('POST', '/threat/email', {
      json: request as unknown as Record<string, unknown>,
    });
    return (response as { data: EmailScanResult }).data ?? (response as EmailScanResult);
  }

  // ===========================================================================
  // Service Status
  // ===========================================================================

  /**
   * Get threat intelligence service status.
   *
   * Shows which providers are configured and available.
   *
   * @returns Service status for all providers
   */
  async getStatus(): Promise<ThreatIntelStatus> {
    const response = await this.client.request<ThreatIntelStatus | { data: ThreatIntelStatus }>('GET', '/threat/status');
    return (response as { data: ThreatIntelStatus }).data ?? (response as ThreatIntelStatus);
  }
}
