/**
 * Backups Namespace API
 *
 * Provides REST API endpoints for backup and disaster recovery:
 * - List and manage backups
 * - Trigger manual backups
 * - Verify backup integrity
 * - Test restore (dry-run)
 * - Cleanup expired backups
 */

/**
 * Backup types.
 */
export type BackupType = 'full' | 'incremental' | 'differential';

/**
 * Backup status.
 */
export type BackupStatus =
  | 'pending'
  | 'in_progress'
  | 'completed'
  | 'verified'
  | 'failed'
  | 'expired';

/**
 * Backup record.
 */
export interface Backup {
  id: string;
  source_path: string;
  backup_path: string;
  backup_type: BackupType;
  status: BackupStatus;
  verified: boolean;
  created_at: string;
  completed_at?: string;
  size_bytes: number;
  compressed_size_bytes: number;
  checksum?: string;
  metadata?: Record<string, unknown>;
}

/**
 * Backup verification result.
 */
export interface VerificationResult {
  backup_id: string;
  verified: boolean;
  checksum_valid: boolean;
  restore_tested: boolean;
  tables_valid: boolean;
  row_counts_valid: boolean;
  errors: string[];
  warnings: string[];
  verified_at: string;
  duration_seconds: number;
}

/**
 * Comprehensive verification result.
 */
export interface ComprehensiveVerificationResult extends VerificationResult {
  schema_valid: boolean;
  referential_integrity_valid: boolean;
  per_table_checksums: Record<string, string>;
  orphan_count: number;
}

/**
 * Retention policy configuration.
 */
export interface RetentionPolicy {
  keep_daily: number;
  keep_weekly: number;
  keep_monthly: number;
  min_backups: number;
}

/**
 * Backup statistics.
 */
export interface BackupStats {
  total_backups: number;
  verified_backups: number;
  failed_backups: number;
  total_size_bytes: number;
  total_size_mb: number;
  latest_backup: Backup | null;
  retention_policy: RetentionPolicy;
}

/**
 * Backups API for disaster recovery operations.
 */
export class BackupsAPI {
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
   * List backups with filtering and pagination.
   *
   * @param options - Filter and pagination options
   */
  async list(options?: {
    limit?: number;
    offset?: number;
    source?: string;
    status?: BackupStatus;
    since?: string;
    backup_type?: BackupType;
  }): Promise<{
    backups: Backup[];
    pagination: {
      limit: number;
      offset: number;
      total: number;
      has_more: boolean;
    };
  }> {
    const params = new URLSearchParams();
    if (options?.limit) params.set('limit', options.limit.toString());
    if (options?.offset) params.set('offset', options.offset.toString());
    if (options?.source) params.set('source', options.source);
    if (options?.status) params.set('status', options.status);
    if (options?.since) params.set('since', options.since);
    if (options?.backup_type) params.set('backup_type', options.backup_type);

    const url = `${this.baseUrl}/api/v2/backups${params.toString() ? `?${params}` : ''}`;
    const response = await fetch(url, {
      method: 'GET',
      headers: this.headers,
    });
    if (!response.ok) throw new Error(`Failed to list backups: ${response.statusText}`);
    return response.json();
  }

  /**
   * Get a specific backup by ID.
   *
   * @param backupId - Backup ID
   */
  async get(backupId: string): Promise<Backup> {
    const response = await fetch(`${this.baseUrl}/api/v2/backups/${backupId}`, {
      method: 'GET',
      headers: this.headers,
    });
    if (!response.ok) throw new Error(`Failed to get backup: ${response.statusText}`);
    return response.json();
  }

  /**
   * Create a new backup.
   *
   * @param sourcePath - Path to the database to backup
   * @param options - Backup options
   */
  async create(
    sourcePath: string,
    options?: {
      backup_type?: BackupType;
      metadata?: Record<string, unknown>;
    }
  ): Promise<{ backup: Backup; message: string }> {
    const response = await fetch(`${this.baseUrl}/api/v2/backups`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        source_path: sourcePath,
        backup_type: options?.backup_type ?? 'full',
        metadata: options?.metadata,
      }),
    });
    if (!response.ok) throw new Error(`Failed to create backup: ${response.statusText}`);
    return response.json();
  }

  /**
   * Verify backup integrity with restore test.
   *
   * @param backupId - Backup ID to verify
   */
  async verify(backupId: string): Promise<VerificationResult> {
    const response = await fetch(
      `${this.baseUrl}/api/v2/backups/${backupId}/verify`,
      {
        method: 'POST',
        headers: this.headers,
      }
    );
    if (!response.ok) throw new Error(`Failed to verify backup: ${response.statusText}`);
    return response.json();
  }

  /**
   * Perform comprehensive verification of a backup.
   *
   * @param backupId - Backup ID to verify
   */
  async verifyComprehensive(backupId: string): Promise<ComprehensiveVerificationResult> {
    const response = await fetch(
      `${this.baseUrl}/api/v2/backups/${backupId}/verify-comprehensive`,
      {
        method: 'POST',
        headers: this.headers,
      }
    );
    if (!response.ok)
      throw new Error(`Failed to verify backup: ${response.statusText}`);
    return response.json();
  }

  /**
   * Test restore a backup (dry-run).
   *
   * @param backupId - Backup ID to test
   * @param targetPath - Optional target path for restore test info
   */
  async testRestore(
    backupId: string,
    targetPath?: string
  ): Promise<{
    backup_id: string;
    restore_test_passed: boolean;
    target_path: string;
    dry_run: boolean;
    message: string;
  }> {
    const response = await fetch(
      `${this.baseUrl}/api/v2/backups/${backupId}/restore-test`,
      {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify({ target_path: targetPath }),
      }
    );
    if (!response.ok) throw new Error(`Failed to test restore: ${response.statusText}`);
    return response.json();
  }

  /**
   * Delete a backup.
   *
   * @param backupId - Backup ID to delete
   */
  async delete(
    backupId: string
  ): Promise<{ deleted: boolean; backup_id: string; message: string }> {
    const response = await fetch(`${this.baseUrl}/api/v2/backups/${backupId}`, {
      method: 'DELETE',
      headers: this.headers,
    });
    if (!response.ok) throw new Error(`Failed to delete backup: ${response.statusText}`);
    return response.json();
  }

  /**
   * Run retention policy cleanup.
   *
   * @param dryRun - If true, only report what would be deleted (default true)
   */
  async cleanup(dryRun = true): Promise<{
    dry_run: boolean;
    backup_ids: string[];
    count: number;
    message: string;
  }> {
    const response = await fetch(`${this.baseUrl}/api/v2/backups/cleanup`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({ dry_run: dryRun }),
    });
    if (!response.ok) throw new Error(`Failed to cleanup backups: ${response.statusText}`);
    return response.json();
  }

  /**
   * Get backup statistics.
   */
  async getStats(): Promise<{ stats: BackupStats; generated_at: string }> {
    const response = await fetch(`${this.baseUrl}/api/v2/backups/stats`, {
      method: 'GET',
      headers: this.headers,
    });
    if (!response.ok) throw new Error(`Failed to get stats: ${response.statusText}`);
    return response.json();
  }
}
