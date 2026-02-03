/**
 * Backups Namespace Tests
 *
 * Comprehensive tests for the backups namespace API including:
 * - Backup CRUD operations
 * - Verification
 * - Restore testing
 * - Cleanup
 * - Statistics
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { BackupsAPI } from '../backups';

interface MockClient {
  request: Mock;
}

describe('BackupsAPI Namespace', () => {
  let api: BackupsAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new BackupsAPI(mockClient as any);
  });

  // ===========================================================================
  // Backup Listing
  // ===========================================================================

  describe('Backup Listing', () => {
    it('should list backups', async () => {
      const mockBackups = {
        backups: [
          {
            id: 'b1',
            source_path: '/data/db',
            backup_type: 'full',
            status: 'completed',
            size_bytes: 1000000,
          },
          {
            id: 'b2',
            source_path: '/data/db',
            backup_type: 'incremental',
            status: 'completed',
            size_bytes: 50000,
          },
        ],
        pagination: { limit: 50, offset: 0, total: 2, has_more: false },
      };
      mockClient.request.mockResolvedValue(mockBackups);

      const result = await api.list();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v2/backups', { params: undefined });
      expect(result.backups).toHaveLength(2);
    });

    it('should list backups with filters', async () => {
      const mockBackups = {
        backups: [{ id: 'b1', status: 'verified' }],
        pagination: { limit: 10, offset: 0, total: 1, has_more: false },
      };
      mockClient.request.mockResolvedValue(mockBackups);

      await api.list({
        limit: 10,
        status: 'verified',
        backup_type: 'full',
        since: '2024-01-01',
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v2/backups', {
        params: { limit: 10, status: 'verified', backup_type: 'full', since: '2024-01-01' },
      });
    });

    it('should get backup by ID', async () => {
      const mockBackup = {
        id: 'b1',
        source_path: '/data/db',
        backup_path: '/backups/b1.tar.gz',
        backup_type: 'full',
        status: 'completed',
        verified: true,
        size_bytes: 1000000,
        compressed_size_bytes: 500000,
        checksum: 'sha256:abc123',
      };
      mockClient.request.mockResolvedValue(mockBackup);

      const result = await api.get('b1');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v2/backups/b1');
      expect(result.verified).toBe(true);
    });
  });

  // ===========================================================================
  // Backup Creation
  // ===========================================================================

  describe('Backup Creation', () => {
    it('should create full backup', async () => {
      const mockBackup = {
        backup: {
          id: 'b_new',
          source_path: '/data/db',
          backup_type: 'full',
          status: 'pending',
        },
        message: 'Backup initiated',
      };
      mockClient.request.mockResolvedValue(mockBackup);

      const result = await api.create('/data/db');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v2/backups', {
        json: { source_path: '/data/db', backup_type: 'full', metadata: undefined },
      });
      expect(result.backup.id).toBe('b_new');
    });

    it('should create incremental backup with metadata', async () => {
      const mockBackup = {
        backup: { id: 'b_inc', backup_type: 'incremental' },
        message: 'Incremental backup initiated',
      };
      mockClient.request.mockResolvedValue(mockBackup);

      const result = await api.create('/data/db', {
        backup_type: 'incremental',
        metadata: { triggered_by: 'schedule' },
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v2/backups', {
        json: {
          source_path: '/data/db',
          backup_type: 'incremental',
          metadata: { triggered_by: 'schedule' },
        },
      });
    });
  });

  // ===========================================================================
  // Verification
  // ===========================================================================

  describe('Verification', () => {
    it('should verify backup', async () => {
      const mockVerification = {
        backup_id: 'b1',
        verified: true,
        checksum_valid: true,
        restore_tested: true,
        tables_valid: true,
        row_counts_valid: true,
        errors: [],
        warnings: [],
        verified_at: '2024-01-20T10:00:00Z',
        duration_seconds: 120,
      };
      mockClient.request.mockResolvedValue(mockVerification);

      const result = await api.verify('b1');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v2/backups/b1/verify');
      expect(result.verified).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should verify backup with errors', async () => {
      const mockVerification = {
        backup_id: 'b1',
        verified: false,
        checksum_valid: false,
        restore_tested: false,
        tables_valid: true,
        row_counts_valid: false,
        errors: ['Checksum mismatch', 'Row count mismatch in table users'],
        warnings: ['Backup is older than recommended'],
        verified_at: '2024-01-20T10:00:00Z',
        duration_seconds: 60,
      };
      mockClient.request.mockResolvedValue(mockVerification);

      const result = await api.verify('b1');

      expect(result.verified).toBe(false);
      expect(result.errors).toHaveLength(2);
    });

    it('should perform comprehensive verification', async () => {
      const mockVerification = {
        backup_id: 'b1',
        verified: true,
        checksum_valid: true,
        restore_tested: true,
        tables_valid: true,
        row_counts_valid: true,
        schema_valid: true,
        referential_integrity_valid: true,
        per_table_checksums: { users: 'sha256:abc', debates: 'sha256:def' },
        orphan_count: 0,
        errors: [],
        warnings: [],
        verified_at: '2024-01-20T10:00:00Z',
        duration_seconds: 300,
      };
      mockClient.request.mockResolvedValue(mockVerification);

      const result = await api.verifyComprehensive('b1');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v2/backups/b1/verify-comprehensive');
      expect(result.schema_valid).toBe(true);
      expect(result.referential_integrity_valid).toBe(true);
    });
  });

  // ===========================================================================
  // Restore Testing
  // ===========================================================================

  describe('Restore Testing', () => {
    it('should test restore', async () => {
      const mockRestore = {
        backup_id: 'b1',
        restore_test_passed: true,
        target_path: '/tmp/restore_test',
        dry_run: true,
        message: 'Restore test successful',
      };
      mockClient.request.mockResolvedValue(mockRestore);

      const result = await api.testRestore('b1');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v2/backups/b1/restore-test', {
        json: { target_path: undefined },
      });
      expect(result.restore_test_passed).toBe(true);
    });

    it('should test restore to specific path', async () => {
      const mockRestore = {
        backup_id: 'b1',
        restore_test_passed: true,
        target_path: '/custom/path',
        dry_run: true,
        message: 'Restore test successful',
      };
      mockClient.request.mockResolvedValue(mockRestore);

      const result = await api.testRestore('b1', '/custom/path');

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v2/backups/b1/restore-test', {
        json: { target_path: '/custom/path' },
      });
    });
  });

  // ===========================================================================
  // Deletion
  // ===========================================================================

  describe('Deletion', () => {
    it('should delete backup', async () => {
      mockClient.request.mockResolvedValue({
        deleted: true,
        backup_id: 'b1',
        message: 'Backup deleted',
      });

      const result = await api.delete('b1');

      expect(mockClient.request).toHaveBeenCalledWith('DELETE', '/api/v2/backups/b1');
      expect(result.deleted).toBe(true);
    });
  });

  // ===========================================================================
  // Cleanup
  // ===========================================================================

  describe('Cleanup', () => {
    it('should run cleanup dry run', async () => {
      const mockCleanup = {
        dry_run: true,
        backup_ids: ['b1', 'b2', 'b3'],
        count: 3,
        message: 'Would delete 3 backups',
      };
      mockClient.request.mockResolvedValue(mockCleanup);

      const result = await api.cleanup();

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v2/backups/cleanup', {
        json: { dry_run: true },
      });
      expect(result.dry_run).toBe(true);
      expect(result.count).toBe(3);
    });

    it('should run actual cleanup', async () => {
      const mockCleanup = {
        dry_run: false,
        backup_ids: ['b1', 'b2'],
        count: 2,
        message: 'Deleted 2 backups',
      };
      mockClient.request.mockResolvedValue(mockCleanup);

      const result = await api.cleanup(false);

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v2/backups/cleanup', {
        json: { dry_run: false },
      });
      expect(result.dry_run).toBe(false);
    });
  });

  // ===========================================================================
  // Statistics
  // ===========================================================================

  describe('Statistics', () => {
    it('should get backup stats', async () => {
      const mockStats = {
        stats: {
          total_backups: 50,
          verified_backups: 45,
          failed_backups: 2,
          total_size_bytes: 5000000000,
          total_size_mb: 5000,
          latest_backup: {
            id: 'b50',
            status: 'verified',
            created_at: '2024-01-20T10:00:00Z',
          },
          retention_policy: {
            keep_daily: 7,
            keep_weekly: 4,
            keep_monthly: 12,
            min_backups: 5,
          },
        },
        generated_at: '2024-01-20T10:05:00Z',
      };
      mockClient.request.mockResolvedValue(mockStats);

      const result = await api.getStats();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v2/backups/stats');
      expect(result.stats.total_backups).toBe(50);
      expect(result.stats.retention_policy.keep_daily).toBe(7);
    });
  });
});
