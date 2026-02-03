/**
 * Compliance Namespace Tests
 *
 * Comprehensive tests for the compliance namespace API including:
 * - Compliance status
 * - SOC 2 reporting
 * - GDPR compliance (export, right to be forgotten)
 * - Audit trail verification
 * - SIEM event export
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { ComplianceAPI } from '../compliance';

interface MockClient {
  request: Mock;
}

describe('ComplianceAPI Namespace', () => {
  let api: ComplianceAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new ComplianceAPI(mockClient as any);
  });

  // ===========================================================================
  // Compliance Status
  // ===========================================================================

  describe('Compliance Status', () => {
    it('should get overall compliance status', async () => {
      const mockStatus = {
        soc2: {
          compliant: true,
          last_audit: '2024-01-15',
          findings_count: 0,
        },
        gdpr: {
          compliant: true,
          data_processing_agreement: true,
          dpo_configured: true,
        },
        hipaa: {
          compliant: false,
          baa_signed: false,
        },
        overall_status: 'partial',
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.getStatus();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v2/compliance/status');
      expect(result.soc2.compliant).toBe(true);
      expect(result.gdpr.compliant).toBe(true);
      expect(result.overall_status).toBe('partial');
    });
  });

  // ===========================================================================
  // SOC 2 Compliance
  // ===========================================================================

  describe('SOC 2 Compliance', () => {
    it('should generate SOC 2 report', async () => {
      const mockReport = {
        report_id: 'rpt_soc2_2024',
        period_start: '2024-01-01',
        period_end: '2024-12-31',
        controls: [
          {
            control_id: 'CC1.1',
            control_name: 'Logical Access Security',
            category: 'Security',
            status: 'pass',
            evidence_count: 15,
          },
        ],
        findings: [],
        overall_assessment: 'pass',
        generated_at: '2024-01-20T10:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockReport);

      const result = await api.generateSoc2Report({
        startDate: '2024-01-01',
        endDate: '2024-12-31',
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v2/compliance/soc2-report', {
        params: {
          start_date: '2024-01-01',
          end_date: '2024-12-31',
        },
      });
      expect(result.overall_assessment).toBe('pass');
      expect(result.controls).toHaveLength(1);
    });

    it('should generate SOC 2 report with specific controls', async () => {
      const mockReport = {
        report_id: 'rpt_soc2_partial',
        controls: [{ control_id: 'CC1.1', status: 'pass' }],
        findings: [],
        overall_assessment: 'pass',
      };
      mockClient.request.mockResolvedValue(mockReport);

      const result = await api.generateSoc2Report({
        controls: ['CC1.1', 'CC1.2', 'CC2.1'],
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v2/compliance/soc2-report', {
        params: {
          controls: 'CC1.1,CC1.2,CC2.1',
        },
      });
      expect(result.overall_assessment).toBe('pass');
    });

    it('should generate SOC 2 report with findings', async () => {
      const mockReport = {
        report_id: 'rpt_soc2_findings',
        controls: [{ control_id: 'CC1.1', status: 'fail' }],
        findings: [
          {
            finding_id: 'f1',
            severity: 'medium',
            control_id: 'CC1.1',
            description: 'MFA not enforced for all users',
            remediation: 'Enable MFA enforcement',
            status: 'open',
          },
        ],
        overall_assessment: 'pass_with_exceptions',
      };
      mockClient.request.mockResolvedValue(mockReport);

      const result = await api.generateSoc2Report();

      expect(result.overall_assessment).toBe('pass_with_exceptions');
      expect(result.findings).toHaveLength(1);
      expect(result.findings[0].severity).toBe('medium');
    });
  });

  // ===========================================================================
  // GDPR Compliance
  // ===========================================================================

  describe('GDPR Compliance', () => {
    it('should export user data (Article 15)', async () => {
      const mockExport = {
        export_id: 'exp_123',
        user_id: 'user_456',
        format: 'json',
        data: {
          profile: { name: 'John Doe', email: 'john@example.com' },
          debates: [{ id: 'd1', topic: 'Test debate' }],
          activity: [{ action: 'login', timestamp: '2024-01-01' }],
        },
        generated_at: '2024-01-20T10:00:00Z',
        expires_at: '2024-01-27T10:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockExport);

      const result = await api.gdprExport('user_456');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v2/compliance/gdpr-export', {
        params: { user_id: 'user_456', format: 'json' },
      });
      expect(result.user_id).toBe('user_456');
      expect(result.data.profile).toBeDefined();
    });

    it('should export user data in CSV format', async () => {
      const mockExport = {
        export_id: 'exp_124',
        user_id: 'user_456',
        format: 'csv',
        data: {},
      };
      mockClient.request.mockResolvedValue(mockExport);

      const result = await api.gdprExport('user_456', 'csv');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v2/compliance/gdpr-export', {
        params: { user_id: 'user_456', format: 'csv' },
      });
      expect(result.format).toBe('csv');
    });

    it('should execute right to be forgotten (Article 17)', async () => {
      const mockDeletion = {
        deletion_id: 'del_123',
        user_id: 'user_456',
        status: 'completed',
        data_deleted: ['profile', 'debates', 'activity', 'preferences'],
        data_retained: ['invoices', 'audit_logs'],
        retention_reason: 'Legal compliance requirement',
        completed_at: '2024-01-20T10:05:00Z',
      };
      mockClient.request.mockResolvedValue(mockDeletion);

      const result = await api.gdprRightToBeForgotten('user_456', {
        confirm: true,
        reason: 'User requested account deletion',
      });

      expect(mockClient.request).toHaveBeenCalledWith(
        'POST',
        '/api/v2/compliance/gdpr/right-to-be-forgotten',
        {
          json: {
            user_id: 'user_456',
            confirm: true,
            reason: 'User requested account deletion',
          },
        }
      );
      expect(result.status).toBe('completed');
      expect(result.data_deleted).toContain('profile');
      expect(result.data_retained).toContain('invoices');
    });

    it('should handle pending deletion request', async () => {
      const mockDeletion = {
        deletion_id: 'del_124',
        user_id: 'user_789',
        status: 'pending',
        data_deleted: [],
        data_retained: [],
      };
      mockClient.request.mockResolvedValue(mockDeletion);

      const result = await api.gdprRightToBeForgotten('user_789');

      expect(result.status).toBe('pending');
    });
  });

  // ===========================================================================
  // Audit Trail
  // ===========================================================================

  describe('Audit Trail', () => {
    it('should verify audit trail integrity', async () => {
      const mockVerification = {
        verified: true,
        period_start: '2024-01-01',
        period_end: '2024-01-31',
        events_checked: 15000,
        anomalies: [],
        integrity_hash: 'sha256:abc123...',
        verified_at: '2024-01-20T10:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockVerification);

      const result = await api.verifyAuditTrail({
        startDate: '2024-01-01',
        endDate: '2024-01-31',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v2/compliance/audit-verify', {
        json: {
          start_date: '2024-01-01',
          end_date: '2024-01-31',
        },
      });
      expect(result.verified).toBe(true);
      expect(result.anomalies).toHaveLength(0);
    });

    it('should detect audit trail anomalies', async () => {
      const mockVerification = {
        verified: false,
        period_start: '2024-01-01',
        period_end: '2024-01-31',
        events_checked: 15000,
        anomalies: [
          {
            anomaly_id: 'anom_1',
            type: 'gap',
            severity: 'high',
            description: 'Missing events between 2024-01-15 10:00 and 10:30',
            event_ids: [],
            detected_at: '2024-01-20T10:00:00Z',
          },
          {
            anomaly_id: 'anom_2',
            type: 'sequence_error',
            severity: 'medium',
            description: 'Event sequence number mismatch',
            event_ids: ['evt_100', 'evt_102'],
            detected_at: '2024-01-20T10:00:00Z',
          },
        ],
        integrity_hash: 'sha256:def456...',
        verified_at: '2024-01-20T10:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockVerification);

      const result = await api.verifyAuditTrail();

      expect(result.verified).toBe(false);
      expect(result.anomalies).toHaveLength(2);
      expect(result.anomalies[0].type).toBe('gap');
    });

    it('should export audit events for SIEM', async () => {
      const mockExport = {
        events: [
          {
            event_id: 'evt_1',
            event_type: 'authentication',
            timestamp: '2024-01-20T09:00:00Z',
            user_id: 'user_123',
            action: 'login',
            outcome: 'success',
            ip_address: '192.168.1.1',
          },
          {
            event_id: 'evt_2',
            event_type: 'data_access',
            timestamp: '2024-01-20T09:05:00Z',
            user_id: 'user_123',
            resource_type: 'debate',
            resource_id: 'd_456',
            action: 'read',
            outcome: 'success',
          },
        ],
        total_count: 1500,
        period_start: '2024-01-01',
        period_end: '2024-01-31',
        format: 'json',
        exported_at: '2024-01-20T10:00:00Z',
      };
      mockClient.request.mockResolvedValue(mockExport);

      const result = await api.exportAuditEvents('2024-01-01', '2024-01-31', {
        eventTypes: ['authentication', 'data_access'],
        limit: 1000,
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v2/compliance/audit-events', {
        params: {
          start_date: '2024-01-01',
          end_date: '2024-01-31',
          format: 'json',
          limit: 1000,
          event_types: 'authentication,data_access',
        },
      });
      expect(result.events).toHaveLength(2);
      expect(result.total_count).toBe(1500);
    });

    it('should export audit events in elasticsearch format', async () => {
      const mockExport = {
        events: [],
        total_count: 0,
        format: 'elasticsearch',
      };
      mockClient.request.mockResolvedValue(mockExport);

      const result = await api.exportAuditEvents('2024-01-01', '2024-01-31', {
        format: 'elasticsearch',
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v2/compliance/audit-events', {
        params: {
          start_date: '2024-01-01',
          end_date: '2024-01-31',
          format: 'elasticsearch',
          limit: 1000,
        },
      });
      expect(result.format).toBe('elasticsearch');
    });
  });
});
