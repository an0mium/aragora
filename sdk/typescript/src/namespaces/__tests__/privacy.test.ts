/**
 * Privacy Namespace Tests
 *
 * Comprehensive tests for the privacy namespace API including:
 * - Data export (GDPR Article 15)
 * - Data inventory
 * - Privacy preferences
 * - Account deletion (GDPR Article 17)
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { PrivacyAPI } from '../privacy';

interface MockClient {
  request: Mock;
}

describe('PrivacyAPI Namespace', () => {
  let api: PrivacyAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new PrivacyAPI(mockClient as any);
  });

  // ===========================================================================
  // Data Export
  // ===========================================================================

  describe('Data Export', () => {
    it('should export data in JSON format', async () => {
      const mockExport = {
        profile: {
          id: 'u123',
          email: 'user@example.com',
          name: 'Test User',
          role: 'member',
          is_active: true,
          email_verified: true,
          created_at: '2024-01-01T00:00:00Z',
          mfa_enabled: false,
        },
        organization: {
          id: 'org123',
          name: 'Acme Corp',
          slug: 'acme',
          tier: 'pro',
          role: 'member',
        },
        oauth_providers: [{ provider: 'google', linked_at: '2024-01-01' }],
        _export_metadata: {
          exported_at: '2024-01-20T10:00:00Z',
          format: 'json',
          data_controller: 'Aragora',
          contact: 'privacy@aragora.ai',
          legal_basis: 'GDPR Article 15',
        },
      };
      mockClient.request.mockResolvedValue(mockExport);

      const result = await api.exportData();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/privacy/export', {
        params: { format: 'json' },
      });
      expect(result.profile.email).toBe('user@example.com');
      expect(result._export_metadata.format).toBe('json');
    });

    it('should export data in CSV format', async () => {
      const mockExport = {
        profile: { id: 'u123', email: 'user@example.com' },
        _export_metadata: { format: 'csv' },
      };
      mockClient.request.mockResolvedValue(mockExport);

      const result = await api.exportData('csv');

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/privacy/export', {
        params: { format: 'csv' },
      });
      expect(result._export_metadata.format).toBe('csv');
    });

    it('should include audit log in export', async () => {
      const mockExport = {
        profile: { id: 'u123' },
        audit_log: [
          { timestamp: '2024-01-20T10:00:00Z', action: 'login', resource_type: 'session' },
          { timestamp: '2024-01-20T09:00:00Z', action: 'create', resource_type: 'debate', resource_id: 'd1' },
        ],
        _export_metadata: {},
      };
      mockClient.request.mockResolvedValue(mockExport);

      const result = await api.exportData();

      expect(result.audit_log).toHaveLength(2);
    });
  });

  // ===========================================================================
  // Data Inventory
  // ===========================================================================

  describe('Data Inventory', () => {
    it('should get data inventory', async () => {
      const mockInventory = {
        categories: [
          {
            name: 'Account Information',
            examples: ['email', 'name', 'profile picture'],
            purpose: 'Account management and authentication',
            retention: '5 years after account closure',
          },
          {
            name: 'Usage Data',
            examples: ['debates created', 'API calls', 'login history'],
            purpose: 'Service improvement and billing',
            retention: '2 years',
          },
        ],
        third_party_sharing: {
          llm_providers: {
            recipients: ['Anthropic', 'OpenAI', 'Google'],
            data_shared: 'Debate content for AI processing',
            purpose: 'AI-powered debate orchestration',
          },
          analytics: {
            recipients: ['PostHog'],
            data_shared: 'Anonymized usage metrics',
            purpose: 'Product improvement',
          },
        },
        data_sold: false,
        opt_out_available: true,
      };
      mockClient.request.mockResolvedValue(mockInventory);

      const result = await api.getDataInventory();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/privacy/data-inventory');
      expect(result.categories).toHaveLength(2);
      expect(result.data_sold).toBe(false);
      expect(result.third_party_sharing.llm_providers.recipients).toContain('Anthropic');
    });
  });

  // ===========================================================================
  // Privacy Preferences
  // ===========================================================================

  describe('Privacy Preferences', () => {
    it('should get privacy preferences', async () => {
      const mockPreferences = {
        do_not_sell: false,
        marketing_opt_out: true,
        analytics_opt_out: false,
        third_party_sharing: true,
      };
      mockClient.request.mockResolvedValue(mockPreferences);

      const result = await api.getPreferences();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/privacy/preferences');
      expect(result.marketing_opt_out).toBe(true);
    });

    it('should update privacy preferences', async () => {
      const mockResponse = {
        message: 'Preferences updated',
        preferences: {
          do_not_sell: true,
          marketing_opt_out: true,
          analytics_opt_out: true,
          third_party_sharing: false,
        },
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.updatePreferences({
        do_not_sell: true,
        analytics_opt_out: true,
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/privacy/preferences', {
        json: { do_not_sell: true, analytics_opt_out: true },
      });
      expect(result.preferences.do_not_sell).toBe(true);
    });
  });

  // ===========================================================================
  // Account Deletion
  // ===========================================================================

  describe('Account Deletion', () => {
    it('should delete account', async () => {
      const mockResponse = {
        message: 'Account scheduled for deletion',
        deletion_id: 'del_123',
        data_deleted: ['profile', 'debates', 'preferences', 'api_keys'],
        retention_note: 'Billing records retained for 7 years per legal requirements',
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.deleteAccount({
        password: 'mypassword123',
        confirm: true,
        reason: 'No longer using the service',
      });

      expect(mockClient.request).toHaveBeenCalledWith('DELETE', '/api/v1/privacy/account', {
        json: {
          password: 'mypassword123',
          confirm: true,
          reason: 'No longer using the service',
        },
      });
      expect(result.deletion_id).toBe('del_123');
      expect(result.data_deleted).toContain('profile');
    });

    it('should delete account without reason', async () => {
      const mockResponse = {
        message: 'Account scheduled for deletion',
        deletion_id: 'del_124',
        data_deleted: ['profile'],
        retention_note: '',
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.deleteAccount({
        password: 'mypassword123',
        confirm: true,
      });

      expect(mockClient.request).toHaveBeenCalledWith('DELETE', '/api/v1/privacy/account', {
        json: { password: 'mypassword123', confirm: true },
      });
    });
  });
});
