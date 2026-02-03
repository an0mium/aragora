/**
 * SSO Namespace Tests
 *
 * Comprehensive tests for the SSO namespace API including:
 * - Status checks
 * - Login flow
 * - Callback handling
 * - Logout
 * - SAML metadata
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { SSONamespace } from '../sso';

interface MockClient {
  request: Mock;
}

describe('SSONamespace', () => {
  let api: SSONamespace;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new SSONamespace(mockClient as any);
  });

  // ===========================================================================
  // Status
  // ===========================================================================

  describe('Status', () => {
    it('should get SSO status', async () => {
      const mockStatus = {
        enabled: true,
        provider_type: 'saml',
        idp_url: 'https://idp.example.com',
        entity_id: 'urn:aragora:sp',
        configured: true,
        last_login: '2024-01-20T10:00:00Z',
        total_logins: 150,
      };
      mockClient.request.mockResolvedValue(mockStatus);

      const result = await api.getStatus();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/auth/sso/status');
      expect(result.enabled).toBe(true);
      expect(result.provider_type).toBe('saml');
    });

    it('should check if enabled', async () => {
      mockClient.request.mockResolvedValue({ enabled: true, configured: true });

      const result = await api.isEnabled();

      expect(result).toBe(true);
    });

    it('should return false when not enabled', async () => {
      mockClient.request.mockResolvedValue({ enabled: false, configured: false });

      const result = await api.isEnabled();

      expect(result).toBe(false);
    });

    it('should check if configured', async () => {
      mockClient.request.mockResolvedValue({ enabled: true, configured: true });

      const result = await api.isConfigured();

      expect(result).toBe(true);
    });

    it('should return false when not configured', async () => {
      mockClient.request.mockResolvedValue({ enabled: true, configured: false });

      const result = await api.isConfigured();

      expect(result).toBe(false);
    });
  });

  // ===========================================================================
  // Login Flow
  // ===========================================================================

  describe('Login Flow', () => {
    it('should initiate login', async () => {
      const mockLogin = {
        redirect_url: 'https://idp.example.com/saml/login?...',
        state: 'state123',
        provider: 'saml',
      };
      mockClient.request.mockResolvedValue(mockLogin);

      const result = await api.login();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/auth/sso/login');
      expect(result.redirect_url).toContain('idp.example.com');
    });

    it('should initiate login with return URL', async () => {
      const mockLogin = {
        redirect_url: 'https://idp.example.com/saml/login?...',
        state: 'state123',
        provider: 'saml',
      };
      mockClient.request.mockResolvedValue(mockLogin);

      const result = await api.login({ returnUrl: '/dashboard' });

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/auth/sso/login?return_url=%2Fdashboard'
      );
    });

    it('should initiate login with OIDC prompt', async () => {
      const mockLogin = {
        redirect_url: 'https://idp.example.com/oauth2/authorize?...',
        state: 'state123',
        nonce: 'nonce456',
        provider: 'oidc',
      };
      mockClient.request.mockResolvedValue(mockLogin);

      const result = await api.login({ prompt: 'consent' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/auth/sso/login?prompt=consent');
    });

    it('should initiate login with both options', async () => {
      const mockLogin = { redirect_url: 'https://...', provider: 'oidc' };
      mockClient.request.mockResolvedValue(mockLogin);

      await api.login({ returnUrl: '/settings', prompt: 'login' });

      expect(mockClient.request).toHaveBeenCalledWith(
        'GET',
        '/auth/sso/login?return_url=%2Fsettings&prompt=login'
      );
    });
  });

  // ===========================================================================
  // Callback
  // ===========================================================================

  describe('Callback', () => {
    it('should handle OIDC callback', async () => {
      const mockResult = {
        success: true,
        user: {
          id: 'u123',
          email: 'user@example.com',
          name: 'Test User',
          provider: 'oidc',
        },
        token: 'access_token_123',
        refresh_token: 'refresh_token_456',
        expires_in: 3600,
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.callback({ code: 'auth_code_123', state: 'state123' });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/auth/sso/callback', {
        body: { code: 'auth_code_123', state: 'state123' },
      });
      expect(result.success).toBe(true);
      expect(result.user?.email).toBe('user@example.com');
    });

    it('should handle SAML callback', async () => {
      const mockResult = {
        success: true,
        user: {
          id: 'u123',
          email: 'user@example.com',
          groups: ['Engineering', 'Admin'],
          provider: 'saml',
        },
        token: 'access_token_123',
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.callback({
        SAMLResponse: 'base64_encoded_response',
        RelayState: 'state123',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/auth/sso/callback', {
        body: { SAMLResponse: 'base64_encoded_response', RelayState: 'state123' },
      });
      expect(result.user?.groups).toContain('Engineering');
    });

    it('should handle callback error', async () => {
      const mockResult = {
        success: false,
        error: 'invalid_grant',
        error_description: 'Authorization code expired',
      };
      mockClient.request.mockResolvedValue(mockResult);

      const result = await api.callback({ code: 'expired_code', state: 'state123' });

      expect(result.success).toBe(false);
      expect(result.error).toBe('invalid_grant');
    });
  });

  // ===========================================================================
  // Logout
  // ===========================================================================

  describe('Logout', () => {
    it('should logout', async () => {
      const mockLogout = {
        success: true,
        message: 'Logged out successfully',
      };
      mockClient.request.mockResolvedValue(mockLogout);

      const result = await api.logout();

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/auth/sso/logout', {
        body: undefined,
      });
      expect(result.success).toBe(true);
    });

    it('should logout from everywhere (SLO)', async () => {
      const mockLogout = {
        success: true,
        redirect_url: 'https://idp.example.com/saml/logout',
        message: 'Single logout initiated',
      };
      mockClient.request.mockResolvedValue(mockLogout);

      const result = await api.logout({ everywhere: true });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/auth/sso/logout', {
        body: { everywhere: true },
      });
      expect(result.redirect_url).toContain('logout');
    });
  });

  // ===========================================================================
  // SAML Metadata
  // ===========================================================================

  describe('SAML Metadata', () => {
    it('should get SAML metadata', async () => {
      const mockMetadata = {
        entity_id: 'urn:aragora:sp',
        acs_url: 'https://api.aragora.ai/auth/sso/callback',
        slo_url: 'https://api.aragora.ai/auth/sso/logout',
        certificate: 'MIICpDCCAYwCCQ...',
        metadata_xml: '<?xml version="1.0"?><EntityDescriptor>...</EntityDescriptor>',
      };
      mockClient.request.mockResolvedValue(mockMetadata);

      const result = await api.getMetadata();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/auth/sso/metadata');
      expect(result.entity_id).toBe('urn:aragora:sp');
      expect(result.acs_url).toContain('/auth/sso/callback');
    });

    it('should get metadata XML', async () => {
      const mockMetadata = {
        metadata_xml: '<?xml version="1.0"?><EntityDescriptor>...</EntityDescriptor>',
      };
      mockClient.request.mockResolvedValue(mockMetadata);

      const result = await api.getMetadataXML();

      expect(result).toContain('<?xml');
      expect(result).toContain('EntityDescriptor');
    });

    it('should return empty string if no metadata XML', async () => {
      mockClient.request.mockResolvedValue({ entity_id: 'test' });

      const result = await api.getMetadataXML();

      expect(result).toBe('');
    });
  });
});
