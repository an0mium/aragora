/**
 * OAuth Namespace Tests
 *
 * Comprehensive tests for the oauth namespace API including:
 * - Provider listing
 * - Authorization URLs
 * - Account linking/unlinking
 * - Callback handling
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { OAuthAPI } from '../oauth';

interface MockClient {
  request: Mock;
}

describe('OAuthAPI Namespace', () => {
  let api: OAuthAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new OAuthAPI(mockClient as any);
  });

  // ===========================================================================
  // Provider Listing
  // ===========================================================================

  describe('Provider Listing', () => {
    it('should get available providers', async () => {
      const mockProviders = {
        providers: [
          { provider: 'google', enabled: true, name: 'Google', icon_url: '/icons/google.svg' },
          { provider: 'github', enabled: true, name: 'GitHub', icon_url: '/icons/github.svg' },
          { provider: 'microsoft', enabled: true, name: 'Microsoft', icon_url: '/icons/microsoft.svg' },
          { provider: 'apple', enabled: false, name: 'Apple', icon_url: '/icons/apple.svg' },
        ],
      };
      mockClient.request.mockResolvedValue(mockProviders);

      const result = await api.getProviders();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/auth/oauth/providers');
      expect(result.providers).toHaveLength(4);
      expect(result.providers[0].provider).toBe('google');
    });

    it('should get linked providers for user', async () => {
      const mockLinked = {
        providers: [
          {
            provider: 'google',
            provider_user_id: 'g_123',
            email: 'user@gmail.com',
            name: 'John Doe',
            linked_at: '2024-01-15T10:00:00Z',
          },
          {
            provider: 'github',
            provider_user_id: 'gh_456',
            email: 'user@users.noreply.github.com',
            linked_at: '2024-01-16T10:00:00Z',
          },
        ],
      };
      mockClient.request.mockResolvedValue(mockLinked);

      const result = await api.getLinkedProviders();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/user/oauth-providers');
      expect(result.providers).toHaveLength(2);
    });
  });

  // ===========================================================================
  // Authorization URLs
  // ===========================================================================

  describe('Authorization URLs', () => {
    it('should get Google auth URL', async () => {
      const mockResponse = {
        authorization_url: 'https://accounts.google.com/o/oauth2/v2/auth?...',
        state: 'state_123',
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.getGoogleAuthUrl();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/auth/oauth/google', {
        params: { redirect_uri: undefined, state: undefined },
      });
      expect(result.authorization_url).toContain('accounts.google.com');
    });

    it('should get Google auth URL with params', async () => {
      const mockResponse = { authorization_url: 'https://...', state: 'state_123' };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.getGoogleAuthUrl({
        redirect_uri: 'https://myapp.com/callback',
        state: 'custom_state',
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/auth/oauth/google', {
        params: { redirect_uri: 'https://myapp.com/callback', state: 'custom_state' },
      });
    });

    it('should get GitHub auth URL', async () => {
      const mockResponse = {
        authorization_url: 'https://github.com/login/oauth/authorize?...',
        state: 'state_456',
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.getGitHubAuthUrl();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/auth/oauth/github', {
        params: { redirect_uri: undefined, state: undefined },
      });
      expect(result.authorization_url).toContain('github.com');
    });

    it('should get Microsoft auth URL', async () => {
      const mockResponse = {
        authorization_url: 'https://login.microsoftonline.com/...',
        state: 'state_789',
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.getMicrosoftAuthUrl();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/auth/oauth/microsoft', {
        params: { redirect_uri: undefined, state: undefined },
      });
      expect(result.authorization_url).toContain('microsoftonline.com');
    });

    it('should get Apple auth URL', async () => {
      const mockResponse = {
        authorization_url: 'https://appleid.apple.com/auth/authorize?...',
        state: 'state_abc',
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.getAppleAuthUrl();

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/auth/oauth/apple', {
        params: { redirect_uri: undefined, state: undefined },
      });
      expect(result.authorization_url).toContain('appleid.apple.com');
    });

    it('should get OIDC auth URL', async () => {
      const mockResponse = {
        authorization_url: 'https://idp.example.com/authorize?...',
        state: 'state_oidc',
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.getOIDCAuthUrl({ provider_id: 'custom_idp' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/auth/oauth/oidc', {
        params: { redirect_uri: undefined, state: undefined, provider_id: 'custom_idp' },
      });
    });
  });

  // ===========================================================================
  // Account Linking
  // ===========================================================================

  describe('Account Linking', () => {
    it('should link OAuth account', async () => {
      const mockResponse = {
        success: true,
        provider: 'github',
        email: 'user@github.com',
        message: 'Account linked successfully',
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.linkAccount({
        provider: 'github',
        code: 'auth_code_123',
        redirect_uri: 'https://myapp.com/callback',
      });

      expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/auth/oauth/link', {
        body: {
          provider: 'github',
          code: 'auth_code_123',
          redirect_uri: 'https://myapp.com/callback',
        },
      });
      expect(result.success).toBe(true);
      expect(result.provider).toBe('github');
    });

    it('should unlink OAuth account', async () => {
      const mockResponse = {
        success: true,
        message: 'Account unlinked successfully',
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.unlinkAccount({ provider: 'google' });

      expect(mockClient.request).toHaveBeenCalledWith('DELETE', '/api/v1/auth/oauth/unlink', {
        body: { provider: 'google' },
      });
      expect(result.success).toBe(true);
    });
  });

  // ===========================================================================
  // Callback Handling
  // ===========================================================================

  describe('Callback Handling', () => {
    it('should handle Google callback', async () => {
      const mockResponse = {
        success: true,
        access_token: 'access_123',
        refresh_token: 'refresh_456',
        user_id: 'u_789',
        email: 'user@gmail.com',
        name: 'John Doe',
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.handleCallback('google', {
        code: 'auth_code',
        state: 'state_123',
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/auth/oauth/google/callback', {
        params: { code: 'auth_code', state: 'state_123', redirect_uri: undefined },
      });
      expect(result.success).toBe(true);
      expect(result.email).toBe('user@gmail.com');
    });

    it('should handle GitHub callback', async () => {
      const mockResponse = {
        success: true,
        access_token: 'gho_xxx',
        user_id: 'gh_123',
        email: 'user@github.com',
      };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.handleCallback('github', {
        code: 'github_code',
        redirect_uri: 'https://myapp.com/callback',
      });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/auth/oauth/github/callback', {
        params: { code: 'github_code', state: undefined, redirect_uri: 'https://myapp.com/callback' },
      });
    });

    it('should handle OIDC callback', async () => {
      const mockResponse = { success: true, user_id: 'oidc_user' };
      mockClient.request.mockResolvedValue(mockResponse);

      await api.handleCallback('oidc', { code: 'oidc_code' });

      expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/auth/oauth/oidc/callback', {
        params: { code: 'oidc_code', state: undefined, redirect_uri: undefined },
      });
    });

    it('should handle callback error', async () => {
      const mockResponse = {
        success: false,
        error: 'invalid_grant',
        error_description: 'Authorization code expired',
      };
      mockClient.request.mockResolvedValue(mockResponse);

      const result = await api.handleCallback('google', { code: 'expired_code' });

      expect(result.success).toBe(false);
      expect(result.error).toBe('invalid_grant');
    });
  });
});
