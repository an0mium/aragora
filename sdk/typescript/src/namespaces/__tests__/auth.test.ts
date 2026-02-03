/**
 * Auth Namespace Tests
 *
 * Comprehensive tests for the auth namespace API including:
 * - Registration and login
 * - Token refresh and verification
 * - Profile management
 * - Password operations
 * - OAuth flows
 * - MFA setup and verification
 * - Session management
 * - API key management
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { AuthAPI } from '../auth';

interface MockClient {
  registerUser: Mock;
  login: Mock;
  refreshToken: Mock;
  verifyEmail: Mock;
  getCurrentUser: Mock;
  updateProfile: Mock;
  changePassword: Mock;
  requestPasswordReset: Mock;
  resetPassword: Mock;
  getOAuthUrl: Mock;
  completeOAuth: Mock;
  setupMFA: Mock;
  verifyMFASetup: Mock;
  disableMFA: Mock;
  listSessions: Mock;
  revokeSession: Mock;
  logoutAll: Mock;
  listApiKeys: Mock;
  createApiKey: Mock;
  revokeApiKey: Mock;
  logout: Mock;
  request: Mock;
}

describe('AuthAPI Namespace', () => {
  let api: AuthAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      registerUser: vi.fn(),
      login: vi.fn(),
      refreshToken: vi.fn(),
      verifyEmail: vi.fn(),
      getCurrentUser: vi.fn(),
      updateProfile: vi.fn(),
      changePassword: vi.fn(),
      requestPasswordReset: vi.fn(),
      resetPassword: vi.fn(),
      getOAuthUrl: vi.fn(),
      completeOAuth: vi.fn(),
      setupMFA: vi.fn(),
      verifyMFASetup: vi.fn(),
      disableMFA: vi.fn(),
      listSessions: vi.fn(),
      revokeSession: vi.fn(),
      logoutAll: vi.fn(),
      listApiKeys: vi.fn(),
      createApiKey: vi.fn(),
      revokeApiKey: vi.fn(),
      logout: vi.fn(),
      request: vi.fn(),
    };
    api = new AuthAPI(mockClient as any);
  });

  // ===========================================================================
  // Registration and Login
  // ===========================================================================

  describe('Registration and Login', () => {
    it('should register a new user', async () => {
      const mockResponse = {
        user: { id: 'u1', email: 'test@example.com', name: 'Test User' },
        token: { access_token: 'token123', refresh_token: 'refresh123' },
      };
      mockClient.registerUser.mockResolvedValue(mockResponse);

      const result = await api.register({
        email: 'test@example.com',
        password: 'securePassword123',
        name: 'Test User',
      });

      expect(mockClient.registerUser).toHaveBeenCalledWith({
        email: 'test@example.com',
        password: 'securePassword123',
        name: 'Test User',
      });
      expect(result.user.email).toBe('test@example.com');
    });

    it('should login with credentials', async () => {
      const mockToken = {
        access_token: 'access123',
        refresh_token: 'refresh123',
        expires_in: 3600,
      };
      mockClient.login.mockResolvedValue(mockToken);

      const result = await api.login({
        email: 'test@example.com',
        password: 'password123',
      });

      expect(mockClient.login).toHaveBeenCalledWith({
        email: 'test@example.com',
        password: 'password123',
      });
      expect(result.access_token).toBe('access123');
    });

    it('should refresh token', async () => {
      const mockToken = {
        access_token: 'newAccess123',
        refresh_token: 'newRefresh123',
        expires_in: 3600,
      };
      mockClient.refreshToken.mockResolvedValue(mockToken);

      const result = await api.refresh({ refresh_token: 'oldRefresh123' });

      expect(mockClient.refreshToken).toHaveBeenCalledWith({
        refresh_token: 'oldRefresh123',
      });
      expect(result.access_token).toBe('newAccess123');
    });

    it('should logout', async () => {
      mockClient.logout.mockResolvedValue({ success: true });

      const result = await api.logout();

      expect(mockClient.logout).toHaveBeenCalled();
      expect(result.success).toBe(true);
    });
  });

  // ===========================================================================
  // Profile Management
  // ===========================================================================

  describe('Profile Management', () => {
    it('should get current user', async () => {
      const mockUser = {
        id: 'u1',
        email: 'test@example.com',
        name: 'Test User',
        mfa_enabled: false,
      };
      mockClient.getCurrentUser.mockResolvedValue(mockUser);

      const result = await api.me();

      expect(mockClient.getCurrentUser).toHaveBeenCalled();
      expect(result.email).toBe('test@example.com');
    });

    it('should update profile', async () => {
      const mockResponse = {
        user: { id: 'u1', email: 'test@example.com', name: 'Updated Name' },
      };
      mockClient.updateProfile.mockResolvedValue(mockResponse);

      const result = await api.updateProfile({ name: 'Updated Name' });

      expect(mockClient.updateProfile).toHaveBeenCalledWith({
        name: 'Updated Name',
      });
      expect(result.user.name).toBe('Updated Name');
    });
  });

  // ===========================================================================
  // Password Operations
  // ===========================================================================

  describe('Password Operations', () => {
    it('should change password', async () => {
      mockClient.changePassword.mockResolvedValue({ success: true });

      const result = await api.changePassword({
        current_password: 'oldPass123',
        new_password: 'newPass456',
      });

      expect(mockClient.changePassword).toHaveBeenCalledWith({
        current_password: 'oldPass123',
        new_password: 'newPass456',
      });
      expect(result.success).toBe(true);
    });

    it('should request password reset', async () => {
      mockClient.requestPasswordReset.mockResolvedValue(undefined);

      await api.requestPasswordReset({
        email: 'test@example.com',
      });

      expect(mockClient.requestPasswordReset).toHaveBeenCalledWith({
        email: 'test@example.com',
      });
    });

    it('should reset password with token', async () => {
      mockClient.resetPassword.mockResolvedValue({ success: true });

      const result = await api.resetPassword({
        token: 'resetToken123',
        new_password: 'newPass456',
      });

      expect(mockClient.resetPassword).toHaveBeenCalledWith({
        token: 'resetToken123',
        new_password: 'newPass456',
      });
      expect(result.success).toBe(true);
    });
  });

  // ===========================================================================
  // MFA Operations
  // ===========================================================================

  describe('MFA Operations', () => {
    it('should setup MFA', async () => {
      const mockSetup = {
        secret: 'JBSWY3DPEHPK3PXP',
        qr_code: 'data:image/png;base64,...',
        backup_codes: ['code1', 'code2', 'code3'],
      };
      mockClient.setupMFA.mockResolvedValue(mockSetup);

      const result = await api.setupMFA({ type: 'totp' });

      expect(mockClient.setupMFA).toHaveBeenCalledWith({ type: 'totp' });
      expect(result.secret).toBe('JBSWY3DPEHPK3PXP');
      expect(result.backup_codes).toHaveLength(3);
    });

    it('should verify MFA code', async () => {
      const mockVerify = { verified: true, backup_codes: ['code1', 'code2'] };
      mockClient.verifyMFASetup.mockResolvedValue(mockVerify);

      const result = await api.verifyMFASetup({ code: '123456' });

      expect(mockClient.verifyMFASetup).toHaveBeenCalledWith({ code: '123456' });
      expect(result.verified).toBe(true);
    });

    it('should disable MFA', async () => {
      mockClient.disableMFA.mockResolvedValue({ disabled: true });

      const result = await api.disableMFA();

      expect(mockClient.disableMFA).toHaveBeenCalled();
      expect(result.disabled).toBe(true);
    });
  });

  // ===========================================================================
  // Session Management
  // ===========================================================================

  describe('Session Management', () => {
    it('should list active sessions', async () => {
      const mockSessions = {
        sessions: [
          { id: 's1', device: 'Chrome on Windows', ip_address: '192.168.1.1', current: true },
          { id: 's2', device: 'Safari on macOS', ip_address: '192.168.1.2', current: false },
        ],
      };
      mockClient.listSessions.mockResolvedValue(mockSessions);

      const result = await api.listSessions();

      expect(mockClient.listSessions).toHaveBeenCalled();
      expect(result.sessions).toHaveLength(2);
    });

    it('should revoke a session', async () => {
      mockClient.revokeSession.mockResolvedValue({ revoked: true });

      const result = await api.revokeSession('s2');

      expect(mockClient.revokeSession).toHaveBeenCalledWith('s2');
      expect(result.revoked).toBe(true);
    });

    it('should revoke all other sessions', async () => {
      mockClient.logoutAll.mockResolvedValue({ logged_out: true, sessions_revoked: 3 });

      const result = await api.logoutAll();

      expect(mockClient.logoutAll).toHaveBeenCalled();
      expect(result.sessions_revoked).toBe(3);
    });
  });

  // ===========================================================================
  // API Key Management
  // ===========================================================================

  describe('API Key Management', () => {
    it('should list API keys', async () => {
      const mockKeys = {
        api_keys: [
          { id: 'k1', name: 'Production', prefix: 'ara_prod_', created_at: '2024-01-01' },
          { id: 'k2', name: 'Development', prefix: 'ara_dev_', created_at: '2024-01-02' },
        ],
      };
      mockClient.listApiKeys.mockResolvedValue(mockKeys);

      const result = await api.listApiKeys();

      expect(mockClient.listApiKeys).toHaveBeenCalled();
      expect(result.api_keys).toHaveLength(2);
    });

    it('should create API key', async () => {
      const mockKey = {
        id: 'k3',
        key: 'ara_live_abc123xyz',
        prefix: 'ara_',
      };
      mockClient.createApiKey.mockResolvedValue(mockKey);

      const result = await api.createApiKey('New Key', 30);

      expect(mockClient.createApiKey).toHaveBeenCalledWith('New Key', 30);
      expect(result.key).toBe('ara_live_abc123xyz');
    });

    it('should revoke API key', async () => {
      mockClient.revokeApiKey.mockResolvedValue({ revoked: true });

      const result = await api.revokeApiKey('k2');

      expect(mockClient.revokeApiKey).toHaveBeenCalledWith('k2');
      expect(result.revoked).toBe(true);
    });
  });

  // ===========================================================================
  // OAuth Operations
  // ===========================================================================

  describe('OAuth Operations', () => {
    it('should get OAuth URL', async () => {
      const mockUrl = {
        url: 'https://accounts.google.com/oauth2/auth?...',
        state: 'randomState123',
      };
      mockClient.getOAuthUrl.mockResolvedValue(mockUrl);

      const result = await api.getOAuthUrl({
        provider: 'google',
        redirect_uri: 'https://app.aragora.ai/callback',
      });

      expect(mockClient.getOAuthUrl).toHaveBeenCalledWith({
        provider: 'google',
        redirect_uri: 'https://app.aragora.ai/callback',
      });
      expect(result.url).toContain('accounts.google.com');
    });

    it('should handle OAuth callback', async () => {
      const mockToken = {
        access_token: 'oauthToken123',
        refresh_token: 'oauthRefresh123',
        user: { id: 'u1', email: 'oauth@example.com' },
      };
      mockClient.completeOAuth.mockResolvedValue(mockToken);

      const result = await api.completeOAuth({
        provider: 'google',
        code: 'authCode123',
        state: 'randomState123',
      });

      expect(mockClient.completeOAuth).toHaveBeenCalledWith({
        provider: 'google',
        code: 'authCode123',
        state: 'randomState123',
      });
      expect(result.access_token).toBe('oauthToken123');
    });
  });

  // ===========================================================================
  // Email Verification
  // ===========================================================================

  describe('Email Verification', () => {
    it('should verify email with token', async () => {
      const mockVerify = { verified: true, email: 'test@example.com' };
      mockClient.verifyEmail.mockResolvedValue(mockVerify);

      const result = await api.verifyEmail({ token: 'verifyToken123' });

      expect(mockClient.verifyEmail).toHaveBeenCalledWith({ token: 'verifyToken123' });
      expect(result.verified).toBe(true);
    });
  });
});
