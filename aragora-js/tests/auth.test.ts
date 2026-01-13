/**
 * Aragora SDK Auth API Tests
 *
 * Tests for authentication, MFA, and user management endpoints.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { AragoraClient, AragoraError } from '../src/client';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('AuthAPI', () => {
  let client: AragoraClient;

  beforeEach(() => {
    client = new AragoraClient({
      baseUrl: 'http://localhost:8080',
      apiKey: 'test-api-key',
    });
    mockFetch.mockReset();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  // ===========================================================================
  // Registration
  // ===========================================================================

  describe('register()', () => {
    it('should register a new user', async () => {
      const mockResponse = {
        user_id: 'user-123',
        email: 'test@example.com',
        message: 'Registration successful',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.register({
        email: 'test@example.com',
        password: 'securePassword123',
        name: 'Test User',
      });

      expect(result.user_id).toBe('user-123');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/auth/register',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            email: 'test@example.com',
            password: 'securePassword123',
            name: 'Test User',
          }),
        })
      );
    });

    it('should register without optional name', async () => {
      const mockResponse = {
        user_id: 'user-456',
        email: 'test2@example.com',
        message: 'Registration successful',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.register({
        email: 'test2@example.com',
        password: 'securePassword123',
      });

      expect(result.user_id).toBe('user-456');
    });

    it('should throw on duplicate email', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 409,
        json: () => Promise.resolve({
          error: 'Email already registered',
          code: 'EMAIL_EXISTS',
        }),
      });

      await expect(
        client.auth.register({
          email: 'existing@example.com',
          password: 'password123',
        })
      ).rejects.toThrow(AragoraError);
    });

    it('should throw on invalid email format', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Invalid email format',
          code: 'VALIDATION_ERROR',
        }),
      });

      await expect(
        client.auth.register({
          email: 'not-an-email',
          password: 'password123',
        })
      ).rejects.toThrow(AragoraError);
    });
  });

  // ===========================================================================
  // Login
  // ===========================================================================

  describe('login()', () => {
    it('should authenticate user and return tokens', async () => {
      const mockResponse = {
        access_token: 'access-token-123',
        refresh_token: 'refresh-token-456',
        token_type: 'Bearer',
        expires_in: 3600,
        user: {
          user_id: 'user-123',
          email: 'test@example.com',
          name: 'Test User',
          role: 'member',
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.login({
        email: 'test@example.com',
        password: 'securePassword123',
      });

      expect(result.access_token).toBe('access-token-123');
      expect(result.refresh_token).toBe('refresh-token-456');
      expect(result.user.email).toBe('test@example.com');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/auth/login',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            email: 'test@example.com',
            password: 'securePassword123',
          }),
        })
      );
    });

    it('should login with MFA code', async () => {
      const mockResponse = {
        access_token: 'access-token-789',
        refresh_token: 'refresh-token-abc',
        token_type: 'Bearer',
        expires_in: 3600,
        user: {
          user_id: 'user-mfa',
          email: 'mfa@example.com',
          name: 'MFA User',
          role: 'member',
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.login({
        email: 'mfa@example.com',
        password: 'password123',
        mfa_code: '123456',
      });

      expect(result.access_token).toBe('access-token-789');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/auth/login',
        expect.objectContaining({
          body: JSON.stringify({
            email: 'mfa@example.com',
            password: 'password123',
            mfa_code: '123456',
          }),
        })
      );
    });

    it('should throw on invalid credentials', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({
          error: 'Invalid email or password',
          code: 'INVALID_CREDENTIALS',
        }),
      });

      await expect(
        client.auth.login({
          email: 'test@example.com',
          password: 'wrongpassword',
        })
      ).rejects.toThrow(AragoraError);
    });

    it('should throw on MFA required', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 403,
        json: () => Promise.resolve({
          error: 'MFA code required',
          code: 'MFA_REQUIRED',
          mfa_required: true,
        }),
      });

      try {
        await client.auth.login({
          email: 'mfa@example.com',
          password: 'password123',
        });
      } catch (error) {
        expect(error).toBeInstanceOf(AragoraError);
        if (error instanceof AragoraError) {
          expect(error.code).toBe('MFA_REQUIRED');
          expect(error.status).toBe(403);
        }
      }
    });

    it('should throw on account locked', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 403,
        json: () => Promise.resolve({
          error: 'Account locked due to too many failed attempts',
          code: 'ACCOUNT_LOCKED',
          locked_until: '2024-01-01T01:00:00Z',
        }),
      });

      await expect(
        client.auth.login({
          email: 'locked@example.com',
          password: 'password123',
        })
      ).rejects.toThrow(AragoraError);
    });
  });

  // ===========================================================================
  // Logout
  // ===========================================================================

  describe('logout()', () => {
    it('should logout current session', async () => {
      const mockResponse = {
        message: 'Logged out successfully',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.logout();
      expect(result.message).toBe('Logged out successfully');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/auth/logout',
        expect.objectContaining({
          method: 'POST',
        })
      );
    });
  });

  describe('logoutAll()', () => {
    it('should logout all sessions', async () => {
      const mockResponse = {
        message: 'All sessions invalidated',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.logoutAll();
      expect(result.message).toBe('All sessions invalidated');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/auth/logout-all',
        expect.objectContaining({
          method: 'POST',
        })
      );
    });
  });

  // ===========================================================================
  // Token Refresh
  // ===========================================================================

  describe('refresh()', () => {
    it('should refresh access token', async () => {
      const mockResponse = {
        access_token: 'new-access-token',
        expires_in: 3600,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.refresh({
        refresh_token: 'refresh-token-456',
      });

      expect(result.access_token).toBe('new-access-token');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/auth/refresh',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            refresh_token: 'refresh-token-456',
          }),
        })
      );
    });

    it('should throw on expired refresh token', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({
          error: 'Refresh token expired',
          code: 'TOKEN_EXPIRED',
        }),
      });

      await expect(
        client.auth.refresh({ refresh_token: 'expired-token' })
      ).rejects.toThrow(AragoraError);
    });
  });

  // ===========================================================================
  // Token Revocation
  // ===========================================================================

  describe('revoke()', () => {
    it('should revoke a specific token', async () => {
      const mockResponse = {
        message: 'Token revoked',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.revoke({
        token: 'token-to-revoke',
      });

      expect(result.message).toBe('Token revoked');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/auth/revoke',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            token: 'token-to-revoke',
          }),
        })
      );
    });
  });

  // ===========================================================================
  // User Profile
  // ===========================================================================

  describe('me()', () => {
    it('should get current user info', async () => {
      const mockResponse = {
        user_id: 'user-123',
        email: 'test@example.com',
        name: 'Test User',
        role: 'member',
        org_id: 'org-456',
        mfa_enabled: false,
        created_at: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.me();
      expect(result.user_id).toBe('user-123');
      expect(result.email).toBe('test@example.com');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/auth/me',
        expect.objectContaining({
          method: 'GET',
        })
      );
    });

    it('should throw on unauthorized', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({
          error: 'Unauthorized',
          code: 'UNAUTHORIZED',
        }),
      });

      await expect(client.auth.me()).rejects.toThrow(AragoraError);
    });
  });

  describe('updateMe()', () => {
    it('should update current user info', async () => {
      const mockResponse = {
        user_id: 'user-123',
        email: 'test@example.com',
        name: 'Updated Name',
        role: 'member',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.updateMe({
        name: 'Updated Name',
      });

      expect(result.name).toBe('Updated Name');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/auth/me',
        expect.objectContaining({
          method: 'PUT',
          body: JSON.stringify({
            name: 'Updated Name',
          }),
        })
      );
    });
  });

  // ===========================================================================
  // Password Change
  // ===========================================================================

  describe('changePassword()', () => {
    it('should change password', async () => {
      const mockResponse = {
        message: 'Password changed successfully',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.changePassword({
        current_password: 'oldPassword123',
        new_password: 'newPassword456',
      });

      expect(result.message).toBe('Password changed successfully');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/auth/password',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            current_password: 'oldPassword123',
            new_password: 'newPassword456',
          }),
        })
      );
    });

    it('should throw on incorrect current password', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({
          error: 'Current password is incorrect',
          code: 'INVALID_PASSWORD',
        }),
      });

      await expect(
        client.auth.changePassword({
          current_password: 'wrongPassword',
          new_password: 'newPassword456',
        })
      ).rejects.toThrow(AragoraError);
    });

    it('should throw on weak new password', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Password does not meet requirements',
          code: 'WEAK_PASSWORD',
        }),
      });

      await expect(
        client.auth.changePassword({
          current_password: 'oldPassword123',
          new_password: '123',
        })
      ).rejects.toThrow(AragoraError);
    });
  });

  // ===========================================================================
  // API Keys
  // ===========================================================================

  describe('createApiKey()', () => {
    it('should create an API key', async () => {
      const mockResponse = {
        api_key: 'aragora_key_abc123xyz',
        prefix: 'aragora_key_abc1',
        created_at: '2024-01-01T00:00:00Z',
        expires_at: null,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.createApiKey({
        name: 'Production Key',
      });

      expect(result.api_key).toBe('aragora_key_abc123xyz');
      expect(result.prefix).toBe('aragora_key_abc1');
    });

    it('should create API key with expiration', async () => {
      const mockResponse = {
        api_key: 'aragora_key_def456uvw',
        prefix: 'aragora_key_def4',
        created_at: '2024-01-01T00:00:00Z',
        expires_at: '2024-12-31T23:59:59Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.createApiKey({
        name: 'Temporary Key',
        expires_in_days: 365,
      });

      expect(result.expires_at).toBe('2024-12-31T23:59:59Z');
    });

    it('should create API key without options', async () => {
      const mockResponse = {
        api_key: 'aragora_key_ghi789rst',
        prefix: 'aragora_key_ghi7',
        created_at: '2024-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.createApiKey();
      expect(result.api_key).toBeDefined();
    });
  });

  describe('revokeApiKey()', () => {
    it('should revoke API key', async () => {
      const mockResponse = {
        message: 'API key revoked',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.revokeApiKey();
      expect(result.message).toBe('API key revoked');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/auth/api-key',
        expect.objectContaining({
          method: 'DELETE',
        })
      );
    });
  });

  // ===========================================================================
  // MFA (Multi-Factor Authentication)
  // ===========================================================================

  describe('mfaSetup()', () => {
    it('should setup MFA and return QR code', async () => {
      const mockResponse = {
        secret: 'JBSWY3DPEHPK3PXP',
        qr_code: 'data:image/png;base64,iVBORw0KGgo...',
        provisioning_uri: 'otpauth://totp/Aragora:test@example.com?secret=JBSWY3DPEHPK3PXP&issuer=Aragora',
        message: 'Scan the QR code with your authenticator app',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.mfaSetup();
      expect(result.secret).toBe('JBSWY3DPEHPK3PXP');
      expect(result.qr_code).toContain('data:image/png;base64');
      expect(result.provisioning_uri).toContain('otpauth://totp');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/auth/mfa/setup',
        expect.objectContaining({
          method: 'POST',
        })
      );
    });

    it('should throw if MFA already enabled', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'MFA is already enabled',
          code: 'MFA_ALREADY_ENABLED',
        }),
      });

      await expect(client.auth.mfaSetup()).rejects.toThrow(AragoraError);
    });
  });

  describe('mfaEnable()', () => {
    it('should enable MFA with verification code', async () => {
      const mockResponse = {
        message: 'MFA enabled successfully',
        backup_codes: [
          'abc123def',
          'ghi456jkl',
          'mno789pqr',
          'stu012vwx',
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.mfaEnable({
        code: '123456',
      });

      expect(result.message).toBe('MFA enabled successfully');
      expect(result.backup_codes).toHaveLength(4);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/auth/mfa/enable',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            code: '123456',
          }),
        })
      );
    });

    it('should throw on invalid verification code', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Invalid verification code',
          code: 'INVALID_MFA_CODE',
        }),
      });

      await expect(
        client.auth.mfaEnable({ code: '000000' })
      ).rejects.toThrow(AragoraError);
    });
  });

  describe('mfaDisable()', () => {
    it('should disable MFA', async () => {
      const mockResponse = {
        message: 'MFA disabled successfully',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.mfaDisable({
        code: '123456',
        password: 'password123',
      });

      expect(result.message).toBe('MFA disabled successfully');
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/auth/mfa/disable',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            code: '123456',
            password: 'password123',
          }),
        })
      );
    });

    it('should throw on invalid MFA code during disable', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Invalid MFA code',
          code: 'INVALID_MFA_CODE',
        }),
      });

      await expect(
        client.auth.mfaDisable({ code: '000000', password: 'password123' })
      ).rejects.toThrow(AragoraError);
    });
  });

  describe('mfaVerify()', () => {
    it('should verify MFA code', async () => {
      const mockResponse = {
        valid: true,
        message: 'Code verified',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.mfaVerify({
        code: '123456',
      });

      expect(result.valid).toBe(true);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/auth/mfa/verify',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            code: '123456',
          }),
        })
      );
    });

    it('should verify backup code', async () => {
      const mockResponse = {
        valid: true,
        message: 'Backup code used',
        backup_codes_remaining: 3,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.mfaVerify({
        code: 'abc123def',
      });

      expect(result.valid).toBe(true);
    });
  });

  describe('mfaBackupCodes()', () => {
    it('should generate new backup codes', async () => {
      const mockResponse = {
        backup_codes: [
          'new123abc',
          'new456def',
          'new789ghi',
          'new012jkl',
        ],
        message: 'New backup codes generated. Previous codes invalidated.',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await client.auth.mfaBackupCodes();
      expect(result.backup_codes).toHaveLength(4);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8080/api/auth/mfa/backup-codes',
        expect.objectContaining({
          method: 'POST',
        })
      );
    });

    it('should throw if MFA not enabled', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'MFA is not enabled',
          code: 'MFA_NOT_ENABLED',
        }),
      });

      await expect(client.auth.mfaBackupCodes()).rejects.toThrow(AragoraError);
    });
  });

  // ===========================================================================
  // Error Handling
  // ===========================================================================

  describe('error handling', () => {
    // Use a client with retries disabled for error handling tests
    let noRetryClient: AragoraClient;

    beforeEach(() => {
      noRetryClient = new AragoraClient({
        baseUrl: 'http://localhost:8080',
        apiKey: 'test-api-key',
        retry: { maxRetries: 0 },
      });
    });

    it('should include error code in AragoraError', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 429,
        json: () => Promise.resolve({
          error: 'Too many login attempts',
          code: 'RATE_LIMIT',
        }),
      });

      try {
        await noRetryClient.auth.login({
          email: 'test@example.com',
          password: 'password123',
        });
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(AragoraError);
        if (error instanceof AragoraError) {
          expect(error.status).toBe(429);
          expect(error.code).toBe('RATE_LIMIT');
          expect(error.message).toBe('Too many login attempts');
          expect(error.retryable).toBe(true); // 429 is retryable
        }
      }
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      await expect(
        noRetryClient.auth.login({
          email: 'test@example.com',
          password: 'password123',
        })
      ).rejects.toThrow(AragoraError);
    });

    it('should handle malformed JSON response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.reject(new Error('Invalid JSON')),
      });

      await expect(
        noRetryClient.auth.login({
          email: 'test@example.com',
          password: 'password123',
        })
      ).rejects.toThrow(AragoraError);
    });
  });
});
