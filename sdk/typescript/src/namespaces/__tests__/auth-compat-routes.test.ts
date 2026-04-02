import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import { AuthRoutesAPI } from '../auth';

interface MockClient {
  request: Mock;
}

describe('AuthRoutesAPI compatibility routes', () => {
  let api: AuthRoutesAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn().mockResolvedValue({}),
    };
    api = new AuthRoutesAPI(mockClient as any);
  });

  it('uses POST for legacy password reset compatibility routes', async () => {
    await api.forgotPasswordAlt({ email: 'user@example.com' });
    await api.resetPasswordAlt({ token: 'reset-token', new_password: 'new-password' });

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'POST', '/api/auth/forgot-password', {
      json: { email: 'user@example.com' },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(2, 'POST', '/api/auth/reset-password', {
      json: { token: 'reset-token', new_password: 'new-password' },
    });
  });

  it('uses POST for the legacy combined MFA compatibility route', async () => {
    await api.mfa({ action: 'setup', method: 'totp' });

    expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/auth/mfa', {
      json: { action: 'setup', method: 'totp' },
    });
  });

  it('uses POST for auth verification and invite compatibility routes', async () => {
    await api.verifyEmail({ token: 'verify-token' });
    await api.resendVerification();
    await api.resendVerificationAlt({ email: 'user@example.com' });
    await api.setupOrganization({ name: 'Acme', slug: 'acme' });
    await api.invite({ email: 'teammate@example.com', role: 'member' });
    await api.acceptInvite({ token: 'invite-token' });

    expect(mockClient.request).toHaveBeenNthCalledWith(1, 'POST', '/api/auth/verify-email', {
      json: { token: 'verify-token' },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(2, 'POST', '/api/auth/verify-email/resend');
    expect(mockClient.request).toHaveBeenNthCalledWith(3, 'POST', '/api/auth/resend-verification', {
      json: { email: 'user@example.com' },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(4, 'POST', '/api/auth/setup-organization', {
      json: { name: 'Acme', slug: 'acme' },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(5, 'POST', '/api/auth/invite', {
      json: { email: 'teammate@example.com', role: 'member' },
    });
    expect(mockClient.request).toHaveBeenNthCalledWith(6, 'POST', '/api/auth/accept-invite', {
      json: { token: 'invite-token' },
    });
  });
});
