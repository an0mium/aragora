/**
 * Additional Tests for AuthContext
 *
 * Tests cover features NOT in the root __tests__/AuthContext.test.tsx:
 * - setTokens (OAuth callback flow)
 * - Organization switching
 * - getCurrentOrgRole
 * - useAuth hook throws outside provider
 * - useRequireAuth redirects when not authenticated
 */

import React from 'react';
import { render, screen, waitFor, act, renderHook } from '@testing-library/react';
import { AuthProvider, useAuth, useRequireAuth } from '../AuthContext';

// Mock localStorage
const mockLocalStorage: Record<string, string> = {};
const localStorageMock = {
  getItem: jest.fn((key: string) => mockLocalStorage[key] || null),
  setItem: jest.fn((key: string, value: string) => {
    mockLocalStorage[key] = value;
  }),
  removeItem: jest.fn((key: string) => {
    delete mockLocalStorage[key];
  }),
  clear: jest.fn(() => {
    Object.keys(mockLocalStorage).forEach(key => delete mockLocalStorage[key]);
  }),
};

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

// Mock window.location using delete + assignment pattern
const originalLocation = window.location;
delete (window as { location?: Location }).location;
window.location = {
  ...originalLocation,
  href: '',
  assign: jest.fn(),
  replace: jest.fn(),
  reload: jest.fn(),
} as unknown as Location;

// Mock data
const mockUser = {
  id: 'user-123',
  email: 'test@example.com',
  name: 'Test User',
  role: 'user',
  org_id: 'org-123',
  is_active: true,
  created_at: '2026-01-01T00:00:00Z',
};

const mockTokens = {
  access_token: 'access-token-123',
  refresh_token: 'refresh-token-123',
  expires_at: new Date(Date.now() + 3600000).toISOString(), // 1 hour from now
};

const mockOrganization = {
  id: 'org-123',
  name: 'Test Org',
  slug: 'test-org',
  tier: 'pro',
  owner_id: 'user-123',
};

const mockUserOrganization = {
  user_id: 'user-123',
  org_id: 'org-123',
  organization: mockOrganization,
  role: 'owner' as const,
  is_default: true,
  joined_at: '2026-01-01T00:00:00Z',
};

// Test consumer component
function TestConsumer({ onAuth }: { onAuth?: (auth: ReturnType<typeof useAuth>) => void }) {
  const auth = useAuth();
  React.useEffect(() => {
    onAuth?.(auth);
  }, [auth, onAuth]);
  return (
    <div>
      <span data-testid="is-authenticated">{auth.isAuthenticated.toString()}</span>
      <span data-testid="is-loading">{auth.isLoading.toString()}</span>
      <span data-testid="user-email">{auth.user?.email || 'none'}</span>
      <span data-testid="org-name">{auth.organization?.name || 'none'}</span>
    </div>
  );
}

// Test consumer for useRequireAuth
function RequireAuthConsumer() {
  const auth = useRequireAuth();
  return (
    <div>
      <span data-testid="is-authenticated">{auth.isAuthenticated.toString()}</span>
    </div>
  );
}

describe('AuthContext', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorageMock.clear();
    mockLocation.href = '';
    (global.fetch as jest.Mock).mockReset();
  });

  describe('Initial State', () => {
    it('starts with unauthenticated state when no stored auth', async () => {
      render(
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      );

      // Initial loading state
      expect(screen.getByTestId('is-loading').textContent).toBe('true');

      // After mount, should be not authenticated
      await waitFor(() => {
        expect(screen.getByTestId('is-loading').textContent).toBe('false');
      });
      expect(screen.getByTestId('is-authenticated').textContent).toBe('false');
      expect(screen.getByTestId('user-email').textContent).toBe('none');
    });

    it('restores auth from localStorage on mount', async () => {
      // Pre-populate localStorage
      localStorageMock.setItem('aragora_tokens', JSON.stringify(mockTokens));
      localStorageMock.setItem('aragora_user', JSON.stringify(mockUser));
      localStorageMock.setItem('aragora_active_org', JSON.stringify(mockOrganization));
      localStorageMock.setItem('aragora_user_orgs', JSON.stringify([mockUserOrganization]));

      // Mock organizations fetch
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ organizations: [mockUserOrganization] }),
      });

      render(
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-authenticated').textContent).toBe('true');
      });
      expect(screen.getByTestId('user-email').textContent).toBe('test@example.com');
      expect(screen.getByTestId('org-name').textContent).toBe('Test Org');
    });

    it('clears expired tokens on mount', async () => {
      const expiredTokens = {
        ...mockTokens,
        expires_at: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago
      };

      localStorageMock.setItem('aragora_tokens', JSON.stringify(expiredTokens));
      localStorageMock.setItem('aragora_user', JSON.stringify(mockUser));

      render(
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-loading').textContent).toBe('false');
      });
      expect(screen.getByTestId('is-authenticated').textContent).toBe('false');
    });
  });

  describe('Login', () => {
    it('successfully logs in and stores auth', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          user: mockUser,
          tokens: mockTokens,
          organization: mockOrganization,
          organizations: [mockUserOrganization],
        }),
      });

      let authRef: ReturnType<typeof useAuth> | null = null;

      render(
        <AuthProvider>
          <TestConsumer onAuth={(auth) => { authRef = auth; }} />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-loading').textContent).toBe('false');
      });

      // Perform login
      await act(async () => {
        const result = await authRef!.login('test@example.com', 'password123');
        expect(result.success).toBe(true);
      });

      expect(screen.getByTestId('is-authenticated').textContent).toBe('true');
      expect(screen.getByTestId('user-email').textContent).toBe('test@example.com');

      // Verify localStorage was updated
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        'aragora_tokens',
        expect.any(String)
      );
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        'aragora_user',
        expect.any(String)
      );
    });

    it('handles login failure', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        json: async () => ({ error: 'Invalid credentials' }),
      });

      let authRef: ReturnType<typeof useAuth> | null = null;

      render(
        <AuthProvider>
          <TestConsumer onAuth={(auth) => { authRef = auth; }} />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-loading').textContent).toBe('false');
      });

      await act(async () => {
        const result = await authRef!.login('test@example.com', 'wrongpassword');
        expect(result.success).toBe(false);
        expect(result.error).toBe('Invalid credentials');
      });

      expect(screen.getByTestId('is-authenticated').textContent).toBe('false');
    });

    it('handles network error during login', async () => {
      (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

      let authRef: ReturnType<typeof useAuth> | null = null;

      render(
        <AuthProvider>
          <TestConsumer onAuth={(auth) => { authRef = auth; }} />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-loading').textContent).toBe('false');
      });

      await act(async () => {
        const result = await authRef!.login('test@example.com', 'password123');
        expect(result.success).toBe(false);
        expect(result.error).toBe('Network error. Please try again.');
      });
    });
  });

  describe('Register', () => {
    it('successfully registers and logs in', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          user: mockUser,
          tokens: mockTokens,
          organization: mockOrganization,
          organizations: [mockUserOrganization],
        }),
      });

      let authRef: ReturnType<typeof useAuth> | null = null;

      render(
        <AuthProvider>
          <TestConsumer onAuth={(auth) => { authRef = auth; }} />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-loading').textContent).toBe('false');
      });

      await act(async () => {
        const result = await authRef!.register('test@example.com', 'password123', 'Test User', 'Test Org');
        expect(result.success).toBe(true);
      });

      expect(screen.getByTestId('is-authenticated').textContent).toBe('true');
      expect(screen.getByTestId('user-email').textContent).toBe('test@example.com');
    });

    it('handles registration failure', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        json: async () => ({ error: 'Email already exists' }),
      });

      let authRef: ReturnType<typeof useAuth> | null = null;

      render(
        <AuthProvider>
          <TestConsumer onAuth={(auth) => { authRef = auth; }} />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-loading').textContent).toBe('false');
      });

      await act(async () => {
        const result = await authRef!.register('test@example.com', 'password123');
        expect(result.success).toBe(false);
        expect(result.error).toBe('Email already exists');
      });
    });
  });

  describe('Logout', () => {
    it('clears auth state and localStorage on logout', async () => {
      // Pre-populate localStorage
      localStorageMock.setItem('aragora_tokens', JSON.stringify(mockTokens));
      localStorageMock.setItem('aragora_user', JSON.stringify(mockUser));

      // Mock logout API call
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: async () => ({}),
      });

      let authRef: ReturnType<typeof useAuth> | null = null;

      render(
        <AuthProvider>
          <TestConsumer onAuth={(auth) => { authRef = auth; }} />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-authenticated').textContent).toBe('true');
      });

      await act(async () => {
        await authRef!.logout();
      });

      expect(screen.getByTestId('is-authenticated').textContent).toBe('false');
      expect(screen.getByTestId('user-email').textContent).toBe('none');
      expect(localStorageMock.removeItem).toHaveBeenCalledWith('aragora_tokens');
      expect(localStorageMock.removeItem).toHaveBeenCalledWith('aragora_user');
    });

    it('handles logout even when API call fails', async () => {
      localStorageMock.setItem('aragora_tokens', JSON.stringify(mockTokens));
      localStorageMock.setItem('aragora_user', JSON.stringify(mockUser));

      (global.fetch as jest.Mock).mockRejectedValue(new Error('Network error'));

      let authRef: ReturnType<typeof useAuth> | null = null;

      render(
        <AuthProvider>
          <TestConsumer onAuth={(auth) => { authRef = auth; }} />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-authenticated').textContent).toBe('true');
      });

      await act(async () => {
        await authRef!.logout();
      });

      // Should still log out locally even if API fails
      expect(screen.getByTestId('is-authenticated').textContent).toBe('false');
    });
  });

  describe('Token Refresh', () => {
    it('successfully refreshes token', async () => {
      localStorageMock.setItem('aragora_tokens', JSON.stringify(mockTokens));
      localStorageMock.setItem('aragora_user', JSON.stringify(mockUser));

      const newTokens = {
        ...mockTokens,
        access_token: 'new-access-token',
        expires_at: new Date(Date.now() + 7200000).toISOString(),
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ tokens: newTokens }),
      });

      let authRef: ReturnType<typeof useAuth> | null = null;

      render(
        <AuthProvider>
          <TestConsumer onAuth={(auth) => { authRef = auth; }} />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-authenticated').textContent).toBe('true');
      });

      await act(async () => {
        const result = await authRef!.refreshToken();
        expect(result).toBe(true);
      });
    });

    it('clears auth when refresh fails', async () => {
      localStorageMock.setItem('aragora_tokens', JSON.stringify(mockTokens));
      localStorageMock.setItem('aragora_user', JSON.stringify(mockUser));

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 401,
      });

      let authRef: ReturnType<typeof useAuth> | null = null;

      render(
        <AuthProvider>
          <TestConsumer onAuth={(auth) => { authRef = auth; }} />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-authenticated').textContent).toBe('true');
      });

      await act(async () => {
        const result = await authRef!.refreshToken();
        expect(result).toBe(false);
      });

      expect(screen.getByTestId('is-authenticated').textContent).toBe('false');
    });

    it('returns false when no refresh token available', async () => {
      let authRef: ReturnType<typeof useAuth> | null = null;

      render(
        <AuthProvider>
          <TestConsumer onAuth={(auth) => { authRef = auth; }} />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-loading').textContent).toBe('false');
      });

      await act(async () => {
        const result = await authRef!.refreshToken();
        expect(result).toBe(false);
      });
    });
  });

  describe('setTokens (OAuth flow)', () => {
    it('successfully sets tokens and fetches user profile', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({
          user: mockUser,
          organization: mockOrganization,
          organizations: [mockUserOrganization],
        }),
      });

      let authRef: ReturnType<typeof useAuth> | null = null;

      render(
        <AuthProvider>
          <TestConsumer onAuth={(auth) => { authRef = auth; }} />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-loading').textContent).toBe('false');
      });

      await act(async () => {
        await authRef!.setTokens('new-access-token', 'new-refresh-token');
      });

      expect(screen.getByTestId('is-authenticated').textContent).toBe('true');
      expect(screen.getByTestId('user-email').textContent).toBe('test@example.com');
    });

    it('clears tokens on 401 response', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 401,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => ({ error: 'Invalid token' }),
      });

      let authRef: ReturnType<typeof useAuth> | null = null;

      render(
        <AuthProvider>
          <TestConsumer onAuth={(auth) => { authRef = auth; }} />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-loading').textContent).toBe('false');
      });

      await act(async () => {
        await expect(authRef!.setTokens('invalid-token', 'refresh-token')).rejects.toThrow(
          'Authentication failed: Invalid tokens'
        );
      });

      expect(localStorageMock.removeItem).toHaveBeenCalledWith('aragora_tokens');
    });
  });

  describe('Organization Switching', () => {
    it('successfully switches organization', async () => {
      localStorageMock.setItem('aragora_tokens', JSON.stringify(mockTokens));
      localStorageMock.setItem('aragora_user', JSON.stringify(mockUser));
      localStorageMock.setItem('aragora_user_orgs', JSON.stringify([mockUserOrganization]));

      const newOrg = {
        id: 'org-456',
        name: 'New Org',
        slug: 'new-org',
        tier: 'enterprise',
        owner_id: 'user-123',
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ organization: newOrg }),
      });

      let authRef: ReturnType<typeof useAuth> | null = null;

      render(
        <AuthProvider>
          <TestConsumer onAuth={(auth) => { authRef = auth; }} />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-authenticated').textContent).toBe('true');
      });

      await act(async () => {
        const result = await authRef!.switchOrganization('org-456');
        expect(result.success).toBe(true);
      });

      expect(screen.getByTestId('org-name').textContent).toBe('New Org');
    });

    it('returns error when not authenticated', async () => {
      let authRef: ReturnType<typeof useAuth> | null = null;

      render(
        <AuthProvider>
          <TestConsumer onAuth={(auth) => { authRef = auth; }} />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-loading').textContent).toBe('false');
      });

      await act(async () => {
        const result = await authRef!.switchOrganization('org-456');
        expect(result.success).toBe(false);
        expect(result.error).toBe('Not authenticated');
      });
    });
  });

  describe('getCurrentOrgRole', () => {
    it('returns the correct role for current organization', async () => {
      localStorageMock.setItem('aragora_tokens', JSON.stringify(mockTokens));
      localStorageMock.setItem('aragora_user', JSON.stringify(mockUser));
      localStorageMock.setItem('aragora_active_org', JSON.stringify(mockOrganization));
      localStorageMock.setItem('aragora_user_orgs', JSON.stringify([mockUserOrganization]));

      let authRef: ReturnType<typeof useAuth> | null = null;

      render(
        <AuthProvider>
          <TestConsumer onAuth={(auth) => { authRef = auth; }} />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-authenticated').textContent).toBe('true');
      });

      expect(authRef!.getCurrentOrgRole()).toBe('owner');
    });

    it('returns null when no organization is active', async () => {
      localStorageMock.setItem('aragora_tokens', JSON.stringify(mockTokens));
      localStorageMock.setItem('aragora_user', JSON.stringify(mockUser));

      let authRef: ReturnType<typeof useAuth> | null = null;

      render(
        <AuthProvider>
          <TestConsumer onAuth={(auth) => { authRef = auth; }} />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-authenticated').textContent).toBe('true');
      });

      expect(authRef!.getCurrentOrgRole()).toBeNull();
    });
  });

  describe('useAuth hook', () => {
    it('throws error when used outside AuthProvider', () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

      expect(() => {
        render(<TestConsumer />);
      }).toThrow('useAuth must be used within an AuthProvider');

      consoleSpy.mockRestore();
    });
  });

  describe('useRequireAuth hook', () => {
    it('redirects to login when not authenticated', async () => {
      render(
        <AuthProvider>
          <RequireAuthConsumer />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(mockLocation.href).toBe('/auth/login');
      });
    });

    it('does not redirect when authenticated', async () => {
      localStorageMock.setItem('aragora_tokens', JSON.stringify(mockTokens));
      localStorageMock.setItem('aragora_user', JSON.stringify(mockUser));

      render(
        <AuthProvider>
          <RequireAuthConsumer />
        </AuthProvider>
      );

      await waitFor(() => {
        expect(screen.getByTestId('is-authenticated').textContent).toBe('true');
      });

      expect(mockLocation.href).toBe('');
    });
  });
});
