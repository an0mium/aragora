'use client';

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { API_BASE_URL } from '@/config';

interface User {
  id: string;
  email: string;
  name: string;
  role: string;
  org_id: string | null;
  is_active: boolean;
  created_at: string;
}

interface Tokens {
  access_token: string;
  refresh_token: string;
  expires_at: string;
}

interface Organization {
  id: string;
  name: string;
  slug: string;
  tier: string;
  owner_id: string;
}

/**
 * Represents a user's membership in an organization (multi-org support).
 */
interface UserOrganization {
  user_id: string;
  org_id: string;
  organization: Organization;
  role: 'member' | 'admin' | 'owner';
  is_default: boolean;
  joined_at: string;
}

interface AuthState {
  user: User | null;
  /** Currently active organization */
  organization: Organization | null;
  /** All organizations the user belongs to (multi-org support) */
  organizations: UserOrganization[];
  tokens: Tokens | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  /** Whether organizations are being loaded */
  isLoadingOrganizations: boolean;
}

interface AuthContextType extends AuthState {
  login: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
  register: (email: string, password: string, name?: string, organization?: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => Promise<void>;
  refreshToken: () => Promise<boolean>;
  setTokens: (accessToken: string, refreshToken: string) => Promise<void>;
  /** Switch to a different organization context */
  switchOrganization: (orgId: string, setAsDefault?: boolean) => Promise<{ success: boolean; error?: string }>;
  /** Refresh the list of user's organizations */
  refreshOrganizations: () => Promise<void>;
  /** Get the user's role in the current organization */
  getCurrentOrgRole: () => 'member' | 'admin' | 'owner' | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const API_BASE = API_BASE_URL;

// Storage keys
const TOKENS_KEY = 'aragora_tokens';
const USER_KEY = 'aragora_user';
const ACTIVE_ORG_KEY = 'aragora_active_org';
const USER_ORGS_KEY = 'aragora_user_orgs';

function getStoredTokens(): Tokens | null {
  if (typeof window === 'undefined') return null;
  const stored = localStorage.getItem(TOKENS_KEY);
  if (!stored) return null;
  try {
    return JSON.parse(stored);
  } catch {
    return null;
  }
}

function getStoredUser(): User | null {
  if (typeof window === 'undefined') return null;
  const stored = localStorage.getItem(USER_KEY);
  if (!stored) return null;
  try {
    return JSON.parse(stored);
  } catch {
    return null;
  }
}

function storeAuth(user: User, tokens: Tokens): void {
  localStorage.setItem(TOKENS_KEY, JSON.stringify(tokens));
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

function storeActiveOrg(org: Organization | null): void {
  if (org) {
    localStorage.setItem(ACTIVE_ORG_KEY, JSON.stringify(org));
  } else {
    localStorage.removeItem(ACTIVE_ORG_KEY);
  }
}

function getStoredActiveOrg(): Organization | null {
  if (typeof window === 'undefined') return null;
  const stored = localStorage.getItem(ACTIVE_ORG_KEY);
  if (!stored) return null;
  try {
    return JSON.parse(stored);
  } catch {
    return null;
  }
}

function storeUserOrgs(orgs: UserOrganization[]): void {
  localStorage.setItem(USER_ORGS_KEY, JSON.stringify(orgs));
}

function getStoredUserOrgs(): UserOrganization[] {
  if (typeof window === 'undefined') return [];
  const stored = localStorage.getItem(USER_ORGS_KEY);
  if (!stored) return [];
  try {
    return JSON.parse(stored);
  } catch {
    return [];
  }
}

function clearAuth(): void {
  localStorage.removeItem(TOKENS_KEY);
  localStorage.removeItem(USER_KEY);
  localStorage.removeItem(ACTIVE_ORG_KEY);
  localStorage.removeItem(USER_ORGS_KEY);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({
    user: null,
    organization: null,
    organizations: [],
    tokens: null,
    isLoading: true,
    isAuthenticated: false,
    isLoadingOrganizations: false,
  });

  // Fetch user's organizations
  const fetchOrganizations = useCallback(async (accessToken: string): Promise<UserOrganization[]> => {
    try {
      const response = await fetch(`${API_BASE}/api/v1/user/organizations`, {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        return data.organizations || [];
      }
      return [];
    } catch {
      return [];
    }
  }, []);

  // Refresh organizations list
  const refreshOrganizations = useCallback(async () => {
    const tokens = state.tokens || getStoredTokens();
    if (!tokens?.access_token) return;

    setState(prev => ({ ...prev, isLoadingOrganizations: true }));

    try {
      const orgs = await fetchOrganizations(tokens.access_token);
      storeUserOrgs(orgs);
      setState(prev => ({
        ...prev,
        organizations: orgs,
        isLoadingOrganizations: false,
      }));
    } catch {
      setState(prev => ({ ...prev, isLoadingOrganizations: false }));
    }
  }, [state.tokens, fetchOrganizations]);

  // Switch organization context
  const switchOrganization = useCallback(async (orgId: string, setAsDefault = false): Promise<{ success: boolean; error?: string }> => {
    const tokens = state.tokens || getStoredTokens();
    if (!tokens?.access_token) {
      return { success: false, error: 'Not authenticated' };
    }

    try {
      const response = await fetch(`${API_BASE}/api/v1/user/organizations/switch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${tokens.access_token}`,
        },
        body: JSON.stringify({ org_id: orgId, set_as_default: setAsDefault }),
      });

      const data = await response.json();

      if (!response.ok) {
        return { success: false, error: data.error || 'Failed to switch organization' };
      }

      const newOrg = data.organization;
      storeActiveOrg(newOrg);

      // Update organizations list if default was changed
      if (setAsDefault && state.organizations.length > 0) {
        const updatedOrgs = state.organizations.map(o => ({
          ...o,
          is_default: o.org_id === orgId,
        }));
        storeUserOrgs(updatedOrgs);
        setState(prev => ({
          ...prev,
          organization: newOrg,
          organizations: updatedOrgs,
        }));
      } else {
        setState(prev => ({
          ...prev,
          organization: newOrg,
        }));
      }

      // If a new token was issued with org context, update it
      if (data.access_token) {
        const newTokens = {
          ...tokens,
          access_token: data.access_token,
        };
        localStorage.setItem(TOKENS_KEY, JSON.stringify(newTokens));
        setState(prev => ({
          ...prev,
          tokens: newTokens,
        }));
      }

      return { success: true };
    } catch {
      return { success: false, error: 'Network error. Please try again.' };
    }
  }, [state.tokens, state.organizations]);

  // Get current org role
  const getCurrentOrgRole = useCallback((): 'member' | 'admin' | 'owner' | null => {
    if (!state.organization) return null;
    const membership = state.organizations.find(o => o.org_id === state.organization?.id);
    return membership?.role || null;
  }, [state.organization, state.organizations]);

  // Check for stored auth on mount
  useEffect(() => {
    const tokens = getStoredTokens();
    const user = getStoredUser();
    const activeOrg = getStoredActiveOrg();
    const userOrgs = getStoredUserOrgs();

    if (tokens && user) {
      // Check if token is expired
      const expiresAt = new Date(tokens.expires_at);
      if (expiresAt > new Date()) {
        setState({
          user,
          organization: activeOrg,
          organizations: userOrgs,
          tokens,
          isLoading: false,
          isAuthenticated: true,
          isLoadingOrganizations: false,
        });
        return;
      }
    }

    setState(prev => ({ ...prev, isLoading: false }));
  }, []);

  // Fetch organizations when authenticated
  useEffect(() => {
    if (state.isAuthenticated && state.tokens?.access_token && state.organizations.length === 0) {
      refreshOrganizations();
    }
  }, [state.isAuthenticated, state.tokens?.access_token, state.organizations.length, refreshOrganizations]);

  const login = useCallback(async (email: string, password: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (!response.ok) {
        return { success: false, error: data.error || 'Login failed' };
      }

      const { user, tokens } = data;
      const organization = data.organization || null;
      const organizations = data.organizations || [];

      storeAuth(user, tokens);
      storeActiveOrg(organization);
      storeUserOrgs(organizations);

      setState({
        user,
        organization,
        organizations,
        tokens,
        isLoading: false,
        isAuthenticated: true,
        isLoadingOrganizations: false,
      });

      return { success: true };
    } catch {
      return { success: false, error: 'Network error. Please try again.' };
    }
  }, []);

  const register = useCallback(async (
    email: string,
    password: string,
    name?: string,
    organization?: string
  ) => {
    try {
      const response = await fetch(`${API_BASE}/api/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, name, organization }),
      });

      const data = await response.json();

      if (!response.ok) {
        return { success: false, error: data.error || 'Registration failed' };
      }

      const { user, tokens } = data;
      const org = data.organization || null;
      const orgs = data.organizations || [];

      storeAuth(user, tokens);
      storeActiveOrg(org);
      storeUserOrgs(orgs);

      setState({
        user,
        organization: org,
        organizations: orgs,
        tokens,
        isLoading: false,
        isAuthenticated: true,
        isLoadingOrganizations: false,
      });

      return { success: true };
    } catch {
      return { success: false, error: 'Network error. Please try again.' };
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      if (state.tokens?.access_token) {
        await fetch(`${API_BASE}/api/auth/logout`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${state.tokens.access_token}`,
          },
        });
      }
    } catch {
      // Ignore logout errors
    }

    clearAuth();
    setState({
      user: null,
      organization: null,
      organizations: [],
      tokens: null,
      isLoading: false,
      isAuthenticated: false,
      isLoadingOrganizations: false,
    });
  }, [state.tokens?.access_token]);

  const refreshToken = useCallback(async () => {
    const tokens = state.tokens || getStoredTokens();
    if (!tokens?.refresh_token) return false;

    try {
      const response = await fetch(`${API_BASE}/api/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: tokens.refresh_token }),
      });

      if (!response.ok) {
        clearAuth();
        setState({
          user: null,
          organization: null,
          organizations: [],
          tokens: null,
          isLoading: false,
          isAuthenticated: false,
          isLoadingOrganizations: false,
        });
        return false;
      }

      const data = await response.json();
      const newTokens = data.tokens;
      const user = state.user || getStoredUser();

      if (user) {
        storeAuth(user, newTokens);
        setState(prev => ({
          ...prev,
          tokens: newTokens,
        }));
      }

      return true;
    } catch {
      return false;
    }
  }, [state.tokens, state.user]);

  // Set tokens from OAuth callback - fetches user profile from API
  const setTokens = useCallback(async (accessToken: string, refreshTokenValue: string) => {
    console.log('[AuthContext] setTokens called');

    // Calculate expiry (default 1 hour from now if not provided)
    const expiresAt = new Date(Date.now() + 3600 * 1000).toISOString();

    const tokens: Tokens = {
      access_token: accessToken,
      refresh_token: refreshTokenValue,
      expires_at: expiresAt,
    };

    // Fetch user profile using the access token FIRST to validate
    try {
      console.log('[AuthContext] Fetching user profile to validate tokens...');
      const response = await fetch(`${API_BASE}/api/v1/auth/me`, {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });

      console.log('[AuthContext] /me response:', {
        status: response.status,
        ok: response.ok,
        contentType: response.headers.get('content-type'),
      });

      if (response.ok) {
        const data = await response.json();
        const user = data.user;
        const organization = data.organization || null;
        const organizations = data.organizations || [];

        console.log('[AuthContext] User profile fetched successfully:', user?.email);

        // Store tokens, user and orgs - only AFTER validation succeeds
        localStorage.setItem(TOKENS_KEY, JSON.stringify(tokens));
        localStorage.setItem(USER_KEY, JSON.stringify(user));
        storeActiveOrg(organization);
        storeUserOrgs(organizations);

        // Update state
        setState({
          user,
          organization,
          organizations,
          tokens,
          isLoading: false,
          isAuthenticated: true,
          isLoadingOrganizations: false,
        });
      } else if (response.status === 401) {
        // 401 means tokens are invalid - DO NOT authenticate
        console.error('[AuthContext] Token validation failed: 401 Unauthorized');
        const contentType = response.headers.get('content-type') || '';
        let errorDetail = '';
        if (contentType.includes('application/json')) {
          try {
            const errData = await response.json();
            errorDetail = errData.error || errData.message || '';
          } catch { /* ignore */ }
        }
        console.error('[AuthContext] Error detail:', errorDetail || '(no detail)');

        // Clear any partial auth data
        clearAuth();
        throw new Error('Authentication failed: Invalid tokens');
      } else {
        // Other error (500, 404, etc.) - log but don't store invalid tokens
        console.error('[AuthContext] Unexpected /me response:', response.status);
        clearAuth();
        throw new Error(`Authentication failed: ${response.status}`);
      }
    } catch (err) {
      console.error('[AuthContext] setTokens error:', err);
      // Re-throw so callback page can handle it
      throw err;
    }
  }, []);

  // Auto-refresh token before expiry
  useEffect(() => {
    if (!state.tokens?.expires_at) return;

    const expiresAt = new Date(state.tokens.expires_at);
    const refreshTime = expiresAt.getTime() - Date.now() - 60000; // 1 min before expiry

    if (refreshTime <= 0) {
      refreshToken();
      return;
    }

    const timeout = setTimeout(refreshToken, refreshTime);
    return () => clearTimeout(timeout);
  }, [state.tokens?.expires_at, refreshToken]);

  return (
    <AuthContext.Provider value={{
      ...state,
      login,
      register,
      logout,
      refreshToken,
      setTokens,
      switchOrganization,
      refreshOrganizations,
      getCurrentOrgRole,
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export function useRequireAuth() {
  const auth = useAuth();

  useEffect(() => {
    if (!auth.isLoading && !auth.isAuthenticated) {
      window.location.href = '/auth/login';
    }
  }, [auth.isLoading, auth.isAuthenticated]);

  return auth;
}

// Export types for external use
export type { User, Tokens, Organization, UserOrganization, AuthState, AuthContextType };
