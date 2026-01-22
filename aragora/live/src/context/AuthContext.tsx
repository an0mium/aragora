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

interface AuthState {
  user: User | null;
  organization: Organization | null;
  tokens: Tokens | null;
  isLoading: boolean;
  isAuthenticated: boolean;
}

interface AuthContextType extends AuthState {
  login: (email: string, password: string) => Promise<{ success: boolean; error?: string }>;
  register: (email: string, password: string, name?: string, organization?: string) => Promise<{ success: boolean; error?: string }>;
  logout: () => Promise<void>;
  refreshToken: () => Promise<boolean>;
  setTokens: (accessToken: string, refreshToken: string) => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const API_BASE = API_BASE_URL;

// Storage keys
const TOKENS_KEY = 'aragora_tokens';
const USER_KEY = 'aragora_user';

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

function clearAuth(): void {
  localStorage.removeItem(TOKENS_KEY);
  localStorage.removeItem(USER_KEY);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({
    user: null,
    organization: null,
    tokens: null,
    isLoading: true,
    isAuthenticated: false,
  });

  // Check for stored auth on mount
  useEffect(() => {
    const tokens = getStoredTokens();
    const user = getStoredUser();

    if (tokens && user) {
      // Check if token is expired
      const expiresAt = new Date(tokens.expires_at);
      if (expiresAt > new Date()) {
        setState({
          user,
          organization: null,
          tokens,
          isLoading: false,
          isAuthenticated: true,
        });
        return;
      }
    }

    setState(prev => ({ ...prev, isLoading: false }));
  }, []);

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
      storeAuth(user, tokens);

      setState({
        user,
        organization: data.organization || null,
        tokens,
        isLoading: false,
        isAuthenticated: true,
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
      storeAuth(user, tokens);

      setState({
        user,
        organization: data.organization || null,
        tokens,
        isLoading: false,
        isAuthenticated: true,
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
      tokens: null,
      isLoading: false,
      isAuthenticated: false,
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
          tokens: null,
          isLoading: false,
          isAuthenticated: false,
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
  const setTokens = useCallback(async (accessToken: string, refreshToken: string) => {
    // Calculate expiry (default 1 hour from now if not provided)
    const expiresAt = new Date(Date.now() + 3600 * 1000).toISOString();

    const tokens: Tokens = {
      access_token: accessToken,
      refresh_token: refreshToken,
      expires_at: expiresAt,
    };

    // Store tokens first
    localStorage.setItem(TOKENS_KEY, JSON.stringify(tokens));

    // Fetch user profile using the access token
    try {
      const response = await fetch(`${API_BASE}/api/user/me`, {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        const user = data.user;

        // Store user
        localStorage.setItem(USER_KEY, JSON.stringify(user));

        // Update state
        setState({
          user,
          organization: data.organization || null,
          tokens,
          isLoading: false,
          isAuthenticated: true,
        });
      } else {
        // Token might be invalid - clear and throw
        clearAuth();
        throw new Error('Failed to fetch user profile');
      }
    } catch {
      // If user fetch fails, still store tokens but mark as loading
      // The user might be created but profile endpoint doesn't exist
      setState(prev => ({
        ...prev,
        tokens,
        isLoading: false,
        isAuthenticated: true,
      }));
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
    <AuthContext.Provider value={{ ...state, login, register, logout, refreshToken, setTokens }}>
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
