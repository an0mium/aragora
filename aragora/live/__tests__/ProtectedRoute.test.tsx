/**
 * Tests for ProtectedRoute component
 *
 * Tests cover:
 * - Loading state during auth check
 * - Redirect to login when not authenticated
 * - Rendering children when authenticated
 * - Tier requirement enforcement
 */

import { render, screen, waitFor } from '@testing-library/react';
import { ProtectedRoute } from '../src/components/auth/ProtectedRoute';
import { mockRouter } from 'next/navigation';

// Mock AuthContext
const mockUseAuth = jest.fn();
jest.mock('../src/context/AuthContext', () => ({
  useAuth: () => mockUseAuth(),
}));

// Mock MatrixRain components
jest.mock('../src/components/MatrixRain', () => ({
  Scanlines: () => null,
  CRTVignette: () => null,
}));

describe('ProtectedRoute', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockRouter.push.mockClear();
  });

  describe('Loading State', () => {
    it('shows loading state while auth is being checked', () => {
      mockUseAuth.mockReturnValue({
        isLoading: true,
        isAuthenticated: false,
        organization: null,
      });

      render(
        <ProtectedRoute>
          <div>Protected Content</div>
        </ProtectedRoute>
      );

      expect(screen.getByText(/authenticating/i)).toBeInTheDocument();
      expect(screen.queryByText('Protected Content')).not.toBeInTheDocument();
    });
  });

  describe('Authentication Check', () => {
    it('redirects to login when not authenticated', async () => {
      mockUseAuth.mockReturnValue({
        isLoading: false,
        isAuthenticated: false,
        organization: null,
      });

      // Mock window.location.pathname
      Object.defineProperty(window, 'location', {
        value: { pathname: '/billing' },
        writable: true,
      });

      render(
        <ProtectedRoute>
          <div>Protected Content</div>
        </ProtectedRoute>
      );

      await waitFor(() => {
        expect(mockRouter.push).toHaveBeenCalledWith(
          '/auth/login?returnUrl=%2Fbilling'
        );
      });

      expect(screen.queryByText('Protected Content')).not.toBeInTheDocument();
    });

    it('renders children when authenticated', async () => {
      mockUseAuth.mockReturnValue({
        isLoading: false,
        isAuthenticated: true,
        organization: { id: 'org-1', tier: 'starter' },
      });

      render(
        <ProtectedRoute>
          <div>Protected Content</div>
        </ProtectedRoute>
      );

      await waitFor(() => {
        expect(screen.getByText('Protected Content')).toBeInTheDocument();
      });

      expect(mockRouter.push).not.toHaveBeenCalled();
    });
  });

  describe('Tier Requirements', () => {
    it('shows upgrade message when tier requirement not met', async () => {
      mockUseAuth.mockReturnValue({
        isLoading: false,
        isAuthenticated: true,
        organization: { id: 'org-1', tier: 'free' },
      });

      render(
        <ProtectedRoute requiredTier="professional">
          <div>Premium Content</div>
        </ProtectedRoute>
      );

      await waitFor(() => {
        expect(screen.getByText(/upgrade required/i)).toBeInTheDocument();
      });

      expect(screen.queryByText('Premium Content')).not.toBeInTheDocument();
      expect(screen.getByText(/professional/i)).toBeInTheDocument();
    });

    it('renders content when tier requirement is met', async () => {
      mockUseAuth.mockReturnValue({
        isLoading: false,
        isAuthenticated: true,
        organization: { id: 'org-1', tier: 'professional' },
      });

      render(
        <ProtectedRoute requiredTier="starter">
          <div>Premium Content</div>
        </ProtectedRoute>
      );

      await waitFor(() => {
        expect(screen.getByText('Premium Content')).toBeInTheDocument();
      });
    });

    it('renders content when tier is enterprise (highest)', async () => {
      mockUseAuth.mockReturnValue({
        isLoading: false,
        isAuthenticated: true,
        organization: { id: 'org-1', tier: 'enterprise' },
      });

      render(
        <ProtectedRoute requiredTier="professional">
          <div>Enterprise Content</div>
        </ProtectedRoute>
      );

      await waitFor(() => {
        expect(screen.getByText('Enterprise Content')).toBeInTheDocument();
      });
    });

    it('shows current tier in upgrade message', async () => {
      mockUseAuth.mockReturnValue({
        isLoading: false,
        isAuthenticated: true,
        organization: { id: 'org-1', tier: 'starter' },
      });

      render(
        <ProtectedRoute requiredTier="enterprise">
          <div>Enterprise Content</div>
        </ProtectedRoute>
      );

      await waitFor(() => {
        expect(screen.getByText(/starter/i)).toBeInTheDocument();
        expect(screen.getByText(/enterprise/i)).toBeInTheDocument();
      });
    });
  });

  describe('Custom Redirect', () => {
    it('uses custom redirectTo path', async () => {
      mockUseAuth.mockReturnValue({
        isLoading: false,
        isAuthenticated: false,
        organization: null,
      });

      render(
        <ProtectedRoute redirectTo="/custom-return">
          <div>Protected Content</div>
        </ProtectedRoute>
      );

      await waitFor(() => {
        expect(mockRouter.push).toHaveBeenCalledWith(
          '/auth/login?returnUrl=%2Fcustom-return'
        );
      });
    });
  });
});
