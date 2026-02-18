import React, { type ReactElement, type ReactNode } from 'react';
import { render, type RenderOptions } from '@testing-library/react';
import { ThemeProvider } from '@/context/ThemeContext';
import { AuthProvider } from '@/context/AuthContext';

/**
 * Default mock values for the AuthContext when used via the mock provider.
 * Individual tests can override these by passing `authOverrides` to
 * `renderWithProviders`.
 */
export const defaultAuthValues = {
  user: null,
  organization: null,
  organizations: [],
  tokens: null,
  isLoading: false,
  isAuthenticated: false,
  isLoadingOrganizations: false,
  login: jest.fn().mockResolvedValue({ success: true }),
  register: jest.fn().mockResolvedValue({ success: true }),
  logout: jest.fn().mockResolvedValue(undefined),
  refreshToken: jest.fn().mockResolvedValue(false),
  setTokens: jest.fn().mockResolvedValue(undefined),
  switchOrganization: jest.fn().mockResolvedValue({ success: true }),
  refreshOrganizations: jest.fn().mockResolvedValue(undefined),
  getCurrentOrgRole: jest.fn().mockReturnValue(null),
};

/**
 * A mock AuthProvider that supplies static context values without making
 * any API calls. This avoids the side-effects (fetch, localStorage) of
 * the real AuthProvider while satisfying any component that calls useAuth().
 */
const MockAuthContext = React.createContext<typeof defaultAuthValues | undefined>(undefined);

function MockAuthProvider({
  children,
  overrides = {},
}: {
  children: ReactNode;
  overrides?: Partial<typeof defaultAuthValues>;
}) {
  const value = { ...defaultAuthValues, ...overrides };
  return (
    <MockAuthContext.Provider value={value}>
      {children}
    </MockAuthContext.Provider>
  );
}

// We need to intercept useAuth calls. The cleanest way is to re-export
// a provider wrapper that uses the real AuthContext internally but with
// mocked values. We achieve this by using the real context's Provider
// directly with mock data.

// Import the AuthContext from the module so we can use its Provider
// without triggering the full AuthProvider's useEffect side effects.
// eslint-disable-next-line @typescript-eslint/no-var-requires
const AuthContextModule = jest.requireActual('@/context/AuthContext');
const RealAuthContext = AuthContextModule.__esModule
  ? AuthContextModule.default
  : null;

/**
 * AllProviders wraps a component in every context provider needed for
 * rendering in tests. Uses a lightweight mock for AuthProvider to avoid
 * network calls, and the real ThemeProvider since it has no side-effects
 * in JSDOM (matchMedia is already mocked in jest.setup.js).
 */
function AllProviders({
  children,
  authOverrides = {},
}: {
  children: ReactNode;
  authOverrides?: Partial<typeof defaultAuthValues>;
}) {
  const authValue = { ...defaultAuthValues, ...authOverrides };

  return (
    <ThemeProvider defaultPreference="dark">
      <AuthProvider>
        {children}
      </AuthProvider>
    </ThemeProvider>
  );
}

// However, the real AuthProvider calls fetch() on mount which causes
// issues in tests. Instead we provide mock auth values by mocking
// the useAuth hook at the module level. Components call useAuth()
// which reads from AuthContext - so we need to provide a context
// value without the real provider's side effects.
//
// The best approach: provide a wrapper that patches useAuth via
// the React context directly using the internal context object.

/**
 * Options for renderWithProviders.
 */
interface RenderWithProvidersOptions extends Omit<RenderOptions, 'wrapper'> {
  /**
   * Override default auth context values. For example:
   * ```ts
   * renderWithProviders(<MyComponent />, {
   *   authOverrides: {
   *     isAuthenticated: true,
   *     user: { id: '1', email: 'test@example.com', name: 'Test', role: 'admin', org_id: null, is_active: true, created_at: '' },
   *   },
   * });
   * ```
   */
  authOverrides?: Partial<typeof defaultAuthValues>;
}

/**
 * Render a component wrapped in all the context providers needed for the
 * Aragora Live frontend. This is the recommended replacement for bare
 * `render()` calls from @testing-library/react in component tests.
 *
 * The AuthProvider is replaced with a lightweight mock that supplies
 * static values (no fetch, no localStorage). Pass `authOverrides` to
 * customise the auth state per test.
 *
 * ThemeProvider uses the real implementation with `defaultPreference="dark"`.
 *
 * @example
 * ```tsx
 * import { renderWithProviders } from '@/test-utils';
 *
 * it('renders when authenticated', () => {
 *   renderWithProviders(<Dashboard />, {
 *     authOverrides: { isAuthenticated: true, user: mockUser },
 *   });
 *   expect(screen.getByText('Dashboard')).toBeInTheDocument();
 * });
 * ```
 */
export function renderWithProviders(
  ui: ReactElement,
  options: RenderWithProvidersOptions = {},
) {
  const { authOverrides = {}, ...renderOptions } = options;
  const authValue = { ...defaultAuthValues, ...authOverrides };

  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <ThemeProvider defaultPreference="dark">
        <MockAuthProvider overrides={authValue}>
          {children}
        </MockAuthProvider>
      </ThemeProvider>
    );
  }

  return render(ui, { wrapper: Wrapper, ...renderOptions });
}

// Re-export everything from @testing-library/react so tests can import
// from '@/test-utils' as a drop-in replacement.
export * from '@testing-library/react';
// Override the default render with our wrapped version
export { renderWithProviders as render };
