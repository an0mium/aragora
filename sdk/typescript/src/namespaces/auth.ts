/**
 * Auth Namespace API
 *
 * Provides a namespaced interface for authentication operations.
 * This wraps the flat client methods for a more intuitive API.
 */

import type {
  RegisterRequest,
  RegisterResponse,
  LoginRequest,
  AuthToken,
  RefreshRequest,
  VerifyEmailRequest,
  VerifyResponse,
  User,
  UpdateProfileRequest,
  UpdateProfileResponse,
  ChangePasswordRequest,
  ForgotPasswordRequest,
  ResetPasswordRequest,
  OAuthUrlParams,
  OAuthUrl,
  OAuthCallbackRequest,
  MFASetupRequest,
  MFASetupResponse,
  MFAVerifyRequest,
  MFAVerifyResponse,
} from '../types';

/**
 * Session information.
 */
export interface SessionInfo {
  /** Session ID */
  id: string;
  /** Device/browser info */
  device: string;
  /** IP address */
  ip_address: string;
  /** Session creation time */
  created_at: string;
  /** Last activity time */
  last_active: string;
  /** Whether this is the current session */
  current: boolean;
}

/**
 * API key information.
 */
export interface ApiKeyInfo {
  /** Key ID */
  id: string;
  /** Key name */
  name: string;
  /** Key prefix (first few characters) */
  prefix: string;
  /** Creation time */
  created_at: string;
  /** Last used time */
  last_used?: string;
}

/**
 * OAuth provider information.
 */
export interface OAuthProviderInfo {
  /** Provider type (google, github, etc.) */
  type: string;
  /** Display name */
  name: string;
  /** Whether enabled */
  enabled: boolean;
  /** Authorization URL */
  auth_url: string;
}

/**
 * Interface for the internal client methods used by AuthAPI.
 */
interface AuthClientInterface {
  registerUser(body: RegisterRequest): Promise<RegisterResponse>;
  login(body: LoginRequest): Promise<AuthToken>;
  refreshToken(body: RefreshRequest): Promise<AuthToken>;
  logout(): Promise<void>;
  verifyEmail(body: VerifyEmailRequest): Promise<VerifyResponse>;
  getCurrentUser(): Promise<User>;
  updateProfile(body: UpdateProfileRequest): Promise<UpdateProfileResponse>;
  changePassword(body: ChangePasswordRequest): Promise<void>;
  requestPasswordReset(body: ForgotPasswordRequest): Promise<void>;
  resetPassword(body: ResetPasswordRequest): Promise<void>;
  getOAuthUrl(params: OAuthUrlParams): Promise<OAuthUrl>;
  completeOAuth(body: OAuthCallbackRequest): Promise<AuthToken>;
  setupMFA(body: MFASetupRequest): Promise<MFASetupResponse>;
  verifyMFASetup(body: MFAVerifyRequest): Promise<MFAVerifyResponse>;
  disableMFA(): Promise<void>;
  enableMFA(code: string): Promise<{ enabled: boolean }>;
  generateBackupCodes(): Promise<{ codes: string[] }>;
  logoutAll(): Promise<{ logged_out: boolean; sessions_revoked: number }>;
  resendVerification(email: string): Promise<{ sent: boolean }>;
  listSessions(): Promise<{ sessions: SessionInfo[] }>;
  revokeSession(sessionId: string): Promise<{ revoked: boolean }>;
  listApiKeys(): Promise<{ keys: ApiKeyInfo[] }>;
  createApiKey(name: string, expiresIn?: number): Promise<{ id: string; key: string; prefix: string }>;
  revokeApiKey(keyId: string): Promise<{ revoked: boolean }>;
  listOAuthProviders(): Promise<{ providers: OAuthProviderInfo[] }>;
  linkOAuthProvider(provider: string, code: string): Promise<{ linked: boolean }>;
  unlinkOAuthProvider(provider: string): Promise<{ unlinked: boolean }>;
  initiateSSOLogin(provider?: string, redirectUrl?: string): Promise<{ authorization_url: string; state: string; provider: string; expires_in: number }>;
  listSSOProviders(): Promise<{
    providers: Array<{ type: string; name: string; enabled: boolean }>;
    sso_enabled: boolean;
  }>;
}

/**
 * Auth API namespace.
 *
 * Provides methods for authentication:
 * - User registration and login
 * - Token management
 * - Password reset
 * - MFA setup and verification
 * - OAuth integration
 * - Session and API key management
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Register a new user
 * const { user, token } = await client.auth.register({
 *   email: 'user@example.com',
 *   password: 'secure123',
 *   name: 'John Doe',
 * });
 *
 * // Login
 * const token = await client.auth.login({
 *   email: 'user@example.com',
 *   password: 'secure123',
 * });
 *
 * // Get current user
 * const user = await client.auth.me();
 *
 * // Setup MFA
 * const mfa = await client.auth.setupMFA({ method: 'totp' });
 * ```
 */
export class AuthAPI {
  constructor(private client: AuthClientInterface) {}

  /**
   * Register a new user.
   */
  async register(body: RegisterRequest): Promise<RegisterResponse> {
    return this.client.registerUser(body);
  }

  /**
   * Login with email and password.
   */
  async login(body: LoginRequest): Promise<AuthToken> {
    return this.client.login(body);
  }

  /**
   * Refresh an access token.
   */
  async refresh(body: RefreshRequest): Promise<AuthToken> {
    return this.client.refreshToken(body);
  }

  /**
   * Logout the current session.
   */
  async logout(): Promise<void> {
    return this.client.logout();
  }

  /**
   * Logout all sessions.
   */
  async logoutAll(): Promise<{ logged_out: boolean; sessions_revoked: number }> {
    return this.client.logoutAll();
  }

  /**
   * Verify email with token.
   */
  async verifyEmail(body: VerifyEmailRequest): Promise<VerifyResponse> {
    return this.client.verifyEmail(body);
  }

  /**
   * Resend verification email.
   */
  async resendVerification(email: string): Promise<{ sent: boolean }> {
    return this.client.resendVerification(email);
  }

  /**
   * Get the current authenticated user.
   */
  async me(): Promise<User> {
    return this.client.getCurrentUser();
  }

  /**
   * Update user profile.
   */
  async updateProfile(body: UpdateProfileRequest): Promise<UpdateProfileResponse> {
    return this.client.updateProfile(body);
  }

  /**
   * Change password.
   */
  async changePassword(body: ChangePasswordRequest): Promise<void> {
    return this.client.changePassword(body);
  }

  /**
   * Request a password reset.
   */
  async requestPasswordReset(body: ForgotPasswordRequest): Promise<void> {
    return this.client.requestPasswordReset(body);
  }

  /**
   * Reset password with token.
   */
  async resetPassword(body: ResetPasswordRequest): Promise<void> {
    return this.client.resetPassword(body);
  }

  /**
   * Get OAuth authorization URL.
   */
  async getOAuthUrl(params: OAuthUrlParams): Promise<OAuthUrl> {
    return this.client.getOAuthUrl(params);
  }

  /**
   * Complete OAuth authentication.
   */
  async completeOAuth(body: OAuthCallbackRequest): Promise<AuthToken> {
    return this.client.completeOAuth(body);
  }

  /**
   * Setup MFA.
   */
  async setupMFA(body: MFASetupRequest): Promise<MFASetupResponse> {
    return this.client.setupMFA(body);
  }

  /**
   * Verify MFA setup.
   */
  async verifyMFASetup(body: MFAVerifyRequest): Promise<MFAVerifyResponse> {
    return this.client.verifyMFASetup(body);
  }

  /**
   * Enable MFA with verification code.
   */
  async enableMFA(code: string): Promise<{ enabled: boolean }> {
    return this.client.enableMFA(code);
  }

  /**
   * Disable MFA.
   */
  async disableMFA(): Promise<void> {
    return this.client.disableMFA();
  }

  /**
   * Generate backup codes for MFA.
   */
  async generateBackupCodes(): Promise<{ codes: string[] }> {
    return this.client.generateBackupCodes();
  }

  /**
   * List active sessions.
   */
  async listSessions(): Promise<{ sessions: SessionInfo[] }> {
    return this.client.listSessions();
  }

  /**
   * Revoke a session.
   */
  async revokeSession(sessionId: string): Promise<{ revoked: boolean }> {
    return this.client.revokeSession(sessionId);
  }

  /**
   * List API keys.
   */
  async listApiKeys(): Promise<{ keys: ApiKeyInfo[] }> {
    return this.client.listApiKeys();
  }

  /**
   * Create a new API key.
   */
  async createApiKey(name: string, expiresIn?: number): Promise<{ id: string; key: string; prefix: string }> {
    return this.client.createApiKey(name, expiresIn);
  }

  /**
   * Revoke an API key.
   */
  async revokeApiKey(keyId: string): Promise<{ revoked: boolean }> {
    return this.client.revokeApiKey(keyId);
  }

  /**
   * List available OAuth providers.
   */
  async listOAuthProviders(): Promise<{ providers: OAuthProviderInfo[] }> {
    return this.client.listOAuthProviders();
  }

  /**
   * Link an OAuth provider to the account.
   */
  async linkOAuthProvider(provider: string, code: string): Promise<{ linked: boolean }> {
    return this.client.linkOAuthProvider(provider, code);
  }

  /**
   * Unlink an OAuth provider from the account.
   */
  async unlinkOAuthProvider(provider: string): Promise<{ unlinked: boolean }> {
    return this.client.unlinkOAuthProvider(provider);
  }

  /**
   * Initiate SSO login.
   */
  async initiateSSOLogin(provider?: string, redirectUrl?: string): Promise<{ authorization_url: string; state: string; provider: string; expires_in: number }> {
    return this.client.initiateSSOLogin(provider, redirectUrl);
  }

  /**
   * List available SSO providers.
   */
  async listSSOProviders(): Promise<{
    providers: Array<{ type: string; name: string; enabled: boolean }>;
    sso_enabled: boolean;
  }> {
    return this.client.listSSOProviders();
  }
}
