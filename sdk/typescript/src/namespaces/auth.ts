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
  setupOrganization(request: { name: string; slug?: string }): Promise<{ organization: import('../types').Tenant }>;
  inviteTeamMember(request: { email: string; role?: string }): Promise<{ invite_token: string; invite_url: string; expires_in: number }>;
  checkInvite(token: string): Promise<{ valid: boolean; email: string; organization_id: string; role: string; expires_at: number }>;
  acceptInvite(token: string): Promise<{ organization_id: string; role: string }>;
  listPendingInvites(): Promise<{ invites: Array<{ id: string; email: string; role: string; expires_at: string; created_at: string }> }>;
  revokeInvite(inviteId: string): Promise<{ revoked: boolean }>;
  getAuthHealth(): Promise<{ status: string; services: Record<string, string> }>;
  getProfile(): Promise<import('../types').User>;
  mfa(request: { action: string; code?: string; method?: string }): Promise<{ status: string; secret?: string; qr_code_uri?: string; backup_codes?: string[] }>;
  getOAuthAuthorizeUrl(params: { provider: string; redirect_uri?: string; state?: string }): Promise<{ authorization_url: string }>;
  getOAuthDiagnostics(): Promise<{ providers: Record<string, unknown>; status: Record<string, string> }>;
  getOAuthCallback(params: { code: string; state?: string }): Promise<{ access_token: string; user: import('../types').User }>;
  forgotPassword(email: string): Promise<{ sent: boolean }>;
  resetPasswordAlt(request: { token: string; new_password: string }): Promise<{ reset: boolean }>;
  resendVerificationAlt(email?: string): Promise<{ sent: boolean }>;
  checkInviteAlt(token: string): Promise<{ valid: boolean; email: string; organization_id: string; role: string; expires_at: number }>;
  acceptInviteAlt(token: string): Promise<{ organization_id: string; role: string }>;
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

  /**
   * Setup a new organization after registration.
   */
  async setupOrganization(request: { name: string; slug?: string }): Promise<{ organization: import('../types').Tenant }> {
    return this.client.setupOrganization(request);
  }

  /**
   * Invite a team member to the organization.
   */
  async inviteTeamMember(request: { email: string; role?: string }): Promise<{ invite_token: string; invite_url: string; expires_in: number }> {
    return this.client.inviteTeamMember(request);
  }

  /**
   * Check if an invite token is valid.
   */
  async checkInvite(token: string): Promise<{ valid: boolean; email: string; organization_id: string; role: string; expires_at: number }> {
    return this.client.checkInvite(token);
  }

  /**
   * Accept an organization invite.
   */
  async acceptInvite(token: string): Promise<{ organization_id: string; role: string }> {
    return this.client.acceptInvite(token);
  }

  /**
   * List pending invitations for the organization.
   */
  async listPendingInvites(): Promise<{ invites: Array<{ id: string; email: string; role: string; expires_at: string; created_at: string }> }> {
    return this.client.listPendingInvites();
  }

  /**
   * Revoke a pending invitation.
   */
  async revokeInvite(inviteId: string): Promise<{ revoked: boolean }> {
    return this.client.revokeInvite(inviteId);
  }

  /**
   * Check authentication service health.
   */
  async health(): Promise<{ status: string; services: Record<string, string> }> {
    return this.client.getAuthHealth();
  }

  /**
   * Get the authenticated user's profile.
   *
   * Alternative to me(), uses /api/auth/profile endpoint.
   */
  async getProfile(): Promise<import('../types').User> {
    return this.client.getProfile();
  }

  /**
   * Combined MFA setup and verification endpoint.
   *
   * @param request - MFA request with action, optional code, and optional method
   */
  async mfa(request: { action: string; code?: string; method?: string }): Promise<{
    status: string;
    secret?: string;
    qr_code_uri?: string;
    backup_codes?: string[];
  }> {
    return this.client.mfa(request);
  }

  /**
   * Get OAuth authorization URL via the authorize endpoint.
   *
   * @param params - Provider, optional redirect URI, and optional state
   */
  async getOAuthAuthorizeUrl(params: {
    provider: string;
    redirect_uri?: string;
    state?: string;
  }): Promise<{ authorization_url: string }> {
    return this.client.getOAuthAuthorizeUrl(params);
  }

  /**
   * Get OAuth configuration diagnostics.
   *
   * Useful for troubleshooting authentication issues.
   */
  async getOAuthDiagnostics(): Promise<{
    providers: Record<string, unknown>;
    status: Record<string, string>;
  }> {
    return this.client.getOAuthDiagnostics();
  }

  /**
   * Handle OAuth callback with authorization code.
   *
   * @param params - Authorization code and optional state
   */
  async getOAuthCallback(params: {
    code: string;
    state?: string;
  }): Promise<{ access_token: string; user: import('../types').User }> {
    return this.client.getOAuthCallback(params);
  }

  /**
   * Request a password reset via the forgot-password endpoint.
   *
   * Alternative to requestPasswordReset(), uses /api/auth/forgot-password.
   */
  async forgotPassword(email: string): Promise<{ sent: boolean }> {
    return this.client.forgotPassword(email);
  }

  /**
   * Reset password via the reset-password endpoint.
   *
   * Alternative to resetPassword(), uses /api/auth/reset-password.
   */
  async resetPasswordAlt(request: {
    token: string;
    new_password: string;
  }): Promise<{ reset: boolean }> {
    return this.client.resetPasswordAlt(request);
  }

  /**
   * Resend email verification via the resend-verification endpoint.
   *
   * Alternative to resendVerification(), uses /api/auth/resend-verification.
   */
  async resendVerificationAlt(email?: string): Promise<{ sent: boolean }> {
    return this.client.resendVerificationAlt(email);
  }

  /**
   * Check invitation validity via the check-invite endpoint.
   *
   * Alternative to checkInvite(), uses /api/auth/check-invite.
   */
  async checkInviteAlt(token: string): Promise<{
    valid: boolean;
    email: string;
    organization_id: string;
    role: string;
    expires_at: number;
  }> {
    return this.client.checkInviteAlt(token);
  }

  /**
   * Accept a team invitation via the accept-invite endpoint.
   *
   * Alternative to acceptInvite(), uses /api/auth/accept-invite.
   */
  async acceptInviteAlt(token: string): Promise<{
    organization_id: string;
    role: string;
  }> {
    return this.client.acceptInviteAlt(token);
  }
}

// =============================================================================
// Route-annotated API for SDK parity detection
// =============================================================================

/**
 * Interface for a raw HTTP request client.
 */
interface AuthRawClientInterface {
  request(method: string, path: string, options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }): Promise<unknown>;
}

/**
 * Auth routes namespace with direct request calls for SDK parity.
 *
 * These methods provide direct route coverage for all auth handler endpoints.
 * They supplement the AuthAPI class which delegates to typed client methods.
 */
export class AuthRoutesAPI {
  constructor(private client: AuthRawClientInterface) {}

  /**
   * Change user password.
   * @route POST /api/auth/password/change
   */
  async changePassword(body: { current_password: string; new_password: string }): Promise<unknown> {
    return this.client.request('POST', '/api/auth/password/change', { json: body });
  }

  /**
   * Request a password reset email.
   * @route POST /api/auth/password/forgot
   */
  async forgotPassword(body: { email: string }): Promise<unknown> {
    return this.client.request('POST', '/api/auth/password/forgot', { json: body });
  }

  /**
   * Reset password with token.
   * @route POST /api/auth/password/reset
   */
  async resetPassword(body: { token: string; new_password: string }): Promise<unknown> {
    return this.client.request('POST', '/api/auth/password/reset', { json: body });
  }

  /**
   * Request a password reset via the forgot-password endpoint.
   * @route POST /api/auth/forgot-password
   */
  async forgotPasswordAlt(body: { email: string }): Promise<unknown> {
    return this.client.request('GET', '/api/auth/forgot-password', { json: body });
  }

  /**
   * Reset password via the reset-password endpoint.
   * @route POST /api/auth/reset-password
   */
  async resetPasswordAlt(body: { token: string; new_password: string }): Promise<unknown> {
    return this.client.request('GET', '/api/auth/reset-password', { json: body });
  }

  /**
   * Get the authenticated user's profile.
   * @route GET /api/auth/profile
   */
  async getProfile(): Promise<unknown> {
    return this.client.request('GET', '/api/auth/profile');
  }

  /**
   * List API keys for the current user.
   * @route GET /api/auth/api-keys
   */
  async listApiKeys(): Promise<unknown> {
    return this.client.request('GET', '/api/auth/api-keys');
  }

  /**
   * Revoke an API key by prefix.
   * @route DELETE /api/auth/api-keys/{prefix}
   */
  async revokeApiKeyByPrefix(prefix: string): Promise<unknown> {
    return this.client.request('DELETE', `/api/auth/api-keys/${prefix}`);
  }

  /**
   * Combined MFA endpoint (setup, verify, enable, disable).
   * @route POST /api/auth/mfa
   */
  async mfa(body: { action: string; code?: string; method?: string }): Promise<unknown> {
    return this.client.request('GET', '/api/auth/mfa', { json: body });
  }

  /**
   * Verify email address with token.
   * @route POST /api/auth/verify-email
   */
  async verifyEmail(body: { token: string }): Promise<unknown> {
    return this.client.request('GET', '/api/auth/verify-email', { json: body });
  }

  /**
   * Resend email verification link.
   * @route POST /api/auth/verify-email/resend
   */
  async resendVerification(): Promise<unknown> {
    return this.client.request('GET', '/api/auth/verify-email/resend');
  }

  /**
   * Resend email verification via the resend-verification endpoint.
   * @route POST /api/auth/resend-verification
   */
  async resendVerificationAlt(body?: { email?: string }): Promise<unknown> {
    return this.client.request('GET', '/api/auth/resend-verification', { json: body });
  }

  /**
   * Set up a new organization after registration.
   * @route POST /api/auth/setup-organization
   */
  async setupOrganization(body: { name: string; slug?: string }): Promise<unknown> {
    return this.client.request('GET', '/api/auth/setup-organization', { json: body });
  }

  /**
   * Invite a team member to the organization.
   * @route POST /api/auth/invite
   */
  async invite(body: { email: string; role?: string }): Promise<unknown> {
    return this.client.request('GET', '/api/auth/invite', { json: body });
  }

  /**
   * Check if an invite token is valid.
   * @route GET /api/auth/check-invite
   */
  async checkInvite(token: string): Promise<unknown> {
    return this.client.request('GET', '/api/auth/check-invite', { params: { token } });
  }

  /**
   * Accept a team invitation.
   * @route POST /api/auth/accept-invite
   */
  async acceptInvite(body: { token: string }): Promise<unknown> {
    return this.client.request('GET', '/api/auth/accept-invite', { json: body });
  }

  /**
   * Check authentication service health.
   * @route GET /api/auth/health
   */
  async health(): Promise<unknown> {
    return this.client.request('GET', '/api/auth/health');
  }

  /**
   * Get OAuth authorization URL.
   * @route GET /api/auth/oauth/url
   */
  async getOAuthUrl(params: { provider: string; redirect_uri?: string; state?: string }): Promise<unknown> {
    return this.client.request('GET', '/api/auth/oauth/url', { params });
  }

  /**
   * Get OAuth authorization URL via the authorize endpoint.
   * @route GET /api/auth/oauth/authorize
   */
  async getOAuthAuthorizeUrl(params: { provider: string; redirect_uri?: string; state?: string }): Promise<unknown> {
    return this.client.request('GET', '/api/auth/oauth/authorize', { params });
  }

  /**
   * Handle OAuth callback with authorization code.
   * @route GET /api/auth/oauth/callback
   */
  async getOAuthCallback(params: { code: string; state?: string }): Promise<unknown> {
    return this.client.request('GET', '/api/auth/oauth/callback', { params });
  }

  /**
   * Get OAuth configuration diagnostics.
   * @route GET /api/auth/oauth/diagnostics
   */
  async getOAuthDiagnostics(): Promise<unknown> {
    return this.client.request('GET', '/api/auth/oauth/diagnostics');
  }
}
