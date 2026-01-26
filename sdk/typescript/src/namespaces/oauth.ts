/**
 * OAuth Namespace API
 *
 * Provides a namespaced interface for OAuth authentication flows.
 * Supports Google, GitHub, Microsoft, Apple, and generic OIDC providers.
 */

/**
 * OAuth provider types.
 */
export type OAuthProvider = 'google' | 'github' | 'microsoft' | 'apple' | 'oidc';

/**
 * OAuth provider configuration.
 */
export interface OAuthProviderConfig {
  provider: OAuthProvider;
  enabled: boolean;
  client_id?: string;
  name: string;
  icon_url?: string;
}

/**
 * Response from OAuth callback.
 */
export interface OAuthCallbackResponse {
  success: boolean;
  access_token?: string;
  refresh_token?: string;
  user_id?: string;
  email?: string;
  name?: string;
  error?: string;
  error_description?: string;
}

/**
 * Linked OAuth account.
 */
export interface LinkedOAuthAccount {
  provider: OAuthProvider;
  provider_user_id: string;
  email: string;
  name?: string;
  linked_at: string;
}

/**
 * Response for listing available providers.
 */
export interface ProvidersResponse {
  providers: OAuthProviderConfig[];
}

/**
 * Response for user's linked providers.
 */
export interface LinkedProvidersResponse {
  providers: LinkedOAuthAccount[];
}

/**
 * Request to link an OAuth account.
 */
export interface LinkAccountRequest {
  provider: OAuthProvider;
  code: string;
  redirect_uri: string;
}

/**
 * Response from linking an account.
 */
export interface LinkAccountResponse {
  success: boolean;
  provider: OAuthProvider;
  email?: string;
  message?: string;
}

/**
 * Request to unlink an OAuth account.
 */
export interface UnlinkAccountRequest {
  provider: OAuthProvider;
}

/**
 * Response from unlinking an account.
 */
export interface UnlinkAccountResponse {
  success: boolean;
  message: string;
}

/**
 * OAuth authorization URL response.
 */
export interface AuthorizationUrlResponse {
  authorization_url: string;
  state: string;
}

/**
 * Interface for the internal client methods used by OAuthAPI.
 */
interface OAuthClientInterface {
  request<T>(method: string, path: string, options?: { params?: Record<string, string | number | boolean | undefined>; body?: unknown }): Promise<T>;
}

/**
 * OAuth API namespace.
 *
 * Provides methods for OAuth authentication:
 * - Getting authorization URLs for various providers
 * - Handling OAuth callbacks
 * - Linking/unlinking OAuth accounts
 * - Listing available and linked providers
 *
 * @example
 * ```typescript
 * const client = createClient({ baseUrl: 'https://api.aragora.ai' });
 *
 * // Get available providers
 * const { providers } = await client.oauth.getProviders();
 *
 * // Get authorization URL for Google
 * const { authorization_url } = await client.oauth.getAuthUrl('google', {
 *   redirect_uri: 'https://myapp.com/callback'
 * });
 *
 * // Link a new OAuth account
 * await client.oauth.linkAccount({
 *   provider: 'github',
 *   code: 'authorization_code',
 *   redirect_uri: 'https://myapp.com/callback'
 * });
 * ```
 */
export class OAuthAPI {
  constructor(private client: OAuthClientInterface) {}

  /**
   * Get list of available OAuth providers.
   */
  async getProviders(): Promise<ProvidersResponse> {
    return this.client.request<ProvidersResponse>('GET', '/api/v1/auth/oauth/providers');
  }

  /**
   * Get user's linked OAuth providers.
   */
  async getLinkedProviders(): Promise<LinkedProvidersResponse> {
    return this.client.request<LinkedProvidersResponse>('GET', '/api/v1/user/oauth-providers');
  }

  /**
   * Get authorization URL for Google OAuth.
   * Redirect user to this URL to start the OAuth flow.
   */
  async getGoogleAuthUrl(params?: { redirect_uri?: string; state?: string }): Promise<AuthorizationUrlResponse> {
    return this.client.request<AuthorizationUrlResponse>('GET', '/api/v1/auth/oauth/google', {
      params: {
        redirect_uri: params?.redirect_uri,
        state: params?.state,
      },
    });
  }

  /**
   * Get authorization URL for GitHub OAuth.
   */
  async getGitHubAuthUrl(params?: { redirect_uri?: string; state?: string }): Promise<AuthorizationUrlResponse> {
    return this.client.request<AuthorizationUrlResponse>('GET', '/api/v1/auth/oauth/github', {
      params: {
        redirect_uri: params?.redirect_uri,
        state: params?.state,
      },
    });
  }

  /**
   * Get authorization URL for Microsoft OAuth.
   */
  async getMicrosoftAuthUrl(params?: { redirect_uri?: string; state?: string }): Promise<AuthorizationUrlResponse> {
    return this.client.request<AuthorizationUrlResponse>('GET', '/api/v1/auth/oauth/microsoft', {
      params: {
        redirect_uri: params?.redirect_uri,
        state: params?.state,
      },
    });
  }

  /**
   * Get authorization URL for Apple OAuth.
   */
  async getAppleAuthUrl(params?: { redirect_uri?: string; state?: string }): Promise<AuthorizationUrlResponse> {
    return this.client.request<AuthorizationUrlResponse>('GET', '/api/v1/auth/oauth/apple', {
      params: {
        redirect_uri: params?.redirect_uri,
        state: params?.state,
      },
    });
  }

  /**
   * Get authorization URL for generic OIDC provider.
   */
  async getOIDCAuthUrl(params?: { redirect_uri?: string; state?: string; provider_id?: string }): Promise<AuthorizationUrlResponse> {
    return this.client.request<AuthorizationUrlResponse>('GET', '/api/v1/auth/oauth/oidc', {
      params: {
        redirect_uri: params?.redirect_uri,
        state: params?.state,
        provider_id: params?.provider_id,
      },
    });
  }

  /**
   * Link an OAuth account to the current user.
   * Use after receiving authorization code from OAuth provider.
   */
  async linkAccount(request: LinkAccountRequest): Promise<LinkAccountResponse> {
    return this.client.request<LinkAccountResponse>('POST', '/api/v1/auth/oauth/link', {
      body: request,
    });
  }

  /**
   * Unlink an OAuth provider from the current user.
   */
  async unlinkAccount(request: UnlinkAccountRequest): Promise<UnlinkAccountResponse> {
    return this.client.request<UnlinkAccountResponse>('DELETE', '/api/v1/auth/oauth/unlink', {
      body: request,
    });
  }

  /**
   * Handle OAuth callback (exchange code for tokens).
   * This is typically called by your server, not the client SDK.
   */
  async handleCallback(
    provider: OAuthProvider,
    params: { code: string; state?: string; redirect_uri?: string }
  ): Promise<OAuthCallbackResponse> {
    const path = provider === 'oidc'
      ? '/api/v1/auth/oauth/oidc/callback'
      : `/api/v1/auth/oauth/${provider}/callback`;

    return this.client.request<OAuthCallbackResponse>('GET', path, {
      params: {
        code: params.code,
        state: params.state,
        redirect_uri: params.redirect_uri,
      },
    });
  }
}
