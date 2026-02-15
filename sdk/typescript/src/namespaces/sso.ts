/**
 * SSO (Single Sign-On) Namespace API
 *
 * Provides endpoints for enterprise SSO authentication via SAML 2.0 and OIDC.
 */

import type { AragoraClient } from '../client';

/**
 * SSO provider types
 */
export type SSOProviderType = 'saml' | 'oidc' | 'oauth2';

/**
 * SSO login response with redirect URL
 */
export interface SSOLoginResponse {
  redirect_url: string;
  state?: string;
  nonce?: string;
  provider: SSOProviderType;
}

/**
 * SSO user information from identity provider
 */
export interface SSOUser {
  id: string;
  email: string;
  name?: string;
  given_name?: string;
  family_name?: string;
  groups?: string[];
  roles?: string[];
  organization_id?: string;
  attributes?: Record<string, unknown>;
  provider: SSOProviderType;
  provider_id?: string;
}

/**
 * SSO callback result after IdP authentication
 */
export interface SSOCallbackResult {
  success: boolean;
  user?: SSOUser;
  token?: string;
  refresh_token?: string;
  expires_in?: number;
  error?: string;
  error_description?: string;
}

/**
 * SSO provider status
 */
export interface SSOStatus {
  enabled: boolean;
  provider_type?: SSOProviderType;
  idp_url?: string;
  entity_id?: string;
  configured: boolean;
  last_login?: string;
  total_logins?: number;
}

/**
 * SAML SP metadata
 */
export interface SAMLMetadata {
  entity_id: string;
  acs_url: string;
  slo_url?: string;
  certificate?: string;
  metadata_xml?: string;
}

/**
 * SSO logout response
 */
export interface SSOLogoutResponse {
  success: boolean;
  redirect_url?: string;
  message?: string;
}

/**
 * SSO namespace for Single Sign-On authentication.
 *
 * @example
 * ```typescript
 * // Check if SSO is configured
 * const status = await client.sso.getStatus();
 * if (status.enabled) {
 *   // Initiate SSO login
 *   const login = await client.sso.login({ returnUrl: '/dashboard' });
 *   window.location.href = login.redirect_url;
 * }
 *
 * // Get SAML metadata for IdP configuration
 * const metadata = await client.sso.getMetadata();
 * ```
 */
export class SSONamespace {
  constructor(private client: AragoraClient) {}

  /**
   * Get SSO configuration status.
   *
   * Returns whether SSO is enabled and configured.
   */
  async getStatus(): Promise<SSOStatus> {
    return this.client.request<SSOStatus>('GET', '/auth/sso/status');
  }

  /**
   * Initiate SSO login flow.
   *
   * Returns a redirect URL to the identity provider.
   *
   * @param options.returnUrl - URL to redirect after successful login
   * @param options.prompt - OIDC prompt parameter (e.g., 'login', 'consent')
   */
  async login(options?: { returnUrl?: string; prompt?: string }): Promise<SSOLoginResponse> {
    const params = new URLSearchParams();
    if (options?.returnUrl) params.set('return_url', options.returnUrl);
    if (options?.prompt) params.set('prompt', options.prompt);

    const query = params.toString();
    const path = query ? `/auth/sso/login?${query}` : '/auth/sso/login';
    return this.client.request<SSOLoginResponse>('GET', path);
  }

  /**
   * Handle SSO callback from identity provider.
   *
   * This is typically called by the server, not the client directly.
   * Use this to validate and exchange the authorization code.
   *
   * @param params - Callback parameters from IdP (code, state, etc.)
   */
  async callback(params: {
    code?: string;
    state?: string;
    SAMLResponse?: string;
    RelayState?: string;
  }): Promise<SSOCallbackResult> {
    return this.client.request<SSOCallbackResult>('POST', '/auth/sso/callback', { body: params });
  }

  /**
   * Initiate SSO logout.
   *
   * @param options.everywhere - Log out from all sessions (SLO)
   */
  async logout(options?: { everywhere?: boolean }): Promise<SSOLogoutResponse> {
    const body = options?.everywhere ? { everywhere: true } : undefined;
    return this.client.request<SSOLogoutResponse>('POST', '/auth/sso/logout', { body });
  }

  /**
   * Get SAML Service Provider metadata.
   *
   * Returns XML metadata for configuring your Identity Provider.
   * Only available for SAML-based SSO.
   */
  async getMetadata(): Promise<SAMLMetadata> {
    return this.client.request<SAMLMetadata>('GET', '/auth/sso/metadata');
  }

  /**
   * Get raw SAML metadata XML.
   *
   * @returns SAML SP metadata as XML string
   */
  async getMetadataXML(): Promise<string> {
    const metadata = await this.getMetadata();
    return metadata.metadata_xml || '';
  }

  /**
   * Check if SSO is enabled.
   */
  async isEnabled(): Promise<boolean> {
    const status = await this.getStatus();
    return status.enabled;
  }

  /**
   * Check if SSO is properly configured.
   */
  async isConfigured(): Promise<boolean> {
    const status = await this.getStatus();
    return status.configured;
  }
}
