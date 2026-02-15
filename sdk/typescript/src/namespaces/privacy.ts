/**
 * Privacy Namespace API
 *
 * Provides GDPR/CCPA compliant privacy endpoints for data export,
 * data inventory, account deletion, and privacy preferences.
 */

/**
 * Data category in the inventory.
 */
export interface DataCategory {
  name: string;
  examples: string[];
  purpose: string;
  retention: string;
}

/**
 * Third party sharing information.
 */
export interface ThirdPartySharing {
  recipients: string[];
  data_shared: string;
  purpose: string;
}

/**
 * Data inventory response.
 */
export interface DataInventory {
  categories: DataCategory[];
  third_party_sharing: {
    llm_providers: ThirdPartySharing;
    analytics: ThirdPartySharing;
  };
  data_sold: boolean;
  opt_out_available: boolean;
}

/**
 * Privacy preferences.
 */
export interface PrivacyPreferences {
  do_not_sell: boolean;
  marketing_opt_out: boolean;
  analytics_opt_out: boolean;
  third_party_sharing: boolean;
}

/**
 * Export metadata.
 */
export interface ExportMetadata {
  exported_at: string;
  format: string;
  data_controller: string;
  contact: string;
  legal_basis: string;
}

/**
 * User data export response.
 */
export interface DataExport {
  profile: {
    id: string;
    email: string;
    name: string;
    role: string;
    is_active: boolean;
    email_verified: boolean;
    created_at: string | null;
    updated_at: string | null;
    last_login_at: string | null;
    mfa_enabled: boolean;
  };
  api_key?: {
    prefix: string;
    created_at: string | null;
    expires_at: string | null;
  };
  organization?: {
    id: string;
    name: string;
    slug: string;
    tier: string;
    role: string;
    joined_at: string | null;
  };
  oauth_providers?: Array<{
    provider: string;
    linked_at: string;
  }>;
  preferences?: Record<string, unknown>;
  audit_log?: Array<{
    timestamp: string;
    action: string;
    resource_type: string;
    resource_id?: string;
  }>;
  usage_summary?: Record<string, unknown>;
  consent_records?: Record<string, unknown>;
  _export_metadata: ExportMetadata;
}

/**
 * Account deletion response.
 */
export interface AccountDeletionResponse {
  message: string;
  deletion_id: string;
  data_deleted: string[];
  retention_note: string;
}

/**
 * Account deletion request.
 */
export interface AccountDeletionRequest {
  password: string;
  confirm: boolean;
  reason?: string;
}

/**
 * Client interface for making HTTP requests.
 */
interface PrivacyClientInterface {
  request<T = unknown>(
    method: string,
    path: string,
    options?: { params?: Record<string, unknown>; json?: Record<string, unknown> }
  ): Promise<T>;
}

/**
 * Privacy API for GDPR/CCPA compliance.
 */
export class PrivacyAPI {
  constructor(private client: PrivacyClientInterface) {}

  /**
   * Export all user data (GDPR Article 15, CCPA Right to Know).
   *
   * @param format - Export format (json or csv)
   */
  async exportData(format: 'json' | 'csv' = 'json'): Promise<DataExport> {
    return this.client.request('GET', '/api/v1/privacy/export', {
      params: { format },
    });
  }

  /**
   * Get inventory of data categories collected.
   */
  async getDataInventory(): Promise<DataInventory> {
    return this.client.request('GET', '/api/v1/privacy/data-inventory');
  }

  /**
   * Get current privacy preferences.
   */
  async getPreferences(): Promise<PrivacyPreferences> {
    return this.client.request('GET', '/api/v1/privacy/preferences');
  }

  /**
   * Update privacy preferences (CCPA Do Not Sell).
   *
   * @param preferences - Privacy preferences to update
   */
  async updatePreferences(
    preferences: Partial<PrivacyPreferences>
  ): Promise<{ message: string; preferences: PrivacyPreferences }> {
    return this.client.request('POST', '/api/v1/privacy/preferences', {
      json: preferences as Record<string, unknown>,
    });
  }

  /**
   * Delete user account (GDPR Article 17, CCPA Right to Delete).
   *
   * @param request - Deletion request with password confirmation
   */
  async deleteAccount(
    request: AccountDeletionRequest
  ): Promise<AccountDeletionResponse> {
    return this.client.request('DELETE', '/api/v1/privacy/account', {
      json: request as unknown as Record<string, unknown>,
    });
  }
}
