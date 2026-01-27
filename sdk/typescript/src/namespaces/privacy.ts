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
 * Privacy API for GDPR/CCPA compliance.
 */
export class PrivacyAPI {
  private baseUrl: string;
  private headers: HeadersInit;

  constructor(baseUrl: string, apiKey: string) {
    this.baseUrl = baseUrl;
    this.headers = {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    };
  }

  /**
   * Export all user data (GDPR Article 15, CCPA Right to Know).
   *
   * @param format - Export format (json or csv)
   */
  async exportData(format: 'json' | 'csv' = 'json'): Promise<DataExport> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/privacy/export?format=${format}`,
      {
        method: 'GET',
        headers: this.headers,
      }
    );
    if (!response.ok) throw new Error(`Export failed: ${response.statusText}`);
    return response.json();
  }

  /**
   * Get inventory of data categories collected.
   */
  async getDataInventory(): Promise<DataInventory> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/privacy/data-inventory`,
      {
        method: 'GET',
        headers: this.headers,
      }
    );
    if (!response.ok) throw new Error(`Failed to get data inventory: ${response.statusText}`);
    return response.json();
  }

  /**
   * Get current privacy preferences.
   */
  async getPreferences(): Promise<PrivacyPreferences> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/privacy/preferences`,
      {
        method: 'GET',
        headers: this.headers,
      }
    );
    if (!response.ok) throw new Error(`Failed to get preferences: ${response.statusText}`);
    return response.json();
  }

  /**
   * Update privacy preferences (CCPA Do Not Sell).
   *
   * @param preferences - Privacy preferences to update
   */
  async updatePreferences(
    preferences: Partial<PrivacyPreferences>
  ): Promise<{ message: string; preferences: PrivacyPreferences }> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/privacy/preferences`,
      {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(preferences),
      }
    );
    if (!response.ok) throw new Error(`Failed to update preferences: ${response.statusText}`);
    return response.json();
  }

  /**
   * Delete user account (GDPR Article 17, CCPA Right to Delete).
   *
   * @param request - Deletion request with password confirmation
   */
  async deleteAccount(
    request: AccountDeletionRequest
  ): Promise<AccountDeletionResponse> {
    const response = await fetch(
      `${this.baseUrl}/api/v1/privacy/account`,
      {
        method: 'DELETE',
        headers: this.headers,
        body: JSON.stringify(request),
      }
    );
    if (!response.ok) throw new Error(`Account deletion failed: ${response.statusText}`);
    return response.json();
  }
}
