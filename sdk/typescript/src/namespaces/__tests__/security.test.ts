import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

import { SecurityAPI } from '../security';

interface MockClient {
  request: Mock;
}

describe('SecurityAPI Namespace', () => {
  let api: SecurityAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new SecurityAPI(mockClient as any);
  });

  it('returns the live security status payload shape', async () => {
    mockClient.request.mockResolvedValue({
      crypto_available: true,
      active_key_id: 'key_active',
      key_version: 7,
      key_age_days: 61,
      rotation_recommended: true,
      rotation_required: false,
      total_keys: 2,
    });

    const result = await api.getStatus();

    expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/admin/security/status');
    expect(result.active_key_id).toBe('key_active');
    expect(result.rotation_recommended).toBe(true);
  });

  it('returns the live security health payload shape', async () => {
    mockClient.request.mockResolvedValue({
      status: 'degraded',
      checks: {
        crypto_available: true,
        service_initialized: true,
        key_rotation_scheduler: {
          status: 'healthy',
          total_rotations: 5,
        },
      },
      issues: [],
      warnings: ['Key is 61 days old, rotation recommended'],
    });

    const result = await api.getHealthChecks();

    expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/admin/security/health');
    expect(result.status).toBe('degraded');
    expect(result.warnings).toHaveLength(1);
  });

  it('returns the live security key list payload shape', async () => {
    mockClient.request.mockResolvedValue({
      keys: [
        {
          key_id: 'key_active',
          version: 7,
          is_active: true,
          created_at: '2024-01-01T00:00:00Z',
          age_days: 61,
        },
      ],
      active_key_id: 'key_active',
      total_keys: 1,
    });

    const result = await api.listKeys();

    expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/admin/security/keys');
    expect(result.total_keys).toBe(1);
    expect(result.keys[0]?.is_active).toBe(true);
  });

  it('posts rotation requests using the live handler body shape', async () => {
    mockClient.request.mockResolvedValue({
      success: true,
      dry_run: true,
      old_key_version: 6,
      new_key_version: 7,
      stores_processed: ['receipts'],
      records_reencrypted: 0,
      failed_records: 0,
      duration_seconds: 0.1,
      errors: [],
    });

    const result = await api.rotateKey({
      dry_run: true,
      stores: ['receipts'],
      force: false,
    });

    expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/admin/security/rotate-key', {
      body: {
        dry_run: true,
        stores: ['receipts'],
        force: false,
      },
    });
    expect(result.dry_run).toBe(true);
  });
});
