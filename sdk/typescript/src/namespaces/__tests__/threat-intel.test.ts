import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import { ThreatIntelAPI } from '../threat-intel';

interface MockClient {
  request: Mock;
}

describe('ThreatIntelAPI', () => {
  let api: ThreatIntelAPI;
  let mockClient: MockClient;

  beforeEach(() => {
    mockClient = {
      request: vi.fn(),
    };
    api = new ThreatIntelAPI(mockClient as any);
  });

  it('posts URL checks to the versioned threat endpoint with a request body', async () => {
    mockClient.request.mockResolvedValue({ is_malicious: true });

    await api.checkURL({ url: 'https://example.com' });

    expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/threat/url', {
      body: { url: 'https://example.com' },
    });
  });

  it('encodes IPs in versioned threat paths', async () => {
    mockClient.request.mockResolvedValue({ is_malicious: false });

    await api.checkIP('2001:db8::1');

    expect(mockClient.request).toHaveBeenCalledWith(
      'GET',
      '/api/v1/threat/ip/2001%3Adb8%3A%3A1'
    );
  });

  it('posts hash batches to the versioned threat endpoint with a request body', async () => {
    mockClient.request.mockResolvedValue({ results: [], summary: { total: 0, malware: 0, clean: 0 } });

    await api.checkHashesBatch({ hashes: ['abc'] });

    expect(mockClient.request).toHaveBeenCalledWith('POST', '/api/v1/threat/hashes', {
      body: { hashes: ['abc'] },
    });
  });

  it('reads service status from the versioned threat status endpoint', async () => {
    const expected = {
      virustotal: { enabled: true, has_key: true, rate_limit: 4 },
      abuseipdb: { enabled: true, has_key: true, rate_limit: 1000 },
      phishtank: { enabled: true, has_key: false, rate_limit: 0 },
      caching: true,
      cache_ttl_hours: 24,
    };
    mockClient.request.mockResolvedValue(expected);

    const status = await api.getStatus();

    expect(mockClient.request).toHaveBeenCalledWith('GET', '/api/v1/threat/status');
    expect(status.cache_ttl_hours).toBe(24);
  });
});
