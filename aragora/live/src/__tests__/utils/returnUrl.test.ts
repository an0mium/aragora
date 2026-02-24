import { DEFAULT_RETURN_URL, normalizeReturnUrl } from '@/utils/returnUrl';

describe('normalizeReturnUrl', () => {
  it('returns fallback for empty values', () => {
    expect(normalizeReturnUrl(null)).toBe(DEFAULT_RETURN_URL);
    expect(normalizeReturnUrl(undefined)).toBe(DEFAULT_RETURN_URL);
    expect(normalizeReturnUrl('')).toBe(DEFAULT_RETURN_URL);
  });

  it('accepts local relative paths', () => {
    expect(normalizeReturnUrl('/debates/abc')).toBe('/debates/abc');
    expect(normalizeReturnUrl('/debates/abc?tab=timeline')).toBe('/debates/abc?tab=timeline');
  });

  it('rejects external or malformed URLs', () => {
    expect(normalizeReturnUrl('https://evil.example.com')).toBe(DEFAULT_RETURN_URL);
    expect(normalizeReturnUrl('//evil.example.com')).toBe(DEFAULT_RETURN_URL);
    expect(normalizeReturnUrl('debates/abc')).toBe(DEFAULT_RETURN_URL);
  });

  it('rejects auth-loop destinations', () => {
    expect(normalizeReturnUrl('/auth/login')).toBe(DEFAULT_RETURN_URL);
    expect(normalizeReturnUrl('/auth/callback')).toBe(DEFAULT_RETURN_URL);
    expect(normalizeReturnUrl('/login?returnUrl=/debates/abc')).toBe(DEFAULT_RETURN_URL);
    expect(normalizeReturnUrl('/signup')).toBe(DEFAULT_RETURN_URL);
  });
});
