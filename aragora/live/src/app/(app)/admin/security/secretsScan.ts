const DEFAULT_CODEBASE_REPO_ID = 'default';

export function buildSecretsScanUrl(apiBase: string, scanId?: string): string {
  const base = `${apiBase}/api/v1/codebase/${DEFAULT_CODEBASE_REPO_ID}/scan/secrets`;
  return scanId ? `${base}/${scanId}` : base;
}
