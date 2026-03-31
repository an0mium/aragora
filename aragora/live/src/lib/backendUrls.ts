export function joinBackendPath(base: string | undefined, path: string): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  const normalizedBase = base?.trim().replace(/\/$/, '') ?? '';
  return normalizedBase ? `${normalizedBase}${normalizedPath}` : normalizedPath;
}
