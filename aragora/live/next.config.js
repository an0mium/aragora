/** @type {import('next').NextConfig} */
const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
});

const requestedOutput = process.env.NEXT_OUTPUT || process.env.ARAGORA_NEXT_OUTPUT;
const isExport = requestedOutput === 'export';

// Embed build SHA at build time (set by CI/CD, falls back to git)
const { execSync } = require('child_process');
const buildSha = process.env.NEXT_PUBLIC_BUILD_SHA
  || (() => { try { return execSync('git rev-parse HEAD', { encoding: 'utf-8' }).trim(); } catch { return 'unknown'; } })();
const buildTime = process.env.NEXT_PUBLIC_BUILD_TIME || new Date().toISOString();

const nextConfig = {
  // Use 'standalone' for Docker, 'export' for static hosting
  output: requestedOutput || 'standalone',
  // Ensure dev and build use separate output dirs to prevent stale-artifact
  // race conditions (ENOENT on _ssgManifest.js / pages-manifest.json).
  isolatedDevBuild: true,
  // Force-clean the .next build output before each build so no stale build-ID
  // artifacts can conflict with the new compilation.
  cleanDistDir: true,
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  env: {
    NEXT_PUBLIC_BUILD_SHA: buildSha,
    NEXT_PUBLIC_BUILD_TIME: buildTime,
  },
  // redirects and rewrites are not supported with output: 'export'.
  // When exporting statically, these are handled by the hosting platform
  // (e.g. Cloudflare Pages _redirects file, Vercel vercel.json, etc.)
  ...(isExport
    ? {}
    : {
        async redirects() {
          return [
            {
              source: '/docs',
              destination: 'https://docs.aragora.ai',
              permanent: false,
            },
            {
              source: '/docs/:path*',
              destination: 'https://docs.aragora.ai/docs/:path*',
              permanent: false,
            },
          ];
        },
        async rewrites() {
          const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://api.aragora.ai';
          return [
            {
              source: '/api/:path*',
              destination: `${apiUrl}/api/:path*`,
            },
          ];
        },
      }),
}

module.exports = withBundleAnalyzer(nextConfig)
