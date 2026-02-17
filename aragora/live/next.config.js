/** @type {import('next').NextConfig} */
const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
});

const requestedOutput = process.env.NEXT_OUTPUT || process.env.ARAGORA_NEXT_OUTPUT;
const isExport = requestedOutput === 'export';

const nextConfig = {
  // Use 'standalone' for Docker, 'export' for static hosting
  output: requestedOutput || 'standalone',
  trailingSlash: true,
  images: {
    unoptimized: true,
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
