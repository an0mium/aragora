/** @type {import('next').NextConfig} */
const requestedOutput = process.env.NEXT_OUTPUT || process.env.ARAGORA_NEXT_OUTPUT;

const nextConfig = {
  // Use 'standalone' for Docker, 'export' for static hosting
  output: requestedOutput || 'standalone',
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
}

module.exports = nextConfig
