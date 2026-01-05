/** @type {import('next').NextConfig} */
const nextConfig = {
  // Use 'standalone' for Docker, 'export' for static hosting
  output: process.env.DOCKER_BUILD ? 'standalone' : 'export',
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
}

module.exports = nextConfig
