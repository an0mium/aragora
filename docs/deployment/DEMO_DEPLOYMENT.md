# Demo Page Deployment

The interactive demo page at `/demo` showcases Aragora's multi-agent debate process
with simulated agents, proposals, critiques, votes, and decision receipts.

## Demo Page Details

- **Source:** `aragora/live/src/app/(app)/demo/page.tsx`
- **Route:** `/demo/`
- **Backend required:** No -- the demo uses purely client-side mock data
- **Build type:** Statically pre-rendered (`'use client'` component with no server dependencies)

## Deployment Methods

### Method 1: Cloudflare Pages (Recommended)

The demo is deployed as part of the full Aragora frontend via Cloudflare Pages.

**Automated deployment** triggers on push to `main` when files in `aragora/live/` change:

```bash
# Triggered automatically by:
# .github/workflows/deploy-frontend.yml
# .github/workflows/deploy-secure.yml (deploy-cloudflare job)
```

**Manual deployment** via GitHub Actions:

1. Go to **Actions** > **Deploy Frontend** > **Run workflow**
2. Select target: `cloudflare`
3. The workflow will build, verify the demo page exists, and deploy

**Required GitHub Secrets:**
- `CLOUDFLARE_API_TOKEN` -- Cloudflare API token with Pages edit permissions
- `CLOUDFLARE_ACCOUNT_ID` -- Cloudflare account ID

**Access URL:**
- `https://aragora.pages.dev/demo/` (Cloudflare Pages default domain)
- `https://aragora.ai/demo/` (if custom domain is configured in Cloudflare)

### Method 2: Docker

Build and run the frontend container:

```bash
cd aragora/live
docker build \
  --build-arg NEXT_PUBLIC_API_URL=https://api.aragora.ai \
  --build-arg NEXT_PUBLIC_WS_URL=wss://api.aragora.ai \
  -t aragora-frontend .

docker run -p 3000:3000 aragora-frontend
# Demo available at http://localhost:3000/demo/
```

Or deploy via GitHub Actions:

1. Go to **Actions** > **Deploy Frontend** > **Run workflow**
2. Select target: `docker`
3. The image is pushed to `ghcr.io/<org>/aragora-frontend:latest`

### Method 3: Local Development

```bash
cd aragora/live
npm install
npm run dev
# Demo available at http://localhost:3000/demo/
```

### Method 4: Standalone Node.js Server

```bash
cd aragora/live
npm install
npm run build    # Uses NEXT_OUTPUT=standalone by default
node .next/standalone/server.js
# Demo available at http://localhost:3000/demo/
```

## Build Verification

The `deploy-frontend.yml` workflow includes a verification step that confirms the
demo page HTML is present in the build output before deploying. If the demo page is
missing, the workflow fails with a clear error message.

To verify locally:

```bash
cd aragora/live
npm run build
# Check the demo page exists:
ls -la .next/server/app/demo.html
```

## Architecture Notes

### Static Export Limitations

The full Aragora frontend cannot be exported as a purely static site (`output: 'export'`)
because it contains dynamic routes (e.g., `/debates/[id]`, `/spectate/[debateId]`) that
require server-side rendering. The `next.config.js` conditionally excludes `redirects`
and `rewrites` when `NEXT_OUTPUT=export` is set to avoid build errors, but the dynamic
routes still prevent a full static export.

The demo page itself is a `'use client'` component with no server dependencies. It is
statically pre-rendered during the standard `standalone` build and works without any
backend API connection.

### Deployment Pipeline

```
Push to main (aragora/live/** changes)
    |
    v
deploy-frontend.yml (or deploy-secure.yml)
    |
    +-- build job: npm ci + npm run build (standalone)
    |       |
    |       +-- Verifies /demo page exists in build output
    |       +-- Uploads standalone artifact
    |
    +-- deploy-cloudflare job: wrangler pages deploy
    |       |
    |       +-- Deploys to aragora.pages.dev
    |
    +-- deploy-docker job (manual): docker build + push to GHCR
```

## Troubleshooting

**Build fails with "redirects/rewrites not supported with export":**
The `next.config.js` conditionally excludes these when `NEXT_OUTPUT=export`. If you
see this warning, ensure the build is using `standalone` mode (the default).

**Demo page not found in build output:**
The demo page at `aragora/live/src/app/(app)/demo/page.tsx` must be a valid React
component. Check for TypeScript errors by running `npx tsc --noEmit` in `aragora/live/`.

**Cloudflare deployment fails with auth errors:**
Verify the `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` secrets are configured
in GitHub repository settings. The API token needs the "Cloudflare Pages: Edit"
permission.
