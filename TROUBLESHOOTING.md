# Aragora.ai Troubleshooting Guide

## Root Cause: "Authentication Required" After Login

**What's actually happening:** You log in successfully (email shows in sidebar), but when you
submit a debate you get "Authentication required". This is a **frontend-side block**, not a
backend 403.

The auth flow:
1. Login succeeds → tokens stored in localStorage → email appears in sidebar
2. `AuthContext` calls `GET /api/auth/me` to validate the stored token on page load
3. If that call **fails** (backend down, CORS blocked, network error), the frontend sets
   `isAuthenticated = false` and clears the session
4. The sidebar still shows your email from the brief moment before validation failed
5. When you submit a debate, the frontend blocks it before ever calling the backend

**Most likely cause:** The backend at `api.aragora.ai` is not responding to `/api/auth/me`,
OR CORS is blocking the request from `aragora.ai`.

---

## Quick Diagnostic (run these in order)

### Step 1: Is the backend alive?

```bash
# From your local machine:
curl -v https://api.aragora.ai/healthz

# Expected: HTTP 200 with JSON health response
# If this fails: backend is down → go to "Fix: Backend Not Running"
```

### Step 2: Is CORS blocking requests?

```bash
# Simulate a browser preflight request from aragora.ai:
curl -v -X OPTIONS https://api.aragora.ai/api/auth/me \
  -H "Origin: https://aragora.ai" \
  -H "Access-Control-Request-Method: GET" \
  -H "Access-Control-Request-Headers: Authorization"

# Expected: HTTP 200 with these headers:
#   Access-Control-Allow-Origin: https://aragora.ai
#   Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
#   Access-Control-Allow-Headers: Content-Type, Authorization
#
# If missing or wrong origin: go to "Fix: CORS Configuration"
```

### Step 3: Does login actually work?

```bash
# Test login (replace YOUR_EMAIL and YOUR_PASSWORD):
curl -s -X POST https://api.aragora.ai/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"YOUR_EMAIL","password":"YOUR_PASSWORD"}'

# Expected: JSON with { "user": {...}, "tokens": { "access_token": "...", ... } }
# If error: go to "Fix: JWT Configuration"
```

### Step 4: Does token validation work?

```bash
# Extract token from Step 3, then:
TOKEN="paste-access-token-here"
curl -v https://api.aragora.ai/api/auth/me \
  -H "Authorization: Bearer $TOKEN"

# Expected: HTTP 200 with user data
# If 401: JWT secret mismatch or token expired
# If 403: RBAC permission issue
```

### Step 5: Does debate creation work?

```bash
TOKEN="paste-access-token-here"
curl -X POST https://api.aragora.ai/api/v1/debates \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"question":"Test debate","agents":"anthropic-api,openai-api","rounds":3}'

# Expected: HTTP 200/201 with debate data
```

---

## Fixes

### Fix: Backend Not Running

SSH into your EC2 instance:

```bash
# Check if the process is running
systemctl status aragora
# or
ps aux | grep "aragora.server"

# If not running, start it:
sudo systemctl start aragora

# Check logs for errors:
journalctl -u aragora -n 50 --no-pager

# If no systemd service exists, start manually:
cd /opt/aragora  # or wherever your code lives
source venv/bin/activate
python -m aragora.server --host 0.0.0.0 --http-port 8080 --port 8765
```

### Fix: CORS Configuration

The backend reads CORS origins from `ARAGORA_ALLOWED_ORIGINS` env var.
Production defaults include `https://aragora.ai` and `https://www.aragora.ai`.

If you're using a custom domain or the defaults aren't working:

```bash
# In your EC2 .env file:
ARAGORA_ALLOWED_ORIGINS=https://aragora.ai,https://www.aragora.ai,https://live.aragora.ai
```

**Important:** `ARAGORA_ENV` must be set for production mode:
```bash
ARAGORA_ENV=production
```

Without `ARAGORA_ENV`, the server runs in dev mode (which includes localhost origins
but may behave differently).

### Fix: JWT Configuration

The system uses `ARAGORA_JWT_SECRET` (NOT Supabase) for auth. This must be:
- Set on the backend EC2 instance
- At least 32 characters long
- The same across all backend instances

```bash
# Generate a secure secret:
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Add to your EC2 .env:
ARAGORA_JWT_SECRET=your-generated-secret-here
ARAGORA_ENV=production
```

If you change the secret, all existing tokens are invalidated. Users must re-login.

### Fix: Supabase History Panel

The "Supabase not configured" warning is **separate from auth**. It only affects the
history panel at the bottom of the dashboard. To fix it:

1. Go to Cloudflare Dashboard → Pages → your project → Settings → Environment Variables
2. Add:
   ```
   NEXT_PUBLIC_SUPABASE_URL = https://YOUR-PROJECT.supabase.co
   NEXT_PUBLIC_SUPABASE_ANON_KEY = eyJ... (your anon/public key from Supabase Dashboard > Settings > API)
   ```
3. Redeploy

### Fix: Cloudflare Pages Environment Variables

These should be set in Cloudflare Pages dashboard:

```
NEXT_PUBLIC_WS_URL = wss://api.aragora.ai/ws
NEXT_PUBLIC_API_URL = https://api.aragora.ai
NEXT_OUTPUT = export
```

Without `NEXT_PUBLIC_WS_URL`, WebSocket features (live debate streaming) won't work
and a red error banner appears.

---

## Backend Environment Variables (Complete)

```bash
# REQUIRED
ARAGORA_ENV=production
ARAGORA_JWT_SECRET=<random-secret-min-32-chars>

# At least ONE AI provider key:
ANTHROPIC_API_KEY=sk-ant-...
# and/or:
OPENAI_API_KEY=sk-...

# CORS (optional - defaults to aragora.ai domains)
ARAGORA_ALLOWED_ORIGINS=https://aragora.ai,https://www.aragora.ai

# Database (optional - defaults to SQLite)
DATABASE_URL=postgresql://user:pass@localhost:5432/aragora

# Recommended
OPENROUTER_API_KEY=...    # Fallback when primary APIs rate-limit
```

---

## Architecture

```
Browser (aragora.ai on Cloudflare Pages)
    │
    ├── Static assets → Cloudflare CDN
    │
    ├── API calls → https://api.aragora.ai (direct, via config.ts)
    │   Frontend resolves API_BASE_URL to https://api.{hostname}
    │   in production (see aragora/live/src/config.ts:resolveApiBaseUrl)
    │
    └── WebSocket → wss://api.aragora.ai/ws

api.aragora.ai (EC2 behind Cloudflare LB)
    ├── /healthz              → Health check
    ├── /api/auth/login       → Login (returns JWT)
    ├── /api/auth/me          → Token validation
    ├── /api/v1/debates       → Create debate (requires debates:create permission)
    ├── /ws                   → WebSocket streaming
    └── ...2000+ endpoints

JWT Flow:
    Login → create_token_pair() using ARAGORA_JWT_SECRET (HS256)
    Subsequent requests → decode_jwt() validates signature + expiry
    Token payload: { sub, email, org_id, role, iat, exp, type, tv }

RBAC Roles (with debates:create permission):
    member, debate_creator, admin, owner
    (viewer and analyst do NOT have debates:create)
```

## Common Error Codes

| Error | Meaning | Fix |
|-------|---------|-----|
| Frontend "Authentication required" | `isAuthenticated` is false — /api/auth/me failed | Check backend is running + CORS |
| 401 Missing token | No Authorization header | Check frontend token storage |
| 401 Invalid token | JWT validation failed | Check ARAGORA_JWT_SECRET matches |
| 402 Quota exceeded | Monthly debate limit hit | Check billing/quota settings |
| 403 Permission denied | RBAC check failed | User needs member+ role |
| 429 Rate limited | Too many requests (5/min for auth) | Wait and retry |
| 502/503 | Backend down | Restart backend on EC2 |
| "Supabase not configured" | Missing env vars on Cloudflare Pages | Add NEXT_PUBLIC_SUPABASE_URL/KEY |
