# Aragora.ai Troubleshooting Guide

## Quick Diagnostic Checklist

### 1. Backend Health (EC2)
```bash
# SSH into EC2, then:
curl http://localhost:8080/api/health          # Should return 200
curl http://localhost:8080/api/auth/me          # Should return 401 (no token)
curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"YOUR_EMAIL","password":"YOUR_PASSWORD"}'
# Should return: { "user": {...}, "tokens": { "access_token": "...", ... } }
```

### 2. Backend Environment Variables (EC2 .env)
```bash
# Required (at least one LLM key):
ANTHROPIC_API_KEY=sk-ant-...
# or
OPENAI_API_KEY=sk-...

# Required for JWT validation:
SUPABASE_JWT_SECRET=your-jwt-secret    # From Supabase Dashboard > Settings > API
SUPABASE_URL=https://xxx.supabase.co   # Your Supabase project URL

# Optional but recommended:
OPENROUTER_API_KEY=...                 # Fallback for rate-limited providers
ARAGORA_ALLOWED_ORIGINS=https://aragora.ai,https://www.aragora.ai
```

### 3. CORS Configuration (EC2)
If frontend calls backend directly (not through proxy), CORS must allow:
- Origin: `https://aragora.ai`
- Methods: `GET, POST, PUT, DELETE, OPTIONS`
- Headers: `Content-Type, Authorization`
- Credentials: `true`

### 4. Cloudflare Pages Environment Variables
Set in: Cloudflare Dashboard > Pages > Settings > Environment Variables

```
NEXT_PUBLIC_WS_URL=wss://api.aragora.ai/ws
NEXT_PUBLIC_API_URL=https://api.aragora.ai
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
NEXT_OUTPUT=export
```

### 5. Test Auth Flow End-to-End
```bash
# 1. Login
TOKEN=$(curl -s -X POST https://api.aragora.ai/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"YOUR_EMAIL","password":"YOUR_PASSWORD"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['tokens']['access_token'])")

# 2. Validate token
curl -H "Authorization: Bearer $TOKEN" https://api.aragora.ai/api/auth/me

# 3. Test debate creation
curl -X POST https://api.aragora.ai/api/v1/debates \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"question":"Test debate","agents":"anthropic-api,openai-api","rounds":3}'
```

### 6. Common Error Codes
| Error | Meaning | Fix |
|-------|---------|-----|
| 401 Missing token | No Authorization header sent | Check frontend token storage |
| 401 Invalid token | JWT validation failed | Check SUPABASE_JWT_SECRET matches |
| 402 Quota exceeded | Monthly debate limit hit | Check billing/quota settings |
| 403 Permission denied | RBAC check failed | User needs "debates:create" permission (member+ role) |
| 404 Endpoint not found | Route not registered | Check backend is running correct version |
| 429 Rate limited | Too many requests | Wait and retry |
| 502/503 | Backend down | Restart backend service on EC2 |

## Architecture Reference

```
User Browser (aragora.ai)
    |
    ├── Static assets → Cloudflare Pages CDN
    |
    ├── /api/* requests → Two possible paths:
    |   ├── Path A: Direct to https://api.aragora.ai (via NEXT_PUBLIC_API_URL)
    |   └── Path B: Cloudflare Pages Function proxy (functions/api/[[catchall]].ts)
    |
    └── WebSocket → wss://api.aragora.ai/ws

api.aragora.ai (EC2 us-east-2)
    ├── /api/auth/*     → Auth handler (login, register, refresh, me)
    ├── /api/v1/debates → Debate handler (RBAC: debates:create)
    ├── /ws             → WebSocket streaming
    └── /api/health     → Health check

Supabase (optional)
    └── History data only (nomic_cycles, stream_events, debate_artifacts)
```
