# Session Management

Aragora provides JWT-based session management with activity tracking, device detection, and multi-session support. This allows users to view and manage their active sessions across devices.

## Overview

The session management system tracks:
- Active sessions per user (up to 10 by default)
- Last activity timestamp for each session
- Device/client information (parsed from user agent)
- IP address at session creation
- Session expiration times

## API Endpoints

### List Active Sessions

```
GET /api/auth/sessions
Authorization: Bearer <access_token>
```

Returns all active sessions for the authenticated user.

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "jti_abc123...",
      "user_id": "user-456",
      "created_at": "2024-01-15T10:30:00Z",
      "last_activity": "2024-01-15T14:22:15Z",
      "ip_address": "192.168.1.100",
      "device_name": "Chrome on Mac",
      "is_current": true,
      "expires_at": "2024-02-14T10:30:00Z"
    },
    {
      "session_id": "jti_def789...",
      "user_id": "user-456",
      "created_at": "2024-01-14T08:00:00Z",
      "last_activity": "2024-01-14T18:45:30Z",
      "ip_address": "10.0.0.50",
      "device_name": "Safari on iPhone",
      "is_current": false,
      "expires_at": "2024-02-13T08:00:00Z"
    }
  ],
  "total": 2
}
```

**Notes:**
- Sessions are sorted by `last_activity` (most recent first)
- `is_current` is `true` for the session making the request
- Inactive sessions (past inactivity timeout) are excluded by default

### Revoke a Session

```
DELETE /api/auth/sessions/:session_id
Authorization: Bearer <access_token>
```

Revokes a specific session. Users cannot revoke their current session (use `/api/auth/logout` instead).

**Response:**
```json
{
  "success": true,
  "message": "Session revoked successfully",
  "session_id": "jti_def789..."
}
```

**Error Cases:**
- `400` - Cannot revoke current session
- `400` - Invalid session ID
- `401` - Not authenticated
- `404` - Session not found

### Related Auth Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/logout` | POST | Logout current session |
| `/api/auth/logout-all` | POST | Terminate all sessions |
| `/api/auth/refresh` | POST | Refresh access token |

## Session Lifecycle

### 1. Session Creation

Sessions are created automatically on:
- Successful login (`POST /api/auth/login`)
- Token refresh (`POST /api/auth/refresh`)
- Registration (`POST /api/auth/register`)

```
Login → JWT issued → Session tracked with:
  - Token JTI as session_id
  - Client IP address
  - User agent string
  - Expiration time
```

### 2. Activity Tracking

Each authenticated request updates the session's `last_activity` timestamp. This is used to:
- Determine session freshness
- Detect inactive sessions
- Power "Last seen" UI features

### 3. Session Expiration

Sessions end when:
- **Token expires** - Configured via `ARAGORA_JWT_SESSION_TTL` (default: 30 days)
- **Inactivity timeout** - No activity for `ARAGORA_SESSION_INACTIVITY_TIMEOUT` (default: 24 hours)
- **User revokes** - Via session management UI or API
- **Logout all** - Via `/api/auth/logout-all`
- **Token version bump** - Admin/security invalidation

## Configuration

Configure session behavior via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_JWT_SESSION_TTL` | `2592000` | Maximum session lifetime in seconds (30 days) |
| `ARAGORA_MAX_SESSIONS_PER_USER` | `10` | Maximum concurrent sessions per user |
| `ARAGORA_SESSION_INACTIVITY_TIMEOUT` | `86400` | Inactivity timeout in seconds (24 hours) |

### Example Configuration

```bash
# Allow longer sessions (90 days)
export ARAGORA_JWT_SESSION_TTL=7776000

# Limit to 5 concurrent sessions
export ARAGORA_MAX_SESSIONS_PER_USER=5

# Shorter inactivity timeout (4 hours)
export ARAGORA_SESSION_INACTIVITY_TIMEOUT=14400
```

## Device Detection

The system automatically parses user agent strings to provide human-readable device names:

| User Agent Contains | Device Name |
|--------------------|-------------|
| `iphone` | iPhone |
| `ipad` | iPad |
| `android` + `mobile` | Android Phone |
| `android` | Android Tablet |
| `macintosh` + `chrome` | Chrome on Mac |
| `macintosh` + `safari` | Safari on Mac |
| `macintosh` + `firefox` | Firefox on Mac |
| `windows` + `chrome` | Chrome on Windows |
| `windows` + `edge` | Edge on Windows |
| `linux` + `chrome` | Chrome on Linux |
| `curl` | cURL |
| `python` | Python Client |
| `postman` | Postman |

## Rate Limits

Session endpoints have the following rate limits:

| Endpoint | Limit | Notes |
|----------|-------|-------|
| `GET /api/auth/sessions` | 30/min | Listing is lightweight |
| `DELETE /api/auth/sessions/:id` | 10/min | Prevents session revocation abuse |

## Implementation Details

### Session Storage

Sessions are stored in-memory with LRU eviction by default. For production deployments:
- Use Redis for distributed session storage
- In-memory store resets on server restart
- Sessions are validated against token blacklist

### Maximum Sessions Enforcement

When a user exceeds `MAX_SESSIONS_PER_USER`:
1. Oldest sessions are automatically evicted
2. Eviction is logged for audit purposes
3. Users are not notified of eviction

### Thread Safety

The session manager is thread-safe:
- All operations use locking
- OrderedDict maintains insertion order for LRU
- Periodic cleanup runs every 5 minutes

## Security Considerations

### Session Fixation

Each login creates a new session ID (derived from JWT JTI). Old sessions are not reused.

### Session Hijacking Mitigation

- IP addresses are logged for audit
- Device fingerprinting via user agent
- Consider implementing additional checks for sensitive operations

### Immediate Revocation

The session tracking system complements but doesn't replace the token blacklist:
- `revoke_session()` removes session tracking
- For immediate token invalidation, use `logout-all` to bump token version
- Natural token expiration provides eventual revocation

## Example Usage

### Python SDK

```python
import requests

BASE_URL = "https://api.aragora.ai"
TOKEN = "your_access_token"

headers = {"Authorization": f"Bearer {TOKEN}"}

# List sessions
response = requests.get(f"{BASE_URL}/api/auth/sessions", headers=headers)
sessions = response.json()["sessions"]

print(f"You have {len(sessions)} active sessions:")
for s in sessions:
    current = " (current)" if s["is_current"] else ""
    print(f"  - {s['device_name']}: last active {s['last_activity']}{current}")

# Revoke a session
session_to_revoke = next(s for s in sessions if not s["is_current"])
response = requests.delete(
    f"{BASE_URL}/api/auth/sessions/{session_to_revoke['session_id']}",
    headers=headers
)
print(response.json()["message"])
```

### JavaScript/TypeScript

```typescript
const API_BASE = 'https://api.aragora.ai';

async function listSessions(token: string) {
  const response = await fetch(`${API_BASE}/api/auth/sessions`, {
    headers: { Authorization: `Bearer ${token}` }
  });
  const data = await response.json();
  return data.sessions;
}

async function revokeSession(token: string, sessionId: string) {
  const response = await fetch(`${API_BASE}/api/auth/sessions/${sessionId}`, {
    method: 'DELETE',
    headers: { Authorization: `Bearer ${token}` }
  });
  return response.json();
}

// Usage
const sessions = await listSessions(accessToken);
const otherSessions = sessions.filter(s => !s.is_current);

if (otherSessions.length > 0) {
  await revokeSession(accessToken, otherSessions[0].session_id);
  console.log('Revoked oldest session');
}
```

### cURL

```bash
# List sessions
curl -H "Authorization: Bearer $TOKEN" \
  https://api.aragora.ai/api/auth/sessions

# Revoke a session
curl -X DELETE \
  -H "Authorization: Bearer $TOKEN" \
  https://api.aragora.ai/api/auth/sessions/jti_abc123...
```

## Troubleshooting

### "Session not found" Error

- Session may have expired due to inactivity
- Session may have been evicted due to max sessions limit
- Token version may have been bumped (check if other sessions work)

### Sessions Not Persisting After Restart

- Default in-memory storage doesn't persist
- Configure Redis for production deployments
- See `docs/REDIS_HA.md` for Redis setup

### Activity Not Updating

- Ensure requests include valid JWT
- Check if session middleware is applied to route
- Verify session manager is initialized

## Related Documentation

- [API Rate Limits](./API_RATE_LIMITS.md) - Rate limiting configuration
- [OAuth Setup](./OAUTH_SETUP.md) - OAuth provider configuration
- [SSO Setup](./SSO_SETUP.md) - Enterprise SSO integration
- [Security Patterns](./SECURITY_PATTERNS.md) - Security best practices
