# RBAC Exemption Registry

This document lists endpoints that are intentionally exempted from standard RBAC (Role-Based Access Control) checks. These exemptions exist because the endpoints use alternative authentication mechanisms or must be publicly accessible.

## Webhook Endpoints (Platform-Authenticated)

These endpoints are authenticated via platform-specific signatures rather than user RBAC:

| Endpoint Pattern | Platform | Auth Method |
|-----------------|----------|-------------|
| `/api/chat/slack/webhook` | Slack | HMAC signature (X-Slack-Signature) |
| `/api/chat/teams/webhook` | Teams | Bot Framework Bearer token |
| `/api/chat/discord/webhook` | Discord | Ed25519 signature |
| `/api/chat/telegram/webhook` | Telegram | Secret token verification |
| `/api/chat/whatsapp/webhook` | WhatsApp | HMAC signature |
| `/api/bots/slack/*` | Slack | Slack signature verification |
| `/api/bots/teams/*` | Teams | Bot Framework auth |
| `/api/bots/telegram/*` | Telegram | Secret token |
| `/api/bots/discord/*` | Discord | Ed25519 signature |
| `/api/bots/whatsapp/*` | WhatsApp | HMAC signature |

**Rationale:** Webhook endpoints receive events from external platforms. These platforms authenticate requests using their own signature/token schemes, not user JWT tokens.

## OAuth/OIDC Flows

OAuth endpoints are part of the authentication flow itself:

| Endpoint Pattern | Purpose |
|-----------------|---------|
| `/auth/oidc/callback` | OIDC callback (receives auth code) |
| `/auth/oidc/authorize` | OIDC authorization initiation |
| `/auth/saml/callback` | SAML assertion consumer |
| `/auth/oauth/*/callback` | OAuth provider callbacks |
| `/.well-known/openid-configuration` | OIDC discovery |
| `/.well-known/jwks.json` | JWT key set |

**Rationale:** These endpoints are the authentication mechanism itself - they cannot require authentication.

## Public Endpoints

Endpoints that must be accessible without authentication:

| Endpoint | Purpose |
|----------|---------|
| `/api/health` | Load balancer health checks |
| `/api/ready` | Kubernetes readiness probe |
| `/api/live` | Kubernetes liveness probe |
| `/api/plans` | Public pricing information |
| `/api/version` | API version information |

**Rationale:** Infrastructure endpoints need to be public for orchestration. Pricing info is public marketing content.

## Internal Service Endpoints

Endpoints called by internal services with service-to-service auth:

| Endpoint Pattern | Service | Auth Method |
|-----------------|---------|-------------|
| `/internal/metrics/*` | Prometheus | mTLS / IP allowlist |
| `/internal/traces/*` | OpenTelemetry | mTLS |

**Rationale:** Internal observability endpoints use network-level security rather than user RBAC.

## Updating This Document

When adding a new exemption:

1. Verify the endpoint truly cannot use standard RBAC
2. Document the alternative authentication mechanism
3. Add to the appropriate section above
4. Update the RBAC audit script if needed to recognize the pattern

## Related Documentation

- [SECURITY.md](./SECURITY.md) - Overall security architecture
- [RBAC_PERMISSION_REFERENCE.md](./RBAC_PERMISSION_REFERENCE.md) - Permission definitions
- [SSO_SETUP.md](./SSO_SETUP.md) - SSO configuration
