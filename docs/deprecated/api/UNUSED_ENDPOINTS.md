# Unused API Endpoints

> **Deprecated:** This inventory is historical. For current endpoint coverage,
> see `docs/API_ENDPOINTS.md` and `docs/API_REFERENCE.md`.

This document tracks API endpoints that are implemented but not yet used by the frontend.
These are candidates for either:
1. **UI Development** - Build frontend components to expose these features
2. **Deprecation** - Remove if no longer needed

Last audited: 2026-01-14

## Summary

| Handler | Endpoints | Purpose | Recommendation |
|---------|-----------|---------|----------------|
| training | 5 | Training data export | Build UI (high value for ML teams) |
| gallery | 3 | Public debate gallery | Build UI (community feature) |
| social | 5 | Twitter/YouTube publishing | Already has partial UI support |
| persona | 4 | Agent persona management | Build UI (power user feature) |
| auditing | 3 | Audit log endpoints | Admin console integration |
| sso | 4 | SAML/OIDC SSO | Enterprise feature (on demand) |

---

## Training Data Export (`/api/training/*`)

**File:** `aragora/server/handlers/training.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/training/export/sft` | POST | Export debates for Supervised Fine-Tuning |
| `/api/training/export/dpo` | POST | Export debates for Direct Preference Optimization |
| `/api/training/export/gauntlet` | POST | Export gauntlet adversarial data |
| `/api/training/stats` | GET | Get export statistics |
| `/api/training/formats` | GET | List available export formats |

**Recommendation:** HIGH VALUE - Build a "Training Export" page in admin console for ML teams.

---

## Public Gallery (`/api/gallery/*`)

**File:** `aragora/server/handlers/gallery.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/gallery` | GET | List public debates |
| `/api/gallery/:debate_id` | GET | Get specific debate with full history |
| `/api/gallery/:debate_id/embed` | GET | Get embeddable debate summary |

**Recommendation:** MEDIUM VALUE - Build public gallery page for community engagement.

---

## Social Media Publishing (`/api/youtube/*`, `/api/debates/*/publish/*`)

**File:** `aragora/server/handlers/social.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/youtube/auth` | GET | Get YouTube OAuth authorization URL |
| `/api/youtube/callback` | GET | Handle YouTube OAuth callback |
| `/api/youtube/status` | GET | Get YouTube connector status |
| `/api/debates/:id/publish/twitter` | POST | Publish debate to Twitter/X |
| `/api/debates/:id/publish/youtube` | POST | Publish debate to YouTube |

**Note:** Some social publishing buttons exist in frontend but OAuth flow may need testing.

**Recommendation:** MEDIUM VALUE - Complete OAuth integration in settings page.

---

## Agent Personas (`/api/persona/*`)

**File:** `aragora/server/handlers/persona.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/persona` | GET | List all personas |
| `/api/persona/:name` | GET | Get specific persona |
| `/api/persona/:name` | PUT | Update persona |
| `/api/persona/:name/stats` | GET | Get persona performance stats |

**Recommendation:** MEDIUM VALUE - Build persona editor for power users.

---

## Audit Logging (`/api/audit/*`)

**File:** `aragora/server/handlers/auditing.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/audit/events` | GET | List audit events |
| `/api/audit/events/:id` | GET | Get specific audit event |
| `/api/audit/summary` | GET | Get audit summary statistics |

**Recommendation:** LOW VALUE - Integrate into admin console "Activity" tab.

---

## Single Sign-On (`/api/sso/*`)

**File:** `aragora/server/handlers/sso.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sso/saml/metadata` | GET | Get SAML service provider metadata |
| `/api/sso/saml/login` | GET | Initiate SAML login |
| `/api/sso/saml/callback` | POST | Handle SAML assertion |
| `/api/sso/oidc/login` | GET | Initiate OIDC login |

**Recommendation:** ENTERPRISE FEATURE - Build on-demand for enterprise customers.

---

## Decision Matrix

| Handler | Decision | Status | Timeline |
|---------|----------|--------|----------|
| training | **BUILD UI** | DONE | Q1 2026 |
| gallery | **BUILD UI** | DONE | Q1 2026 |
| social | **KEEP** | Partial UI exists, OAuth needs completion | Q2 2026 |
| persona | **BUILD UI** | DONE | Q2 2026 |
| auditing | **KEEP** | DONE | Q1 2026 |
| sso | **KEEP** | Enterprise feature, build on demand | On request |

**No endpoints scheduled for deprecation.** All endpoints now have UI coverage.

---

## Action Items

### Phase 1: High-Value Features (Q1 2026)
- [x] Training Export UI in admin console (`/admin/training`)
- [x] Public Gallery page (`/gallery`)
- [x] Audit log integration (`/admin/audit`)

### Phase 2: Power User Features (Q2 2026)
- [x] Persona Editor (`/admin/personas`)
- [x] Complete Social OAuth flow (`/settings` > Account > Connected Accounts)

### Phase 3: Enterprise Features (On Demand)
- [ ] SSO configuration page (SAML/OIDC)
