# SDK-API Parity Report

Generated: 2026-01-13

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Backend Endpoints** | ~207 |
| **SDK-Covered Endpoints** | ~127 |
| **Missing from SDK** | ~80 |
| **Coverage Percentage** | **61%** |
| **SDK API Classes** | 23 |
| **Orphaned SDK Methods** | 2 |

The TypeScript SDK (`aragora-js`) provides good coverage of core debate functionality but lacks support for several backend capabilities including: OAuth, Slack integration, broadcast/podcast generation, gallery, reviews, evolution A/B testing, admin operations, webhooks, plugins, SSO, and many agent analytics endpoints.

---

## Coverage by Domain

### 1. Debates API (DebatesAPI)
**Coverage: 15/22 (68%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `POST /api/debates` | POST | `debates.create()` | :white_check_mark: |
| `GET /api/debates` | GET | `debates.list()` | :white_check_mark: |
| `GET /api/debates/{id}` | GET | `debates.get()` | :white_check_mark: |
| `GET /api/debates/slug/{slug}` | GET | `debates.get()` | :white_check_mark: |
| `GET /api/debates/{id}/impasse` | GET | `debates.impasse()` | :white_check_mark: |
| `GET /api/debates/{id}/convergence` | GET | `debates.convergence()` | :white_check_mark: |
| `GET /api/debates/{id}/citations` | GET | `debates.citations()` | :white_check_mark: |
| `GET /api/debates/{id}/messages` | GET | `debates.messages()` | :white_check_mark: |
| `GET /api/debates/{id}/evidence` | GET | `debates.evidence()` | :white_check_mark: |
| `GET /api/debates/{id}/summary` | GET | `debates.summary()` | :white_check_mark: |
| `GET /api/debates/{id}/followups` | GET | `debates.followupSuggestions()` | :white_check_mark: |
| `POST /api/debates/{id}/fork` | POST | `debates.fork()` | :white_check_mark: |
| `POST /api/debates/{id}/followup` | POST | `debates.followup()` | :white_check_mark: |
| `GET /api/debates/{id}/export/{format}` | GET | `debates.export()` | :white_check_mark: |
| `PATCH /api/debates/{id}` | PATCH | - | :x: Missing |
| `POST /api/debates/{id}/verify` | POST | - | :x: Missing |
| `GET /api/debates/{id}/verification-report` | GET | - | :x: Missing |
| `GET /api/debates/{id}/forks` | GET | - | :x: Missing |
| `GET /api/search` | GET | - | :x: Missing |
| `GET /api/debate/{id}/meta-critique` | GET | - | :x: Missing |
| `GET /api/debate/{id}/graph/stats` | GET | - | :x: Missing |

### 2. Batch Debates API (BatchDebatesAPI)
**Coverage: 3/3 (100%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `POST /api/debates/batch` | POST | `batchDebates.submit()` | :white_check_mark: |
| `GET /api/debates/batch/{id}/status` | GET | `batchDebates.status()` | :white_check_mark: |
| `GET /api/debates/queue/status` | GET | `batchDebates.queueStatus()` | :white_check_mark: |

### 3. Graph Debates API (GraphDebatesAPI)
**Coverage: 3/3 (100%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `POST /api/debates/graph` | POST | `graphDebates.create()` | :white_check_mark: |
| `GET /api/debates/graph/{id}` | GET | `graphDebates.get()` | :white_check_mark: |
| `GET /api/debates/graph/{id}/branches` | GET | `graphDebates.getBranches()` | :white_check_mark: |

### 4. Matrix Debates API (MatrixDebatesAPI)
**Coverage: 3/3 (100%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `POST /api/debates/matrix` | POST | `matrixDebates.create()` | :white_check_mark: |
| `GET /api/debates/matrix/{id}` | GET | `matrixDebates.get()` | :white_check_mark: |
| `GET /api/debates/matrix/{id}/conclusions` | GET | `matrixDebates.getConclusions()` | :white_check_mark: |

### 5. Agents API (AgentsAPI)
**Coverage: 5/28 (18%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `GET /api/agents` | GET | `agents.list()` | :white_check_mark: |
| `GET /api/agents/{id}` | GET | `agents.get()` | :white_check_mark: |
| `GET /api/agent/{name}/history` | GET | `agents.history()` | :white_check_mark: |
| `GET /api/agent/{name}/rivals` | GET | `agents.rivals()` | :white_check_mark: |
| `GET /api/agent/{name}/allies` | GET | `agents.allies()` | :white_check_mark: |
| `GET /api/agents/local` | GET | - | :x: Missing |
| `GET /api/agents/local/status` | GET | - | :x: Missing |
| `GET /api/rankings` | GET | - | :x: Missing |
| `GET /api/matches/recent` | GET | - | :x: Missing |
| `GET /api/agent/compare` | GET | - | :x: Missing |
| `GET /api/agent/{name}/profile` | GET | - | :x: Missing |
| `GET /api/agent/{name}/calibration` | GET | - | :x: Missing |
| `GET /api/agent/{name}/consistency` | GET | - | :x: Missing |
| `GET /api/agent/{name}/flips` | GET | - | :x: Missing |
| `GET /api/agent/{name}/network` | GET | - | :x: Missing |
| `GET /api/agent/{name}/moments` | GET | - | :x: Missing |
| `GET /api/agent/{name}/positions` | GET | - | :x: Missing |
| `GET /api/agent/{name}/domains` | GET | - | :x: Missing |
| `GET /api/agent/{name}/performance` | GET | - | :x: Missing |
| `GET /api/agent/{name}/head-to-head/{opponent}` | GET | - | :x: Missing |
| `GET /api/agent/{name}/opponent-briefing/{opponent}` | GET | - | :x: Missing |
| `GET /api/flips/recent` | GET | - | :x: Missing (use insights.flips) |
| `GET /api/flips/summary` | GET | - | :x: Missing |

### 6. Leaderboard API (LeaderboardAPI)
**Coverage: 1/1 (100%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `GET /api/leaderboard` | GET | `leaderboard.get()` | :white_check_mark: |

### 7. Verification API (VerificationAPI)
**Coverage: 3/4 (75%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `POST /api/verify/claim` | POST | `verification.verify()` | :white_check_mark: |
| `POST /api/verify/batch` | POST | `verification.verifyBatch()` | :white_check_mark: |
| `GET /api/verify/status` | GET | `verification.status()` | :white_check_mark: |
| `POST /api/verify/translate` | POST | - | :x: Missing |

### 8. Memory API (MemoryAPI)
**Coverage: 9/12 (75%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `GET /api/memory/analytics` | GET | `memory.analytics()` | :white_check_mark: |
| `GET /api/memory/analytics/tier/{tier}` | GET | `memory.tierStats()` | :white_check_mark: |
| `POST /api/memory/analytics/snapshot` | POST | `memory.snapshot()` | :white_check_mark: |
| `GET /api/memory/continuum/retrieve` | GET | `memory.retrieve()` | :white_check_mark: |
| `POST /api/memory/continuum/consolidate` | POST | `memory.consolidate()` | :white_check_mark: |
| `POST /api/memory/continuum/cleanup` | POST | `memory.cleanup()` | :white_check_mark: |
| `GET /api/memory/tier-stats` | GET | `memory.tiers()` | :white_check_mark: |
| `GET /api/memory/archive-stats` | GET | `memory.archiveStats()` | :white_check_mark: |
| `GET /api/memory/pressure` | GET | `memory.pressure()` | :white_check_mark: |
| `DELETE /api/memory/continuum/{id}` | DELETE | `memory.delete()` | :white_check_mark: |
| `GET /api/memory/tiers` | GET | - | :x: Missing (different from tier-stats) |
| `GET /api/memory/search` | GET | - | :x: Missing |
| `GET /api/memory/critiques` | GET | - | :x: Missing |

### 9. Gauntlet API (GauntletAPI)
**Coverage: 2/8 (25%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `POST /api/gauntlet/run` | POST | `gauntlet.run()` | :white_check_mark: |
| `GET /api/gauntlet/{id}/receipt` | GET | `gauntlet.getReceipt()` | :white_check_mark: |
| `GET /api/gauntlet/personas` | GET | - | :x: Missing |
| `GET /api/gauntlet/results` | GET | - | :x: Missing |
| `GET /api/gauntlet/{id}` | GET | - | :x: Missing |
| `GET /api/gauntlet/{id}/heatmap` | GET | - | :x: Missing |
| `GET /api/gauntlet/{id}/compare/{id2}` | GET | - | :x: Missing |
| `DELETE /api/gauntlet/{id}` | DELETE | - | :x: Missing |

### 10. Replay API (ReplayAPI)
**Coverage: 4/4 (100%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `GET /api/replays` | GET | `replays.list()` | :white_check_mark: |
| `GET /api/replays/{id}` | GET | `replays.get()` | :white_check_mark: |
| `DELETE /api/replays/{id}` | DELETE | `replays.delete()` | :white_check_mark: |
| `GET /api/replays/{id}/export` | GET | `replays.export()` | :white_check_mark: |

### 11. Pulse API (PulseAPI)
**Coverage: 10/10 (100%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `GET /api/pulse/trending` | GET | `pulse.trending()` | :white_check_mark: |
| `GET /api/pulse/suggest` | GET | `pulse.suggest()` | :white_check_mark: |
| `GET /api/pulse/analytics` | GET | `pulse.analytics()` | :white_check_mark: |
| `POST /api/pulse/debate-topic` | POST | `pulse.debateTopic()` | :white_check_mark: |
| `GET /api/pulse/scheduler/status` | GET | `pulse.schedulerStatus()` | :white_check_mark: |
| `POST /api/pulse/scheduler/start` | POST | `pulse.schedulerStart()` | :white_check_mark: |
| `POST /api/pulse/scheduler/stop` | POST | `pulse.schedulerStop()` | :white_check_mark: |
| `POST /api/pulse/scheduler/pause` | POST | `pulse.schedulerPause()` | :white_check_mark: |
| `POST /api/pulse/scheduler/resume` | POST | `pulse.schedulerResume()` | :white_check_mark: |
| `PUT /api/pulse/scheduler/config` | PUT | `pulse.schedulerConfig()` | :white_check_mark: |
| `GET /api/pulse/scheduler/history` | GET | `pulse.schedulerHistory()` | :white_check_mark: |

### 12. Documents API (DocumentsAPI)
**Coverage: 5/5 (100%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `GET /api/documents` | GET | `documents.list()` | :white_check_mark: |
| `GET /api/documents/formats` | GET | `documents.formats()` | :white_check_mark: |
| `GET /api/documents/{id}` | GET | `documents.get()` | :white_check_mark: |
| `DELETE /api/documents/{id}` | DELETE | `documents.delete()` | :white_check_mark: |
| `POST /api/documents/upload` | POST | `documents.upload()` | :white_check_mark: |

### 13. Breakpoints API (BreakpointsAPI)
**Coverage: 3/3 (100%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `GET /api/breakpoints/pending` | GET | `breakpoints.pending()` | :white_check_mark: |
| `GET /api/breakpoints/{id}/status` | GET | `breakpoints.status()` | :white_check_mark: |
| `POST /api/breakpoints/{id}/resolve` | POST | `breakpoints.resolve()` | :white_check_mark: |

### 14. Tournaments API (TournamentsAPI)
**Coverage: 2/2 (100%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `GET /api/tournaments` | GET | `tournaments.list()` | :white_check_mark: |
| `GET /api/tournaments/{id}/standings` | GET | `tournaments.standings()` | :white_check_mark: |

### 15. Organizations API (OrganizationsAPI)
**Coverage: 9/9 (100%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `GET /api/org/{id}` | GET | `organizations.get()` | :white_check_mark: |
| `PUT /api/org/{id}` | PUT | `organizations.update()` | :white_check_mark: |
| `GET /api/org/{id}/members` | GET | `organizations.members()` | :white_check_mark: |
| `POST /api/org/{id}/invite` | POST | `organizations.invite()` | :white_check_mark: |
| `GET /api/org/{id}/invitations` | GET | `organizations.invitations()` | :white_check_mark: |
| `DELETE /api/org/{id}/invitations/{invId}` | DELETE | `organizations.revokeInvitation()` | :white_check_mark: |
| `DELETE /api/org/{id}/members/{userId}` | DELETE | `organizations.removeMember()` | :white_check_mark: |
| `PUT /api/org/{id}/members/{userId}/role` | PUT | `organizations.updateMemberRole()` | :white_check_mark: |
| `GET /api/invitations/pending` | GET | `organizations.myPendingInvitations()` | :white_check_mark: |
| `POST /api/invitations/{token}/accept` | POST | `organizations.acceptInvitation()` | :white_check_mark: |

### 16. Analytics API (AnalyticsAPI)
**Coverage: 3/9 (33%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `GET /api/analytics` | GET | `analytics.overview()` | :white_check_mark: |
| `GET /api/analytics/agent/{id}` | GET | `analytics.agent()` | :white_check_mark: |
| `GET /api/analytics/debates` | GET | `analytics.debates()` | :white_check_mark: |
| `GET /api/analytics/disagreements` | GET | - | :x: Missing |
| `GET /api/analytics/role-rotation` | GET | - | :x: Missing |
| `GET /api/analytics/early-stops` | GET | - | :x: Missing |
| `GET /api/analytics/consensus-quality` | GET | - | :x: Missing |
| `GET /api/ranking/stats` | GET | - | :x: Missing |
| `GET /api/memory/stats` | GET | - | :x: Missing |

### 17. Auth API (AuthAPI)
**Coverage: 15/16 (94%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `POST /api/auth/register` | POST | `auth.register()` | :white_check_mark: |
| `POST /api/auth/login` | POST | `auth.login()` | :white_check_mark: |
| `POST /api/auth/logout` | POST | `auth.logout()` | :white_check_mark: |
| `POST /api/auth/logout-all` | POST | `auth.logoutAll()` | :white_check_mark: |
| `POST /api/auth/refresh` | POST | `auth.refresh()` | :white_check_mark: |
| `POST /api/auth/revoke` | POST | `auth.revoke()` | :white_check_mark: |
| `GET /api/auth/me` | GET | `auth.me()` | :white_check_mark: |
| `PUT /api/auth/me` | PUT | `auth.updateMe()` | :white_check_mark: |
| `POST /api/auth/password` | POST | `auth.changePassword()` | :white_check_mark: |
| `POST /api/auth/api-key` | POST | `auth.createApiKey()` | :white_check_mark: |
| `DELETE /api/auth/api-key` | DELETE | `auth.revokeApiKey()` | :white_check_mark: |
| `POST /api/auth/mfa/setup` | POST | `auth.mfaSetup()` | :white_check_mark: |
| `POST /api/auth/mfa/enable` | POST | `auth.mfaEnable()` | :white_check_mark: |
| `POST /api/auth/mfa/disable` | POST | `auth.mfaDisable()` | :white_check_mark: |
| `POST /api/auth/mfa/verify` | POST | `auth.mfaVerify()` | :white_check_mark: |
| `POST /api/auth/mfa/backup-codes` | POST | `auth.mfaBackupCodes()` | :white_check_mark: |

### 18. Billing API (BillingAPI)
**Coverage: 10/11 (91%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `GET /api/billing/plans` | GET | `billing.plans()` | :white_check_mark: |
| `GET /api/billing/usage` | GET | `billing.usage()` | :white_check_mark: |
| `GET /api/billing/subscription` | GET | `billing.subscription()` | :white_check_mark: |
| `POST /api/billing/checkout` | POST | `billing.checkout()` | :white_check_mark: |
| `POST /api/billing/portal` | POST | `billing.portal()` | :white_check_mark: |
| `POST /api/billing/cancel` | POST | `billing.cancel()` | :white_check_mark: |
| `POST /api/billing/resume` | POST | `billing.resume()` | :white_check_mark: |
| `GET /api/billing/audit-log` | GET | `billing.auditLog()` | :white_check_mark: |
| `GET /api/billing/usage/export` | GET | `billing.exportUsage()` | :white_check_mark: |
| `GET /api/billing/usage/forecast` | GET | `billing.forecast()` | :white_check_mark: |
| `GET /api/billing/invoices` | GET | `billing.invoices()` | :white_check_mark: |
| `POST /api/webhooks/stripe` | POST | - | :x: Missing (server-side only) |

### 19. Evidence API (EvidenceAPI)
**Coverage: 8/8 (100%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `GET /api/evidence` | GET | `evidence.list()` | :white_check_mark: |
| `GET /api/evidence/{id}` | GET | `evidence.get()` | :white_check_mark: |
| `POST /api/evidence/search` | POST | `evidence.search()` | :white_check_mark: |
| `POST /api/evidence/collect` | POST | `evidence.collect()` | :white_check_mark: |
| `GET /api/evidence/debate/{id}` | GET | `evidence.forDebate()` | :white_check_mark: |
| `POST /api/evidence/debate/{id}` | POST | `evidence.associateWithDebate()` | :white_check_mark: |
| `GET /api/evidence/statistics` | GET | `evidence.statistics()` | :white_check_mark: |
| `DELETE /api/evidence/{id}` | DELETE | `evidence.delete()` | :white_check_mark: |

### 20. Calibration API (CalibrationAPI)
**Coverage: 4/4 (100%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `GET /api/calibration/leaderboard` | GET | `calibration.leaderboard()` | :white_check_mark: |
| `GET /api/agent/{name}/calibration-curve` | GET | `calibration.curve()` | :white_check_mark: |
| `GET /api/agent/{name}/calibration-summary` | GET | `calibration.summary()` | :white_check_mark: |
| `GET /api/calibration/visualization` | GET | `calibration.visualization()` | :white_check_mark: |

### 21. Insights API (InsightsAPI)
**Coverage: 3/3 (100%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `GET /api/insights/recent` | GET | `insights.recent()` | :white_check_mark: |
| `GET /api/flips/recent` | GET | `insights.flips()` | :white_check_mark: |
| `POST /api/insights/extract-detailed` | POST | `insights.extractDetailed()` | :white_check_mark: |

### 22. Belief Network API (BeliefNetworkAPI)
**Coverage: 4/4 (100%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `GET /api/belief-network/{id}/cruxes` | GET | `beliefNetwork.cruxes()` | :white_check_mark: |
| `GET /api/belief-network/{id}/load-bearing-claims` | GET | `beliefNetwork.loadBearingClaims()` | :white_check_mark: |
| `GET /api/provenance/{id}/claims/{claimId}/support` | GET | `beliefNetwork.claimSupport()` | :white_check_mark: |
| `GET /api/debate/{id}/graph-stats` | GET | `beliefNetwork.graphStats()` | :white_check_mark: |

### 23. Consensus API (ConsensusAPI)
**Coverage: 7/7 (100%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `GET /api/consensus/similar` | GET | `consensus.similar()` | :white_check_mark: |
| `GET /api/consensus/settled` | GET | `consensus.settled()` | :white_check_mark: |
| `GET /api/consensus/stats` | GET | `consensus.stats()` | :white_check_mark: |
| `GET /api/consensus/dissents` | GET | `consensus.dissents()` | :white_check_mark: |
| `GET /api/consensus/contrarian-views` | GET | `consensus.contrarianViews()` | :white_check_mark: |
| `GET /api/consensus/risk-warnings` | GET | `consensus.riskWarnings()` | :white_check_mark: |
| `GET /api/consensus/domain/{domain}` | GET | `consensus.domainHistory()` | :white_check_mark: |

### 24. Health API (Client Methods)
**Coverage: 2/2 (100%)**

| Endpoint | Method | SDK Method | Status |
|----------|--------|------------|--------|
| `GET /api/health` | GET | `client.health()` | :white_check_mark: |
| `GET /api/health/deep` | GET | `client.healthDeep()` | :white_check_mark: |

---

## Endpoints Missing from SDK (By Priority)

### HIGH Priority - Core Functionality Gaps

These endpoints support core features that SDK users would expect:

| Handler | Endpoint | Method | Description |
|---------|----------|--------|-------------|
| debates.py | `PATCH /api/debates/{id}` | PATCH | Update debate metadata |
| debates.py | `POST /api/debates/{id}/verify` | POST | Verify debate claims |
| debates.py | `GET /api/debates/{id}/verification-report` | GET | Get verification results |
| debates.py | `GET /api/search` | GET | Search debates |
| agents.py | `GET /api/agent/{name}/profile` | GET | Full agent profile |
| agents.py | `GET /api/agent/{name}/calibration` | GET | Agent calibration data |
| agents.py | `GET /api/agent/{name}/performance` | GET | Agent performance metrics |
| verification.py | `POST /api/verify/translate` | POST | Translate to formal notation |
| gauntlet.py | `GET /api/gauntlet/{id}` | GET | Get gauntlet run details |
| gauntlet.py | `DELETE /api/gauntlet/{id}` | DELETE | Delete gauntlet run |

### MEDIUM Priority - Enhanced Features

These endpoints provide useful but not essential functionality:

| Handler | Endpoint | Method | Description |
|---------|----------|--------|-------------|
| agents.py | `GET /api/agent/{name}/head-to-head/{opponent}` | GET | Head-to-head comparison |
| agents.py | `GET /api/agent/{name}/opponent-briefing/{opponent}` | GET | Strategic briefing |
| agents.py | `GET /api/agent/{name}/consistency` | GET | Agent consistency metrics |
| agents.py | `GET /api/agent/{name}/flips` | GET | Agent position flips |
| agents.py | `GET /api/agent/{name}/network` | GET | Agent relationship network |
| agents.py | `GET /api/agent/{name}/moments` | GET | Notable agent moments |
| agents.py | `GET /api/agent/{name}/positions` | GET | Agent historical positions |
| agents.py | `GET /api/agent/{name}/domains` | GET | Agent domain expertise |
| agents.py | `GET /api/agents/local` | GET | Local agent availability |
| agents.py | `GET /api/agents/local/status` | GET | Local agent status |
| agents.py | `GET /api/rankings` | GET | Full rankings list |
| agents.py | `GET /api/matches/recent` | GET | Recent match history |
| agents.py | `GET /api/agent/compare` | GET | Multi-agent comparison |
| agents.py | `GET /api/flips/summary` | GET | Flip statistics summary |
| analytics.py | `GET /api/analytics/disagreements` | GET | Disagreement analytics |
| analytics.py | `GET /api/analytics/role-rotation` | GET | Role rotation patterns |
| analytics.py | `GET /api/analytics/early-stops` | GET | Early termination analysis |
| analytics.py | `GET /api/analytics/consensus-quality` | GET | Consensus quality metrics |
| analytics.py | `GET /api/ranking/stats` | GET | Ranking system statistics |
| analytics.py | `GET /api/memory/stats` | GET | Memory system statistics |
| gauntlet.py | `GET /api/gauntlet/personas` | GET | Available adversarial personas |
| gauntlet.py | `GET /api/gauntlet/results` | GET | All gauntlet results |
| gauntlet.py | `GET /api/gauntlet/{id}/heatmap` | GET | Challenge response heatmap |
| gauntlet.py | `GET /api/gauntlet/{id}/compare/{id2}` | GET | Compare two gauntlet runs |
| memory.py | `GET /api/memory/tiers` | GET | Memory tier configuration |
| memory.py | `GET /api/memory/search` | GET | Search memories |
| memory.py | `GET /api/memory/critiques` | GET | Stored critiques |
| debates.py | `GET /api/debates/{id}/forks` | GET | List debate forks |
| debates.py | `GET /api/debate/{id}/meta-critique` | GET | Meta-critique analysis |
| debates.py | `GET /api/debate/{id}/graph/stats` | GET | Argument graph statistics |

### LOW Priority - Specialized/Admin Features

These endpoints are for admin, integrations, or niche use cases:

| Handler | Endpoint | Method | Description |
|---------|----------|--------|-------------|
| **OAuth** | | | |
| oauth.py | `GET /api/auth/oauth/google` | GET | Initiate Google OAuth |
| oauth.py | `POST /api/auth/oauth/google` | POST | Initiate Google OAuth |
| oauth.py | `GET /api/auth/oauth/google/callback` | GET | OAuth callback |
| oauth.py | `POST /api/auth/oauth/link` | POST | Link OAuth account |
| oauth.py | `DELETE /api/auth/oauth/unlink` | DELETE | Unlink OAuth account |
| oauth.py | `GET /api/auth/oauth/providers` | GET | List OAuth providers |
| **Slack Integration** | | | |
| slack.py | `POST /api/integrations/slack/commands` | POST | Slack slash commands |
| slack.py | `POST /api/integrations/slack/interactive` | POST | Slack interactive components |
| slack.py | `POST /api/integrations/slack/events` | POST | Slack events webhook |
| slack.py | `GET /api/integrations/slack/status` | GET | Slack integration status |
| **Broadcast/Podcast** | | | |
| broadcast.py | `POST /api/debates/{id}/broadcast` | POST | Generate podcast audio |
| broadcast.py | `POST /api/debates/{id}/broadcast/full` | POST | Full broadcast pipeline |
| broadcast.py | `GET /api/podcast/feed.xml` | GET | RSS podcast feed |
| **Gallery** | | | |
| gallery.py | `GET /api/gallery` | GET | Public debate gallery |
| gallery.py | `GET /api/gallery/{id}` | GET | Gallery entry details |
| gallery.py | `GET /api/gallery/{id}/embed` | GET | Embeddable widget |
| **Reviews** | | | |
| reviews.py | `GET /api/reviews` | GET | Code review list |
| reviews.py | `GET /api/reviews/{id}` | GET | Specific review |
| **Evolution A/B Testing** | | | |
| evolution_ab_testing.py | `GET /api/evolution/ab-tests` | GET | List A/B tests |
| evolution_ab_testing.py | `POST /api/evolution/ab-tests` | POST | Start A/B test |
| evolution_ab_testing.py | `GET /api/evolution/ab-tests/{id}` | GET | Get A/B test |
| evolution_ab_testing.py | `GET /api/evolution/ab-tests/{agent}/active` | GET | Active test for agent |
| evolution_ab_testing.py | `POST /api/evolution/ab-tests/{id}/record` | POST | Record test result |
| evolution_ab_testing.py | `POST /api/evolution/ab-tests/{id}/conclude` | POST | Conclude test |
| evolution_ab_testing.py | `DELETE /api/evolution/ab-tests/{id}` | DELETE | Cancel test |
| **Admin** | | | |
| admin.py | `POST /api/admin/gc` | POST | Trigger garbage collection |
| admin.py | `POST /api/admin/reindex` | POST | Reindex search |
| admin.py | `GET /api/admin/stats` | GET | Admin statistics |
| admin.py | `DELETE /api/admin/cache` | DELETE | Clear caches |
| **Webhooks** | | | |
| webhooks.py | `GET /api/webhooks` | GET | List webhooks |
| webhooks.py | `POST /api/webhooks` | POST | Create webhook |
| webhooks.py | `GET /api/webhooks/{id}` | GET | Get webhook |
| webhooks.py | `DELETE /api/webhooks/{id}` | DELETE | Delete webhook |
| webhooks.py | `POST /api/webhooks/{id}/test` | POST | Test webhook |
| **Plugins** | | | |
| plugins.py | `GET /api/plugins` | GET | List plugins |
| plugins.py | `POST /api/plugins` | POST | Install plugin |
| plugins.py | `GET /api/plugins/{id}` | GET | Plugin details |
| plugins.py | `DELETE /api/plugins/{id}` | DELETE | Uninstall plugin |
| plugins.py | `POST /api/plugins/{id}/enable` | POST | Enable plugin |
| plugins.py | `POST /api/plugins/{id}/disable` | POST | Disable plugin |
| **SSO** | | | |
| sso.py | `GET /api/auth/sso/metadata` | GET | SAML metadata |
| sso.py | `POST /api/auth/sso/acs` | POST | SAML assertion consumer |
| sso.py | `GET /api/auth/sso/initiate` | GET | Initiate SSO flow |
| **Moments** | | | |
| moments.py | `GET /api/moments` | GET | List notable moments |
| moments.py | `GET /api/moments/{id}` | GET | Moment details |
| **Genesis** | | | |
| genesis.py | `POST /api/genesis/agent` | POST | Create new agent |
| genesis.py | `GET /api/genesis/templates` | GET | Agent templates |
| **Relationships** | | | |
| relationship.py | `GET /api/relationships` | GET | Agent relationships |
| relationship.py | `GET /api/relationships/{agent}` | GET | Relationships for agent |
| **Introspection** | | | |
| introspection.py | `GET /api/introspection/{agent}` | GET | Agent introspection |
| introspection.py | `POST /api/introspection/{agent}/analyze` | POST | Analyze agent |
| **Laboratory** | | | |
| laboratory.py | `GET /api/lab/experiments` | GET | List experiments |
| laboratory.py | `POST /api/lab/experiments` | POST | Create experiment |
| laboratory.py | `GET /api/lab/experiments/{id}` | GET | Experiment details |
| **Learning** | | | |
| learning.py | `GET /api/learning/progress` | GET | Learning progress |
| learning.py | `GET /api/learning/curriculum` | GET | Learning curriculum |
| **Training** | | | |
| training.py | `POST /api/training/session` | POST | Start training session |
| training.py | `GET /api/training/sessions` | GET | List training sessions |
| **Metrics** | | | |
| metrics.py | `GET /api/metrics` | GET | System metrics |
| metrics.py | `GET /api/metrics/prometheus` | GET | Prometheus format |
| **Notifications** | | | |
| notifications.py | `GET /api/notifications` | GET | User notifications |
| notifications.py | `PUT /api/notifications/{id}/read` | PUT | Mark notification read |
| **Persona** | | | |
| persona.py | `GET /api/personas` | GET | List personas |
| persona.py | `POST /api/personas` | POST | Create persona |
| persona.py | `GET /api/personas/{id}` | GET | Persona details |
| **Probes** | | | |
| probes.py | `GET /api/probes/liveness` | GET | Liveness probe |
| probes.py | `GET /api/probes/readiness` | GET | Readiness probe |
| **Routing** | | | |
| routing.py | `POST /api/routing/suggest` | POST | Suggest routing |
| **Sharing** | | | |
| sharing.py | `POST /api/debates/{id}/share` | POST | Create share link |
| sharing.py | `GET /api/share/{token}` | GET | Access shared debate |
| **Social** | | | |
| social.py | `POST /api/debates/{id}/react` | POST | Add reaction |
| social.py | `GET /api/debates/{id}/reactions` | GET | Get reactions |
| social.py | `POST /api/debates/{id}/comment` | POST | Add comment |
| social.py | `GET /api/debates/{id}/comments` | GET | Get comments |
| **Audio** | | | |
| audio.py | `GET /audio/{id}.mp3` | GET | Get audio file |
| audio.py | `GET /api/audio/debates` | GET | Debates with audio |
| **Dashboard** | | | |
| dashboard.py | `GET /api/dashboard` | GET | Dashboard summary |
| dashboard.py | `GET /api/dashboard/activity` | GET | Recent activity |
| **Features** | | | |
| features.py | `GET /api/features` | GET | Feature flags |
| features.py | `PUT /api/features/{flag}` | PUT | Toggle feature |
| **Evolution** | | | |
| evolution.py | `GET /api/evolution/{agent}/history` | GET | Evolution history |
| evolution.py | `POST /api/evolution/{agent}/evolve` | POST | Trigger evolution |
| **Auditing** | | | |
| auditing.py | `GET /api/audit/events` | GET | Audit events |
| auditing.py | `GET /api/audit/events/{id}` | GET | Audit event details |
| **Critique** | | | |
| critique.py | `POST /api/debates/{id}/critique` | POST | Submit critique |
| critique.py | `GET /api/debates/{id}/critiques` | GET | Get critiques |
| **System** | | | |
| system.py | `GET /api/system/config` | GET | System configuration |
| system.py | `GET /api/system/version` | GET | System version |

---

## Orphaned SDK Methods

These SDK methods reference endpoints that don't exist or have different paths:

| SDK Class | Method | Expected Endpoint | Issue |
|-----------|--------|------------------|-------|
| `AnalyticsAPI` | `overview()` | `/api/analytics` | Backend may use different path |
| `AnalyticsAPI` | `debates()` | `/api/analytics/debates` | Backend may use different path |

---

## Recommendations for 1.0 Release

### Phase 1: Critical Gaps (Before GA)

1. **Add core debate methods:**
   - `debates.update()` -> `PATCH /api/debates/{id}`
   - `debates.verify()` -> `POST /api/debates/{id}/verify`
   - `debates.verificationReport()` -> `GET /api/debates/{id}/verification-report`
   - `debates.search()` -> `GET /api/search`

2. **Add essential agent methods:**
   - `agents.profile()` -> `GET /api/agent/{name}/profile`
   - `agents.calibration()` -> `GET /api/agent/{name}/calibration`
   - `agents.performance()` -> `GET /api/agent/{name}/performance`

3. **Complete verification API:**
   - `verification.translate()` -> `POST /api/verify/translate`

### Phase 2: Enhanced Features (Post-GA)

1. **Expand agents API with competitive analysis:**
   - Head-to-head comparisons
   - Opponent briefings
   - Consistency and flip tracking

2. **Add gauntlet management:**
   - `gauntlet.get()`, `gauntlet.delete()`, `gauntlet.personas()`
   - Heatmap and comparison views

3. **Extend analytics API:**
   - Disagreement analysis
   - Consensus quality metrics
   - Role rotation patterns

### Phase 3: Integrations (Future)

1. **OAuth integration** - if needed for web apps
2. **Slack integration** - for enterprise customers
3. **Broadcast/Podcast API** - for content generation
4. **Gallery/Reviews** - for public-facing features

---

## File References

### Backend Handlers
- `/Users/armand/Development/aragora/aragora/server/handlers/` (61+ handler modules)

### SDK Source
- `/Users/armand/Development/aragora/aragora-js/src/client.ts` (2100 lines, 23 API classes)
