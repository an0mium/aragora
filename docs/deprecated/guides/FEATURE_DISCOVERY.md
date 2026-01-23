# Aragora Feature Discovery Guide

> Deprecated: This inventory reflects an earlier UI snapshot and is no longer
> maintained. For current capability coverage, see
> [FEATURES.md](../../FEATURES.md),
> [FRONTEND_ROUTES.md](../../FRONTEND_ROUTES.md), and
> [CONTROL_PLANE.md](../../CONTROL_PLANE.md).

Complete catalog of Aragora's **100+ features** organized by use case.

**Codebase Statistics:**
- 1000+ Python modules across 50+ directories
- 34,400+ tests with exceptional coverage
- 117 debate modules with 9-round structured protocol
- 65 HTTP handlers + 15 WebSocket streams
- 85% commercially ready (see [COMMERCIAL_OVERVIEW.md](COMMERCIAL_OVERVIEW.md))

## Quick Access Reference

### Top Navigation Quick Links
These are always visible in the dashboard header:

| Link | Route | Description |
|------|-------|-------------|
| ARCHIVE | `/debates` | Historical debate archive |
| GRAPH | `/debates/graph` | Graph-based debate visualization |
| MATRIX | `/debates/matrix` | Matrix view of debates |
| AGENTS | `/agents` | Agent directory and profiles |
| NETWORK | `/network` | Agent relationship network |
| INSIGHTS | `/insights` | Extracted debate insights |
| EVIDENCE | `/evidence` | Evidence explorer |
| TRAINING | `/training` | Training data export |
| PULSE | `/pulse` | Trending topics from social |
| GAUNTLET | `/gauntlet` | Challenge suite for agents |
| RANKS | `/leaderboard` | ELO rankings leaderboard |
| ANALYTICS | `/analytics` | System analytics |
| PROBE | `/probe` | Capability probes |
| SAVES | `/checkpoints` | Debate checkpoints |
| PROOFS | `/verify` | Formal verification |
| QUALITY | `/quality` | Quality metrics |
| CALIBRATE | `/calibration` | Confidence calibration |
| MODES | `/modes` | Operational modes |
| COMPARE | `/compare` | Agent comparison |
| CRUX | `/crux` | Crux point analysis |
| REDTEAM | `/red-team` | Red team analysis |
| MEM | `/memory-analytics` | Memory analytics |
| HOOKS | `/webhooks` | Webhook management |
| ADMIN | `/admin` | Admin panel |
| DEV | `/developer` | Developer console |

---

## Features by Category

### 1. Core Debate Features (Always Visible)

| Feature | Component | Description |
|---------|-----------|-------------|
| Document Upload | `DocumentUpload` | Upload context documents for debates |
| User Participation | `UserParticipation` | Vote on positions, suggest improvements |
| Trickster Detection | `TricksterAlertPanel` | Detect hollow consensus patterns |
| Rhetorical Analysis | `RhetoricalObserverPanel` | Analyze argumentation quality |
| Citations | `CitationsPanel` | Review evidence sources |
| History | `HistoryPanel` | Debate history tracking |

### 2. Browse & Discovery Features

| Feature | Component | Description |
|---------|-----------|-------------|
| Trending Topics | `TrendingTopicsPanel` | Live pulse from social platforms |
| Debate List | `DebateListPanel` | Browse all debates |
| Debate Browser | `DebateBrowser` | Search and filter past debates |
| Replay Browser | `ReplayBrowser` | Replay debate recordings |

### 3. Agent Analysis Features

| Feature | Component | Description |
|---------|-----------|-------------|
| Agent Compare | `AgentComparePanel` | Side-by-side agent comparison |
| Agent Network | `AgentNetworkPanel` | Relationship graph visualization |
| Mood Tracker | `MoodTrackerPanel` | Agent sentiment over time |
| Leaderboard | `LeaderboardPanel` | ELO rankings with history |
| Calibration | `CalibrationPanel` | Confidence calibration tracking |
| Tournament | `TournamentPanel` | Run agent tournaments |
| Tournament Viewer | `TournamentViewerPanel` | View tournament results |

### 4. Insights & Learning Features

| Feature | Component | Description |
|---------|-----------|-------------|
| Moments Timeline | `MomentsTimeline` | Key debate moments |
| Uncertainty | `UncertaintyPanel` | Confidence analysis |
| Evidence Visualizer | `EvidenceVisualizerPanel` | Evidence network graph |
| Consensus Quality | `ConsensusQualityDashboard` | Consensus metrics |
| Learning Dashboard | `LearningDashboard` | Cross-cycle improvements |
| Insights | `InsightsPanel` | Pattern extraction |
| Crux Analysis | `CruxPanel` | Identify debate cruxes |
| Contrary Views | `ContraryViewsPanel` | Alternative perspectives |
| Risk Warnings | `RiskWarningsPanel` | Risk assessment |
| Evolution | `EvolutionPanel` | Prompt evolution tracking |
| Evidence Panel | `EvidencePanel` | Evidence for current debate |

### 5. System Tools (Section 5 - Collapsed by Default)

| Feature | Component | Description |
|---------|-----------|-------------|
| Capability Probe | `CapabilityProbePanel` | Test agent capabilities |
| Operational Modes | `OperationalModesPanel` | Configure debate modes |
| Red Team Analysis | `RedTeamAnalysisPanel` | Adversarial analysis |
| Gauntlet | `GauntletPanel` | Gauntlet test results |
| Reviews | `ReviewsPanel` | Code review interface |
| Plugin Marketplace | `PluginMarketplacePanel` | Browse plugins |
| Laboratory | `LaboratoryPanel` | Experimental features |
| Breakpoints | `BreakpointsPanel` | Debug checkpoints |
| Batch Debate | `BatchDebatePanel` | Run multiple debates |
| Pulse Scheduler | `PulseSchedulerControlPanel` | Control pulse scheduler |
| Broadcast | `BroadcastPanel` | Share debates |

### 6. Analysis & Exploration (Section 6 - Collapsed)

| Feature | Component | Description |
|---------|-----------|-------------|
| Lineage Browser | `LineageBrowser` | Debate genealogy |
| Influence Graph | `InfluenceGraph` | Agent influence tracking |
| Evolution Timeline | `EvolutionTimeline` | Agent evolution history |
| Genesis Explorer | `GenesisExplorer` | First debate analysis |
| Public Gallery | `PublicGallery` | Public debate showcase |
| Graph Debate Browser | `GraphDebateBrowser` | Graph-based exploration |
| Scenario Matrix | `ScenarioMatrixView` | Scenario analysis |
| Tournament Bracket | `TournamentBracket` | Tournament visualization |
| Proof Tree | `ProofTreeVisualization` | Formal proof visualization |
| A/B Test Results | `ABTestResultsPanel` | A/B test results |
| Gauntlet Runner | `GauntletRunner` | Run gauntlet challenges |
| Training Export | `TrainingExportPanel` | Export training data |
| Token Stream | `TokenStreamViewer` | Token-level debugging |

### 7. Advanced / Debug (Section 7 - Collapsed)

| Feature | Component | Description |
|---------|-----------|-------------|
| Deep Analytics | `AnalyticsPanel` | Advanced analytics |
| Server Metrics | `MetricsPanel` | Server performance metrics |
| Knowledge Base | `ConsensusKnowledgeBase` | Knowledge base browser |
| Memory Inspector | `MemoryInspector` | Memory inspection |
| Memory Explorer | `MemoryExplorerPanel` | Memory exploration |
| Memory Analytics | `MemoryAnalyticsPanel` | Memory statistics |
| Impasse Detection | `ImpasseDetectionPanel` | Deadlock detection |
| API Explorer | `ApiExplorerPanel` | REST API explorer |
| Checkpoint Panel | `CheckpointPanel` | Checkpoint management |
| Proof Visualizer | `ProofVisualizerPanel` | Proof visualization |
| Settings | `SettingsPanel` | Application settings |

### 8. Enterprise Features (Production-Ready)

| Feature | Component/Location | Status |
|---------|-------------------|--------|
| **Authentication** | | |
| OIDC SSO | `aragora/auth/oidc.py` | Ready |
| SAML SSO | `aragora/auth/saml.py` | Ready |
| MFA (TOTP/HOTP) | `aragora/auth/mfa.py` | Ready |
| API Key Management | `aragora/auth/api_keys.py` | Ready |
| Session Management | `aragora/auth/session.py` | Ready |
| Account Lockout | `aragora/auth/lockout.py` | Ready |
| **Multi-Tenancy** | | |
| Tenant Isolation | `aragora/tenancy/isolation.py` | Ready |
| Resource Quotas | `aragora/tenancy/quotas.py` | Ready |
| Usage Metering | `aragora/billing/metering.py` | Ready |
| **Security** | | |
| Encryption (AES-256-GCM) | `aragora/security/encryption.py` | Ready |
| Input Validation | `aragora/server/validation/` | Ready |
| Rate Limiting | `aragora/server/middleware/rate_limiter.py` | Ready |
| Circuit Breaker | `aragora/resilience.py` | Ready |
| **Compliance** | | |
| Audit Trail | `aragora/audit/` | Ready |
| SOC 2 Documentation | `docs/COMPLIANCE.md` | Ready |
| GDPR Support | `aragora/privacy/`, `docs/GDPR.md` | Ready |
| Data Classification | `docs/DATA_CLASSIFICATION.md` | Ready |
| **Observability** | | |
| Prometheus Metrics | `aragora/observability/metrics.py` | Ready |
| Grafana Dashboards | `deploy/grafana/` | Ready |
| OpenTelemetry Tracing | `aragora/observability/tracing.py` | Ready |
| Structured Logging | `aragora/observability/logging.py` | Ready |
| SLO Framework | `aragora/observability/slo.py` | Ready |
| **Connectors (24+)** | | |
| Slack | `aragora/connectors/chat/slack.py` | Ready |
| Discord | `aragora/connectors/chat/discord.py` | Ready |
| Teams | `aragora/connectors/chat/teams.py` | Ready |
| GitHub | `aragora/connectors/enterprise/git/github.py` | Ready |
| SharePoint/Confluence | `aragora/connectors/enterprise/` | Ready |

See [ENTERPRISE_FEATURES.md](ENTERPRISE_FEATURES.md) for complete enterprise documentation.

---

## Feature Flags

Some features require backend configuration. Check if enabled via the Features API:

| Feature ID | Components | Default |
|------------|------------|---------|
| `pulse` | TrendingTopicsPanel, PulseSchedulerControlPanel | Optional |
| `calibration` | CalibrationPanel | Optional |
| `tournaments` | TournamentPanel, TournamentViewerPanel | Optional |
| `evolution` | EvolutionPanel | Optional |
| `plugins` | PluginMarketplacePanel | Optional |
| `laboratory` | LaboratoryPanel, ABTestResultsPanel | Optional |
| `memory` | MemoryInspector, MemoryExplorerPanel | Optional |

Enable features via environment variables or API:
```bash
# Enable all features
export ARAGORA_FEATURES=pulse,calibration,tournaments,evolution,plugins,laboratory,memory
```

---

## REST API Endpoints (275+ endpoints via 65 handlers)

### Core Debate API
```
POST   /api/debate              - Start new debate
GET    /api/debate/{id}         - Get debate details
POST   /api/debate/{id}/vote    - Cast vote
DELETE /api/debate/{id}         - Cancel debate
```

### Agent API
```
GET    /api/agents              - List all agents
GET    /api/agents/{name}       - Agent details
GET    /api/agents/{name}/stats - Agent statistics
```

### Memory API
```
GET    /api/memory              - Memory overview
GET    /api/memory/continuum    - Continuum memory
POST   /api/memory/store        - Store memory
```

### Pulse API
```
GET    /api/pulse/trending      - Get trending topics
POST   /api/pulse/schedule      - Schedule debate from topic
GET    /api/pulse/stats         - Pulse statistics
```

### Evidence API
```
GET    /api/evidence            - List evidence
POST   /api/evidence/search     - Search evidence
GET    /api/evidence/{id}       - Get evidence details
```

### Analytics API
```
GET    /api/analytics           - System analytics
GET    /api/analytics/debates   - Debate analytics
GET    /api/analytics/agents    - Agent analytics
```

Full API documentation: See `/api/docs` or `docs/API_REFERENCE.md`

---

## WebSocket Events

Real-time events via WebSocket at `ws://host/ws`:

| Event | Description |
|-------|-------------|
| `debate_start` | Debate started |
| `round_start` | New round began |
| `agent_message` | Agent response |
| `critique` | Critique submitted |
| `vote` | Vote cast |
| `consensus` | Consensus reached |
| `debate_end` | Debate completed |
| `uncertainty_analysis` | Uncertainty metrics available |
| `grounded_verdict` | Grounded verdict generated |

---

## Dashboard Modes

Toggle between modes using the mode selector:

| Mode | Description |
|------|-------------|
| **Focus** | Minimal UI - core debate only |
| **Explorer** | Full UI - all panels visible |
| **Deep Audit** | Detailed view with token streams |

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+/` | Toggle command palette |
| `Ctrl+K` | Quick search |
| `Escape` | Close modal |

---

## Getting Started

1. **First Time Users**: The OnboardingWizard will guide you through:
   - Architecture Review
   - Security Review
   - Compliance Check
   - Exploration Mode

2. **Power Users**: Expand collapsed sections (6 & 7) for advanced tools

3. **Developers**: Use `/developer` for API exploration and debugging

---

## Enabling Hidden Features

To expand collapsed sections in the dashboard:

1. Click the section header (e.g., "ANALYSIS & EXPLORATION")
2. The section expands to show all panels
3. Your preference is saved automatically

To enable feature-flagged components:

1. Check `/api/features` for available features
2. Enable via environment variable or admin panel
3. Refresh the dashboard

---

## Support

- Documentation: `/docs`
- API Reference: `/api/docs`
- GitHub Issues: https://github.com/an0mium/aragora/issues
