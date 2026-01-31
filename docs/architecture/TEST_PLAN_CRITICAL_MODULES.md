# Test Plan for Critical Untested Modules

**Document Version:** 1.0
**Created:** 2026-01-30
**Status:** Draft

This document outlines comprehensive test plans for the 20 most critical untested files in the Aragora codebase, prioritized by risk (business logic complexity + line count).

---

## Summary

| Priority | Module Category | Files | Est. Total Tests |
|----------|----------------|-------|-----------------|
| P0 | Debate Module | 5 | ~450 |
| P1 | Knowledge Module | 5 | ~380 |
| P1 | Agents Module | 3 | ~200 |
| P2 | Server Module | 3 | ~280 |
| P2 | Connectors Module | 4 | ~240 |
| **Total** | | **20** | **~1,550** |

---

## P0: Debate Module (Highest Risk - Decision Quality)

### 1. `aragora/debate/judge_selector.py` (1,364 lines)

**Module Purpose:** Provides strategies for selecting judges to evaluate debate outcomes, including single-judge strategies (ELO-ranked, calibrated, crux-aware) and multi-judge panel systems with voting aggregation.

**Key Classes/Functions to Test:**
- `JudgeSelector` - Main selection coordinator with circuit breaker integration
- `JudgeScoringMixin` - ELO and calibration score computation
- `JudgeSelectionStrategy` (and implementations):
  - `LastAgentStrategy`, `RandomStrategy`
  - `EloRankedStrategy` - Highest ELO selection
  - `CalibratedStrategy` - Composite ELO + calibration scoring
  - `CruxAwareStrategy` - Historical dissenter preference
  - `VotedStrategy` - Agent voting on judge
- `JudgePanel` - Multi-judge coordination
  - `record_vote()`, `get_result()`, `deliberate_and_vote()`
- `JudgingStrategy` enum (MAJORITY, SUPERMAJORITY, UNANIMOUS, WEIGHTED)
- `create_judge_panel()` - Factory function

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Unit | ~40 | Individual strategy selection, scoring calculations |
| Integration | ~25 | Strategy + ELO system, Strategy + CircuitBreaker |
| Edge Cases | ~20 | Empty agent lists, all agents unavailable, tie-breaking |
| Panel Voting | ~25 | All voting strategies, deliberation rounds |

**Estimated Test Count:** ~110

**Priority Level:** P0
**Risk Assessment:** Judge selection directly affects debate fairness. Biased or incorrect selection can invalidate entire debate outcomes.

---

### 2. `aragora/debate/team_selector.py` (1,113 lines)

**Module Purpose:** Selects and scores agents for debate participation using ELO ratings, calibration, domain capabilities, circuit breaker filtering, and Agent CV profiles.

**Key Classes/Functions to Test:**
- `TeamSelector` - Main selection coordinator
  - `select()` - Core selection method
  - `_filter_available()` - Circuit breaker filtering
  - `_filter_by_domain_capability()` - Domain-based filtering
  - `_filter_by_hierarchy_role()` - Gastown role filtering
  - `_compute_score()` - Multi-factor scoring
  - `_get_agent_cvs_batch()` - CV-based scoring
- `TeamSelectionConfig` - Configuration dataclass
- `DOMAIN_CAPABILITY_MAP` - Domain-to-agent mapping
- `AgentScorer`, `CalibrationScorer` protocols

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Unit | ~35 | Score computation, domain filtering, config validation |
| Integration | ~20 | ELO + Calibration + CV combined scoring |
| Circuit Breaker | ~15 | Availability filtering, fallback behavior |
| Domain Filtering | ~20 | All domain mappings, custom maps, fallback |
| Hierarchy | ~10 | Role-based filtering, Gastown integration |

**Estimated Test Count:** ~100

**Priority Level:** P0
**Risk Assessment:** Team composition directly affects debate quality. Incorrect filtering can exclude critical perspectives.

---

### 3. `aragora/debate/autonomic_executor.py` (1,016 lines)

**Module Purpose:** Provides error handling and timeout management for agent operations, implementing the "autonomic layer" that keeps debates running even when individual agents fail.

**Key Classes/Functions to Test:**
- `AutonomicExecutor` - Core execution wrapper
  - `generate()` - Safe generation with error handling
  - `generate_with_fallback()` - Fallback agent substitution
  - `critique()` - Safe critique generation
  - `vote()` - Safe voting
  - `with_timeout()` - Per-agent timeout management
  - `get_escalated_timeout()` - Timeout escalation logic
- `StreamingContentBuffer` - Partial response capture
  - `append()`, `get_partial()`, `clear()`
- Telemetry emission (`_emit_agent_telemetry()`)
- Wisdom fallback (`_get_wisdom_fallback()`)

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Unit | ~30 | Timeout calculation, buffer operations, error classification |
| Timeout | ~20 | Escalation, max timeout, per-agent tracking |
| Error Handling | ~25 | All exception types, fallback behavior |
| Streaming | ~15 | Partial content capture, buffer limits |
| Integration | ~10 | With CircuitBreaker, with PerformanceMonitor |

**Estimated Test Count:** ~100

**Priority Level:** P0
**Risk Assessment:** Failures here can crash entire debates. Proper error isolation is critical for production reliability.

---

### 4. `aragora/debate/cognitive_limiter_rlm.py` (942 lines)

**Module Purpose:** Extends the base CognitiveLoadLimiter with RLM (Recursive Language Model) support for processing arbitrarily long debate contexts via REPL-based programmatic access.

**Key Classes/Functions to Test:**
- `RLMCognitiveLoadLimiter` - Main RLM-enhanced limiter
  - `query_with_rlm()` - REPL-based context query
  - `compress_context_async()` - Hierarchical compression
  - `_format_messages_for_rlm()` - Message formatting
  - `_fallback_search()` - Keyword fallback
  - `for_stress_level()` - Factory method
- `RLMCognitiveBudget` - Extended budget config
- `CompressedContext` - Compression result dataclass
- RLM factory integration (`get_rlm()`, `get_compressor()`)

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Unit | ~25 | Budget calculations, message formatting |
| Compression | ~20 | Threshold detection, ratio tracking |
| RLM Integration | ~15 | With/without official RLM library |
| Fallback | ~15 | Keyword search, summarization fallback |
| Stress Levels | ~5 | All stress level configurations |

**Estimated Test Count:** ~80

**Priority Level:** P0
**Risk Assessment:** Context management affects agent response quality. Token exhaustion or poor compression degrades debate outcomes.

---

### 5. `aragora/debate/ml_integration.py` (820 lines)

**Module Purpose:** Bridges the ML module with the debate orchestrator, providing ML-powered agent selection, quality gates, and consensus estimation.

**Key Classes/Functions to Test:**
- `MLDelegationStrategy` - ML-powered agent routing
  - `select_agents()` - ML-based selection
  - `score_agent()` - Individual agent scoring
  - `_reorder_agents()` - Order by ML scores
- `QualityGate` - Response quality filtering
  - `score_response()` - Quality scoring
  - `passes_gate()` - Threshold check
  - `filter_responses()`, `filter_messages()` - Batch filtering
- `ConsensusEstimator` - Early termination detection
  - `estimate_consensus()` - Consensus prediction
  - `should_terminate_early()` - Termination decision
- `MLIntegrationConfig` - Configuration
- `create_ml_team_selector()` - Factory function

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Unit | ~25 | Scoring, configuration, caching |
| ML Routing | ~20 | Agent selection, fallback behavior |
| Quality Gates | ~20 | Filtering, threshold logic |
| Consensus | ~10 | Estimation, early termination |
| Cache | ~5 | TTL, invalidation |

**Estimated Test Count:** ~80

**Priority Level:** P0
**Risk Assessment:** ML-based decisions affect team composition and quality filtering. Errors can bias debates systematically.

---

## P1: Knowledge Module (High Risk - Data Integrity)

### 6. `aragora/knowledge/mound/validation.py` (511 lines)

**Module Purpose:** Input validation and production hardening for Knowledge Mound, providing configurable limits, standardized error responses, and resource enforcement.

**Key Classes/Functions to Test:**
- Validators:
  - `validate_content()` - Content size validation
  - `validate_id()` - ID format validation
  - `validate_workspace_id()` - Workspace ID format
  - `validate_topics()` - Topic list validation
  - `validate_metadata()` - Metadata size/format
  - `validate_query()` - Query string validation
  - `validate_graph_params()` - Depth/node limits
  - `validate_pagination()` - Limit/offset validation
- Error classes: `ValidationError`, `ContentTooLargeError`, `InvalidIdError`, `ResourceLimitExceededError`, `NotFoundError`, `AccessDeniedError`
- `ValidationLimits` - Configuration dataclass
- `BoundedList` - Thread-safe FIFO list
- `ConcurrencyLimiter` - Semaphore wrapper

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Unit | ~40 | All validators with valid/invalid inputs |
| Limits | ~15 | Default limits, custom limits |
| Error Types | ~15 | All error classes, HTTP status mapping |
| Thread Safety | ~10 | BoundedList, ConcurrencyLimiter |

**Estimated Test Count:** ~80

**Priority Level:** P1
**Risk Assessment:** Validation failures can allow malformed data, causing downstream corruption or injection attacks.

---

### 7. `aragora/knowledge/mound/quality.py` (329 lines)

**Module Purpose:** Automated quality scoring for knowledge items enabling tier-based auto-curation and lifecycle management across freshness, confidence, usage, relevance, and relationships.

**Key Classes/Functions to Test:**
- `QualityScorer` - Main scoring class
  - `score()` - Compute quality score
  - `_score_freshness()` - Age-based scoring
  - `_score_confidence()` - Source reliability scoring
  - `_score_usage()` - Access/citation scoring
  - `_score_relevance()` - Query match scoring
  - `_score_relationships()` - Graph connectivity scoring
  - `assign_tier()` - Tier assignment
  - `batch_score()` - Bulk scoring
- `QualityScore` - Score breakdown dataclass
- `QualityWeights` - Component weights
- `TierThresholds` - Tier boundaries
- Module functions: `get_scorer()`, `score_item()`, `assign_tier()`

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Unit | ~30 | Each scoring dimension, weight combinations |
| Tier Assignment | ~10 | All tier boundaries, edge cases |
| Batch | ~5 | Bulk scoring performance |
| Weights | ~10 | Custom weights, normalization |

**Estimated Test Count:** ~55

**Priority Level:** P1
**Risk Assessment:** Incorrect quality scoring affects knowledge retention and retrieval. Poor scoring leads to data loss or stale knowledge surfacing.

---

### 8. `aragora/knowledge/mound/adapters/belief_adapter.py` (1,383 lines)

**Module Purpose:** Bridges the Belief Network to the Knowledge Mound, enabling bidirectional integration for claims, propositions, and belief state persistence.

**Key Classes/Functions to Test:**
- `BeliefAdapter` - Main adapter class
  - `store_claim()`, `store_claims_batch()` - Claim storage
  - `store_proposition()` - Proposition storage
  - `store_belief_state()` - Full state persistence
  - `get_claim()`, `get_claims_by_topic()` - Retrieval
  - `search_claims()` - Semantic search
  - `get_agent_belief_history()` - Agent belief tracking
  - `update_claim_confidence()` - Confidence updates
  - `to_knowledge_item()` - KM conversion
- Reverse flow methods:
  - `validate_claim_from_km()` - KM-based validation
  - `sync_validations_from_km()` - Batch sync

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Unit | ~35 | Store/retrieve operations, conversions |
| Batch | ~15 | Bulk operations, indexing |
| Search | ~15 | Topic matching, semantic search |
| Reverse Flow | ~15 | KM validation, sync |

**Estimated Test Count:** ~80

**Priority Level:** P1
**Risk Assessment:** Belief state is core to agent reasoning. Data corruption here undermines decision provenance.

---

### 9. `aragora/knowledge/mound/adapters/insights_adapter.py` (1,276 lines)

**Module Purpose:** Bridges InsightStore and FlipDetector to the Knowledge Mound, enabling persistence of insights, flip events, and pattern clusters with bidirectional sync.

**Key Classes/Functions to Test:**
- `InsightsAdapter` - Main adapter class
  - `store_insight()`, `store_debate_insights()` - Insight storage
  - `store_flip()`, `store_flips_batch()` - Flip event storage
  - `store_pattern()` - Pattern cluster storage
  - `search_similar_insights()` - Relevance search
  - `get_agent_flip_history()` - Flip tracking
  - `to_knowledge_item()`, `flip_to_knowledge_item()` - Conversions
- Reverse flow:
  - `update_flip_thresholds_from_km()` - Threshold tuning
  - `get_agent_flip_baselines()` - Baseline computation
  - `validate_flip_from_km()` - Flip validation
  - `sync_validations_from_km()` - Batch sync
- Result dataclasses: `KMFlipThresholdUpdate`, `KMAgentFlipBaseline`, `KMFlipValidation`

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Unit | ~30 | Store/retrieve, confidence filtering |
| Search | ~15 | Similar insights, type filtering |
| Flip Tracking | ~20 | Agent history, domain flips |
| Reverse Flow | ~15 | Threshold updates, validation |

**Estimated Test Count:** ~80

**Priority Level:** P1
**Risk Assessment:** Insights drive meta-learning. Flip detection is critical for agent consistency tracking.

---

### 10. `aragora/knowledge/mound/governance.py` (estimated 1,020 lines)

**Module Purpose:** Knowledge governance including access control, retention policies, and audit trails for the Knowledge Mound.

**Key Classes/Functions to Test:**
- Governance policies and enforcement
- Access control checks
- Retention policy application
- Audit trail generation
- Compliance verification

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Unit | ~30 | Policy evaluation, access checks |
| Retention | ~20 | Policy enforcement, expiration |
| Audit | ~15 | Trail generation, integrity |
| RBAC | ~20 | Permission checks, role hierarchy |

**Estimated Test Count:** ~85

**Priority Level:** P1
**Risk Assessment:** Governance failures expose sensitive data or violate compliance requirements.

---

## P1: Agents Module (Medium-High Risk)

### 11. `aragora/agents/personas.py` (1,725 lines)

**Module Purpose:** Agent personas with evolving specialization, providing defined personality traits, expertise areas, and performance-based learning.

**Key Classes/Functions to Test:**
- `Persona` - Persona dataclass
  - `top_expertise`, `trait_string`, `to_prompt_context()`
  - `generation_params` - Temperature, top_p, frequency_penalty
- `PersonaManager` (SQLiteStore subclass)
  - `get_persona()`, `create_persona()` - CRUD
  - `record_performance()` - Performance tracking
  - `_update_expertise()` - Expertise evolution
  - `infer_traits()` - Trait inference from behavior
  - `get_all_personas()`, `get_performance_summary()`
- `DEFAULT_PERSONAS` - Built-in persona definitions
- Constants: `EXPERTISE_DOMAINS`, `PERSONALITY_TRAITS`
- Helpers: `get_or_create_persona()`, `apply_persona_to_agent()`, `get_persona_prompt()`

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Unit | ~30 | Persona dataclass, trait validation |
| Database | ~25 | SQLite operations, migrations |
| Learning | ~20 | Performance tracking, expertise evolution |
| Inference | ~10 | Trait inference from patterns |
| Prompts | ~10 | Prompt generation, context building |

**Estimated Test Count:** ~95

**Priority Level:** P1
**Risk Assessment:** Personas influence agent behavior and response diversity. Incorrect expertise tracking skews agent selection.

---

### 12. `aragora/agents/api_agents/openrouter.py` (1,004 lines)

**Module Purpose:** OpenRouter API integration providing unified access to models like DeepSeek, Llama, Mistral via an OpenAI-compatible API with rate limiting, retry, and fallback.

**Key Classes/Functions to Test:**
- `OpenRouterAgent` - Main agent class
  - `generate()` - Response generation with retry
  - `_generate_with_model()` - Model-specific generation with fallback
  - `generate_stream()` - SSE streaming with retry
  - `_build_context_prompt()` - Context formatting
  - `_record_token_usage()` - Usage tracking
- Rate limiter integration (`get_openrouter_limiter()`)
- `OPENROUTER_FALLBACK_MODELS` - Fallback chain mapping
- Model-specific subclasses (DeepSeek, Llama, etc.)

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Unit | ~20 | Prompt building, token recording |
| Rate Limiting | ~20 | Limiter integration, backoff |
| Retry | ~15 | Retry logic, exponential backoff |
| Fallback | ~15 | Model fallback chain |
| Streaming | ~10 | SSE parsing, partial responses |

**Estimated Test Count:** ~80

**Priority Level:** P1
**Risk Assessment:** OpenRouter is a key fallback for primary API failures. Failures here break resilience.

---

### 13. `aragora/agents/rate_limiter.py` (726 lines)

**Module Purpose:** Rate limiting logic for API agents, providing per-provider limits, backoff strategies, and header-based rate tracking.

**Key Classes/Functions to Test:**
- Rate limiter classes for each provider
- `acquire()`, `release()` - Token acquisition
- `update_from_headers()` - Header parsing
- `record_rate_limit_error()` - Error handling with backoff
- `record_success()` - Success state tracking
- Backoff calculation with jitter
- Factory functions (`get_openrouter_limiter()`, etc.)

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Unit | ~25 | Token management, backoff calculation |
| Headers | ~15 | Rate limit header parsing |
| Concurrent | ~15 | Thread-safe acquisition |
| Backoff | ~10 | Exponential backoff, jitter |

**Estimated Test Count:** ~65

**Priority Level:** P1
**Risk Assessment:** Rate limiting failures cause API bans or quota exhaustion, breaking all agent operations.

---

## P2: Server Module (Medium Risk)

### 14. `aragora/server/handlers/compliance_handler.py` (1,765 lines)

**Module Purpose:** REST API endpoints for compliance and audit operations including SOC 2 reports, GDPR exports, right-to-be-forgotten, and SIEM integration.

**Key Classes/Functions to Test:**
- `ComplianceHandler` - Main handler class
  - `_get_status()` - Overall compliance status
  - `_get_soc2_report()` - SOC 2 report generation
  - `_gdpr_export()` - User data export
  - `_right_to_be_forgotten()` - GDPR erasure workflow
  - `_verify_audit()` - Audit trail verification
  - `_get_audit_events()` - SIEM event export
  - `_coordinated_deletion()` - Backup-aware deletion
- Control evaluation (`_evaluate_controls()`)
- Rate limiting and permission checks

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Unit | ~30 | Report generation, status computation |
| GDPR | ~25 | Export, deletion, legal holds |
| Audit | ~20 | Trail verification, event export |
| RBAC | ~15 | Permission enforcement |
| Rate Limits | ~5 | Endpoint throttling |

**Estimated Test Count:** ~95

**Priority Level:** P2
**Risk Assessment:** Compliance failures expose legal liability. Incorrect GDPR handling violates regulations.

---

### 15. `aragora/server/handlers/control_plane.py` (1,726 lines)

**Module Purpose:** REST API endpoints for the enterprise control plane including agent registration, task scheduling, health monitoring, and policy violations.

**Key Classes/Functions to Test:**
- `ControlPlaneHandler` - Main handler class
  - Agent CRUD: `_handle_list_agents()`, `_handle_register_agent()`, `_handle_get_agent()`, `_handle_unregister_agent()`
  - Task management: `_handle_submit_task()`, `_handle_get_task()`, `_handle_complete_task()`, `_handle_fail_task()`
  - Health: `_handle_system_health()`, `_handle_detailed_health()`, `_handle_agent_health()`
  - Metrics: `_handle_get_metrics()`, `_handle_get_queue()`
  - Policy violations: `_handle_list_policy_violations()`, `_handle_get_policy_violation()`
- Event emission (`_emit_event()`)
- Path normalization

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Agent CRUD | ~25 | Registration, heartbeat, unregister |
| Task Management | ~25 | Submit, claim, complete, fail |
| Health | ~20 | System/agent health, breakers |
| Metrics | ~15 | Queue metrics, dashboard |
| Permissions | ~10 | RBAC enforcement |

**Estimated Test Count:** ~95

**Priority Level:** P2
**Risk Assessment:** Control plane is the orchestration layer. Failures break multi-agent coordination.

---

### 16. `aragora/server/handlers/workflows.py` (1,784 lines)

**Module Purpose:** HTTP API handlers for the Visual Workflow Builder providing CRUD, execution control, versioning, and template management.

**Key Classes/Functions to Test:**
- Workflow CRUD: `list_workflows()`, `get_workflow()`, `create_workflow()`, `update_workflow()`, `delete_workflow()`
- Execution: `execute_workflow()`, `get_execution()`, `terminate_execution()`
- Versioning: `get_versions()`, `restore_version()`
- Templates: `list_templates()`, `get_template()`, `create_from_template()`
- Approvals: `list_pending_approvals()`, `approve_step()`, `reject_step()`
- Helper functions: `_step_result_to_dict()`, `_get_store()`

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| CRUD | ~25 | Create, read, update, delete |
| Execution | ~25 | Run, pause, resume, terminate |
| Versioning | ~15 | History, restore, rollback |
| Templates | ~15 | Gallery, instantiation |
| Approvals | ~10 | Human-in-the-loop |

**Estimated Test Count:** ~90

**Priority Level:** P2
**Risk Assessment:** Workflows automate business processes. Execution failures break automation.

---

## P2: Connectors Module (Lower Risk - External Integrations)

### 17. `aragora/connectors/chat/telegram.py` (1,695 lines)

**Module Purpose:** Telegram Bot API connector implementing ChatPlatformConnector with circuit breaker protection for fault tolerance.

**Key Classes/Functions to Test:**
- `TelegramConnector` - Main connector class
  - `_telegram_api_request()` - Resilient API calls
  - `send_message()` - Message sending with formatting
  - `send_file()`, `send_voice()` - Media sending
  - `process_webhook()` - Webhook event handling
  - `verify_webhook()` - Signature verification
  - `get_updates()` - Long-polling
- Error classification (`_classify_telegram_error()`)
- Message parsing and conversion

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Unit | ~25 | Message building, error classification |
| API Calls | ~20 | Send/receive, retry logic |
| Webhook | ~15 | Event parsing, verification |
| Circuit Breaker | ~10 | Failure tracking, recovery |
| Media | ~10 | File/voice handling |

**Estimated Test Count:** ~80

**Priority Level:** P2
**Risk Assessment:** Chat integrations are customer-facing. Failures break user communication.

---

### 18. `aragora/connectors/accounting/qbo.py` (estimated 1,519 lines)

**Module Purpose:** QuickBooks Online connector for financial data integration including invoices, payments, and account synchronization.

**Key Classes/Functions to Test:**
- OAuth2 authentication flow
- Invoice CRUD operations
- Payment processing
- Account synchronization
- Rate limiting and retry
- Error handling

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Auth | ~15 | OAuth2 flow, token refresh |
| Invoices | ~15 | CRUD operations |
| Payments | ~15 | Processing, reconciliation |
| Sync | ~10 | Account synchronization |
| Errors | ~10 | API error handling |

**Estimated Test Count:** ~65

**Priority Level:** P2
**Risk Assessment:** Financial integrations handle sensitive data. Errors affect billing accuracy.

---

### 19. `aragora/connectors/chat/whatsapp.py` (1,191 lines)

**Module Purpose:** WhatsApp Business API connector using Meta Cloud API with circuit breaker protection for resilient message delivery.

**Key Classes/Functions to Test:**
- `WhatsAppConnector` - Main connector class
  - `_whatsapp_api_request()` - Resilient API calls
  - `send_message()` - Text messages
  - `send_interactive()` - Buttons, lists
  - `send_template()` - Message templates
  - `send_media()` - Images, documents, audio
  - `process_webhook()` - Webhook handling
  - `verify_webhook()` - Meta signature verification
- Rate limit handling (codes 4, 80007, 130429)

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| Unit | ~20 | Message building, template formatting |
| API Calls | ~20 | Send/receive, rate limits |
| Webhook | ~15 | Event parsing, verification |
| Interactive | ~10 | Buttons, lists, replies |
| Media | ~10 | File uploads, attachments |

**Estimated Test Count:** ~75

**Priority Level:** P2
**Risk Assessment:** WhatsApp is a primary customer channel. Delivery failures affect user experience.

---

### 20. `aragora/connectors/ecommerce/shopify.py` (estimated 1,147 lines)

**Module Purpose:** Shopify connector for e-commerce integration including orders, products, customers, and inventory synchronization.

**Key Classes/Functions to Test:**
- GraphQL API integration
- Order management
- Product CRUD
- Customer synchronization
- Inventory updates
- Webhook processing

**Test Categories:**

| Category | Tests | Description |
|----------|-------|-------------|
| GraphQL | ~15 | Query building, mutations |
| Orders | ~15 | Create, update, fulfill |
| Products | ~10 | CRUD, variants |
| Customers | ~10 | Sync, segmentation |
| Webhooks | ~10 | Event processing |

**Estimated Test Count:** ~60

**Priority Level:** P2
**Risk Assessment:** E-commerce integrations affect order processing and inventory accuracy.

---

## Test Infrastructure Requirements

### Fixtures Needed
- Mock ELO systems with configurable ratings
- Mock agents with various capabilities
- Mock circuit breakers in different states
- Mock API responses for all external services
- Database fixtures for SQLite stores
- Knowledge Mound test instances

### Test Utilities
- Async test helpers for coroutine testing
- Timeout simulation utilities
- Rate limit simulation
- Webhook payload generators
- API response factories

### Coverage Targets
| Module | Current | Target |
|--------|---------|--------|
| Debate | ~40% | 85% |
| Knowledge | ~55% | 80% |
| Agents | ~45% | 75% |
| Server | ~50% | 75% |
| Connectors | ~35% | 70% |

---

## Implementation Priority

### Phase 1 (P0 - Week 1-2)
1. `judge_selector.py` - Critical for debate fairness
2. `team_selector.py` - Critical for team composition
3. `autonomic_executor.py` - Critical for stability

### Phase 2 (P0/P1 - Week 3-4)
4. `cognitive_limiter_rlm.py` - Context management
5. `ml_integration.py` - ML-based decisions
6. `validation.py` - Data integrity

### Phase 3 (P1 - Week 5-6)
7. `quality.py` - Knowledge lifecycle
8. `belief_adapter.py` - Reasoning provenance
9. `insights_adapter.py` - Meta-learning
10. `personas.py` - Agent behavior

### Phase 4 (P1/P2 - Week 7-8)
11. `openrouter.py` - API resilience
12. `rate_limiter.py` - Quota management
13. `compliance_handler.py` - Legal requirements
14. `control_plane.py` - Orchestration

### Phase 5 (P2 - Week 9-10)
15. `workflows.py` - Automation
16. `governance.py` - Access control
17. `telegram.py` - Chat integration
18. `whatsapp.py` - Chat integration
19. `qbo.py` - Financial integration
20. `shopify.py` - E-commerce integration

---

## Success Criteria

1. All P0 modules achieve >85% line coverage
2. All P1 modules achieve >75% line coverage
3. All P2 modules achieve >70% line coverage
4. No critical paths without integration tests
5. All edge cases documented and tested
6. Regression suite runs in <15 minutes
