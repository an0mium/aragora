# Connector Status Matrix

Last updated: 2026-02-23

## Summary

- **Production**: 99 connectors
- **Beta**: 48 connectors
- **Stub**: 4 connectors

## Status Criteria

| Status | Definition |
|--------|-----------|
| Production | Real API calls with error handling, retry logic, and/or circuit breakers |
| Beta | Real API calls with basic error handling but limited resilience patterns |
| Stub | Methods return empty data, raise NotImplementedError, or are placeholders |

---

## Evidence Connectors (`aragora/connectors/`)

Top-level evidence connectors extend `BaseConnector` and provide `search()`/`fetch()` for debate evidence collection.

| Category | Connector | Status | Features | Tests |
|----------|-----------|--------|----------|-------|
| Research | ArXiv (`arxiv.py`) | Production | httpx, rate limiting, caching, XML parsing, defusedxml | Yes |
| Research | PubMed (`pubmed.py`) | Production | NCBI E-utilities, API key support, caching | Yes |
| Research | Semantic Scholar (`semantic_scholar.py`) | Production | Graph API, API key support, caching | Yes |
| Research | CrossRef (`crossref.py`) | Production | DOI metadata, mailto polite pool, caching | Yes |
| Research | Wikipedia (`wikipedia.py`) | Production | REST + MediaWiki API, caching | Yes |
| Social | HackerNews (`hackernews.py`) | Production | Algolia API, retry, caching | Yes |
| Social | Reddit (`reddit.py`) | Production | JSON API, retry, rate limiting, caching | Yes |
| Social | Twitter/X (`twitter.py`) | Production | API v2, bearer token auth, retry, caching | Yes |
| Social | Twitter Poster (`twitter_poster.py`) | Production | OAuth 1.0a, threads, media upload, circuit breaker | Yes |
| News | NewsAPI (`newsapi.py`) | Production | API key auth, credibility tiers, caching | Yes |
| Finance | SEC EDGAR (`sec.py`) | Production | EDGAR API, form type filtering, caching | Yes |
| Web | Web/DuckDuckGo (`web.py`) | Production | DuckDuckGo search, HTML parsing, domain authority, circuit breaker | Yes |
| Web | Local Docs (`local_docs.py`) | Production | File system search, multi-format, regex | Yes |
| Code | GitHub (`github.py`) | Production | gh CLI + REST API, auth fallback, input validation | Yes |
| Code | Repository Crawler (`repository_crawler.py`) | Production | Git repos, AST parsing, incremental indexing | Yes |
| Database | SQL (`sql.py`) | Beta | PostgreSQL/MySQL/SQLite, parameterized queries | Yes |
| Media | Whisper (`whisper.py`) | Production | OpenAI Whisper API, streaming, retry, rate limiting | Yes |
| Media | YouTube Uploader (`youtube_uploader.py`) | Production | YouTube Data API v3, OAuth 2.0, circuit breaker | Yes |
| Legal | CourtListener (`courtlistener.py`) | Production | REST API v4, API key auth, retry, caching | Yes |
| Legal | GovInfo (`govinfo.py`) | Production | GovInfo Search Service, API key auth, caching | Yes |
| Healthcare | Clinical Tables (`clinical_tables.py`) | Production | NLM ClinicalTables API, ICD-10-CM, caching | Yes |
| Healthcare | NICE Guidance (`nice_guidance.py`) | Beta | NICE API, API key auth, caching | Yes |
| Healthcare | RxNav (`rxnav.py`) | Production | NIH RxNav REST API, drug interactions, caching | Yes |
| Chat Export | Conversation Ingestor (`conversation_ingestor.py`) | Production | ChatGPT/Claude exports, claim extraction, topic clustering | Yes |
| Session | Debate Session (`debate_session.py`) | Production | Cross-channel session tracking, handoff | Yes |
| Infra | Metrics (`metrics.py`) | Production | Prometheus counters, histograms, gauges | Yes |
| Infra | Recovery (`recovery.py`) | Production | Retry, circuit breaker, token refresh, fallback chains | Yes |
| Infra | Runtime Registry (`runtime_registry.py`) | Production | Connector discovery, health checks | Yes |

## Connector Subdirectories (`aragora/connectors/*/`)

### Accounting

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| QuickBooks Online (`qbo.py`) | Production | OAuth 2.0, circuit breaker, multi-company, transaction sync | Yes (5 files) |
| QuickBooks Legacy (`quickbooks.py`) | Beta | Deprecated wrapper, basic API calls | Yes |
| Xero (`xero.py`) | Production | OAuth 2.0, circuit breaker, invoices, bank reconciliation | Yes |
| Plaid (`plaid.py`) | Production | Link integration, circuit breaker, transaction categorization | Yes |
| Gusto (`gusto.py`) | Production | OAuth 2.0, circuit breaker, payroll, journal entries | Yes |
| FASB/GAAP (`gaap.py`) | Beta | Licensed content proxy, configurable endpoints | Yes |
| IRS (`irs.py`) | Beta | Proxy endpoint, configurable API base | Yes |

### Advertising

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Google Ads (`google_ads.py`) | Beta | Campaigns, ad groups, keywords, reporting | Yes |
| LinkedIn Ads (`linkedin_ads.py`) | Beta | B2B campaigns, lead gen forms, analytics | Yes |
| Meta Ads (`meta_ads.py`) | Beta | Facebook/Instagram ads, audiences, insights | Yes |
| Microsoft Ads (`microsoft_ads.py`) | Beta | Bing Ads, campaigns, conversion tracking | Yes |
| TikTok Ads (`tiktok_ads.py`) | Beta | Campaigns, creative management, analytics | Yes |
| Twitter Ads (`twitter_ads.py`) | Beta | Promoted tweets, targeting, analytics | Yes |

### Analytics

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Google Analytics (`google_analytics.py`) | Beta | GA4 Data API, reports, real-time data | Yes |
| Metabase (`metabase.py`) | Beta | Questions, dashboards, query execution | Yes |
| Mixpanel (`mixpanel.py`) | Beta | Events, profiles, reports, exports | Yes |
| Segment (`segment.py`) | Beta | CDP tracking, sources, destinations | Yes |

### Automation

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Zapier (`zapier.py`) | Production | HMAC webhook signatures, triggers, actions | Yes |
| n8n (`n8n.py`) | Production | Webhook triggers, node definitions, community node support | Yes |

### Blockchain

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| ERC-8004 (`connector.py`) | Production | Agent identity, reputation, validation records | Yes (3 files) |

### Browser

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Playwright (`playwright_connector.py`) | Beta | Page navigation, element interaction, screenshots | Yes |

### Calendar

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Google Calendar (`google_calendar.py`) | Beta | OAuth 2.0, events, free/busy, push notifications | Yes |
| Outlook Calendar (`outlook_calendar.py`) | Beta | Microsoft Graph API, events, conflict detection | Yes |

### Chat Platforms (`chat/`)

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Slack (`chat/slack/`) | Production | Bot API, events, threads, message queuing (5 modules) | Yes (5 files) |
| Telegram (`chat/telegram/`) | Production | Bot API, webhooks, inline, media, files (7 modules) | Yes (2+ files) |
| WhatsApp (`chat/whatsapp.py`) | Production | Cloud API, circuit breaker, HMAC verification, tracing | Yes |
| Discord (`chat/discord.py`) | Production | REST API, interactions, NaCl verification, circuit breaker | Yes (2 files) |
| Microsoft Teams (`chat/teams/`) | Production | Graph API, channels, events, adaptive cards (7 modules) | Yes (3+ files) |
| Google Chat (`chat/google_chat.py`) | Production | Chat API, Cards v2, service account auth | Yes |
| Signal (`chat/signal.py`) | Production | signal-cli REST API, circuit breaker | Yes |
| iMessage (`chat/imessage.py`) | Production | BlueBubbles server, circuit breaker | Yes |

### Communication

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| SendGrid (`communication/sendgrid.py`) | Stub | API scaffolding only; `search()`/`fetch()` raise NotImplementedError | No |
| Twilio (`communication/twilio.py`) | Stub | API scaffolding only; `search()`/`fetch()` raise NotImplementedError | No |

### Credentials

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Credential Providers (`credentials/providers.py`) | Production | Env vars, AWS Secrets Manager, chained fallback, TTL caching | Yes |

### CRM

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| HubSpot (`crm/hubspot.py`) | Beta | Contacts, companies, deals, engagements, marketing | Yes |
| Pipedrive (`crm/pipedrive.py`) | Beta | Deals, persons, organizations, activities, webhooks | Yes |

### Devices

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| FCM/APNs/Web Push (`devices/push.py`) | Production | Firebase, APNs, VAPID; retry, circuit breaker, batch | Yes |
| Alexa (`devices/alexa.py`) | Beta | Smart Home Skill, proactive notifications, OAuth linking | Yes |
| Google Home (`devices/google_home.py`) | Beta | Conversational Actions, Home Graph, broadcasts | Yes |

### DevOps

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| PagerDuty (`devops/pagerduty.py`) | Beta | Incident creation, investigation notes, on-call lookup | Yes |

### Documents

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Document Parser (`documents/parser.py`) | Production | PDF, DOCX, XLSX, PPTX, HTML, CSV, JSON, YAML, XML | Yes |
| Document Connector (`documents/connector.py`) | Production | Integrates parser with evidence collection | Yes |

### E-Commerce

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Shopify (`ecommerce/shopify/`) | Beta | Orders, products, inventory, customers, webhooks | Yes |
| WooCommerce (`ecommerce/woocommerce/`) | Beta | Orders, products, webhooks, sync | Yes |
| Amazon (`ecommerce/amazon/`) | Beta | SP-API, orders, inventory, reports | Yes |
| ShipStation (`ecommerce/shipstation.py`) | Beta | Shipments, orders, label generation | Yes |

### Email

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Gmail Sync (`email/gmail_sync.py`) | Beta | Gmail API, label sync, incremental | Yes |
| Outlook Sync (`email/outlook_sync.py`) | Beta | Microsoft Graph, folder sync | Yes |
| Resilience (`email/resilience.py`) | Production | Rate limiting, circuit breaker for email connectors | Yes |

### Enterprise Collaboration

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Jira (`enterprise/collaboration/jira.py`) | Production | Cloud + Data Center, JQL, incremental sync, webhooks | Yes |
| Confluence (`enterprise/collaboration/confluence.py`) | Production | Content extraction, CQL, incremental sync, webhooks | Yes |
| Notion (`enterprise/collaboration/notion.py`) | Production | Pages, databases, block extraction, incremental sync | Yes |
| Slack Enterprise (`enterprise/collaboration/slack.py`) | Production | Channel indexing, threads, files, incremental sync | Yes |
| Teams Enterprise (`enterprise/collaboration/teams.py`) | Production | Graph API, channels, messages, delta sync | Yes |
| Asana (`enterprise/collaboration/asana.py`) | Beta | Tasks, projects, subtasks, custom fields, webhooks | Yes |
| Monday.com (`enterprise/collaboration/monday.py`) | Beta | GraphQL API v2, boards, items, updates | Yes |
| Linear (`enterprise/collaboration/linear.py`) | Beta | GraphQL API, issues, projects, cycles | Yes |

### Enterprise CRM

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Salesforce (`enterprise/crm/salesforce.py`) | Production | SOQL, Bulk API, OAuth 2.0, incremental sync | Yes |

### Enterprise Database

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| PostgreSQL (`enterprise/database/postgres.py`) | Production | Incremental sync, LISTEN/NOTIFY, CDC, connection pooling | Yes |
| MySQL (`enterprise/database/mysql.py`) | Production | Incremental sync, binlog CDC, connection pooling | Yes |
| MongoDB (`enterprise/database/mongodb.py`) | Production | Change streams, aggregation pipelines, incremental sync | Yes |
| Snowflake (`enterprise/database/snowflake.py`) | Production | Time travel, multi-warehouse, CHANGES clause | Yes |
| SQL Server (`enterprise/database/sqlserver.py`) | Production | CDC, Change Tracking, incremental sync | Yes |
| CDC Framework (`enterprise/database/cdc.py`) | Production | Change Data Capture abstraction, event handlers | Yes |

### Enterprise Documents

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| SharePoint (`enterprise/documents/sharepoint.py`) | Production | Microsoft Graph, document libraries, change tokens, webhooks | Yes |
| Google Drive (`enterprise/documents/gdrive.py`) | Production | OAuth 2.0, Changes API, Shared Drives, export | Yes |
| Google Sheets (`enterprise/documents/gsheets.py`) | Production | Sheets API, multi-sheet, named ranges, incremental sync | Yes |
| OneDrive (`enterprise/documents/onedrive.py`) | Production | Microsoft Graph, delta sync, Office export | Yes |
| Dropbox (`enterprise/documents/dropbox.py`) | Production | OAuth 2.0, cursor-based sync, content download | Yes |
| S3 (`enterprise/documents/s3.py`) | Production | AWS S3/MinIO, document parsing, prefix filtering | Yes |

### Enterprise Git

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| GitHub Enterprise (`enterprise/git/github.py`) | Production | Full repo crawling, incremental sync, AST parsing, webhooks | Yes |

### Enterprise Healthcare

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| FHIR R4 (`enterprise/healthcare/fhir.py`) | Production | SMART on FHIR, PHI redaction, audit logging | Yes |
| HL7 v2 (`enterprise/healthcare/hl7v2.py`) | Production | Segment parsing, MLLP transport, PHI redaction | Yes |
| Epic EHR (`enterprise/healthcare/ehr/epic.py`) | Beta | App Orchard auth, MyChart, Epic-specific extensions | Yes |
| Cerner EHR (`enterprise/healthcare/ehr/cerner.py`) | Beta | Millennium auth, bulk data export | Yes |

### Enterprise ITSM

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| ServiceNow (`enterprise/itsm/servicenow.py`) | Production | Incidents, Problems, Changes, incremental sync, webhooks | Yes |

### Enterprise Streaming

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Kafka (`enterprise/streaming/kafka.py`) | Production | Consumer groups, schema registry, DLQ, circuit breaker, backoff | Yes |
| RabbitMQ (`enterprise/streaming/rabbitmq.py`) | Production | Exchange binding, acknowledgment, DLQ, circuit breaker, backoff | Yes |
| AWS SNS/SQS (`enterprise/streaming/snssqs.py`) | Production | Long polling, DLQ, FIFO dedup, circuit breaker, backoff | Yes |

### Feeds

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| RSS/Atom Ingestor (`feeds/ingestor.py`) | Production | RSS 2.0/Atom, backoff retries, circuit breaker, parallel fetch | Yes |

### Knowledge

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Obsidian (`knowledge/obsidian.py`) | Beta | Vault access, frontmatter, wikilinks, tags, REST API | Yes |

### Legal

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| DocuSign (`legal/docusign.py`) | Beta | OAuth 2.0/JWT, envelopes, templates, status tracking | Yes |
| LexisNexis (`legal/lexis.py`) | Beta | Placeholder for licensed content, configurable proxy | Yes |
| Westlaw (`legal/westlaw.py`) | Beta | Placeholder for licensed content, configurable proxy | Yes |

### Low-Code

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Airtable (`lowcode/airtable.py`) | Beta | Records CRUD, views, filtering, attachments | Yes |
| Knack (`lowcode/knack.py`) | Beta | Objects, records CRUD, views, fields | Yes |

### Marketing

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Klaviyo (`marketing/klaviyo.py`) | Beta | Lists, profiles, campaigns, flows, SMS | Yes |
| Mailchimp (`marketing/mailchimp.py`) | Beta | Audiences, campaigns, subscribers, templates | Yes |

### Marketplace

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Walmart (`marketplace/walmart.py`) | Beta | Orders, inventory, catalog, pricing, reports | Yes |

### Memory

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Claude-Mem (`memory/claude_mem.py`) | Production | Local claude-mem API, memory search, observations | Yes |

### Payments

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Stripe (`payments/stripe.py`) | Production | Customers, subscriptions, intents, invoices, webhooks | Yes |
| PayPal (`payments/paypal.py`) | Production | Orders, payments, subscriptions, payouts, webhooks | Yes |
| Square (`payments/square.py`) | Production | Payments, customers, subscriptions, invoices, catalog | Yes |
| Authorize.net (`payments/authorize_net.py`) | Production | CIM, transactions, ARB, fraud detection, webhooks | Yes |

### Productivity

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Trello (`productivity/trello.py`) | Stub | API scaffolding only; `search()`/`fetch()` raise NotImplementedError | No |

### Social

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Instagram (`social/instagram.py`) | Stub | API scaffolding only; `search()`/`fetch()` raise NotImplementedError | No |

### Supermemory

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Supermemory Client (`supermemory/client.py`) | Production | SDK wrapper, privacy filtering, circuit breaker, retry | Yes (3 files) |

### Support

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Zendesk (`support/zendesk.py`) | Beta | Tickets, users, organizations, SLA policies | Yes |
| Freshdesk (`support/freshdesk.py`) | Beta | Tickets, contacts, canned responses, SLA | Yes |
| Intercom (`support/intercom.py`) | Beta | Conversations, contacts, articles, teams | Yes |
| Help Scout (`support/helpscout.py`) | Beta | Conversations, customers, mailboxes, OAuth 2.0 | Yes |

### Tax

| Connector | Status | Features | Tests |
|-----------|--------|----------|-------|
| Generic Tax (`tax/generic.py`) | Beta | Multi-jurisdiction proxy, configurable per-jurisdiction | Yes |

---

## Integration Connectors (`aragora/integrations/`)

Integration connectors post debate results to external platforms and handle bidirectional communication.

| Category | Connector | Status | Features | Tests |
|----------|-----------|--------|----------|-------|
| Chat | Slack (`slack.py`) | Production | Block Kit, webhook, consensus alerts, circuit breaker, tracing | Yes |
| Chat | Discord (`discord.py`) | Production | Rich embeds, webhooks, rate limiting, retry, circuit breaker | Yes |
| Chat | Microsoft Teams (`teams.py`) | Production | Adaptive Cards, webhooks, circuit breaker, tracing | Yes |
| Chat | Telegram (`telegram.py`) | Production | HTML formatting, inline keyboards, rate limiting | Yes |
| Chat | WhatsApp (`whatsapp.py`) | Production | Meta Cloud API + Twilio fallback, templates | Yes |
| Chat | Matrix/Element (`matrix.py`) | Production | Client-Server API, room events, formatted messages | Yes |
| Chat | Zoom (`zoom.py`) | Production | OAuth 2.0, chat messages, webhooks, meeting integration | Yes |
| Email | Email (`email.py`) | Production | SMTP + SendGrid + AWS SES, HTML templates, circuit breaker | Yes (5+ files) |
| Email | Email OAuth (`email_oauth.py`) | Production | Multi-backend token storage (SQLite, Redis, Postgres) | Yes |
| Email | Email Rate Limiter (`email_rate_limiter.py`) | Production | Redis-backed distributed rate limiting, token bucket | Yes |
| Email | Email Reply Loop (`email_reply_loop.py`) | Production | Inbound parse, IMAP polling, debate routing | Yes |
| Voice | Twilio Voice (`twilio_voice.py`) | Production | Inbound/outbound calls, TwiML, TTS, HMAC verification | Yes |
| Automation | Zapier (`zapier.py`) | Production | REST webhooks, HMAC-SHA256, triggers, actions, SSRF protection | Yes |
| Automation | Make/Integromat (`make.py`) | Production | Webhook format, instant triggers, actions, SSRF protection | Yes |
| Automation | n8n (`n8n.py`) | Production | Webhooks, HMAC-SHA256, node definitions, SSRF protection | Yes |
| AI Framework | LangChain (`langchain.py`) | Production | Tool, Retriever, Callback, Chain interfaces | Yes (3+ files) |
| AI Framework | LangChain Package (`langchain/`) | Production | Callbacks, chains, retriever, tools modules | Yes |
| AI Framework | OpenClaw (`openclaw/`) | Production | Client, audit bridge | Yes (2 files) |
| Infra | Webhooks (`webhooks.py`) | Production | Non-blocking dispatcher, bounded queue, HMAC, circuit breaker | Yes |
| Infra | Receipt Webhooks (`receipt_webhooks.py`) | Production | Receipt lifecycle events, webhook dispatch | Yes |
| Infra | Platform Resilience (`platform_resilience.py`) | Production | Platform circuit breakers, DLQ, distributed rate limiting | Yes |

---

## Test Coverage Summary

| Area | Test Files | Coverage |
|------|-----------|----------|
| `tests/connectors/` | 201 | All major connectors covered |
| `tests/integrations/` | 28 | All integration connectors covered |
| **Total** | **229** | |

## Notes

- **Stub connectors** (SendGrid, Twilio, Instagram, Trello) have API scaffolding (env var config, auth headers) but raise `NotImplementedError` on `search()`/`fetch()`. They are ready for community contributions.
- **Beta connectors** have real API call implementations with data models and basic error handling, but typically lack circuit breaker patterns and advanced retry logic.
- **Production connectors** include robust error handling, circuit breakers, rate limiting, caching, and/or retry with exponential backoff.
- The **enterprise connectors** (`enterprise/`) all extend `EnterpriseConnector` with incremental sync, pagination safety caps (`_MAX_PAGES`), and standardized `SyncItem` output.
- The **LexisNexis** and **Westlaw** connectors are documented as "placeholders for licensed content" but implement configurable proxy endpoints, placing them in Beta rather than Stub.
