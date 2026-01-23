# Integrations

Aragora connects to the tools your teams already use. This guide provides the
integration map and points to the detailed setup docs.

## Integration Types

| Type | Purpose | Docs |
|------|---------|------|
| Chat/Bot integrations | Bidirectional chat commands and replies | `docs/BOT_INTEGRATIONS.md` |
| Channel routing | Inbound + outbound routing across channels | `docs/CHANNELS.md` |
| Automation platforms | n8n/Zapier style workflows | `docs/AUTOMATION_INTEGRATIONS.md` |
| Webhooks | Event notifications and receipts | This doc (see below) |
| Developer workflows | PR review + code feedback | `docs/GITHUB_PR_REVIEW.md` |
| Inbox ops | Shared inbox + routing rules | `docs/SHARED_INBOX.md` |
| Accounting/ERP | QuickBooks Online dashboards + Plaid bank feeds | `docs/API_REFERENCE.md` |
| Payroll | Gusto payroll sync + journal entries | `docs/API_REFERENCE.md` |

## Available Integrations (Modules)

| Integration | Module | Notes |
|------------|--------|------|
| Slack | `aragora/integrations/slack.py` | Webhook notifications + bot support |
| Discord | `aragora/integrations/discord.py` | Webhook notifications + bot support |
| Teams | `aragora/integrations/teams.py` | Bot Framework integration |
| Telegram | `aragora/integrations/telegram.py` | Bot commands + replies |
| WhatsApp | `aragora/integrations/whatsapp.py` | WhatsApp Business API |
| Zoom | `aragora/integrations/zoom.py` | Meeting summaries + alerts |
| Email | `aragora/integrations/email.py` | Outbound notifications |
| Outlook/M365 | `aragora/server/handlers/features/outlook.py` | Outlook email integration |
| Receipt webhooks | `aragora/integrations/receipt_webhooks.py` | Decision receipt delivery |
| Generic webhooks | `aragora/integrations/webhooks.py` | Event-driven webhooks |
| Zapier | `aragora/integrations/zapier.py` | Workflow automation |
| n8n | `aragora/integrations/n8n.py` | Workflow automation |
| Twilio Voice | `aragora/integrations/twilio_voice.py` | Voice calls |
| GitHub PR review API | `aragora/server/handlers/github/pr_review.py` | Automated pull request review |
| GitHub audit bridge | `aragora/server/handlers/github/audit_bridge.py` | Sync audit findings to GitHub |
| Knowledge chat bridge | `aragora/services/knowledge_chat_bridge.py` | Chat-to-knowledge context search |
| Accounting (QuickBooks) | `aragora/server/handlers/accounting.py` | QBO dashboards + reports |
| Accounting (Plaid) | `aragora/connectors/accounting/plaid.py` | Bank account sync + transaction feeds |
| Payroll (Gusto) | `aragora/connectors/accounting/gusto.py` | Payroll runs + journal entries |
| Threat intelligence | `aragora/services/threat_intelligence.py` | VirusTotal/AbuseIPDB/PhishTank feeds |
| Shared inbox APIs | `aragora/server/handlers/shared_inbox.py` | Collaborative inbox workflows |

## Webhook Notifications (Slack + Discord)

Use webhooks to broadcast debate status or decision receipts to chat channels.

### Slack Webhook

```python
from aragora.integrations.slack import SlackConfig, SlackIntegration

config = SlackConfig(
    webhook_url="https://hooks.slack.com/services/...",
    channel="#debates",
    bot_name="Aragora",
    icon_emoji=":speech_balloon:",
    notify_on_consensus=True,
)

slack = SlackIntegration(config)
await slack.send_consensus_alert(
    debate_id="abc123",
    topic="Adopt event-driven architecture?",
    consensus_type="majority",
    result={"winner": "Yes", "confidence": 0.86},
)
```

### Discord Webhook

```python
from aragora.integrations.discord import DiscordConfig, DiscordIntegration

config = DiscordConfig(
    webhook_url="https://discord.com/api/webhooks/...",
    username="Aragora Debates",
    include_agent_details=True,
)

discord = DiscordIntegration(config)
await discord.send_debate_start(
    debate_id="abc123",
    topic="Should we migrate to microservices?",
    agents=["anthropic-api", "openai-api", "gemini"],
    config={"rounds": 3, "consensus_mode": "majority"},
)
```

## Notes

- For full bot setup (commands, OAuth, message events), use
  `docs/BOT_INTEGRATIONS.md`.
- For channel routing and message origin tracking, use `docs/CHANNELS.md`.
