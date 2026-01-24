# 15-Minute Onboarding Flow

**Version:** 1.0
**Target:** First debate + decision receipt in <15 minutes
**Audience:** SME teams (5-50 users)

---

## Flow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    15-MINUTE ONBOARDING FLOW                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [1] SIGNUP        [2] WORKSPACE      [3] INTEGRATE    [4] DEBATE  │
│      (2 min)           (2 min)            (5 min)         (5 min)  │
│                                                                     │
│  ┌─────────┐      ┌─────────┐       ┌─────────┐      ┌─────────┐  │
│  │ Email   │──────│ Create  │───────│ Connect │──────│ Run     │  │
│  │ Verify  │      │ Workspace│       │ Slack   │      │ First   │  │
│  │         │      │         │       │         │      │ Debate  │  │
│  └─────────┘      └─────────┘       └─────────┘      └─────────┘  │
│                                                                     │
│                                            ▼                        │
│                                     ┌─────────────┐                │
│                                     │  RECEIPT    │                │
│                                     │  (1 min)    │                │
│                                     └─────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Signup (2 minutes)

### Step 1.1: Landing Page
**URL:** `https://aragora.ai/signup`

**UI Elements:**
- Email input
- "Continue with Google" button (OAuth)
- Terms of service checkbox

**Actions:**
1. User enters email or clicks Google OAuth
2. Verification email sent (if email signup)
3. Redirect to workspace creation

### Step 1.2: Email Verification
**URL:** `https://aragora.ai/verify?token={token}`

**Flow:**
1. User clicks email link
2. Token validated
3. Session created
4. Redirect to workspace creation

---

## Phase 2: Workspace Creation (2 minutes)

### Step 2.1: Organization Details
**URL:** `https://aragora.ai/onboarding/workspace`

**Form Fields:**
| Field | Required | Validation |
|-------|----------|------------|
| Organization Name | Yes | 3-50 characters |
| Workspace URL | Yes | Alphanumeric, unique |
| Team Size | Yes | Dropdown: 1-5, 6-15, 16-50, 50+ |
| Primary Use Case | No | Dropdown: Team decisions, Project planning, Vendor selection, Other |

**Backend:**
```python
POST /api/workspaces
{
    "name": "Acme Corp",
    "slug": "acme-corp",
    "team_size": "6-15",
    "use_case": "team_decisions"
}
```

### Step 2.2: Admin Setup
**Auto-configured:**
- Current user becomes workspace admin
- Default budget: $100/month (adjustable)
- Default agents: Claude, GPT-4, Gemini

---

## Phase 3: Integration Setup (5 minutes)

### Step 3.1: Integration Selection
**URL:** `https://aragora.ai/onboarding/integrations`

**Available Integrations:**

| Integration | Recommended | Setup Time | Use Case |
|-------------|-------------|------------|----------|
| Slack | Yes (default) | 3 min | Debate notifications, team voting |
| Gmail | Optional | 5 min | Email-based debates, document context |
| Google Drive | Optional | 3 min | Document references |
| Outlook | Optional | 5 min | Email-based debates |

**UI:**
- Card-based selection
- "Skip for now" option
- Progress indicator showing 3/4 complete

### Step 3.2: Slack OAuth Flow
**Recommended First Integration**

**Steps:**
1. Click "Connect Slack"
2. OAuth redirect to Slack
3. User approves permissions:
   - `channels:read` - List channels
   - `chat:write` - Post debate results
   - `users:read` - Identify participants
4. Redirect back with code
5. Store tokens securely
6. Test message: "Aragora connected! Reply to start your first debate."

**Backend:**
```python
# OAuth callback
POST /api/integrations/slack/callback
{
    "code": "oauth_code",
    "state": "state_token"
}

# Test connection
POST /api/integrations/slack/test
{
    "channel": "#general"
}
```

### Step 3.3: Connection Verification
**UI:**
- Green checkmark on successful connection
- "Send test message" button
- Channel selector for notifications

---

## Phase 4: First Debate (5 minutes)

### Step 4.1: Template Selection
**URL:** `https://aragora.ai/onboarding/first-debate`

**Quick-Start Templates:**

| Template | Agents | Rounds | Best For |
|----------|--------|--------|----------|
| Quick Decision | Claude, GPT-4 | 2 | Simple yes/no decisions |
| Feature Prioritization | Claude, GPT-4, Gemini | 3 | Product decisions |
| Vendor Comparison | Claude, GPT-4 | 3 | Vendor selection |
| Policy Review | Claude, GPT-4, Gemini | 4 | Policy decisions |

**Recommendation Algorithm:**
- If `use_case == "team_decisions"`: Quick Decision
- If `use_case == "project_planning"`: Feature Prioritization
- If `use_case == "vendor_selection"`: Vendor Comparison
- Default: Quick Decision

### Step 4.2: Topic Entry
**UI Elements:**
- Large text input: "What decision do you need to make?"
- Context textarea: "Add any relevant context (optional)"
- Example topics shown below input

**Examples:**
- "Should we adopt TypeScript for our frontend?"
- "Which CRM should we use: Salesforce or HubSpot?"
- "Should we implement a 4-day work week?"

### Step 4.3: Agent Preview
**UI:**
- Agent cards with avatars
- Brief description of each agent's perspective
- "Customize agents" link (advanced)

### Step 4.4: Run Debate
**Backend:**
```python
POST /api/debates
{
    "topic": "Should we adopt TypeScript for our frontend?",
    "context": "We currently use JavaScript...",
    "agents": ["anthropic-api", "openai-api"],
    "rounds": 2,
    "consensus": "majority"
}
```

**Real-Time UI:**
- Streaming agent responses
- Progress indicator: Round 1/2, Round 2/2
- Estimated time: "~3 minutes remaining"

---

## Phase 5: Decision Receipt (1 minute)

### Step 5.1: Receipt Display
**URL:** `https://aragora.ai/debates/{id}/receipt`

**Receipt Contents:**
- Decision summary (1-2 sentences)
- Confidence score (0-100%)
- Key arguments for each position
- Agent agreement breakdown
- Voting results (if applicable)

### Step 5.2: Export Options
**Actions:**
- "Download PDF" button
- "Copy Markdown" button
- "Share to Slack" button (if connected)
- "Start follow-up debate" button

### Step 5.3: Completion Screen
**URL:** `https://aragora.ai/onboarding/complete`

**UI:**
- Celebration animation
- "You completed your first debate in X minutes!"
- Next steps:
  - "Invite team members"
  - "Explore templates"
  - "Connect more integrations"
  - "Go to dashboard"

---

## Success Metrics

| Checkpoint | Target Time | Measurement |
|------------|-------------|-------------|
| Signup complete | 2 min | Time from landing to workspace created |
| First integration | 5 min | Time to first OAuth complete |
| Debate started | 3 min | Time from integration to debate create |
| Receipt viewed | 5 min | Time from debate start to receipt view |
| **Total** | **<15 min** | End-to-end |

---

## Error Handling

### Common Errors

| Error | User Message | Resolution |
|-------|--------------|------------|
| OAuth failed | "Couldn't connect to Slack. Try again?" | Retry button |
| Debate timeout | "Debate took longer than expected. Here's what we have so far." | Show partial results |
| No API keys | "Aragora needs AI providers configured. Contact support." | Link to docs |

### Recovery Flows

1. **OAuth Cancelled:** Return to integration selection with "Try again" option
2. **Debate Failed:** Show error, offer retry or skip to dashboard
3. **Slow Debate:** Show progress bar, allow cancel

---

## Implementation Files

| Component | File | Notes |
|-----------|------|-------|
| Signup API | `aragora/server/handlers/auth/handler.py` | OAuth and email signup |
| Workspace API | `aragora/server/handlers/organizations.py` | Workspace CRUD |
| Slack OAuth | `aragora/connectors/chat/slack.py` | OAuth flow |
| Debate API | `aragora/server/handlers/debates/handler.py` | Create and run |
| Receipt API | `aragora/gauntlet/receipts.py` | Receipt generation |

---

## Related Documents

- [SME_STARTER_PACK.md](SME_STARTER_PACK.md) - Feature scope
- [INTEGRATION_AUDIT.md](INTEGRATION_AUDIT.md) - Connector details
- [API_REFERENCE.md](API_REFERENCE.md) - API documentation

---

*Created: 2026-01-24*
*Next Review: Sprint 2 Planning*
