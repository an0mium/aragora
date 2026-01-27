# GitHub PR Review API

Aragora provides a GitHub pull request review handler for automated, multi-agent
review flows. This API can trigger a review, fetch PR details, and submit review
comments back to GitHub.

## Overview

Endpoints live under `/api/v1/github/pr`. The handler currently stores review
results in memory; for production use, replace with a durable store.

If you want these routes enabled in the unified server, register
`PRReviewHandler` in the handler registry (`aragora/server/handlers/__init__.py`)
and add any RBAC rules required for your deployment.

If `GITHUB_TOKEN` is not configured, the handler returns demo data for PR
fetches and review submission.

For ad-hoc reviews (snippets, diffs, or PR URLs without GitHub writeback),
use the `/api/v1/code-review/*` endpoints.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/github/pr/review` | Trigger a PR review |
| GET | `/api/v1/github/pr/{pr_number}` | Get PR details (requires `repository` query param) |
| GET | `/api/v1/github/pr/review/{review_id}` | Get review status/result |
| GET | `/api/v1/github/pr/{pr_number}/reviews` | List reviews for a PR |
| POST | `/api/v1/github/pr/{pr_number}/review` | Submit review to GitHub |

## Trigger a Review

```http
POST /api/v1/github/pr/review
Content-Type: application/json

{
  "repository": "owner/repo",
  "pr_number": 42,
  "review_type": "comprehensive"
}
```

Response:

```json
{
  "success": true,
  "review_id": "review_abcd1234",
  "status": "in_progress",
  "pr_number": 42,
  "repository": "owner/repo"
}
```

## Fetch PR Details

```http
GET /api/v1/github/pr/42?repository=owner/repo
```

## Fetch Review Status

```http
GET /api/v1/github/pr/review/review_abcd1234
```

## List Reviews

```http
GET /api/v1/github/pr/42/reviews?repository=owner/repo
```

## Submit a Review

```http
POST /api/v1/github/pr/42/review
Content-Type: application/json

{
  "repository": "owner/repo",
  "event": "APPROVE",
  "body": "LGTM",
  "comments": [
    {
      "path": "aragora/core/decision.py",
      "position": 12,
      "body": "Consider adding a null-check here"
    }
  ]
}
```

## Configuration

- `GITHUB_TOKEN`: Required for live GitHub API calls.
- No token configured: returns demo PR details and accepts reviews in demo mode.

## Notes

- Review runs are asynchronous; poll `/api/v1/github/pr/review/{review_id}` for
  completion.
- The current implementation performs a lightweight heuristic review. Wire this
  into multi-agent vetted decisionmaking for deeper analysis.

## GitHub App Webhooks

For automatic PR review triggering, install the Aragora GitHub App. The app receives webhooks and automatically queues code review debates.

### Webhook Endpoint

```
POST /api/v1/webhooks/github
```

### Supported Events

| Event | Action | Behavior |
|-------|--------|----------|
| `pull_request` | `opened` | Queue code review debate |
| `pull_request` | `synchronize` | Re-trigger on new commits |
| `issues` | `opened` | Queue issue triage debate |
| `push` | - | Track code changes for context |
| `installation` | `created`/`deleted` | Track app installations |
| `ping` | - | Acknowledge webhook setup |

### Security

Webhooks are verified using HMAC-SHA256 signatures. Configure the secret via:

```bash
export GITHUB_WEBHOOK_SECRET=your-secret-here
```

### Setup

1. Create a GitHub App using the manifest in `github-app-manifest.json`
2. Install the app on your repositories
3. Configure `GITHUB_WEBHOOK_SECRET` to match the app's webhook secret
4. Webhooks will automatically trigger debates on PR and issue events

### Status Endpoint

```http
GET /api/v1/webhooks/github/status

# Response:
{
  "status": "configured",
  "webhook_endpoint": "/api/v1/webhooks/github",
  "supported_events": ["pull_request", "issues", "push", "installation", "ping"],
  "features": {
    "pr_auto_review": true,
    "issue_triage": true,
    "push_tracking": true
  }
}
```
