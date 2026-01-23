# GitHub PR Review API

Aragora exposes GitHub pull request review endpoints for automated, multi-agent
review flows. This API can trigger a review, fetch PR details, and submit review
comments back to GitHub.

## Overview

Endpoints live under `/api/v1/github/pr`. The handler currently stores review
results in memory; for production use, replace with a durable store.

If `GITHUB_TOKEN` is not configured, the handler returns demo data for PR
fetches and review submission.

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
  into multi-agent deliberation for deeper analysis.
