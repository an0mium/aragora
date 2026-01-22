---
title: Debates API
description: Create and manage multi-agent debates
sidebar_position: 1
---

# Debates API

The Debates API allows you to create, manage, and query multi-agent debates.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/debates` | Create a new debate |
| GET | `/api/debates` | List debates |
| GET | `/api/debates/:id` | Get debate details |
| GET | `/api/debates/:id/messages` | Get debate messages |
| GET | `/api/debates/:id/consensus` | Get consensus result |

## Create Debate

```bash
POST /api/debates
```

### Request Body

```json
{
  "topic": "string",
  "context": "string (optional)",
  "agents": ["claude", "gpt4", "gemini"],
  "rounds": 3,
  "protocol": {
    "phases": ["opening", "critique", "revision", "vote"],
    "consensus_threshold": 0.75
  }
}
```

### Response

```json
{
  "id": "debate_abc123",
  "status": "running",
  "topic": "...",
  "created_at": "2024-01-15T10:30:00Z"
}
```

## List Debates

```bash
GET /api/debates?status=completed&limit=20
```

### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| status | string | Filter by status (pending, running, completed, failed) |
| limit | number | Max results (default: 20) |
| offset | number | Pagination offset |

## Get Debate

```bash
GET /api/debates/:id
```

Returns full debate details including all rounds and messages.

## Get Consensus

```bash
GET /api/debates/:id/consensus
```

### Response

```json
{
  "debate_id": "debate_abc123",
  "status": "consensus_reached",
  "consensus": {
    "summary": "...",
    "confidence": 0.85,
    "key_points": ["..."],
    "disagreements": ["..."]
  },
  "voting_summary": {
    "agree": 2,
    "agree_with_modifications": 1,
    "disagree": 0
  }
}
```
