"""Consensus endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

CONSENSUS_ENDPOINTS = {
    "/api/consensus/similar": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Similar debates",
            "description": """Find debates similar to a given topic using semantic similarity.

**Use cases:**
- Check if a question has been debated before
- Find related discussions for context
- Avoid redundant debates on settled topics

**Similarity:** Uses embedding-based semantic search to find related debates.""",
            "operationId": "findSimilarDebates",
            "parameters": [
                {
                    "name": "topic",
                    "in": "query",
                    "required": True,
                    "description": "Topic or question to find similar debates for",
                    "schema": {"type": "string"},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of similar debates to return",
                    "schema": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                },
            ],
            "responses": {"200": _ok_response("Similar debates")},
        },
    },
    "/api/consensus/settled": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Settled questions",
            "description": """Get questions where strong consensus has been reached.

**Settled criteria:** Agreement threshold met (default 80%) across multiple debates.

**Use cases:**
- Find authoritative answers from past debates
- Identify topics with high confidence conclusions
- Build knowledge base from debate outcomes""",
            "operationId": "getSettledQuestions",
            "parameters": [
                {
                    "name": "threshold",
                    "in": "query",
                    "description": "Minimum consensus threshold (0.0-1.0)",
                    "schema": {"type": "number", "default": 0.8, "minimum": 0.5, "maximum": 1.0},
                },
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of questions to return",
                    "schema": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
                },
            ],
            "responses": {"200": _ok_response("Settled questions")},
        },
    },
    "/api/consensus/stats": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Consensus statistics",
            "description": """Get aggregate statistics about consensus across all debates.

**Response includes:**
- Overall consensus rate
- Average time to consensus
- Distribution by domain
- Trend over time""",
            "operationId": "getConsensusStats",
            "responses": {"200": _ok_response("Consensus stats")},
        },
    },
    "/api/consensus/dissents": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Dissenting views",
            "description": """Get significant dissenting views from debates.

**Dissents:** Minority positions with strong reasoning that didn't achieve consensus.

**Use cases:**
- Understand alternative perspectives
- Identify potential blind spots in consensus
- Preserve minority viewpoints for future consideration""",
            "operationId": "getDissentingViews",
            "parameters": [
                {
                    "name": "limit",
                    "in": "query",
                    "description": "Maximum number of dissents to return",
                    "schema": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
                }
            ],
            "responses": {"200": _ok_response("Dissenting views")},
        },
    },
    "/api/consensus/contrarian-views": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Contrarian views",
            "description": """Get contrarian views that challenge established consensus.

**Contrarian:** Positions that directly oppose majority conclusions with substantive arguments.

These views are preserved to:
- Prevent groupthink
- Enable future reconsideration
- Document minority reasoning""",
            "operationId": "getContrarianViews",
            "responses": {"200": _ok_response("Contrarian views")},
        },
    },
    "/api/consensus/risk-warnings": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Risk warnings",
            "description": """Get risk warnings from consensus analysis.

**Warning types:**
- Low confidence consensus (threshold barely met)
- Rapidly shifting consensus
- Contradictory historical positions
- Potential bias indicators""",
            "operationId": "getRiskWarnings",
            "responses": {"200": _ok_response("Risk warnings")},
        },
    },
    "/api/consensus/domain/{domain}": {
        "get": {
            "tags": ["Consensus"],
            "summary": "Domain consensus",
            "description": """Get consensus data for a specific domain (e.g., "technology", "policy", "science").

**Response includes:**
- Settled questions in domain
- Active debates
- Domain-specific consensus rates
- Top contributing agents""",
            "operationId": "getDomainConsensus",
            "parameters": [
                {
                    "name": "domain",
                    "in": "path",
                    "required": True,
                    "description": "Domain name (e.g., technology, science, policy)",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {"200": _ok_response("Domain consensus data")},
        },
    },
}
