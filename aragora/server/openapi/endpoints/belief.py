"""Belief network endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response

BELIEF_ENDPOINTS = {
    "/api/belief-network/{debate_id}/cruxes": {
        "get": {
            "tags": ["Belief"],
            "summary": "Get debate cruxes",
            "description": """Get the crux points where agents most strongly disagree in a debate.

**Cruxes** are propositions where changing an agent's belief would change their
overall conclusion. Identifying cruxes helps focus further debate on the most
productive points of disagreement.""",
            "operationId": "getBeliefNetworkCruxes",
            "parameters": [
                {
                    "name": "debate_id",
                    "in": "path",
                    "required": True,
                    "description": "The debate ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {"200": _ok_response("Debate cruxes", "BeliefCruxesResponse")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/belief-network/{debate_id}/load-bearing-claims": {
        "get": {
            "tags": ["Belief"],
            "summary": "Get load-bearing claims",
            "description": """Get claims that are foundational to the debate's argument structure.

**Load-bearing claims** are propositions that, if removed or disproven, would
cause significant parts of the argument structure to collapse. These are the
critical assumptions underlying the debate's conclusions.""",
            "operationId": "getBeliefNetworkLoadBearingClaims",
            "parameters": [
                {
                    "name": "debate_id",
                    "in": "path",
                    "required": True,
                    "description": "The debate ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {"200": _ok_response("Load-bearing claims", "LoadBearingClaimsResponse")},
            "security": [{"bearerAuth": []}],
        },
    },
    "/api/debate/{debate_id}/graph-stats": {
        "get": {
            "tags": ["Belief"],
            "summary": "Get debate graph stats",
            "description": """Get graph-based statistics for the belief network of a debate.

**Includes:** node count, edge count, clustering coefficient, argument depth,
most-connected claims, and structural metrics.""",
            "operationId": "getBeliefGraphStats",
            "parameters": [
                {
                    "name": "debate_id",
                    "in": "path",
                    "required": True,
                    "description": "The debate ID",
                    "schema": {"type": "string"},
                }
            ],
            "responses": {"200": _ok_response("Graph statistics", "BeliefGraphStats")},
            "security": [{"bearerAuth": []}],
        },
    },
}
