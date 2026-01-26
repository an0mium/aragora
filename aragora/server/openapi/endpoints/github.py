"""GitHub PR review endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS, AUTH_REQUIREMENTS


GITHUB_ENDPOINTS = {
    "/api/v1/github/pr/review": {
        "post": {
            "tags": ["GitHub"],
            "summary": "Trigger PR review",
            "operationId": "createGithubPrReview",
            "description": "Trigger an automated PR review run.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/GitHubPRReviewTriggerRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("Review started", "GitHubPRReviewTriggerResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/github/pr/{pr_number}": {
        "get": {
            "tags": ["GitHub"],
            "summary": "Get PR details",
            "operationId": "getGithubPr",
            "description": "Fetch PR metadata and file diffs.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "pr_number",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "integer"},
                },
                {
                    "name": "repository",
                    "in": "query",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("PR details", "GitHubPRDetailsResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/github/pr/review/{review_id}": {
        "get": {
            "tags": ["GitHub"],
            "summary": "Get review status",
            "operationId": "getGithubPrReview",
            "description": "Fetch review status and results.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "review_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Review status", "GitHubPRReviewStatusResponse"),
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/github/pr/{pr_number}/reviews": {
        "get": {
            "tags": ["GitHub"],
            "summary": "List PR reviews",
            "operationId": "getGithubPrReview",
            "description": "List review runs for a PR.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "pr_number",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "integer"},
                },
                {
                    "name": "repository",
                    "in": "query",
                    "required": True,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "200": _ok_response("Review list", "GitHubPRReviewListResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/github/pr/{pr_number}/review": {
        "post": {
            "tags": ["GitHub"],
            "summary": "Submit PR review",
            "operationId": "createGithubPrReview",
            "description": "Submit a review verdict back to GitHub.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "pr_number",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "integer"},
                }
            ],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/GitHubPRSubmitReviewRequest"}
                    }
                },
            },
            "responses": {
                "200": _ok_response("Review submitted", "GitHubPRSubmitReviewResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "403": STANDARD_ERRORS["403"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/github/audit/issues": {
        "post": {
            "tags": ["GitHub"],
            "summary": "Create issues from findings",
            "operationId": "createGithubAuditIssues",
            "description": "Create GitHub issues from audit findings.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Issues created", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/github/audit/issues/bulk": {
        "post": {
            "tags": ["GitHub"],
            "summary": "Bulk create issues",
            "operationId": "createGithubAuditIssuesBulk",
            "description": "Bulk create GitHub issues from findings.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("Issues created", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/github/audit/pr": {
        "post": {
            "tags": ["GitHub"],
            "summary": "Create PR with fixes",
            "operationId": "createGithubAuditPr",
            "description": "Create a remediation PR from audit findings.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}},
            },
            "responses": {
                "200": _ok_response("PR created", "StandardSuccessResponse"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "500": STANDARD_ERRORS["500"],
            },
        }
    },
    "/api/v1/github/audit/sync/{session_id}": {
        "get": {
            "tags": ["GitHub"],
            "summary": "Get sync status",
            "operationId": "getGithubAuditSync",
            "description": "Get status of a GitHub audit sync.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "session_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Sync status", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
        "post": {
            "tags": ["GitHub"],
            "summary": "Sync findings",
            "operationId": "createGithubAuditSync",
            "description": "Sync an audit session to GitHub.",
            "security": AUTH_REQUIREMENTS["optional"]["security"],
            "parameters": [
                {
                    "name": "session_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "responses": {
                "200": _ok_response("Sync started", "StandardSuccessResponse"),
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
}


__all__ = ["GITHUB_ENDPOINTS"]
