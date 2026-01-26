"""OAuth endpoint definitions."""

from aragora.server.openapi.helpers import _ok_response, STANDARD_ERRORS

OAUTH_ENDPOINTS = {
    "/api/auth/oauth/google": {
        "get": {
            "tags": ["OAuth"],
            "summary": "Start Google OAuth flow",
            "operationId": "listAuthOauthGoogle",
            "description": "Redirect user to Google OAuth consent screen for authentication. Supports optional account linking for already-authenticated users.",
            "parameters": [
                {
                    "name": "redirect_url",
                    "in": "query",
                    "description": "URL to redirect after successful authentication (must be in allowlist)",
                    "schema": {"type": "string", "format": "uri"},
                },
            ],
            "responses": {
                "302": {"description": "Redirect to Google OAuth consent screen"},
                "400": STANDARD_ERRORS["400"],
                "429": STANDARD_ERRORS["429"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/auth/oauth/google/callback": {
        "get": {
            "tags": ["OAuth"],
            "summary": "Google OAuth callback",
            "operationId": "listAuthOauthGoogleCallback",
            "description": "Handle the OAuth callback from Google after user consent. Exchanges authorization code for tokens and creates/links user account.",
            "parameters": [
                {
                    "name": "code",
                    "in": "query",
                    "required": True,
                    "description": "Authorization code from Google",
                    "schema": {"type": "string"},
                },
                {
                    "name": "state",
                    "in": "query",
                    "required": True,
                    "description": "State parameter for CSRF protection",
                    "schema": {"type": "string"},
                },
                {
                    "name": "error",
                    "in": "query",
                    "description": "Error code if user denied consent",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "302": {"description": "Redirect to success URL with auth token"},
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/auth/oauth/github": {
        "get": {
            "tags": ["OAuth"],
            "summary": "Start GitHub OAuth flow",
            "operationId": "listAuthOauthGithub",
            "description": "Redirect user to GitHub OAuth consent screen for authentication.",
            "parameters": [
                {
                    "name": "redirect_url",
                    "in": "query",
                    "description": "URL to redirect after successful authentication",
                    "schema": {"type": "string", "format": "uri"},
                },
            ],
            "responses": {
                "302": {"description": "Redirect to GitHub OAuth consent screen"},
                "400": STANDARD_ERRORS["400"],
                "429": STANDARD_ERRORS["429"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/auth/oauth/github/callback": {
        "get": {
            "tags": ["OAuth"],
            "summary": "GitHub OAuth callback",
            "operationId": "listAuthOauthGithubCallback",
            "description": "Handle the OAuth callback from GitHub after user consent.",
            "parameters": [
                {
                    "name": "code",
                    "in": "query",
                    "required": True,
                    "description": "Authorization code from GitHub",
                    "schema": {"type": "string"},
                },
                {
                    "name": "state",
                    "in": "query",
                    "required": True,
                    "description": "State parameter for CSRF protection",
                    "schema": {"type": "string"},
                },
            ],
            "responses": {
                "302": {"description": "Redirect to success URL with auth token"},
                "400": STANDARD_ERRORS["400"],
                "500": STANDARD_ERRORS["500"],
            },
        },
    },
    "/api/auth/oauth/link": {
        "post": {
            "tags": ["OAuth"],
            "summary": "Link OAuth account",
            "operationId": "createAuthOauthLink",
            "description": "Link an OAuth provider account to the current authenticated user.",
            "security": [{"bearerAuth": []}],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "required": ["provider", "code"],
                            "properties": {
                                "provider": {
                                    "type": "string",
                                    "enum": ["google", "github"],
                                    "description": "OAuth provider name",
                                },
                                "code": {
                                    "type": "string",
                                    "description": "Authorization code from provider",
                                },
                            },
                        },
                    },
                },
            },
            "responses": {
                "200": _ok_response("Account linked successfully"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
            },
        },
    },
    "/api/auth/oauth/unlink": {
        "delete": {
            "tags": ["OAuth"],
            "summary": "Unlink OAuth account",
            "operationId": "deleteAuthOauthUnlink",
            "description": "Remove OAuth provider link from the current user's account.",
            "security": [{"bearerAuth": []}],
            "parameters": [
                {
                    "name": "provider",
                    "in": "query",
                    "required": True,
                    "description": "OAuth provider to unlink",
                    "schema": {"type": "string", "enum": ["google", "github"]},
                },
            ],
            "responses": {
                "200": _ok_response("Account unlinked successfully"),
                "400": STANDARD_ERRORS["400"],
                "401": STANDARD_ERRORS["401"],
                "404": STANDARD_ERRORS["404"],
            },
        },
    },
    "/api/auth/oauth/providers": {
        "get": {
            "tags": ["OAuth"],
            "summary": "List available OAuth providers",
            "operationId": "listAuthOauthProviders",
            "description": "Get list of configured OAuth providers available for authentication.",
            "responses": {
                "200": _ok_response("List of available providers", "OAuthProviders"),
            },
        },
    },
    "/api/user/oauth-providers": {
        "get": {
            "tags": ["OAuth"],
            "summary": "Get user's linked providers",
            "operationId": "listUserOauthProviders",
            "description": "Get list of OAuth providers linked to the current user's account.",
            "security": [{"bearerAuth": []}],
            "responses": {
                "200": _ok_response("List of linked providers"),
                "401": STANDARD_ERRORS["401"],
            },
        },
    },
}
