"""
API Explorer Endpoints (FastAPI v2).

Serves the combined OpenAPI specification and interactive documentation:
- GET /api/v2/explorer/openapi.json  - Full merged OpenAPI spec (v1 legacy + v2 FastAPI)
- GET /api/v2/explorer/stats         - Endpoint statistics (counts by tag, method)
- GET /api/v2/explorer/swagger       - Swagger UI pointed at the full spec
- GET /api/v2/explorer/redoc         - ReDoc pointed at the full spec

The full spec merges:
1. The legacy handler-based endpoints (~2000 operations from openapi_impl)
2. The FastAPI v2 endpoints (auto-generated from Pydantic models)

This powers the /api-explorer frontend page.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/explorer", tags=["API Explorer"])

# Cache the merged spec for 10 minutes
_spec_cache: dict[str, Any] | None = None
_spec_cache_time: float = 0.0
_CACHE_TTL = 600.0


def _generate_full_spec(fastapi_app: Any = None) -> dict[str, Any]:
    """Generate the full merged OpenAPI specification.

    Merges the legacy handler-based spec with the FastAPI v2 spec.
    The legacy spec has ~2000 operations; FastAPI adds ~50 more.
    """
    import time

    global _spec_cache, _spec_cache_time

    now = time.time()
    if _spec_cache is not None and (now - _spec_cache_time) < _CACHE_TTL:
        return _spec_cache

    # Start with the legacy spec (the comprehensive one)
    try:
        from aragora.server.openapi_impl import generate_openapi_schema

        spec = generate_openapi_schema()
    except (ImportError, RuntimeError, ValueError) as e:
        logger.warning("Failed to load legacy OpenAPI spec: %s", e)
        spec = {
            "openapi": "3.1.0",
            "info": {
                "title": "Aragora API",
                "version": "2.0.0",
                "description": "Multi-agent debate orchestration platform",
            },
            "paths": {},
            "components": {"schemas": {}, "securitySchemes": {}},
        }

    # Merge FastAPI v2 spec paths
    if fastapi_app is not None:
        try:
            v2_spec = fastapi_app.openapi()
            if v2_spec and "paths" in v2_spec:
                for path, methods in v2_spec["paths"].items():
                    if path not in spec.get("paths", {}):
                        spec.setdefault("paths", {})[path] = methods
                    else:
                        # Merge methods for shared paths
                        for method, operation in methods.items():
                            if method not in spec["paths"][path]:
                                spec["paths"][path][method] = operation

                # Merge v2 component schemas
                v2_schemas = v2_spec.get("components", {}).get("schemas", {})
                if v2_schemas:
                    spec.setdefault("components", {}).setdefault("schemas", {}).update(v2_schemas)
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.debug("Could not merge FastAPI v2 spec: %s", e)

    # Ensure security schemes are comprehensive
    sec_schemes = spec.setdefault("components", {}).setdefault("securitySchemes", {})
    if "bearerAuth" not in sec_schemes:
        sec_schemes["bearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token from /api/v1/auth/login or ARAGORA_API_TOKEN env var.",
        }
    if "apiKeyAuth" not in sec_schemes:
        sec_schemes["apiKeyAuth"] = {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key from /api/v1/auth/api-keys.",
        }

    # Update metadata for the merged spec
    spec["info"] = {
        "title": "Aragora API",
        "version": "2.0.0",
        "description": (
            "Decision Integrity Platform API -- orchestrating 42 agent types to "
            "adversarially vet decisions against your organization's knowledge, "
            "then delivering audit-ready decision receipts to any channel.\n\n"
            "This specification covers all endpoints across v1 (legacy) and v2 (FastAPI) "
            "surfaces. Use the tag filter to browse by category."
        ),
        "contact": {"name": "Aragora Team"},
        "license": {"name": "MIT"},
    }

    # Update servers
    spec["servers"] = [
        {"url": "http://localhost:8080", "description": "Development server"},
        {"url": "https://api.aragora.ai", "description": "Production server"},
    ]

    _spec_cache = spec
    _spec_cache_time = now

    return spec


@router.get(
    "/openapi.json",
    summary="Full OpenAPI specification",
    description="Returns the complete merged OpenAPI 3.1 spec covering all 2000+ API operations.",
    response_class=JSONResponse,
    include_in_schema=False,
)
async def get_full_openapi_spec(request: Request) -> JSONResponse:
    """Serve the full merged OpenAPI specification.

    Combines the legacy handler spec (~2000 ops) with FastAPI v2 routes.
    Cached for 10 minutes.
    """
    try:
        spec = _generate_full_spec(request.app)
        return JSONResponse(
            content=spec,
            headers={
                "Cache-Control": "public, max-age=600",
                "Access-Control-Allow-Origin": "*",
            },
        )
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        logger.exception("Failed to generate full OpenAPI spec: %s", e)
        return JSONResponse(
            content={"error": "Failed to generate OpenAPI spec"},
            status_code=500,
        )


@router.get(
    "/stats",
    summary="API statistics",
    description="Returns endpoint counts grouped by tag and HTTP method.",
)
async def get_api_stats(request: Request) -> dict[str, Any]:
    """Return API statistics: endpoint counts by tag and method."""
    try:
        spec = _generate_full_spec(request.app)
        paths = spec.get("paths", {})

        tag_counts: dict[str, int] = {}
        method_counts: dict[str, int] = {}
        total = 0

        for _path, methods in paths.items():
            for method, details in methods.items():
                if method.startswith("x-") or method == "parameters":
                    continue
                if not isinstance(details, dict):
                    continue
                total += 1
                method_upper = method.upper()
                method_counts[method_upper] = method_counts.get(method_upper, 0) + 1

                tags = details.get("tags", ["Untagged"])
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

        sorted_tags = sorted(tag_counts.items(), key=lambda x: -x[1])

        return {
            "total_endpoints": total,
            "total_paths": len(paths),
            "by_method": method_counts,
            "by_tag": [{"tag": t, "count": c} for t, c in sorted_tags],
        }
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        logger.exception("Failed to generate API stats: %s", e)
        return {"error": "Failed to generate stats", "total_endpoints": 0}


@router.get(
    "/swagger",
    summary="Swagger UI (full spec)",
    description="Interactive Swagger UI with all 2000+ API operations.",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def swagger_ui() -> HTMLResponse:
    """Serve Swagger UI pointed at the full merged spec."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aragora API Explorer</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
    <style>
        html { box-sizing: border-box; overflow-y: scroll; }
        *, *:before, *:after { box-sizing: inherit; }
        body { margin: 0; background: #0a0a0a; color: #e0e0e0; }
        .swagger-ui { background: #0a0a0a; }
        .swagger-ui .topbar { background: #111; border-bottom: 1px solid #00ff41; }
        .swagger-ui .topbar .download-url-wrapper .select-label { color: #00ff41; }
        .swagger-ui .info { margin: 20px 0; }
        .swagger-ui .info .title { font-size: 2em; color: #00ff41; }
        .swagger-ui .info .description p { color: #ccc; }
        .swagger-ui .opblock-tag { color: #00ff41; border-bottom-color: #333; }
        .swagger-ui .opblock .opblock-summary-method { font-family: monospace; }
        .swagger-ui .btn.authorize { color: #00ff41; border-color: #00ff41; }
        .swagger-ui .btn.authorize svg { fill: #00ff41; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            window.ui = SwaggerUIBundle({
                url: "/api/v2/explorer/openapi.json",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                validatorUrl: null,
                docExpansion: "list",
                defaultModelsExpandDepth: 1,
                displayRequestDuration: true,
                filter: true,
                showExtensions: true,
                showCommonExtensions: true,
                persistAuthorization: true,
                tryItOutEnabled: true
            });
        };
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@router.get(
    "/redoc",
    summary="ReDoc (full spec)",
    description="ReDoc documentation viewer with all 2000+ API operations.",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def redoc_ui() -> HTMLResponse:
    """Serve ReDoc pointed at the full merged spec."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aragora API Reference</title>
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700"
          rel="stylesheet">
    <style>
        body { margin: 0; padding: 0; }
    </style>
</head>
<body>
    <redoc spec-url="/api/v2/explorer/openapi.json"
           expand-responses="200,201"
           hide-download-button="false"
           native-scrollbars="true"
           path-in-middle-panel="true"
           theme='{"colors":{"primary":{"main":"#00ff41"}},"typography":{"fontFamily":"Roboto, sans-serif"}}'>
    </redoc>
    <script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>
</body>
</html>"""
    return HTMLResponse(content=html)


def invalidate_cache() -> None:
    """Invalidate the cached spec (call after endpoint registration changes)."""
    global _spec_cache, _spec_cache_time
    _spec_cache = None
    _spec_cache_time = 0.0
