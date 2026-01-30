"""
FastAPI Application Factory.

Creates and configures the FastAPI application with:
- Middleware (auth, RBAC, rate limiting, tracing)
- Route registration
- Server context injection
- Lifespan management
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .middleware.tracing import TracingMiddleware
from .middleware.validation import RequestValidationMiddleware
from .middleware.error_handling import setup_exception_handlers
from .routes import health, debates, decisions

logger = logging.getLogger(__name__)


def _get_allowed_origins() -> list[str]:
    """Get allowed CORS origins from environment."""
    origins_str = os.environ.get("ARAGORA_ALLOWED_ORIGINS", "*")
    if origins_str == "*":
        return ["*"]
    return [o.strip() for o in origins_str.split(",") if o.strip()]


def _build_server_context(nomic_dir: Path | None = None) -> dict[str, Any]:
    """
    Build the server context with initialized subsystems.

    This provides the same context as the legacy server for handler compatibility.
    """
    from aragora.storage.debate_storage import DebateStorage

    ctx: dict[str, Any] = {}

    # Initialize storage
    try:
        storage = DebateStorage(nomic_dir=nomic_dir)
        ctx["storage"] = storage
        logger.info("Initialized DebateStorage")
    except (OSError, IOError, RuntimeError) as e:
        logger.warning(f"Failed to initialize DebateStorage: {e}")
        ctx["storage"] = None

    # Initialize ELO system
    try:
        from aragora.ranking.elo import EloSystem

        ctx["elo_system"] = EloSystem()
        logger.info("Initialized EloSystem")
    except (ImportError, RuntimeError, ValueError) as e:
        logger.warning(f"Failed to initialize EloSystem: {e}")
        ctx["elo_system"] = None

    # Initialize user store (optional)
    try:
        from aragora.storage.user_store import get_user_store

        ctx["user_store"] = get_user_store()
    except (ImportError, OSError, RuntimeError) as e:
        logger.debug(f"User store not available: {e}")
        ctx["user_store"] = None

    # Initialize RBAC checker
    try:
        from aragora.rbac.checker import get_permission_checker

        ctx["rbac_checker"] = get_permission_checker()
    except (ImportError, RuntimeError, ValueError) as e:
        logger.warning(f"Failed to initialize RBAC checker: {e}")
        ctx["rbac_checker"] = None

    # Initialize DecisionService
    try:
        from aragora.debate.decision_service import get_decision_service

        ctx["decision_service"] = get_decision_service()
        logger.info("Initialized DecisionService")
    except (ImportError, RuntimeError, ValueError) as e:
        logger.warning(f"Failed to initialize DecisionService: {e}")
        ctx["decision_service"] = None

    return ctx


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan - startup and shutdown.

    Initializes server context on startup and cleans up on shutdown.
    """
    logger.info("FastAPI server starting up...")

    # Initialize server context
    nomic_dir = Path(os.environ.get("ARAGORA_NOMIC_DIR", "."))
    ctx = _build_server_context(nomic_dir)
    app.state.context = ctx

    logger.info("FastAPI server ready")

    yield

    # Cleanup on shutdown
    logger.info("FastAPI server shutting down...")

    # Cancel any running debates
    decision_service = ctx.get("decision_service")
    if decision_service:
        for task in getattr(decision_service, "_running_tasks", {}).values():
            if not task.done():
                task.cancel()

    if ctx.get("storage"):
        try:
            ctx["storage"].close()
        except (OSError, IOError, RuntimeError) as e:
            logger.debug("Error closing storage during shutdown: %s", e)


def create_app(
    nomic_dir: Path | None = None,
    title: str = "Aragora API",
    version: str = "2.0.0",
    debug: bool = False,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        nomic_dir: Directory for nomic data storage
        title: API title for OpenAPI docs
        version: API version
        debug: Enable debug mode

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        version=version,
        description="Multi-agent debate orchestration platform",
        docs_url="/api/v2/docs",
        redoc_url="/api/v2/redoc",
        openapi_url="/api/v2/openapi.json",
        debug=debug,
        lifespan=lifespan,
    )

    # Add middleware (order matters - first added is outermost)

    # CORS middleware
    allowed_origins = _get_allowed_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Trace-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )

    # Tracing middleware
    app.add_middleware(TracingMiddleware)

    # Request validation middleware (body size, JSON depth, array limits)
    app.add_middleware(RequestValidationMiddleware)

    # Register routes
    app.include_router(health.router)
    app.include_router(debates.router)
    app.include_router(decisions.router)

    # Setup exception handlers
    setup_exception_handlers(app)

    # Add root redirect
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "Aragora API v2", "docs": "/api/v2/docs"}

    logger.info(f"FastAPI app created: {title} v{version}")

    return app


# Default app instance for uvicorn
app = create_app()
