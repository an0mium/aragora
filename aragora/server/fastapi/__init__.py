"""
FastAPI Server Implementation for Aragora.

This module provides an async FastAPI server that can run alongside
the existing ThreadingHTTPServer for gradual migration.

Usage:
    # Run FastAPI server
    uvicorn aragora.server.app:app --host 0.0.0.0 --port 8081

    # Or programmatically
    from aragora.server.app import create_app
    app = create_app()
"""

from .factory import create_app

__all__ = ["create_app"]
