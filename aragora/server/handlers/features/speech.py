"""
Speech-to-text API handlers.

The ``aragora.speech`` module has been removed. All endpoints now return
501 Not Implemented so that callers receive an explicit signal rather than
a silent failure.

Endpoints:
- POST /api/speech/transcribe      -> 501
- POST /api/speech/transcribe-url  -> 501
- GET  /api/speech/providers       -> 501
"""

from __future__ import annotations

import logging

from ..base import BaseHandler, HandlerResult, json_response

logger = logging.getLogger(__name__)

_NOT_IMPLEMENTED_BODY = {
    "error": "Speech transcription has been removed from this server.",
    "code": "NOT_IMPLEMENTED",
    "status": 501,
}


class SpeechHandler(BaseHandler):
    """Stub handler -- speech module has been removed."""

    ROUTES = [
        "/api/v1/speech/transcribe",
        "/api/v1/speech/transcribe-url",
        "/api/v1/speech/providers",
    ]

    def __init__(self, ctx: dict | None = None, server_context: dict | None = None):
        self.ctx = server_context or ctx or {}

    def can_handle(self, path: str) -> bool:
        return path in self.ROUTES

    def handle(self, path: str, query_params: dict, handler) -> HandlerResult | None:
        if path in self.ROUTES:
            return json_response(_NOT_IMPLEMENTED_BODY, status=501)
        return None

    def handle_post(self, path: str, query_params: dict, handler) -> HandlerResult | None:
        if path in self.ROUTES:
            return json_response(_NOT_IMPLEMENTED_BODY, status=501)
        return None
