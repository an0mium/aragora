"""
Shared imports and constants for API-based agents.

This module provides common imports used across all agent implementations
to avoid code duplication.
"""

import asyncio
import aiohttp
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

from aragora.agents.base import CritiqueMixin
from aragora.agents.errors import (
    AgentConnectionError,
    AgentRateLimitError,
    AgentTimeoutError,
    handle_agent_errors,
)
from aragora.agents.registry import AgentRegistry
from aragora.config import DB_TIMEOUT_SECONDS, get_api_key
from aragora.core import Agent, Critique, Message
from aragora.server.error_utils import sanitize_error_text as _sanitize_error_message

logger = logging.getLogger(__name__)

# Maximum buffer size for streaming responses (prevents DoS via memory exhaustion)
MAX_STREAM_BUFFER_SIZE = 10 * 1024 * 1024  # 10MB

__all__ = [
    # Standard library
    "asyncio",
    "aiohttp",
    "json",
    "logging",
    "os",
    "re",
    "threading",
    "time",
    "dataclass",
    "Optional",
    # Aragora imports
    "CritiqueMixin",
    "AgentConnectionError",
    "AgentRateLimitError",
    "AgentTimeoutError",
    "handle_agent_errors",
    "AgentRegistry",
    "DB_TIMEOUT_SECONDS",
    "get_api_key",
    "Agent",
    "Critique",
    "Message",
    "_sanitize_error_message",
    # Module-level
    "logger",
    "MAX_STREAM_BUFFER_SIZE",
]
