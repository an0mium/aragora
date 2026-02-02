"""Skip MCP tests when optional dependency is missing."""

import pytest

pytest.importorskip("mcp")
