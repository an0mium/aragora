"""
Conftest for vector abstraction tests.

Installs mock modules for optional vector DB dependencies (qdrant-client,
chromadb) so that tests using mocked backends can run without the real
libraries installed.  When the real library IS installed the mock is
skipped and the genuine module is used instead.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock


def _install_mock_qdrant_client() -> None:
    """Install a mock ``qdrant_client`` package into sys.modules."""
    if "qdrant_client" in sys.modules and sys.modules["qdrant_client"] is not None:
        return  # Real library already imported

    # Create mock module hierarchy
    qdrant_client = ModuleType("qdrant_client")
    qdrant_client.QdrantClient = MagicMock  # type: ignore[attr-defined]
    qdrant_client.AsyncQdrantClient = MagicMock  # type: ignore[attr-defined]

    qdrant_client_http = ModuleType("qdrant_client.http")

    # Use MagicMock as the models module so any attribute (Distance,
    # PointStruct, PointIdsList, ...) returns a usable mock automatically.
    qdrant_client_http_models = MagicMock()
    qdrant_client_http_models.__name__ = "qdrant_client.http.models"

    qdrant_client_http.models = qdrant_client_http_models  # type: ignore[attr-defined]

    sys.modules["qdrant_client"] = qdrant_client
    sys.modules["qdrant_client.http"] = qdrant_client_http
    sys.modules["qdrant_client.http.models"] = qdrant_client_http_models

    # Reload the production module so it picks up the mock and sets QDRANT_AVAILABLE = True
    prod_mod_name = "aragora.knowledge.mound.vector_abstraction.qdrant"
    if prod_mod_name in sys.modules:
        importlib.reload(sys.modules[prod_mod_name])


def _install_mock_chromadb() -> None:
    """Install a mock ``chromadb`` package into sys.modules."""
    if "chromadb" in sys.modules and sys.modules["chromadb"] is not None:
        return  # Real library already imported

    chromadb_mod = ModuleType("chromadb")
    chromadb_mod.PersistentClient = MagicMock  # type: ignore[attr-defined]
    chromadb_mod.HttpClient = MagicMock  # type: ignore[attr-defined]
    chromadb_mod.ClientAPI = MagicMock  # type: ignore[attr-defined]

    chromadb_config = ModuleType("chromadb.config")
    chromadb_config.Settings = MagicMock  # type: ignore[attr-defined]

    sys.modules["chromadb"] = chromadb_mod
    sys.modules["chromadb.config"] = chromadb_config

    # Reload the production module so it picks up the mock and sets CHROMA_AVAILABLE = True
    prod_mod_name = "aragora.knowledge.mound.vector_abstraction.chroma"
    if prod_mod_name in sys.modules:
        importlib.reload(sys.modules[prod_mod_name])


# Install mocks at import time (before test collection reads the test module)
_install_mock_qdrant_client()
_install_mock_chromadb()
