"""Unified Connector Runtime Registry.

Discovers, registers, and health-checks all installed connectors at runtime.
Unlike the static ``registry.py`` which scans source files on disk, this module
maintains a live, singleton registry that tracks connector health and status
during server operation.

Phase Y: Connector Consolidation.
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ConnectorStatus(Enum):
    """Runtime health status for a connector."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ConnectorInfo:
    """Metadata and live status for a single connector."""

    name: str
    connector_type: str  # "chat", "payment", "ecommerce", "enterprise", "ai", "memory"
    module_path: str
    status: ConnectorStatus = ConnectorStatus.UNKNOWN
    last_health_check: float | None = None
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "name": self.name,
            "connector_type": self.connector_type,
            "module_path": self.module_path,
            "status": self.status.value,
            "last_health_check": self.last_health_check,
            "capabilities": list(self.capabilities),
            "metadata": dict(self.metadata),
        }


class ConnectorRegistry:
    """Central runtime registry for all connectors.

    Provides singleton access, automatic discovery via importability probes,
    per-connector and bulk health checking, and filtering helpers.
    """

    _instance: ConnectorRegistry | None = None

    def __init__(self) -> None:
        self._connectors: dict[str, ConnectorInfo] = {}
        self._discover_connectors()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> ConnectorRegistry:
        """Return the global singleton, creating it on first access."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (primarily for testing)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_connectors(self) -> None:
        """Scan known connector module paths and register importable ones."""

        # Chat connectors
        self._try_register(
            "telegram", "chat", "aragora.connectors.chat.telegram", ["messaging", "webhooks"]
        )
        self._try_register(
            "whatsapp", "chat", "aragora.connectors.chat.whatsapp", ["messaging", "webhooks"]
        )
        self._try_register(
            "signal", "chat", "aragora.connectors.chat.signal", ["messaging", "webhooks"]
        )
        self._try_register("imessage", "chat", "aragora.connectors.chat.imessage", ["messaging"])
        self._try_register(
            "slack", "chat", "aragora.connectors.chat.slack", ["messaging", "slash_commands"]
        )
        self._try_register(
            "teams", "chat", "aragora.connectors.chat.teams", ["messaging", "adaptive_cards"]
        )
        self._try_register(
            "discord", "chat", "aragora.connectors.chat.discord", ["messaging", "slash_commands"]
        )
        self._try_register(
            "google_chat", "chat", "aragora.connectors.chat.google_chat", ["messaging"]
        )

        # Payment connectors
        self._try_register(
            "stripe", "payment", "aragora.connectors.payments.stripe", ["payments", "subscriptions"]
        )
        self._try_register("paypal", "payment", "aragora.connectors.payments.paypal", ["payments"])
        self._try_register("square", "payment", "aragora.connectors.payments.square", ["payments"])
        self._try_register(
            "authorize_net", "payment", "aragora.connectors.payments.authorize_net", ["payments"]
        )

        # Enterprise connectors
        self._try_register(
            "kafka",
            "enterprise",
            "aragora.connectors.enterprise.streaming.kafka",
            ["streaming", "events"],
        )
        self._try_register(
            "rabbitmq",
            "enterprise",
            "aragora.connectors.enterprise.streaming.rabbitmq",
            ["streaming", "events"],
        )
        self._try_register(
            "fasb",
            "ai",
            "aragora.connectors.accounting.gaap",
            ["gaap", "standards"],
        )
        self._try_register(
            "irs",
            "ai",
            "aragora.connectors.accounting.irs",
            ["tax", "guidance"],
        )
        self._try_register(
            "westlaw",
            "ai",
            "aragora.connectors.legal.westlaw",
            ["case_law", "research"],
        )
        self._try_register(
            "lexis",
            "ai",
            "aragora.connectors.legal.lexis",
            ["case_law", "research"],
        )

        # Ecommerce connectors
        self._try_register(
            "shopify",
            "ecommerce",
            "aragora.connectors.ecommerce.shopify",
            ["webhooks", "orders", "products"],
        )
        self._try_register(
            "woocommerce",
            "ecommerce",
            "aragora.connectors.ecommerce.woocommerce",
            ["webhooks", "orders"],
        )
        self._try_register(
            "amazon", "ecommerce", "aragora.connectors.ecommerce.amazon", ["orders", "products"]
        )
        self._try_register(
            "shipstation",
            "ecommerce",
            "aragora.connectors.ecommerce.shipstation",
            ["shipping", "fulfillment"],
        )

        # Evidence / AI connectors
        self._try_register("github", "ai", "aragora.connectors.github", ["search", "issues", "prs"])
        self._try_register("arxiv", "ai", "aragora.connectors.arxiv", ["search", "papers"])
        self._try_register(
            "courtlistener",
            "ai",
            "aragora.connectors.courtlistener",
            ["search", "case_law"],
        )
        self._try_register(
            "govinfo",
            "ai",
            "aragora.connectors.govinfo",
            ["search", "statutes"],
        )
        self._try_register("pubmed", "ai", "aragora.connectors.pubmed", ["search", "papers"])
        self._try_register(
            "semantic_scholar",
            "ai",
            "aragora.connectors.semantic_scholar",
            ["search", "papers"],
        )
        self._try_register(
            "nice_guidance",
            "ai",
            "aragora.connectors.nice_guidance",
            ["search", "guidelines"],
        )
        self._try_register("crossref", "ai", "aragora.connectors.crossref", ["search", "citations"])
        self._try_register(
            "wikipedia", "ai", "aragora.connectors.wikipedia", ["search", "articles"]
        )
        self._try_register("reddit", "ai", "aragora.connectors.reddit", ["search", "posts"])
        self._try_register(
            "hackernews", "ai", "aragora.connectors.hackernews", ["search", "stories"]
        )
        self._try_register("twitter", "ai", "aragora.connectors.twitter", ["search", "posts"])
        self._try_register("web", "ai", "aragora.connectors.web", ["search"])
        self._try_register(
            "clinical_tables",
            "ai",
            "aragora.connectors.clinical_tables",
            ["icd", "lookup"],
        )
        self._try_register("rxnav", "ai", "aragora.connectors.rxnav", ["drug", "lookup"])

        # Memory connectors
        self._try_register(
            "claude_mem",
            "memory",
            "aragora.connectors.memory.claude_mem",
            ["search", "observations"],
        )

    def _try_register(
        self,
        name: str,
        connector_type: str,
        module_path: str,
        capabilities: list[str],
    ) -> None:
        """Attempt to import *module_path* and register the connector.

        If the module is not importable the connector is still recorded but
        with ``UNKNOWN`` status so the management API can report it.
        """
        status = ConnectorStatus.UNKNOWN
        metadata: dict[str, Any] = {}

        try:
            importlib.import_module(module_path)
            status = ConnectorStatus.HEALTHY
            metadata["importable"] = True
        except ImportError as exc:
            status = ConnectorStatus.UNHEALTHY
            metadata["importable"] = False
            metadata["import_error"] = str(exc)
            logger.debug("Connector %s not importable: %s", name, exc)
        except Exception as exc:  # noqa: BLE001
            status = ConnectorStatus.DEGRADED
            metadata["importable"] = False
            metadata["import_error"] = str(exc)
            logger.warning("Connector %s import error: %s", name, exc)

        info = ConnectorInfo(
            name=name,
            connector_type=connector_type,
            module_path=module_path,
            status=status,
            last_health_check=time.time(),
            capabilities=list(capabilities),
            metadata=metadata,
        )
        self._connectors[name] = info

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------

    def register(self, info: ConnectorInfo) -> None:
        """Manually register (or overwrite) a connector."""
        self._connectors[info.name] = info

    def unregister(self, name: str) -> bool:
        """Remove a connector from the registry. Returns True if it existed."""
        return self._connectors.pop(name, None) is not None

    def get(self, name: str) -> ConnectorInfo | None:
        """Return info for *name*, or ``None`` if not registered."""
        return self._connectors.get(name)

    def list_all(self) -> list[ConnectorInfo]:
        """Return all registered connectors sorted by name."""
        return sorted(self._connectors.values(), key=lambda c: c.name)

    def list_by_type(self, connector_type: str) -> list[ConnectorInfo]:
        """Return connectors filtered by *connector_type*."""
        return sorted(
            (c for c in self._connectors.values() if c.connector_type == connector_type),
            key=lambda c: c.name,
        )

    def list_by_status(self, status: ConnectorStatus) -> list[ConnectorInfo]:
        """Return connectors filtered by *status*."""
        return sorted(
            (c for c in self._connectors.values() if c.status == status),
            key=lambda c: c.name,
        )

    # ------------------------------------------------------------------
    # Health checking
    # ------------------------------------------------------------------

    def health_check(self, name: str) -> ConnectorStatus:
        """Run a health check for a single connector.

        The check re-imports the module to verify importability and, if the
        module exposes a ``health_check`` callable, invokes it.
        """
        info = self._connectors.get(name)
        if info is None:
            return ConnectorStatus.UNKNOWN

        try:
            mod = importlib.import_module(info.module_path)
            info.metadata["importable"] = True

            # If the module exposes a health_check helper, call it.
            checker = getattr(mod, "health_check", None)
            if callable(checker):
                result = checker()
                if isinstance(result, bool):
                    info.status = ConnectorStatus.HEALTHY if result else ConnectorStatus.UNHEALTHY
                elif isinstance(result, dict):
                    healthy = result.get("healthy", result.get("is_healthy", False))
                    info.status = ConnectorStatus.HEALTHY if healthy else ConnectorStatus.DEGRADED
                    info.metadata.update(result)
                else:
                    info.status = ConnectorStatus.HEALTHY
            else:
                info.status = ConnectorStatus.HEALTHY

        except ImportError as exc:
            info.status = ConnectorStatus.UNHEALTHY
            info.metadata["importable"] = False
            info.metadata["import_error"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            info.status = ConnectorStatus.DEGRADED
            info.metadata["health_error"] = str(exc)

        info.last_health_check = time.time()
        return info.status

    def health_check_all(self) -> dict[str, ConnectorStatus]:
        """Run health checks on every registered connector."""
        return {name: self.health_check(name) for name in self._connectors}

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Return an aggregate summary of the registry."""
        total = len(self._connectors)
        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}

        for info in self._connectors.values():
            by_type[info.connector_type] = by_type.get(info.connector_type, 0) + 1
            by_status[info.status.value] = by_status.get(info.status.value, 0) + 1

        return {
            "total": total,
            "by_type": dict(sorted(by_type.items())),
            "by_status": dict(sorted(by_status.items())),
            "connectors": [c.to_dict() for c in self.list_all()],
        }


def get_connector_registry() -> ConnectorRegistry:
    """Module-level convenience accessor for the singleton."""
    return ConnectorRegistry.get_instance()
