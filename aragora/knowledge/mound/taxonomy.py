"""
Domain Taxonomy for hierarchical knowledge organization.

Provides hierarchical organization of knowledge domains with auto-classification.
Supports industry-specific taxonomies for enterprise use cases.

Usage:
    from aragora.knowledge.mound.taxonomy import DomainTaxonomy

    taxonomy = DomainTaxonomy(graph_store)

    # Get or create a domain path
    domain_id = await taxonomy.ensure_path(["legal", "contracts", "termination"])

    # Auto-classify content
    domain = await taxonomy.classify("This contract has a 90-day notice period")
    # Returns: "legal/contracts"

    # Get all children of a domain
    children = await taxonomy.get_children("legal")
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from aragora.knowledge.mound.graph_store import KnowledgeGraphStore

logger = logging.getLogger(__name__)


@dataclass
class TaxonomyNode:
    """A node in the domain taxonomy tree."""

    id: str
    name: str
    parent_id: Optional[str]
    full_path: str  # e.g., "legal/contracts/termination"
    description: Optional[str]
    tenant_id: str
    created_at: datetime
    children: List["TaxonomyNode"] = field(default_factory=list)


# Default industry taxonomies
DEFAULT_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "legal": {
        "description": "Legal and compliance domain",
        "children": {
            "contracts": {
                "description": "Contract analysis and review",
                "children": {
                    "termination": {"description": "Contract termination clauses"},
                    "liability": {"description": "Liability and indemnification"},
                    "payment": {"description": "Payment terms and conditions"},
                    "confidentiality": {"description": "NDAs and confidentiality"},
                },
            },
            "compliance": {
                "description": "Regulatory compliance",
                "children": {
                    "gdpr": {"description": "GDPR data protection"},
                    "hipaa": {"description": "Healthcare privacy"},
                    "sox": {"description": "Sarbanes-Oxley compliance"},
                    "pci": {"description": "PCI-DSS payment security"},
                },
            },
            "litigation": {
                "description": "Litigation and disputes",
                "children": {
                    "discovery": {"description": "E-discovery and document review"},
                    "evidence": {"description": "Evidence analysis"},
                },
            },
            "intellectual_property": {
                "description": "IP and patents",
                "children": {
                    "patents": {"description": "Patent analysis"},
                    "trademarks": {"description": "Trademark review"},
                    "copyright": {"description": "Copyright matters"},
                },
            },
        },
    },
    "financial": {
        "description": "Financial and accounting domain",
        "children": {
            "accounting": {
                "description": "Accounting standards and practices",
                "children": {
                    "gaap": {"description": "US GAAP compliance"},
                    "ifrs": {"description": "International standards"},
                    "revenue": {"description": "Revenue recognition"},
                },
            },
            "audit": {
                "description": "Audit and assurance",
                "children": {
                    "internal": {"description": "Internal audit"},
                    "external": {"description": "External audit"},
                    "controls": {"description": "Internal controls"},
                },
            },
            "tax": {
                "description": "Tax planning and compliance",
                "children": {
                    "corporate": {"description": "Corporate tax"},
                    "international": {"description": "International tax"},
                    "transfer_pricing": {"description": "Transfer pricing"},
                },
            },
            "treasury": {
                "description": "Treasury and cash management",
            },
        },
    },
    "technical": {
        "description": "Technical and software domain",
        "children": {
            "architecture": {
                "description": "Software architecture",
                "children": {
                    "microservices": {"description": "Microservices patterns"},
                    "monolith": {"description": "Monolithic systems"},
                    "event_driven": {"description": "Event-driven architecture"},
                },
            },
            "security": {
                "description": "Security and vulnerabilities",
                "children": {
                    "authentication": {"description": "Auth and identity"},
                    "encryption": {"description": "Encryption and cryptography"},
                    "vulnerabilities": {"description": "Security vulnerabilities"},
                    "penetration": {"description": "Penetration testing"},
                },
            },
            "infrastructure": {
                "description": "Infrastructure and DevOps",
                "children": {
                    "cloud": {"description": "Cloud infrastructure"},
                    "containers": {"description": "Containers and orchestration"},
                    "networking": {"description": "Network architecture"},
                },
            },
            "data": {
                "description": "Data engineering and analytics",
                "children": {
                    "pipelines": {"description": "Data pipelines"},
                    "warehousing": {"description": "Data warehousing"},
                    "ml": {"description": "Machine learning"},
                },
            },
            "performance": {
                "description": "Performance optimization",
            },
            "testing": {
                "description": "Testing and QA",
            },
        },
    },
    "healthcare": {
        "description": "Healthcare and medical domain",
        "children": {
            "clinical": {
                "description": "Clinical documentation",
                "children": {
                    "diagnosis": {"description": "Diagnostic records"},
                    "treatment": {"description": "Treatment plans"},
                    "outcomes": {"description": "Patient outcomes"},
                },
            },
            "research": {
                "description": "Medical research",
                "children": {
                    "trials": {"description": "Clinical trials"},
                    "literature": {"description": "Literature review"},
                },
            },
            "compliance": {
                "description": "Healthcare compliance",
                "children": {
                    "hipaa": {"description": "HIPAA compliance"},
                    "fda": {"description": "FDA regulations"},
                },
            },
        },
    },
    "operational": {
        "description": "Business operations domain",
        "children": {
            "hr": {
                "description": "Human resources",
                "children": {
                    "recruitment": {"description": "Hiring and recruitment"},
                    "policies": {"description": "HR policies"},
                    "compensation": {"description": "Compensation and benefits"},
                },
            },
            "procurement": {
                "description": "Procurement and vendors",
            },
            "logistics": {
                "description": "Logistics and supply chain",
            },
        },
    },
    "general": {
        "description": "General/uncategorized knowledge",
    },
}

# Keywords for auto-classification
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "legal/contracts": [
        "contract", "agreement", "clause", "term", "party", "parties",
        "obligation", "breach", "termination", "notice period", "indemnity",
    ],
    "legal/compliance": [
        "compliance", "regulation", "regulatory", "requirement", "audit",
        "violation", "penalty", "enforcement",
    ],
    "legal/compliance/gdpr": [
        "gdpr", "data protection", "privacy", "personal data", "consent",
        "data subject", "controller", "processor",
    ],
    "legal/compliance/hipaa": [
        "hipaa", "phi", "protected health", "covered entity", "baa",
    ],
    "financial/accounting": [
        "accounting", "gaap", "ifrs", "financial statement", "balance sheet",
        "income statement", "revenue", "expense", "accrual",
    ],
    "financial/audit": [
        "audit", "auditor", "internal control", "material weakness",
        "opinion", "assurance", "attestation",
    ],
    "financial/tax": [
        "tax", "taxation", "deduction", "credit", "liability", "irs",
        "transfer pricing", "withholding",
    ],
    "technical/architecture": [
        "architecture", "design pattern", "microservice", "api", "service",
        "scalability", "reliability", "system design",
    ],
    "technical/security": [
        "security", "vulnerability", "authentication", "authorization",
        "encryption", "attack", "threat", "exploit", "cve",
    ],
    "technical/infrastructure": [
        "infrastructure", "cloud", "kubernetes", "docker", "aws", "gcp",
        "azure", "terraform", "deployment", "ci/cd",
    ],
    "technical/data": [
        "data pipeline", "etl", "data warehouse", "analytics", "ml",
        "machine learning", "model", "dataset",
    ],
    "healthcare/clinical": [
        "patient", "diagnosis", "treatment", "clinical", "medical",
        "symptom", "condition", "medication",
    ],
    "healthcare/research": [
        "clinical trial", "study", "research", "hypothesis", "cohort",
        "randomized", "placebo",
    ],
}


class DomainTaxonomy:
    """
    Hierarchical domain organization for knowledge classification.

    Provides auto-classification of content into domain hierarchies,
    with support for industry-specific taxonomies.
    """

    def __init__(
        self,
        graph_store: "KnowledgeGraphStore",
        tenant_id: str = "default",
        custom_taxonomy: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the domain taxonomy.

        Args:
            graph_store: Knowledge graph store for persistence
            tenant_id: Tenant ID for isolation
            custom_taxonomy: Custom taxonomy to merge with defaults
        """
        self._store = graph_store
        self._tenant_id = tenant_id
        self._cache: Dict[str, TaxonomyNode] = {}
        self._initialized = False

        # Merge custom taxonomy with defaults
        self._taxonomy = DEFAULT_TAXONOMY.copy()
        if custom_taxonomy:
            self._merge_taxonomy(self._taxonomy, custom_taxonomy)

    def _merge_taxonomy(
        self, base: Dict[str, Any], custom: Dict[str, Any]
    ) -> None:
        """Recursively merge custom taxonomy into base."""
        for key, value in custom.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_taxonomy(base[key], value)
            else:
                base[key] = value

    async def initialize(self) -> None:
        """Initialize taxonomy, creating default domains if needed."""
        if self._initialized:
            return

        await self._ensure_default_taxonomy()
        await self._load_cache()
        self._initialized = True
        logger.info(f"DomainTaxonomy initialized with {len(self._cache)} domains")

    async def _ensure_default_taxonomy(self) -> None:
        """Create default taxonomy structure if it doesn't exist."""
        await self._create_taxonomy_recursive(self._taxonomy, None, "")

    async def _create_taxonomy_recursive(
        self,
        taxonomy: Dict[str, Any],
        parent_id: Optional[str],
        parent_path: str,
    ) -> None:
        """Recursively create taxonomy nodes."""
        for name, config in taxonomy.items():
            if isinstance(config, dict):
                description = config.get("description", "")
                children = config.get("children", {})
            else:
                description = ""
                children = {}

            current_path = f"{parent_path}/{name}" if parent_path else name

            # Check if already exists
            existing = await self._get_by_path(current_path)
            if existing:
                node_id = existing.id
            else:
                node_id = await self._create_node(
                    name, parent_id, description, current_path
                )

            # Recurse into children
            if children:
                await self._create_taxonomy_recursive(
                    children, node_id, current_path
                )

    async def _create_node(
        self,
        name: str,
        parent_id: Optional[str],
        description: str,
        full_path: str,
    ) -> str:
        """Create a taxonomy node in the database."""
        return await asyncio.to_thread(
            self._sync_create_node, name, parent_id, description, full_path
        )

    def _sync_create_node(
        self,
        name: str,
        parent_id: Optional[str],
        description: str,
        full_path: str,
    ) -> str:
        """Synchronous node creation."""
        node_id = f"dom_{uuid.uuid4().hex[:12]}"

        with self._store.connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO domain_taxonomy
                    (id, name, parent_id, description, tenant_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node_id,
                        name,
                        parent_id,
                        description,
                        self._tenant_id,
                        datetime.now().isoformat(),
                    ),
                )
            except Exception as e:
                if "UNIQUE constraint" in str(e):
                    # Already exists, get existing ID
                    row = self._store.fetch_one(
                        """
                        SELECT id FROM domain_taxonomy
                        WHERE name = ? AND (parent_id = ? OR (parent_id IS NULL AND ? IS NULL))
                        AND tenant_id = ?
                        """,
                        (name, parent_id, parent_id, self._tenant_id),
                    )
                    return row[0] if row else node_id
                raise

        return node_id

    async def _get_by_path(self, path: str) -> Optional[TaxonomyNode]:
        """Get a taxonomy node by its full path."""
        if path in self._cache:
            return self._cache[path]

        return await asyncio.to_thread(self._sync_get_by_path, path)

    def _sync_get_by_path(self, path: str) -> Optional[TaxonomyNode]:
        """Synchronous path lookup."""
        parts = path.split("/")

        parent_id: Optional[str] = None
        current_node: Optional[TaxonomyNode] = None

        for part in parts:
            row = self._store.fetch_one(
                """
                SELECT id, name, parent_id, description, tenant_id, created_at
                FROM domain_taxonomy
                WHERE name = ? AND (parent_id = ? OR (parent_id IS NULL AND ? IS NULL))
                AND tenant_id = ?
                """,
                (part, parent_id, parent_id, self._tenant_id),
            )

            if not row:
                return None

            current_node = TaxonomyNode(
                id=row[0],
                name=row[1],
                parent_id=row[2],
                full_path=path,
                description=row[3],
                tenant_id=row[4],
                created_at=datetime.fromisoformat(row[5]),
            )
            parent_id = row[0]

        return current_node

    async def _load_cache(self) -> None:
        """Load taxonomy into memory cache."""
        nodes = await asyncio.to_thread(self._sync_load_all)
        self._cache = {node.full_path: node for node in nodes}

    def _sync_load_all(self) -> List[TaxonomyNode]:
        """Synchronous load of all taxonomy nodes."""
        rows = self._store.fetch_all(
            """
            SELECT id, name, parent_id, description, tenant_id, created_at
            FROM domain_taxonomy
            WHERE tenant_id = ?
            """,
            (self._tenant_id,),
        )

        # Build parent_id -> children mapping
        nodes_by_id: Dict[str, TaxonomyNode] = {}
        for row in rows:
            node = TaxonomyNode(
                id=row[0],
                name=row[1],
                parent_id=row[2],
                full_path="",  # Will be computed
                description=row[3],
                tenant_id=row[4],
                created_at=datetime.fromisoformat(row[5]),
            )
            nodes_by_id[node.id] = node

        # Compute full paths
        def compute_path(node: TaxonomyNode) -> str:
            if node.parent_id is None:
                return node.name
            parent = nodes_by_id.get(node.parent_id)
            if parent:
                return f"{compute_path(parent)}/{node.name}"
            return node.name

        for node in nodes_by_id.values():
            node.full_path = compute_path(node)

        return list(nodes_by_id.values())

    # =========================================================================
    # Public API
    # =========================================================================

    async def classify(self, content: str) -> str:
        """
        Auto-classify content into a domain.

        Uses keyword matching to determine the most appropriate domain.

        Args:
            content: Text content to classify

        Returns:
            Domain path (e.g., "legal/contracts")
        """
        content_lower = content.lower()
        best_match = "general"
        best_score = 0

        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(
                1 for kw in keywords
                if kw.lower() in content_lower
            )
            if score > best_score:
                best_score = score
                best_match = domain

        return best_match

    async def ensure_path(
        self, path_parts: List[str], description: str = ""
    ) -> str:
        """
        Ensure a domain path exists, creating nodes as needed.

        Args:
            path_parts: List of path components ["legal", "contracts", "termination"]
            description: Description for the leaf node

        Returns:
            Domain path string
        """
        if not self._initialized:
            await self.initialize()

        full_path = "/".join(path_parts)

        # Check cache
        if full_path in self._cache:
            return full_path

        # Create path
        parent_id: Optional[str] = None
        current_path = ""

        for i, part in enumerate(path_parts):
            current_path = f"{current_path}/{part}" if current_path else part

            existing = await self._get_by_path(current_path)
            if existing:
                parent_id = existing.id
            else:
                # Create this node
                node_desc = description if i == len(path_parts) - 1 else ""
                parent_id = await self._create_node(
                    part, parent_id, node_desc, current_path
                )

        # Refresh cache
        await self._load_cache()

        return full_path

    async def get_path(self, domain: str) -> List[str]:
        """
        Get full path from root to domain.

        Args:
            domain: Domain path string

        Returns:
            List of path components
        """
        return domain.split("/")

    async def get_children(
        self, domain: str = ""
    ) -> List[TaxonomyNode]:
        """
        Get child domains of a given domain.

        Args:
            domain: Parent domain path (empty for root)

        Returns:
            List of child taxonomy nodes
        """
        if not self._initialized:
            await self.initialize()

        return await asyncio.to_thread(self._sync_get_children, domain)

    def _sync_get_children(self, domain: str) -> List[TaxonomyNode]:
        """Synchronous children lookup."""
        if domain:
            parent = self._cache.get(domain)
            if not parent:
                return []
            parent_id = parent.id
        else:
            parent_id = None

        rows = self._store.fetch_all(
            """
            SELECT id, name, parent_id, description, tenant_id, created_at
            FROM domain_taxonomy
            WHERE (parent_id = ? OR (parent_id IS NULL AND ? IS NULL))
            AND tenant_id = ?
            """,
            (parent_id, parent_id, self._tenant_id),
        )

        return [
            TaxonomyNode(
                id=row[0],
                name=row[1],
                parent_id=row[2],
                full_path=f"{domain}/{row[1]}" if domain else row[1],
                description=row[3],
                tenant_id=row[4],
                created_at=datetime.fromisoformat(row[5]),
            )
            for row in rows
        ]

    async def get_all_domains(self) -> List[str]:
        """Get all domain paths."""
        if not self._initialized:
            await self.initialize()

        return list(self._cache.keys())

    async def get_stats(self) -> dict:
        """Get taxonomy statistics."""
        if not self._initialized:
            await self.initialize()

        return {
            "total_domains": len(self._cache),
            "root_domains": len([
                d for d in self._cache.values() if d.parent_id is None
            ]),
            "max_depth": max(
                (d.full_path.count("/") + 1 for d in self._cache.values()),
                default=0,
            ),
            "tenant_id": self._tenant_id,
        }


__all__ = [
    "DomainTaxonomy",
    "TaxonomyNode",
    "DEFAULT_TAXONOMY",
    "DOMAIN_KEYWORDS",
]
