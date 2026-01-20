"""
Culture Accumulator for Knowledge Mound.

Accumulates organizational patterns from debates and decisions,
inspired by Agno's organizational learning concept.

This tracks:
- Decision patterns (how the organization typically decides)
- Risk tolerance (conservative vs aggressive)
- Domain expertise distribution
- Preferred agents for different tasks
- Common critique patterns
- Consensus difficulty by topic

Enhanced for the enterprise control plane with:
- Organization-level culture (cross-workspace patterns)
- Promotion of workspace patterns to organization level
- Culture document management for explicit organizational knowledge
- Integration with RBAC for access control
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from aragora.knowledge.mound.types import (
    CulturePattern,
    CulturePatternType,
    CultureProfile,
)

if TYPE_CHECKING:
    from aragora.knowledge.mound.facade import KnowledgeMound

logger = logging.getLogger(__name__)


class CultureDocumentCategory(str, Enum):
    """Categories for explicit culture documents."""

    VALUES = "values"  # Core organizational values
    PRACTICES = "practices"  # Established practices and processes
    STANDARDS = "standards"  # Quality and technical standards
    POLICIES = "policies"  # Organizational policies
    LEARNINGS = "learnings"  # Accumulated learnings from debates


@dataclass
class CultureDocument:
    """
    An explicit culture document representing organizational knowledge.

    Culture documents are organization-level knowledge entries that
    are accessible across all workspaces within an organization.
    They can be created manually or promoted from workspace patterns.
    """

    id: str
    org_id: str
    category: CultureDocumentCategory
    title: str
    content: str
    embeddings: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    version: int = 1
    supersedes: Optional[str] = None  # ID of previous version
    source_workspace_id: Optional[str] = None  # If promoted from workspace
    source_pattern_id: Optional[str] = None  # If promoted from pattern
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "org_id": self.org_id,
            "category": self.category.value,
            "title": self.title,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "version": self.version,
            "supersedes": self.supersedes,
            "source_workspace_id": self.source_workspace_id,
            "source_pattern_id": self.source_pattern_id,
            "metadata": self.metadata,
            "is_active": self.is_active,
        }


@dataclass
class OrganizationCulture:
    """
    Aggregated culture profile for an organization.

    Combines patterns from all workspaces plus explicit culture documents.
    """

    org_id: str
    documents: List[CultureDocument]
    aggregated_patterns: Dict[CulturePatternType, List[CulturePattern]]
    workspace_profiles: Dict[str, CultureProfile]
    generated_at: datetime = field(default_factory=datetime.now)
    total_observations: int = 0
    workspace_count: int = 0
    dominant_traits: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "org_id": self.org_id,
            "documents": [d.to_dict() for d in self.documents],
            "workspace_count": self.workspace_count,
            "generated_at": self.generated_at.isoformat(),
            "total_observations": self.total_observations,
            "dominant_traits": self.dominant_traits,
        }


@dataclass
class DebateObservation:
    """Extracted observations from a debate."""

    debate_id: str
    topic: str
    participating_agents: List[str]
    winning_agents: List[str]
    rounds_to_consensus: int
    consensus_reached: bool
    consensus_strength: str
    critique_patterns: List[str]
    domain: Optional[str]
    created_at: datetime = field(default_factory=datetime.now)


class CultureAccumulator:
    """
    Accumulates organizational patterns from debates and decisions.

    The culture accumulator observes debates and extracts patterns that
    characterize how the organization makes decisions, which agents are
    effective in which domains, and how consensus is typically reached.
    """

    def __init__(
        self,
        mound: "KnowledgeMound",
        min_observations_for_pattern: int = 3,
    ):
        """
        Initialize culture accumulator.

        Args:
            mound: Reference to the Knowledge Mound
            min_observations_for_pattern: Minimum observations before pattern is confident
        """
        self._mound = mound
        self._min_observations = min_observations_for_pattern

        # In-memory pattern cache (synced to storage)
        self._patterns: Dict[str, Dict[CulturePatternType, Dict[str, CulturePattern]]] = (
            defaultdict(lambda: defaultdict(dict))
        )

    async def observe_debate(
        self,
        debate_result: Any,
        workspace_id: str,
    ) -> List[CulturePattern]:
        """
        Extract and store cultural patterns from a completed debate.

        Args:
            debate_result: Result from Arena.run()
            workspace_id: Workspace for pattern storage

        Returns:
            List of patterns created or updated
        """
        patterns_updated: List[CulturePattern] = []

        try:
            # Extract observation from debate result
            observation = self._extract_observation(debate_result)
            if not observation:
                return []

            # Update patterns based on observation
            patterns_updated.extend(await self._update_agent_preferences(observation, workspace_id))
            patterns_updated.extend(await self._update_decision_style(observation, workspace_id))
            patterns_updated.extend(await self._update_debate_dynamics(observation, workspace_id))
            patterns_updated.extend(await self._update_domain_expertise(observation, workspace_id))

            logger.debug(
                f"Extracted {len(patterns_updated)} patterns from debate {observation.debate_id}"
            )

            # Emit MOUND_UPDATED event when culture patterns change
            if patterns_updated:
                self._emit_mound_updated(
                    workspace_id=workspace_id,
                    update_type="culture_patterns",
                    patterns_count=len(patterns_updated),
                    debate_id=observation.debate_id,
                )

        except Exception as e:
            logger.warning(f"Failed to observe debate: {e}")

        return patterns_updated

    def _emit_mound_updated(
        self,
        workspace_id: str,
        update_type: str,
        **kwargs,
    ) -> None:
        """Emit MOUND_UPDATED event for cross-subsystem tracking."""
        if not hasattr(self._mound, "event_emitter") or not self._mound.event_emitter:
            return

        try:
            from aragora.events.types import StreamEvent, StreamEventType

            self._mound.event_emitter.emit(
                StreamEvent(
                    type=StreamEventType.MOUND_UPDATED,
                    data={
                        "workspace_id": workspace_id,
                        "update_type": update_type,
                        **kwargs,
                    },
                )
            )
        except (ImportError, AttributeError, TypeError):
            pass  # Events module not available

    def _extract_observation(self, debate_result: Any) -> Optional[DebateObservation]:
        """Extract observation data from debate result."""
        try:
            # Extract basic info
            debate_id = getattr(debate_result, "debate_id", str(uuid.uuid4())[:8])
            topic = getattr(debate_result, "task", "") or getattr(debate_result, "topic", "")

            # Extract agent info
            participating_agents = []
            winning_agents = []

            if hasattr(debate_result, "proposals"):
                for proposal in debate_result.proposals:
                    if hasattr(proposal, "agent_type"):
                        participating_agents.append(proposal.agent_type)

            if hasattr(debate_result, "winner"):
                if debate_result.winner:
                    winning_agents.append(debate_result.winner)

            # Extract consensus info
            consensus_reached = getattr(debate_result, "consensus_reached", False)
            rounds_used = getattr(debate_result, "rounds_used", 0)
            confidence = getattr(debate_result, "confidence", 0.5)

            # Map confidence to strength
            if confidence >= 0.9:
                consensus_strength = "unanimous"
            elif confidence >= 0.7:
                consensus_strength = "strong"
            elif confidence >= 0.5:
                consensus_strength = "moderate"
            else:
                consensus_strength = "weak"

            # Extract critique patterns (if available)
            critique_patterns = []
            if hasattr(debate_result, "critiques"):
                for critique in debate_result.critiques[:5]:  # Limit to 5
                    if hasattr(critique, "type"):
                        critique_patterns.append(critique.type)

            # Infer domain from topic
            domain = self._infer_domain(topic)

            return DebateObservation(
                debate_id=debate_id,
                topic=topic,
                participating_agents=participating_agents,
                winning_agents=winning_agents,
                rounds_to_consensus=rounds_used,
                consensus_reached=consensus_reached,
                consensus_strength=consensus_strength,
                critique_patterns=critique_patterns,
                domain=domain,
            )

        except Exception as e:
            logger.warning(f"Failed to extract observation: {e}")
            return None

    def _infer_domain(self, topic: str) -> Optional[str]:
        """Infer domain from topic keywords."""
        topic_lower = topic.lower()

        domain_keywords = {
            "security": ["security", "auth", "encrypt", "vulnerability", "attack"],
            "performance": ["performance", "speed", "latency", "optimize", "cache"],
            "architecture": ["architecture", "design", "pattern", "structure", "module"],
            "testing": ["test", "qa", "coverage", "assertion", "mock"],
            "api": ["api", "endpoint", "rest", "graphql", "rpc"],
            "database": ["database", "sql", "query", "index", "schema"],
            "frontend": ["ui", "frontend", "react", "component", "css"],
            "legal": ["contract", "compliance", "regulation", "legal", "terms"],
            "financial": ["financial", "accounting", "audit", "budget", "cost"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in topic_lower for kw in keywords):
                return domain

        return None

    async def _update_agent_preferences(
        self,
        observation: DebateObservation,
        workspace_id: str,
    ) -> List[CulturePattern]:
        """Update agent preference patterns."""
        patterns = []

        # Track winning agents by domain
        if observation.domain and observation.winning_agents:
            for agent in observation.winning_agents:
                pattern_key = f"{observation.domain}:{agent}"

                existing = self._patterns[workspace_id][CulturePatternType.AGENT_PREFERENCES].get(
                    pattern_key
                )

                if existing:
                    existing.observation_count += 1
                    existing.confidence = min(
                        1.0, existing.observation_count / (self._min_observations * 2)
                    )
                    existing.last_observed_at = datetime.now()
                    existing.contributing_debates.append(observation.debate_id)
                    patterns.append(existing)
                else:
                    pattern = CulturePattern(
                        id=f"cp_{uuid.uuid4().hex[:12]}",
                        workspace_id=workspace_id,
                        pattern_type=CulturePatternType.AGENT_PREFERENCES,
                        pattern_key=pattern_key,
                        pattern_value={
                            "agent": agent,
                            "domain": observation.domain,
                            "wins": 1,
                        },
                        observation_count=1,
                        confidence=0.3,
                        first_observed_at=datetime.now(),
                        last_observed_at=datetime.now(),
                        contributing_debates=[observation.debate_id],
                    )
                    self._patterns[workspace_id][CulturePatternType.AGENT_PREFERENCES][
                        pattern_key
                    ] = pattern
                    patterns.append(pattern)

        return patterns

    async def _update_decision_style(
        self,
        observation: DebateObservation,
        workspace_id: str,
    ) -> List[CulturePattern]:
        """Update decision style patterns."""
        patterns = []

        # Track consensus patterns
        style_key = "consensus_pattern"

        existing = self._patterns[workspace_id][CulturePatternType.DECISION_STYLE].get(style_key)

        if existing:
            # Update running averages
            count = existing.observation_count
            existing.pattern_value["avg_rounds"] = (
                existing.pattern_value.get("avg_rounds", 0) * count
                + observation.rounds_to_consensus
            ) / (count + 1)
            existing.pattern_value["consensus_rate"] = (
                existing.pattern_value.get("consensus_rate", 0) * count
                + (1.0 if observation.consensus_reached else 0.0)
            ) / (count + 1)
            existing.pattern_value["strength_counts"][observation.consensus_strength] = (
                existing.pattern_value.get("strength_counts", {}).get(
                    observation.consensus_strength, 0
                )
                + 1
            )
            existing.observation_count += 1
            existing.confidence = min(
                1.0, existing.observation_count / (self._min_observations * 3)
            )
            existing.last_observed_at = datetime.now()
            patterns.append(existing)
        else:
            pattern = CulturePattern(
                id=f"cp_{uuid.uuid4().hex[:12]}",
                workspace_id=workspace_id,
                pattern_type=CulturePatternType.DECISION_STYLE,
                pattern_key=style_key,
                pattern_value={
                    "avg_rounds": observation.rounds_to_consensus,
                    "consensus_rate": 1.0 if observation.consensus_reached else 0.0,
                    "strength_counts": {observation.consensus_strength: 1},
                },
                observation_count=1,
                confidence=0.2,
                first_observed_at=datetime.now(),
                last_observed_at=datetime.now(),
                contributing_debates=[observation.debate_id],
            )
            self._patterns[workspace_id][CulturePatternType.DECISION_STYLE][style_key] = pattern
            patterns.append(pattern)

        return patterns

    async def _update_debate_dynamics(
        self,
        observation: DebateObservation,
        workspace_id: str,
    ) -> List[CulturePattern]:
        """Update debate dynamics patterns."""
        patterns = []

        # Track critique patterns
        for critique_type in observation.critique_patterns:
            pattern_key = f"critique:{critique_type}"

            existing = self._patterns[workspace_id][CulturePatternType.DEBATE_DYNAMICS].get(
                pattern_key
            )

            if existing:
                existing.observation_count += 1
                existing.confidence = min(
                    1.0, existing.observation_count / (self._min_observations * 2)
                )
                existing.last_observed_at = datetime.now()
                patterns.append(existing)
            else:
                pattern = CulturePattern(
                    id=f"cp_{uuid.uuid4().hex[:12]}",
                    workspace_id=workspace_id,
                    pattern_type=CulturePatternType.DEBATE_DYNAMICS,
                    pattern_key=pattern_key,
                    pattern_value={
                        "critique_type": critique_type,
                        "frequency": 1,
                    },
                    observation_count=1,
                    confidence=0.2,
                    first_observed_at=datetime.now(),
                    last_observed_at=datetime.now(),
                    contributing_debates=[observation.debate_id],
                )
                self._patterns[workspace_id][CulturePatternType.DEBATE_DYNAMICS][
                    pattern_key
                ] = pattern
                patterns.append(pattern)

        return patterns

    async def _update_domain_expertise(
        self,
        observation: DebateObservation,
        workspace_id: str,
    ) -> List[CulturePattern]:
        """Update domain expertise patterns."""
        patterns = []

        if not observation.domain:
            return patterns

        pattern_key = observation.domain

        existing = self._patterns[workspace_id][CulturePatternType.DOMAIN_EXPERTISE].get(
            pattern_key
        )

        if existing:
            existing.observation_count += 1
            existing.confidence = min(
                1.0, existing.observation_count / (self._min_observations * 2)
            )
            existing.last_observed_at = datetime.now()
            existing.contributing_debates.append(observation.debate_id)
            patterns.append(existing)
        else:
            pattern = CulturePattern(
                id=f"cp_{uuid.uuid4().hex[:12]}",
                workspace_id=workspace_id,
                pattern_type=CulturePatternType.DOMAIN_EXPERTISE,
                pattern_key=pattern_key,
                pattern_value={
                    "domain": observation.domain,
                    "debate_count": 1,
                },
                observation_count=1,
                confidence=0.3,
                first_observed_at=datetime.now(),
                last_observed_at=datetime.now(),
                contributing_debates=[observation.debate_id],
            )
            self._patterns[workspace_id][CulturePatternType.DOMAIN_EXPERTISE][pattern_key] = pattern
            patterns.append(pattern)

        return patterns

    def get_patterns(
        self,
        workspace_id: str,
        pattern_type: Optional[CulturePatternType] = None,
        min_confidence: float = 0.0,
        min_observations: int = 0,
    ) -> List[CulturePattern]:
        """
        Get accumulated patterns for a workspace.

        Args:
            workspace_id: Workspace to get patterns for
            pattern_type: Optional filter by pattern type
            min_confidence: Minimum confidence threshold (0.0-1.0)
            min_observations: Minimum observation count

        Returns:
            List of matching CulturePattern objects
        """
        workspace_patterns = self._patterns.get(workspace_id, {})

        if not workspace_patterns:
            return []

        patterns: List[CulturePattern] = []

        if pattern_type is not None:
            # Get patterns of specific type
            type_patterns = workspace_patterns.get(pattern_type, {})
            patterns = list(type_patterns.values())
        else:
            # Get all patterns
            for type_patterns in workspace_patterns.values():
                patterns.extend(type_patterns.values())

        # Apply filters
        if min_confidence > 0.0:
            patterns = [p for p in patterns if p.confidence >= min_confidence]

        if min_observations > 0:
            patterns = [p for p in patterns if p.observation_count >= min_observations]

        # Sort by confidence (descending) then observation count
        patterns.sort(key=lambda p: (p.confidence, p.observation_count), reverse=True)

        return patterns

    def get_patterns_summary(
        self,
        workspace_id: str,
    ) -> Dict[str, Any]:
        """
        Get a summary of accumulated patterns for a workspace.

        Args:
            workspace_id: Workspace to summarize

        Returns:
            Dictionary with pattern counts, types, and top patterns
        """
        workspace_patterns = self._patterns.get(workspace_id, {})

        if not workspace_patterns:
            return {
                "workspace_id": workspace_id,
                "total_patterns": 0,
                "patterns_by_type": {},
                "total_observations": 0,
                "top_patterns": [],
            }

        patterns_by_type: Dict[str, int] = {}
        total_observations = 0
        all_patterns: List[CulturePattern] = []

        for pattern_type, type_patterns in workspace_patterns.items():
            patterns_by_type[pattern_type.value] = len(type_patterns)
            for pattern in type_patterns.values():
                total_observations += pattern.observation_count
                all_patterns.append(pattern)

        # Get top patterns by confidence
        all_patterns.sort(key=lambda p: (p.confidence, p.observation_count), reverse=True)
        top_patterns = [
            {
                "id": p.id,
                "type": p.pattern_type.value,
                "key": p.pattern_key,
                "confidence": round(p.confidence, 3),
                "observations": p.observation_count,
            }
            for p in all_patterns[:10]
        ]

        return {
            "workspace_id": workspace_id,
            "total_patterns": len(all_patterns),
            "patterns_by_type": patterns_by_type,
            "total_observations": total_observations,
            "top_patterns": top_patterns,
        }

    async def get_profile(self, workspace_id: str) -> CultureProfile:
        """
        Get aggregated culture profile for a workspace.

        Args:
            workspace_id: Workspace to get profile for

        Returns:
            CultureProfile with all accumulated patterns
        """
        workspace_patterns = self._patterns.get(workspace_id, {})

        # Organize patterns by type
        patterns_by_type: Dict[CulturePatternType, List[CulturePattern]] = {}
        total_observations = 0

        for pattern_type in CulturePatternType:
            type_patterns = list(workspace_patterns.get(pattern_type, {}).values())
            # Filter to confident patterns
            confident_patterns = [
                p
                for p in type_patterns
                if p.confidence >= 0.3 or p.observation_count >= self._min_observations
            ]
            patterns_by_type[pattern_type] = confident_patterns
            total_observations += sum(p.observation_count for p in confident_patterns)

        # Extract dominant traits
        dominant_traits = {}

        # Most effective agents
        agent_patterns = patterns_by_type.get(CulturePatternType.AGENT_PREFERENCES, [])
        if agent_patterns:
            sorted_agents = sorted(agent_patterns, key=lambda p: p.confidence, reverse=True)
            dominant_traits["top_agents"] = [
                p.pattern_value.get("agent") for p in sorted_agents[:3]
            ]

        # Decision style
        decision_patterns = patterns_by_type.get(CulturePatternType.DECISION_STYLE, [])
        if decision_patterns:
            main_pattern = max(decision_patterns, key=lambda p: p.observation_count)
            dominant_traits["avg_consensus_rounds"] = main_pattern.pattern_value.get(
                "avg_rounds", 0
            )
            dominant_traits["consensus_rate"] = main_pattern.pattern_value.get("consensus_rate", 0)

        # Domain expertise
        domain_patterns = patterns_by_type.get(CulturePatternType.DOMAIN_EXPERTISE, [])
        if domain_patterns:
            sorted_domains = sorted(
                domain_patterns, key=lambda p: p.observation_count, reverse=True
            )
            dominant_traits["expertise_areas"] = [p.pattern_key for p in sorted_domains[:5]]

        return CultureProfile(
            workspace_id=workspace_id,
            patterns=patterns_by_type,
            generated_at=datetime.now(),
            total_observations=total_observations,
            dominant_traits=dominant_traits,
        )

    async def recommend_agents(
        self,
        task_type: str,
        workspace_id: str,
    ) -> List[str]:
        """
        Recommend agents based on cultural patterns.

        Args:
            task_type: Type of task (e.g., "security", "architecture")
            workspace_id: Workspace to check patterns for

        Returns:
            List of recommended agent types
        """
        workspace_patterns = self._patterns.get(workspace_id, {})
        agent_patterns = workspace_patterns.get(CulturePatternType.AGENT_PREFERENCES, {})

        # Find patterns matching the task type
        matching = []
        for pattern_key, pattern in agent_patterns.items():
            domain = pattern.pattern_value.get("domain", "")
            if domain == task_type or task_type.lower() in domain.lower():
                matching.append((pattern.pattern_value.get("agent"), pattern.confidence))

        # Sort by confidence and return top agents
        matching.sort(key=lambda x: x[1], reverse=True)
        return [agent for agent, _ in matching[:3]]


class OrganizationCultureManager:
    """
    Manages organization-level culture (cross-workspace patterns).

    This is the "termite mound" - a shared knowledge superstructure
    that all agents across all workspaces contribute to and query from.

    Features:
    - Explicit culture documents (values, practices, standards, policies)
    - Automatic pattern aggregation across workspaces
    - Promotion of workspace patterns to organization level
    - Semantic search over culture documents
    - RBAC integration for access control
    """

    def __init__(
        self,
        mound: "KnowledgeMound",
        culture_accumulator: Optional[CultureAccumulator] = None,
    ):
        """
        Initialize organization culture manager.

        Args:
            mound: Reference to the Knowledge Mound
            culture_accumulator: Workspace-level culture accumulator
        """
        self._mound = mound
        self._accumulator = culture_accumulator

        # Document storage by org
        self._documents: Dict[str, Dict[str, CultureDocument]] = defaultdict(dict)

        # Workspace to org mapping
        self._workspace_orgs: Dict[str, str] = {}

    def register_workspace(self, workspace_id: str, org_id: str) -> None:
        """Register a workspace's organization for culture aggregation."""
        self._workspace_orgs[workspace_id] = org_id

    async def add_document(
        self,
        org_id: str,
        category: CultureDocumentCategory,
        title: str,
        content: str,
        created_by: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CultureDocument:
        """
        Add an explicit culture document.

        Args:
            org_id: Organization to add document to
            category: Document category
            title: Document title
            content: Document content
            created_by: User creating the document
            metadata: Optional metadata

        Returns:
            Created culture document
        """
        doc_id = f"cd_{uuid.uuid4().hex[:12]}"

        # Generate embeddings if vector store is available
        embeddings = None
        if hasattr(self._mound, "_vector_store") and self._mound._vector_store:
            try:
                embeddings = await self._mound._vector_store.embed_text(content)
            except Exception as e:
                logger.debug(f"Could not generate embeddings: {e}")

        doc = CultureDocument(
            id=doc_id,
            org_id=org_id,
            category=category,
            title=title,
            content=content,
            embeddings=embeddings,
            created_by=created_by,
            metadata=metadata or {},
        )

        self._documents[org_id][doc_id] = doc

        logger.info(f"Added culture document {doc_id} to org {org_id}")

        return doc

    async def update_document(
        self,
        doc_id: str,
        org_id: str,
        content: str,
        updated_by: str,
    ) -> CultureDocument:
        """
        Update a culture document, creating a new version.

        Args:
            doc_id: Document to update
            org_id: Organization owning the document
            content: New content
            updated_by: User making the update

        Returns:
            Updated document
        """
        old_doc = self._documents[org_id].get(doc_id)
        if not old_doc:
            raise ValueError(f"Document not found: {doc_id}")

        # Deactivate old version
        old_doc.is_active = False

        # Create new version
        new_doc_id = f"cd_{uuid.uuid4().hex[:12]}"

        embeddings = None
        if hasattr(self._mound, "_vector_store") and self._mound._vector_store:
            try:
                embeddings = await self._mound._vector_store.embed_text(content)
            except Exception as e:
                logger.debug(f"Could not generate embeddings: {e}")

        new_doc = CultureDocument(
            id=new_doc_id,
            org_id=org_id,
            category=old_doc.category,
            title=old_doc.title,
            content=content,
            embeddings=embeddings,
            created_by=updated_by,
            version=old_doc.version + 1,
            supersedes=doc_id,
            source_workspace_id=old_doc.source_workspace_id,
            source_pattern_id=old_doc.source_pattern_id,
            metadata=old_doc.metadata,
        )

        self._documents[org_id][new_doc_id] = new_doc

        logger.info(f"Updated culture document {doc_id} -> {new_doc_id}")

        return new_doc

    async def get_documents(
        self,
        org_id: str,
        category: Optional[CultureDocumentCategory] = None,
        active_only: bool = True,
    ) -> List[CultureDocument]:
        """
        Get culture documents for an organization.

        Args:
            org_id: Organization to query
            category: Optional category filter
            active_only: Only return active (non-superseded) documents

        Returns:
            List of matching documents
        """
        docs = list(self._documents.get(org_id, {}).values())

        if active_only:
            docs = [d for d in docs if d.is_active]

        if category:
            docs = [d for d in docs if d.category == category]

        return sorted(docs, key=lambda d: d.updated_at, reverse=True)

    async def query_culture(
        self,
        org_id: str,
        query: str,
        limit: int = 10,
    ) -> List[CultureDocument]:
        """
        Query culture documents semantically.

        Args:
            org_id: Organization to query
            query: Semantic query
            limit: Maximum results

        Returns:
            Relevant culture documents
        """
        docs = await self.get_documents(org_id, active_only=True)

        if not docs:
            return []

        # If vector store is available, use semantic search
        if hasattr(self._mound, "_vector_store") and self._mound._vector_store:
            try:
                query_embedding = await self._mound._vector_store.embed_text(query)

                # Score documents by cosine similarity
                scored = []
                for doc in docs:
                    if doc.embeddings:
                        similarity = self._cosine_similarity(query_embedding, doc.embeddings)
                        scored.append((doc, similarity))

                scored.sort(key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in scored[:limit]]
            except Exception as e:
                logger.debug(f"Semantic search failed: {e}")

        # Fallback to keyword matching
        query_lower = query.lower()
        scored = []
        for doc in docs:
            score = 0
            if query_lower in doc.title.lower():
                score += 2
            if query_lower in doc.content.lower():
                score += 1
            if score > 0:
                scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:limit]]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def promote_pattern_to_culture(
        self,
        workspace_id: str,
        pattern_id: str,
        promoted_by: str,
        title: Optional[str] = None,
    ) -> CultureDocument:
        """
        Promote a workspace pattern to an organization culture document.

        Args:
            workspace_id: Workspace containing the pattern
            pattern_id: Pattern ID to promote
            promoted_by: User promoting the pattern
            title: Optional title override

        Returns:
            Created culture document
        """
        org_id = self._workspace_orgs.get(workspace_id)
        if not org_id:
            raise ValueError(f"Workspace {workspace_id} not registered to an org")

        if not self._accumulator:
            raise ValueError("Culture accumulator not configured")

        # Find the pattern
        workspace_patterns = self._accumulator._patterns.get(workspace_id, {})
        pattern = None

        for pattern_type in CulturePatternType:
            type_patterns = workspace_patterns.get(pattern_type, {})
            for p in type_patterns.values():
                if p.id == pattern_id:
                    pattern = p
                    break
            if pattern:
                break

        if not pattern:
            raise ValueError(f"Pattern not found: {pattern_id}")

        # Create culture document from pattern
        doc_title = title or f"Learning: {pattern.pattern_key}"
        content = self._pattern_to_content(pattern)

        doc = await self.add_document(
            org_id=org_id,
            category=CultureDocumentCategory.LEARNINGS,
            title=doc_title,
            content=content,
            created_by=promoted_by,
            metadata={
                "promoted_from_pattern": pattern_id,
                "pattern_type": pattern.pattern_type.value,
                "observation_count": pattern.observation_count,
                "confidence": pattern.confidence,
            },
        )

        doc.source_workspace_id = workspace_id
        doc.source_pattern_id = pattern_id

        logger.info(f"Promoted pattern {pattern_id} to culture document {doc.id}")

        return doc

    def _pattern_to_content(self, pattern: CulturePattern) -> str:
        """Convert a pattern to human-readable content."""
        lines = [
            f"Pattern Type: {pattern.pattern_type.value}",
            f"Pattern Key: {pattern.pattern_key}",
            f"Observations: {pattern.observation_count}",
            f"Confidence: {pattern.confidence:.2%}",
            "",
            "Pattern Value:",
        ]

        for key, value in pattern.pattern_value.items():
            lines.append(f"  - {key}: {value}")

        if pattern.contributing_debates:
            lines.append("")
            lines.append(f"Contributing Debates: {len(pattern.contributing_debates)}")

        return "\n".join(lines)

    async def get_organization_culture(
        self,
        org_id: str,
        workspace_ids: Optional[List[str]] = None,
    ) -> OrganizationCulture:
        """
        Get the complete organization culture profile.

        Aggregates patterns from all workspaces plus explicit documents.

        Args:
            org_id: Organization to profile
            workspace_ids: Optional list of workspaces to include

        Returns:
            Complete organization culture profile
        """
        # Get explicit culture documents
        documents = await self.get_documents(org_id, active_only=True)

        # Get workspace profiles
        workspace_profiles: Dict[str, CultureProfile] = {}
        aggregated_patterns: Dict[CulturePatternType, List[CulturePattern]] = {
            pt: [] for pt in CulturePatternType
        }

        # Find all workspaces for this org
        if workspace_ids is None:
            workspace_ids = [ws for ws, org in self._workspace_orgs.items() if org == org_id]

        total_observations = 0

        if self._accumulator:
            for ws_id in workspace_ids:
                try:
                    profile = await self._accumulator.get_profile(ws_id)
                    workspace_profiles[ws_id] = profile
                    total_observations += profile.total_observations

                    # Aggregate patterns
                    for pattern_type, patterns in profile.patterns.items():
                        aggregated_patterns[pattern_type].extend(patterns)
                except Exception as e:
                    logger.debug(f"Could not get profile for workspace {ws_id}: {e}")

        # Extract dominant traits from aggregated data
        dominant_traits = self._extract_dominant_traits(aggregated_patterns)

        return OrganizationCulture(
            org_id=org_id,
            documents=documents,
            aggregated_patterns=aggregated_patterns,
            workspace_profiles=workspace_profiles,
            total_observations=total_observations,
            workspace_count=len(workspace_profiles),
            dominant_traits=dominant_traits,
        )

    def _extract_dominant_traits(
        self,
        patterns: Dict[CulturePatternType, List[CulturePattern]],
    ) -> Dict[str, Any]:
        """Extract dominant traits from aggregated patterns."""
        traits: Dict[str, Any] = {}

        # Top agents across org
        agent_patterns = patterns.get(CulturePatternType.AGENT_PREFERENCES, [])
        if agent_patterns:
            agent_scores: Dict[str, float] = defaultdict(float)
            for p in agent_patterns:
                agent = p.pattern_value.get("agent", "")
                if agent:
                    agent_scores[agent] += p.confidence * p.observation_count

            sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
            traits["top_agents"] = [a for a, _ in sorted_agents[:5]]

        # Domain expertise across org
        domain_patterns = patterns.get(CulturePatternType.DOMAIN_EXPERTISE, [])
        if domain_patterns:
            domain_scores: Dict[str, int] = defaultdict(int)
            for p in domain_patterns:
                domain_scores[p.pattern_key] += p.observation_count

            sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
            traits["expertise_areas"] = [d for d, _ in sorted_domains[:5]]

        # Decision patterns
        decision_patterns = patterns.get(CulturePatternType.DECISION_STYLE, [])
        if decision_patterns:
            total_rounds = 0
            total_consensus = 0
            count = 0

            for p in decision_patterns:
                total_rounds += p.pattern_value.get("avg_rounds", 0)
                total_consensus += p.pattern_value.get("consensus_rate", 0)
                count += 1

            if count > 0:
                traits["avg_consensus_rounds"] = total_rounds / count
                traits["consensus_rate"] = total_consensus / count

        return traits

    async def get_relevant_context(
        self,
        org_id: str,
        task: str,
        max_documents: int = 3,
    ) -> str:
        """
        Get relevant culture context for a task.

        This is used to inject organizational knowledge into agent prompts.

        Args:
            org_id: Organization to query
            task: Task description
            max_documents: Maximum documents to include

        Returns:
            Formatted context string
        """
        docs = await self.query_culture(org_id, task, limit=max_documents)

        if not docs:
            return ""

        lines = ["## Organizational Context", ""]

        for doc in docs:
            lines.append(f"### {doc.title} ({doc.category.value})")
            lines.append(doc.content)
            lines.append("")

        return "\n".join(lines)
