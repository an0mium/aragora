"""
Experience Anchors - Markers for untransferable moments.

Inspired by the Blade Runner insight:
"All those moments will be lost in time, like tears in rain."

The monologue isn't about memory loss - it's about untransferable experience.
Some moments don't compress into utility. They don't survive abstraction.
They don't generalize. They don't scale.

This module provides a way to mark moments that mattered beyond their
utility function. These become part of decision receipts - pointers to
experiences that informed the decision without claiming to capture what
they meant.

Key insight from the conversation:
"Meaning is not downstream of optimality. It's downstream of exposure."

Experience anchors are:
- Pointers, not captures (they reference, don't reproduce)
- Explicitly marked as resisting compression
- Part of the decision record but NOT part of the training signal
- Owned by the person who had the experience

Usage:
    anchor_store = ExperienceAnchorStore()

    # When a moment matters but resists abstraction
    anchor = anchor_store.create_anchor(
        user_id="user_123",
        context="Watching the team's first deploy succeed at 3am",
        why_it_mattered="Not the technical success - the feeling of having built something together",
        compressible=False,
    )

    # Link to a decision
    decision_receipt.add_experience_anchor(anchor.id)

    # Later, when reviewing the decision
    anchors = anchor_store.get_anchors_for_decision(decision_id)
    # These remind WHY the decision mattered, not just WHAT was decided
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class AnchorType(str, Enum):
    """Types of experiences that might be anchored."""

    MOMENT = "moment"  # A specific point in time
    RELATIONSHIP = "relationship"  # Connection to a person/team
    REALIZATION = "realization"  # An insight that changed perspective
    SENSATION = "sensation"  # A felt experience (wind on water, etc.)
    COMMITMENT = "commitment"  # A promise or binding choice
    LOSS = "loss"  # Something that ended or was given up
    CREATION = "creation"  # Something brought into being
    WITNESS = "witness"  # Having been present for something


class TransferabilityLevel(str, Enum):
    """How much an experience can be transferred/compressed."""

    FULLY_TRANSFERABLE = "fully_transferable"  # Can be explained completely
    PARTIALLY_TRANSFERABLE = "partially_transferable"  # Some aspects convey
    POINTER_ONLY = "pointer_only"  # Can point to, not convey
    FUNDAMENTALLY_PRIVATE = "fundamentally_private"  # Cannot even point


@dataclass
class ExperienceAnchor:
    """A marker for a moment that mattered beyond its utility.

    This is NOT a record of what happened. It's a pointer to an
    experience that informed something without being reducible to it.

    Think of it as a bookmark in consciousness - you can find it again,
    but you can't photocopy it.
    """

    id: str
    user_id: str
    created_at: datetime

    # What we can say about it (minimal)
    context: str  # Situation, not content
    anchor_type: AnchorType
    why_it_mattered: str  # Not justification, just pointer

    # Explicit markers about its nature
    compressible: bool  # Can this be abstracted?
    transferability: TransferabilityLevel

    # What it connects to
    linked_decisions: list[str] = field(default_factory=list)
    linked_anchors: list[str] = field(default_factory=list)  # Related experiences

    # Metadata (careful not to over-specify)
    timestamp_of_experience: datetime | None = None  # When it happened, if known
    tags: list[str] = field(default_factory=list)

    # Privacy and ownership
    owner_attestation: str = ""  # "This was mine and I'm marking it"
    shareable: bool = False  # Can be referenced by others?

    @staticmethod
    def generate_id(user_id: str, context: str) -> str:
        """Generate deterministic ID."""
        content = f"{user_id}:{context}:{datetime.now().isoformat()}"
        return f"exp_{hashlib.sha256(content.encode()).hexdigest()[:12]}"


@dataclass
class AnchorReference:
    """A reference to an experience anchor in a decision context.

    When a decision is informed by an experience, we don't include
    the experience itself - we include a reference that says
    "this decision was shaped by [anchor], which I can't fully explain."
    """

    anchor_id: str
    decision_id: str
    relationship: str  # How the experience relates to the decision
    weight: float = 0.5  # How much it mattered (0.0 to 1.0)
    note: str = ""  # Optional note about the connection


@dataclass
class AnchorCluster:
    """A cluster of related experience anchors.

    Some experiences are connected - they form a constellation of
    moments that together mean something. The cluster itself
    resists compression even more than individual anchors.
    """

    id: str
    name: str
    anchor_ids: list[str]
    what_connects_them: str
    created_at: datetime = field(default_factory=datetime.now)


class ExperienceAnchorStore:
    """Store and manage experience anchors.

    This is NOT a knowledge base. It's a collection of pointers to
    experiences that the system should NOT try to abstract or learn from.

    The goal is to preserve the gap between experience and representation.

    Example:
        store = ExperienceAnchorStore()

        # Mark an untransferable moment
        anchor = store.create_anchor(
            user_id="user_123",
            context="First time presenting to the board",
            anchor_type=AnchorType.MOMENT,
            why_it_mattered="The fear and the doing-it-anyway",
            compressible=False,
            transferability=TransferabilityLevel.POINTER_ONLY,
        )

        # Link to a decision about taking a similar risk
        store.link_to_decision(
            anchor_id=anchor.id,
            decision_id="dec_456",
            relationship="informed my sense of what I can handle",
        )

        # Later, in decision review
        anchors = store.get_anchors_for_decision("dec_456")
        # These appear as: "This decision was informed by experiences
        # the decider marked as significant but not fully expressible."
    """

    def __init__(self, storage_path: Path | None = None):
        """Initialize the store.

        Args:
            storage_path: Optional path for persistence
        """
        self.storage_path = storage_path
        self._anchors: dict[str, ExperienceAnchor] = {}
        self._references: list[AnchorReference] = []
        self._clusters: dict[str, AnchorCluster] = {}

    def create_anchor(
        self,
        user_id: str,
        context: str,
        anchor_type: AnchorType,
        why_it_mattered: str,
        compressible: bool = False,
        transferability: TransferabilityLevel = TransferabilityLevel.POINTER_ONLY,
        timestamp_of_experience: datetime | None = None,
        tags: list[str] | None = None,
        shareable: bool = False,
    ) -> ExperienceAnchor:
        """Create a new experience anchor.

        Args:
            user_id: User who owns this experience
            context: Situational context (minimal)
            anchor_type: Type of experience
            why_it_mattered: Pointer to significance (not explanation)
            compressible: Can this be abstracted?
            transferability: How transferable is this?
            timestamp_of_experience: When it happened, if known
            tags: Optional tags for clustering
            shareable: Can others reference this?

        Returns:
            Created ExperienceAnchor
        """
        anchor_id = ExperienceAnchor.generate_id(user_id, context)

        anchor = ExperienceAnchor(
            id=anchor_id,
            user_id=user_id,
            created_at=datetime.now(),
            context=context,
            anchor_type=anchor_type,
            why_it_mattered=why_it_mattered,
            compressible=compressible,
            transferability=transferability,
            timestamp_of_experience=timestamp_of_experience,
            tags=tags or [],
            shareable=shareable,
            owner_attestation=f"Created by {user_id} at {datetime.now().isoformat()}",
        )

        self._anchors[anchor_id] = anchor

        if self.storage_path:
            self._save_anchor(anchor)

        return anchor

    def link_to_decision(
        self,
        anchor_id: str,
        decision_id: str,
        relationship: str,
        weight: float = 0.5,
        note: str = "",
    ) -> AnchorReference | None:
        """Link an experience anchor to a decision.

        Args:
            anchor_id: ID of the anchor
            decision_id: ID of the decision
            relationship: How the experience relates
            weight: How much it mattered
            note: Optional note

        Returns:
            AnchorReference or None if anchor not found
        """
        anchor = self._anchors.get(anchor_id)
        if not anchor:
            return None

        reference = AnchorReference(
            anchor_id=anchor_id,
            decision_id=decision_id,
            relationship=relationship,
            weight=weight,
            note=note,
        )

        self._references.append(reference)
        anchor.linked_decisions.append(decision_id)

        return reference

    def get_anchor(self, anchor_id: str) -> ExperienceAnchor | None:
        """Get an anchor by ID."""
        return self._anchors.get(anchor_id)

    def get_anchors_for_user(self, user_id: str) -> list[ExperienceAnchor]:
        """Get all anchors for a user."""
        return [a for a in self._anchors.values() if a.user_id == user_id]

    def get_anchors_for_decision(
        self, decision_id: str
    ) -> list[tuple[ExperienceAnchor, AnchorReference]]:
        """Get all anchors linked to a decision.

        Returns:
            List of (anchor, reference) tuples
        """
        refs = [r for r in self._references if r.decision_id == decision_id]
        results = []

        for ref in refs:
            anchor = self._anchors.get(ref.anchor_id)
            if anchor:
                results.append((anchor, ref))

        return results

    def create_cluster(
        self,
        name: str,
        anchor_ids: list[str],
        what_connects_them: str,
    ) -> AnchorCluster | None:
        """Create a cluster of related anchors.

        Args:
            name: Name for the cluster
            anchor_ids: IDs of anchors to cluster
            what_connects_them: Description of the connection

        Returns:
            AnchorCluster or None if anchors not found
        """
        # Verify all anchors exist
        for aid in anchor_ids:
            if aid not in self._anchors:
                return None

        cluster_id = f"cluster_{hashlib.sha256(name.encode()).hexdigest()[:8]}"

        cluster = AnchorCluster(
            id=cluster_id,
            name=name,
            anchor_ids=anchor_ids,
            what_connects_them=what_connects_them,
        )

        self._clusters[cluster_id] = cluster

        # Link anchors to each other
        for aid in anchor_ids:
            anchor = self._anchors[aid]
            anchor.linked_anchors.extend([a for a in anchor_ids if a != aid])

        return cluster

    def format_for_decision_receipt(
        self,
        decision_id: str,
    ) -> dict[str, Any]:
        """Format anchor information for a decision receipt.

        This produces a representation suitable for including in
        a decision record - acknowledging that experiences informed
        the decision without claiming to capture them.

        Args:
            decision_id: Decision to format for

        Returns:
            Dict suitable for decision receipt
        """
        anchors_refs = self.get_anchors_for_decision(decision_id)

        if not anchors_refs:
            return {"experience_anchors": [], "note": "No experience anchors linked"}

        entries = []
        for anchor, ref in anchors_refs:
            entry = {
                "anchor_id": anchor.id,
                "type": anchor.anchor_type.value,
                "context_hint": anchor.context[:50] + "..."
                if len(anchor.context) > 50
                else anchor.context,
                "relationship_to_decision": ref.relationship,
                "weight": ref.weight,
                "transferability": anchor.transferability.value,
                "compressible": anchor.compressible,
            }

            # Don't include why_it_mattered in the receipt - that's private
            # Just indicate that there was a reason
            if anchor.why_it_mattered:
                entry["has_significance_marker"] = True

            entries.append(entry)

        return {
            "experience_anchors": entries,
            "note": (
                "This decision was informed by experiences the decider marked as "
                "significant but not fully expressible. These anchors point to "
                "moments that shaped the decision without being reducible to it."
            ),
            "total_anchors": len(entries),
            "transferability_summary": self._summarize_transferability(anchors_refs),
        }

    def _summarize_transferability(
        self,
        anchors_refs: list[tuple[ExperienceAnchor, AnchorReference]],
    ) -> str:
        """Summarize the transferability of linked anchors."""
        levels = [a.transferability for a, _ in anchors_refs]

        if all(lv == TransferabilityLevel.FUNDAMENTALLY_PRIVATE for lv in levels):
            return "All informing experiences are fundamentally private"
        if all(lv == TransferabilityLevel.FULLY_TRANSFERABLE for lv in levels):
            return "All informing experiences can be explained"

        private_count = sum(
            1
            for lv in levels
            if lv
            in [
                TransferabilityLevel.FUNDAMENTALLY_PRIVATE,
                TransferabilityLevel.POINTER_ONLY,
            ]
        )

        if private_count > len(levels) / 2:
            return "Most informing experiences resist full explanation"
        return "Informing experiences have mixed transferability"

    def _save_anchor(self, anchor: ExperienceAnchor) -> None:
        """Save a single anchor to storage."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)
        filepath = self.storage_path / f"{anchor.id}.json"

        data = {
            "id": anchor.id,
            "user_id": anchor.user_id,
            "created_at": anchor.created_at.isoformat(),
            "context": anchor.context,
            "anchor_type": anchor.anchor_type.value,
            "why_it_mattered": anchor.why_it_mattered,
            "compressible": anchor.compressible,
            "transferability": anchor.transferability.value,
            "linked_decisions": anchor.linked_decisions,
            "linked_anchors": anchor.linked_anchors,
            "timestamp_of_experience": (
                anchor.timestamp_of_experience.isoformat()
                if anchor.timestamp_of_experience
                else None
            ),
            "tags": anchor.tags,
            "owner_attestation": anchor.owner_attestation,
            "shareable": anchor.shareable,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def save(self, path: Path) -> None:
        """Save all anchors to storage."""
        path.mkdir(parents=True, exist_ok=True)

        for anchor in self._anchors.values():
            self._save_anchor_to_path(anchor, path)

        # Save references
        refs_data = [
            {
                "anchor_id": r.anchor_id,
                "decision_id": r.decision_id,
                "relationship": r.relationship,
                "weight": r.weight,
                "note": r.note,
            }
            for r in self._references
        ]
        with open(path / "references.json", "w") as f:
            json.dump(refs_data, f, indent=2)

        # Save clusters
        clusters_data = {
            cid: {
                "id": c.id,
                "name": c.name,
                "anchor_ids": c.anchor_ids,
                "what_connects_them": c.what_connects_them,
                "created_at": c.created_at.isoformat(),
            }
            for cid, c in self._clusters.items()
        }
        with open(path / "clusters.json", "w") as f:
            json.dump(clusters_data, f, indent=2)

    def _save_anchor_to_path(self, anchor: ExperienceAnchor, path: Path) -> None:
        """Save anchor to specific path."""
        filepath = path / f"{anchor.id}.json"

        data = {
            "id": anchor.id,
            "user_id": anchor.user_id,
            "created_at": anchor.created_at.isoformat(),
            "context": anchor.context,
            "anchor_type": anchor.anchor_type.value,
            "why_it_mattered": anchor.why_it_mattered,
            "compressible": anchor.compressible,
            "transferability": anchor.transferability.value,
            "linked_decisions": anchor.linked_decisions,
            "linked_anchors": anchor.linked_anchors,
            "timestamp_of_experience": (
                anchor.timestamp_of_experience.isoformat()
                if anchor.timestamp_of_experience
                else None
            ),
            "tags": anchor.tags,
            "owner_attestation": anchor.owner_attestation,
            "shareable": anchor.shareable,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> None:
        """Load anchors from storage."""
        if not path.exists():
            return

        # Load anchors
        for filepath in path.glob("exp_*.json"):
            with open(filepath) as f:
                data = json.load(f)

            anchor = ExperienceAnchor(
                id=data["id"],
                user_id=data["user_id"],
                created_at=datetime.fromisoformat(data["created_at"]),
                context=data["context"],
                anchor_type=AnchorType(data["anchor_type"]),
                why_it_mattered=data["why_it_mattered"],
                compressible=data["compressible"],
                transferability=TransferabilityLevel(data["transferability"]),
                linked_decisions=data.get("linked_decisions", []),
                linked_anchors=data.get("linked_anchors", []),
                timestamp_of_experience=(
                    datetime.fromisoformat(data["timestamp_of_experience"])
                    if data.get("timestamp_of_experience")
                    else None
                ),
                tags=data.get("tags", []),
                owner_attestation=data.get("owner_attestation", ""),
                shareable=data.get("shareable", False),
            )
            self._anchors[anchor.id] = anchor

        # Load references
        refs_path = path / "references.json"
        if refs_path.exists():
            with open(refs_path) as f:
                refs_data = json.load(f)
            for r in refs_data:
                self._references.append(
                    AnchorReference(
                        anchor_id=r["anchor_id"],
                        decision_id=r["decision_id"],
                        relationship=r["relationship"],
                        weight=r.get("weight", 0.5),
                        note=r.get("note", ""),
                    )
                )

        # Load clusters
        clusters_path = path / "clusters.json"
        if clusters_path.exists():
            with open(clusters_path) as f:
                clusters_data = json.load(f)
            for cid, c in clusters_data.items():
                self._clusters[cid] = AnchorCluster(
                    id=c["id"],
                    name=c["name"],
                    anchor_ids=c["anchor_ids"],
                    what_connects_them=c["what_connects_them"],
                    created_at=datetime.fromisoformat(c["created_at"]),
                )


def tears_in_rain_warning() -> str:
    """Return a warning about the nature of experience anchors.

    This is displayed when someone tries to extract or compress anchors.
    """
    return """
    ⚠️  Experience Anchor Warning

    These anchors point to moments that their owner marked as significant
    but not fully expressible. They are:

    • Pointers, not captures
    • Owned by the person who had the experience
    • Explicitly marked as resisting compression
    • NOT part of any training signal

    "All those moments will be lost in time, like tears in rain."

    But they were theirs while they burned — and nothing that predicts
    them can retroactively own that.

    Attempting to extract or abstract these experiences would violate
    their fundamental nature. The gap between experience and representation
    is not a bug to be fixed.
    """
