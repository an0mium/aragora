"""
Data Classification Policy Enforcement.

Tags data with classification levels and enforces handling rules per level.
Integrates with the privacy subsystem for PII detection and the compliance
framework for policy-driven enforcement.

Classification Levels:
- PUBLIC: No restrictions, freely shareable
- INTERNAL: Organization-internal, not for external distribution
- CONFIDENTIAL: Sensitive business data, restricted access
- RESTRICTED: Highly sensitive, strict controls required
- PII: Personally identifiable information, privacy-regulated

Usage:
    from aragora.compliance.data_classification import (
        DataClassifier,
        DataClassification,
        ClassificationMetadata,
        PolicyEnforcer,
    )

    classifier = DataClassifier()

    # Classify data
    level = classifier.classify({"email": "user@example.com"}, context="user_profile")

    # Get policy for that level
    policy = classifier.get_policy(level)

    # Tag data with classification metadata
    metadata = classifier.tag({"email": "user@example.com"}, context="user_profile")
    # metadata.classification == DataClassification.PII
    # metadata.label == "pii"

    # Enforce cross-context access rules
    enforcer = PolicyEnforcer(classifier)
    result = enforcer.enforce_access(
        data={"api_key": "sk-xxx"},
        source_classification=DataClassification.RESTRICTED,
        target_classification=DataClassification.PUBLIC,
    )
    # result.allowed == False -- RESTRICTED data cannot flow to PUBLIC context

    # Validate a proposed operation
    result = classifier.validate_handling(
        data={"email": "user@example.com"},
        classification=level,
        operation="export",
    )
    if not result.allowed:
        print(result.violations)

    # Scan text for PII
    detections = classifier.scan_for_pii("Contact john@acme.com or 555-123-4567")

    # Get the full active policy summary
    summary = classifier.get_active_policy()
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DataClassification(str, Enum):
    """Data classification levels, ordered from least to most sensitive."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"


class Operation(str, Enum):
    """Operations that can be performed on classified data."""

    READ = "read"
    WRITE = "write"
    EXPORT = "export"
    SHARE = "share"
    DELETE = "delete"
    ARCHIVE = "archive"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassificationPolicy:
    """Handling rules enforced for a given classification level."""

    classification: DataClassification
    encryption_required: bool = False
    audit_logging: bool = False
    retention_days: int = 365
    allowed_regions: list[str] = field(default_factory=list)
    requires_consent: bool = False
    allowed_operations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "classification": self.classification.value,
            "encryption_required": self.encryption_required,
            "audit_logging": self.audit_logging,
            "retention_days": self.retention_days,
            "allowed_regions": self.allowed_regions,
            "requires_consent": self.requires_consent,
            "allowed_operations": self.allowed_operations,
        }


@dataclass
class PIIDetection:
    """A single PII detection within scanned text."""

    type: str  # e.g. "email", "phone", "ssn", "credit_card"
    start: int
    end: int
    confidence: float = 0.9

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": self.type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }


@dataclass
class ValidationResult:
    """Result of a handling-validation check."""

    allowed: bool
    violations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "allowed": self.allowed,
            "violations": self.violations,
            "recommendations": self.recommendations,
        }


@dataclass
class ClassificationMetadata:
    """Classification tag attached to a piece of data.

    Carries the assigned level, a human-readable label, the timestamp of
    classification, and optional context about why this level was chosen.
    """

    classification: DataClassification
    label: str
    classified_at: str  # ISO-8601 timestamp
    context: str = ""
    pii_detected: bool = False
    pii_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "classification": self.classification.value,
            "label": self.label,
            "classified_at": self.classified_at,
            "context": self.context,
            "pii_detected": self.pii_detected,
            "pii_types": self.pii_types,
        }


@dataclass
class EnforcementResult:
    """Result of a cross-context enforcement check.

    Determines whether data at a given classification level may flow
    into a target context at a (possibly lower) classification level.
    """

    allowed: bool
    source_classification: DataClassification
    target_classification: DataClassification
    violations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "allowed": self.allowed,
            "source_classification": self.source_classification.value,
            "target_classification": self.target_classification.value,
            "violations": self.violations,
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------------
# Sensitivity ordering helper
# ---------------------------------------------------------------------------

SENSITIVITY_ORDER: list[DataClassification] = list(DataClassification)
"""Classification levels in ascending sensitivity order."""


def sensitivity_index(level: DataClassification) -> int:
    """Return the numeric sensitivity index for *level* (higher = more sensitive)."""
    return SENSITIVITY_ORDER.index(level)


# ---------------------------------------------------------------------------
# Default policies
# ---------------------------------------------------------------------------

DEFAULT_POLICIES: dict[DataClassification, ClassificationPolicy] = {
    DataClassification.PUBLIC: ClassificationPolicy(
        classification=DataClassification.PUBLIC,
        encryption_required=False,
        audit_logging=False,
        retention_days=365,
        allowed_regions=[],  # empty = unrestricted
        requires_consent=False,
        allowed_operations=["read", "write", "export", "share", "delete", "archive"],
    ),
    DataClassification.INTERNAL: ClassificationPolicy(
        classification=DataClassification.INTERNAL,
        encryption_required=False,
        audit_logging=True,
        retention_days=365,
        allowed_regions=[],
        requires_consent=False,
        allowed_operations=["read", "write", "export", "delete", "archive"],
    ),
    DataClassification.CONFIDENTIAL: ClassificationPolicy(
        classification=DataClassification.CONFIDENTIAL,
        encryption_required=True,
        audit_logging=True,
        retention_days=180,
        allowed_regions=["us", "eu", "uk"],
        requires_consent=False,
        allowed_operations=["read", "write", "delete", "archive"],
    ),
    DataClassification.RESTRICTED: ClassificationPolicy(
        classification=DataClassification.RESTRICTED,
        encryption_required=True,
        audit_logging=True,
        retention_days=90,
        allowed_regions=["us", "eu"],
        requires_consent=True,
        allowed_operations=["read", "write", "delete"],
    ),
    DataClassification.PII: ClassificationPolicy(
        classification=DataClassification.PII,
        encryption_required=True,
        audit_logging=True,
        retention_days=90,
        allowed_regions=["us", "eu"],
        requires_consent=True,
        allowed_operations=["read", "write", "delete"],
    ),
}


# ---------------------------------------------------------------------------
# PII regex patterns: (compiled_pattern, pii_type, confidence)
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[re.Pattern[str], str, float]] = [
    (
        re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
        "email",
        0.95,
    ),
    (
        re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "phone",
        0.85,
    ),
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "ssn",
        0.95,
    ),
    (
        re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
        "credit_card",
        0.80,
    ),
]

# Keywords that signal a particular classification when found in data keys
# or context strings.
_CLASSIFICATION_KEYWORDS: dict[DataClassification, list[str]] = {
    DataClassification.PII: [
        "email",
        "phone",
        "ssn",
        "social_security",
        "date_of_birth",
        "dob",
        "passport",
        "driver_license",
        "credit_card",
        "address",
        "national_id",
    ],
    DataClassification.RESTRICTED: [
        "secret",
        "api_key",
        "password",
        "token",
        "credential",
        "private_key",
        "encryption_key",
    ],
    DataClassification.CONFIDENTIAL: [
        "salary",
        "revenue",
        "financial",
        "proprietary",
        "trade_secret",
        "internal_only",
        "contract",
        "medical",
        "health",
        "diagnosis",
    ],
    DataClassification.INTERNAL: [
        "employee",
        "department",
        "project",
        "roadmap",
        "internal",
    ],
}


# ---------------------------------------------------------------------------
# DataClassifier
# ---------------------------------------------------------------------------


class DataClassifier:
    """
    Rule-based data classifier and policy enforcer.

    Assigns a :class:`DataClassification` level to data based on field names,
    content scanning, and context keywords.  Provides per-level policies and
    validates whether a proposed operation is allowed.
    """

    def __init__(
        self,
        policies: dict[DataClassification, ClassificationPolicy] | None = None,
    ) -> None:
        self._policies = policies or DEFAULT_POLICIES.copy()

    # -- classification -----------------------------------------------------

    def classify(self, data: dict[str, Any], context: str = "") -> DataClassification:
        """Classify *data* by inspecting keys, values, and *context*.

        Returns the highest (most sensitive) classification that applies.
        """
        highest = DataClassification.PUBLIC
        sensitivity_order = list(DataClassification)

        combined_text = " ".join(str(k) + " " + str(v) for k, v in data.items()).lower()
        context_lower = context.lower()

        for level in reversed(sensitivity_order):
            keywords = _CLASSIFICATION_KEYWORDS.get(level, [])
            for kw in keywords:
                if kw in combined_text or kw in context_lower:
                    if sensitivity_order.index(level) > sensitivity_order.index(highest):
                        highest = level
                    break

        # Additionally, scan string values for PII patterns
        for value in data.values():
            if isinstance(value, str) and self.scan_for_pii(value):
                if sensitivity_order.index(DataClassification.PII) > sensitivity_order.index(
                    highest
                ):
                    highest = DataClassification.PII

        return highest

    # -- policy lookup ------------------------------------------------------

    def get_policy(self, classification: DataClassification) -> ClassificationPolicy:
        """Return the :class:`ClassificationPolicy` for *classification*."""
        return self._policies[classification]

    # -- validation ---------------------------------------------------------

    def validate_handling(
        self,
        data: dict[str, Any],
        classification: DataClassification,
        operation: str,
        *,
        region: str | None = None,
        has_consent: bool = False,
        is_encrypted: bool = False,
    ) -> ValidationResult:
        """Check whether *operation* is allowed under the policy for *classification*.

        Returns a :class:`ValidationResult` listing any violations and
        recommendations.
        """
        policy = self.get_policy(classification)
        violations: list[str] = []
        recommendations: list[str] = []

        # 1. Operation allowed?
        if policy.allowed_operations and operation not in policy.allowed_operations:
            violations.append(
                f"Operation '{operation}' is not allowed for {classification.value} data"
            )

        # 2. Encryption check
        if policy.encryption_required and not is_encrypted:
            violations.append(f"Encryption is required for {classification.value} data")
            recommendations.append("Encrypt data using AES-256-GCM before processing")

        # 3. Region check
        if policy.allowed_regions and region and region.lower() not in policy.allowed_regions:
            violations.append(f"Region '{region}' is not allowed for {classification.value} data")
            recommendations.append(
                f"Restrict processing to allowed regions: {', '.join(policy.allowed_regions)}"
            )

        # 4. Consent check
        if policy.requires_consent and not has_consent:
            violations.append(f"User consent is required for {classification.value} data")
            recommendations.append("Obtain explicit user consent before processing")

        # 5. Audit logging advisory
        if policy.audit_logging:
            recommendations.append("Ensure this operation is recorded in the audit log")

        allowed = len(violations) == 0
        return ValidationResult(
            allowed=allowed,
            violations=violations,
            recommendations=recommendations,
        )

    # -- PII scanning -------------------------------------------------------

    def scan_for_pii(self, text: str) -> list[PIIDetection]:
        """Scan *text* for PII using regex patterns.

        Returns a list of :class:`PIIDetection` instances describing each
        match found.
        """
        detections: list[PIIDetection] = []
        for pattern, pii_type, confidence in _PII_PATTERNS:
            for match in pattern.finditer(text):
                detections.append(
                    PIIDetection(
                        type=pii_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                    )
                )
        return detections

    # -- tagging ------------------------------------------------------------

    def tag(self, data: dict[str, Any], context: str = "") -> ClassificationMetadata:
        """Classify *data* and return a :class:`ClassificationMetadata` tag.

        The tag includes the classification level, a human-readable label,
        the classification timestamp, and any PII types detected.
        """
        level = self.classify(data, context=context)

        # Gather PII types from all string values
        pii_types: list[str] = []
        for value in data.values():
            if isinstance(value, str):
                for det in self.scan_for_pii(value):
                    if det.type not in pii_types:
                        pii_types.append(det.type)

        return ClassificationMetadata(
            classification=level,
            label=level.value,
            classified_at=datetime.now(timezone.utc).isoformat(),
            context=context,
            pii_detected=len(pii_types) > 0,
            pii_types=pii_types,
        )

    # -- active policy summary ----------------------------------------------

    def get_active_policy(self) -> dict[str, Any]:
        """Return the full active policy as a serializable dictionary.

        Includes metadata about the policy version, all per-level rules,
        and classification keywords.
        """
        return {
            "version": "1.0",
            "name": "Aragora Data Classification Policy",
            "description": (
                "Defines classification levels and handling rules for data "
                "processed by the Aragora platform."
            ),
            "levels": [level.value for level in DataClassification],
            "policies": {
                level.value: self._policies[level].to_dict() for level in DataClassification
            },
            "keywords": {level.value: kws for level, kws in _CLASSIFICATION_KEYWORDS.items()},
            "sensitivity_order": [level.value for level in SENSITIVITY_ORDER],
        }

    # -- all policies accessor -----------------------------------------------

    @property
    def policies(self) -> dict[DataClassification, ClassificationPolicy]:
        """Return the internal policies dictionary (read-only view)."""
        return dict(self._policies)


# ---------------------------------------------------------------------------
# PolicyEnforcer
# ---------------------------------------------------------------------------


class PolicyEnforcer:
    """Prevents data classified at a higher level from being exposed in a
    lower-classification context.

    For example, RESTRICTED data must not flow into a PUBLIC or INTERNAL
    context.  The enforcer checks the source and target levels and produces
    an :class:`EnforcementResult` with any violations.

    Usage::

        enforcer = PolicyEnforcer()
        result = enforcer.enforce_access(
            data={"api_key": "sk-xxx"},
            source_classification=DataClassification.RESTRICTED,
            target_classification=DataClassification.PUBLIC,
        )
        if not result.allowed:
            # block the operation
            ...

    The enforcer also supports adding classification labels to audit log
    entries via :meth:`audit_label`.
    """

    def __init__(
        self,
        classifier: DataClassifier | None = None,
    ) -> None:
        self._classifier = classifier or DataClassifier()

    @property
    def classifier(self) -> DataClassifier:
        """Return the underlying classifier."""
        return self._classifier

    def enforce_access(
        self,
        data: dict[str, Any],
        source_classification: DataClassification,
        target_classification: DataClassification,
        *,
        operation: str = "read",
        region: str | None = None,
        has_consent: bool = False,
        is_encrypted: bool = False,
    ) -> EnforcementResult:
        """Check whether *data* at *source_classification* may flow into
        a context at *target_classification*.

        Data is blocked when the source sensitivity is strictly higher than
        the target sensitivity.  Additional policy checks (encryption,
        consent, region) are applied via the underlying classifier.
        """
        violations: list[str] = []
        recommendations: list[str] = []

        source_idx = sensitivity_index(source_classification)
        target_idx = sensitivity_index(target_classification)

        # Core rule: data cannot flow to a less-sensitive context
        if source_idx > target_idx:
            violations.append(
                f"Data classified as '{source_classification.value}' cannot be "
                f"exposed in '{target_classification.value}' context"
            )
            recommendations.append(
                f"Restrict access to contexts at '{source_classification.value}' level or higher"
            )
            logger.warning(
                "Classification enforcement violation: %s data blocked from %s context",
                source_classification.value,
                target_classification.value,
            )

        # Also apply the standard handling validation for the source level
        handling_result = self._classifier.validate_handling(
            data=data,
            classification=source_classification,
            operation=operation,
            region=region,
            has_consent=has_consent,
            is_encrypted=is_encrypted,
        )
        violations.extend(handling_result.violations)
        recommendations.extend(handling_result.recommendations)

        return EnforcementResult(
            allowed=len(violations) == 0,
            source_classification=source_classification,
            target_classification=target_classification,
            violations=violations,
            recommendations=recommendations,
        )

    def audit_label(
        self,
        data: dict[str, Any],
        context: str = "",
    ) -> dict[str, Any]:
        """Generate a classification label suitable for inclusion in an
        audit log entry.

        Returns a dictionary with ``classification``, ``label``,
        ``pii_detected``, ``audit_logging_required``, and ``timestamp``
        keys.
        """
        metadata = self._classifier.tag(data, context=context)
        policy = self._classifier.get_policy(metadata.classification)

        return {
            "classification": metadata.classification.value,
            "label": metadata.label,
            "pii_detected": metadata.pii_detected,
            "pii_types": metadata.pii_types,
            "audit_logging_required": policy.audit_logging,
            "encryption_required": policy.encryption_required,
            "timestamp": metadata.classified_at,
        }

    def classify_debate_result(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Add classification metadata to a debate result dictionary.

        Inspects the result's content (``outcome``, ``arguments``,
        ``summary``, etc.) and attaches a ``_classification`` key.
        """
        # Build a flat dict of text fields for classification
        fields_to_scan: dict[str, Any] = {}
        for key in ("outcome", "summary", "arguments", "reasoning", "conclusion"):
            if key in result:
                val = result[key]
                if isinstance(val, str):
                    fields_to_scan[key] = val
                elif isinstance(val, list):
                    fields_to_scan[key] = " ".join(str(v) for v in val)
                elif isinstance(val, dict):
                    fields_to_scan[key] = " ".join(str(v) for v in val.values())

        if not fields_to_scan:
            fields_to_scan = {"_raw": str(result)}

        metadata = self._classifier.tag(fields_to_scan)
        enriched = dict(result)
        enriched["_classification"] = metadata.to_dict()
        return enriched

    def classify_knowledge_item(
        self,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        """Add classification metadata to a knowledge item dictionary.

        Knowledge items typically have ``content``, ``title``, ``tags``,
        and ``metadata`` fields.
        """
        fields_to_scan: dict[str, Any] = {}
        for key in ("content", "title", "description", "text", "body"):
            if key in item and isinstance(item[key], str):
                fields_to_scan[key] = item[key]

        context = ""
        if "tags" in item and isinstance(item["tags"], list):
            context = " ".join(str(t) for t in item["tags"])

        if not fields_to_scan:
            fields_to_scan = {"_raw": str(item)}

        metadata = self._classifier.tag(fields_to_scan, context=context)
        enriched = dict(item)
        enriched["_classification"] = metadata.to_dict()
        return enriched

    def classify_api_response(
        self,
        response_data: dict[str, Any],
        context: str = "",
    ) -> dict[str, Any]:
        """Add classification metadata to an API response payload.

        The classification is added as a top-level ``_classification`` key
        so consumers can determine the sensitivity of the response.
        """
        metadata = self._classifier.tag(response_data, context=context)
        enriched = dict(response_data)
        enriched["_classification"] = metadata.to_dict()
        return enriched


__all__ = [
    "DataClassification",
    "ClassificationPolicy",
    "ClassificationMetadata",
    "DataClassifier",
    "EnforcementResult",
    "Operation",
    "PIIDetection",
    "PolicyEnforcer",
    "ValidationResult",
    "DEFAULT_POLICIES",
    "SENSITIVITY_ORDER",
    "sensitivity_index",
]
