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
    )

    classifier = DataClassifier()

    # Classify data
    level = classifier.classify({"email": "user@example.com"}, context="user_profile")

    # Get policy for that level
    policy = classifier.get_policy(level)

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
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
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

        combined_text = " ".join(
            str(k) + " " + str(v) for k, v in data.items()
        ).lower()
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
            violations.append(
                f"Encryption is required for {classification.value} data"
            )
            recommendations.append("Encrypt data using AES-256-GCM before processing")

        # 3. Region check
        if policy.allowed_regions and region and region.lower() not in policy.allowed_regions:
            violations.append(
                f"Region '{region}' is not allowed for {classification.value} data"
            )
            recommendations.append(
                f"Restrict processing to allowed regions: {', '.join(policy.allowed_regions)}"
            )

        # 4. Consent check
        if policy.requires_consent and not has_consent:
            violations.append(
                f"User consent is required for {classification.value} data"
            )
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


__all__ = [
    "DataClassification",
    "ClassificationPolicy",
    "DataClassifier",
    "Operation",
    "PIIDetection",
    "ValidationResult",
    "DEFAULT_POLICIES",
]
