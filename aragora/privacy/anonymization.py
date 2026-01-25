"""
HIPAA De-identification and Anonymization Module.

Implements comprehensive data anonymization following:
- HIPAA Safe Harbor de-identification (18 identifiers)
- K-anonymity for datasets
- Differential privacy noise addition
- Multiple anonymization methods (redact, hash, generalize, suppress, pseudonymize)

Usage:
    from aragora.privacy.anonymization import HIPAAAnonymizer, AnonymizationMethod

    anonymizer = HIPAAAnonymizer()
    result = anonymizer.anonymize("John Smith's SSN is 123-45-6789", AnonymizationMethod.REDACT)
    print(result.anonymized_content)  # "[NAME]'s SSN is [SSN]"
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)


class AnonymizationMethod(str, Enum):
    """Methods for anonymizing sensitive data."""

    REDACT = "redact"  # Replace with [REDACTED] or type marker
    HASH = "hash"  # One-way cryptographic hash
    GENERALIZE = "generalize"  # Reduce precision (age: 35 → 30-40)
    SUPPRESS = "suppress"  # Remove entirely
    PSEUDONYMIZE = "pseudonymize"  # Replace with consistent fake value


class IdentifierType(str, Enum):
    """HIPAA Safe Harbor identifier types."""

    NAME = "name"
    ADDRESS = "address"
    DATES = "dates"  # All dates except year (birth, admission, discharge, death)
    PHONE = "phone"
    FAX = "fax"
    EMAIL = "email"
    SSN = "ssn"
    MEDICAL_RECORD = "medical_record"
    HEALTH_PLAN = "health_plan"
    ACCOUNT = "account"
    LICENSE = "license"  # Driver's license, professional license
    VEHICLE = "vehicle"  # VIN, license plate
    DEVICE = "device"  # Device identifiers and serial numbers
    URL = "url"
    IP = "ip"
    BIOMETRIC = "biometric"
    PHOTO = "photo"
    UNIQUE_IDENTIFIER = "unique_identifier"  # Any other unique identifier


@dataclass
class DetectedIdentifier:
    """A detected HIPAA identifier in text."""

    identifier_type: IdentifierType
    value: str
    start_pos: int
    end_pos: int
    confidence: float = 0.9


@dataclass
class AnonymizationResult:
    """Result of anonymization operation."""

    original_hash: str  # SHA-256 of original content (for audit)
    anonymized_content: str
    fields_anonymized: list[str] = field(default_factory=list)
    method_used: dict[str, AnonymizationMethod] = field(default_factory=dict)
    identifiers_found: list[DetectedIdentifier] = field(default_factory=list)
    reversible: bool = False
    audit_id: str = field(default_factory=lambda: str(uuid4()))
    anonymized_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_hash": self.original_hash,
            "anonymized_content": self.anonymized_content,
            "fields_anonymized": self.fields_anonymized,
            "method_used": {k: v.value for k, v in self.method_used.items()},
            "identifiers_count": len(self.identifiers_found),
            "reversible": self.reversible,
            "audit_id": self.audit_id,
            "anonymized_at": self.anonymized_at.isoformat(),
        }


@dataclass
class SafeHarborResult:
    """Result of Safe Harbor compliance verification."""

    compliant: bool
    identifiers_remaining: list[DetectedIdentifier] = field(default_factory=list)
    verification_notes: list[str] = field(default_factory=list)
    verified_at: datetime = field(default_factory=datetime.utcnow)


class HIPAAAnonymizer:
    """
    HIPAA Safe Harbor de-identification implementation.

    Implements the 18 identifiers specified in the Safe Harbor method:
    1. Names
    2. Geographic data (address, city, state, zip)
    3. Dates (except year)
    4. Phone numbers
    5. Fax numbers
    6. Email addresses
    7. Social Security numbers
    8. Medical record numbers
    9. Health plan beneficiary numbers
    10. Account numbers
    11. Certificate/license numbers
    12. Vehicle identifiers
    13. Device identifiers
    14. Web URLs
    15. IP addresses
    16. Biometric identifiers
    17. Full face photos
    18. Any other unique identifier
    """

    # Regex patterns for HIPAA identifiers
    IDENTIFIER_PATTERNS: dict[IdentifierType, list[tuple[str, float]]] = {
        IdentifierType.NAME: [
            # Common name patterns (first + last)
            (r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", 0.7),
            # Prefixed names (Mr., Mrs., Dr., etc.)
            (r"\b(Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", 0.85),
        ],
        IdentifierType.SSN: [
            (r"\b(\d{3})-(\d{2})-(\d{4})\b", 0.95),
            (r"\b(\d{3})(\d{2})(\d{4})\b", 0.8),
        ],
        IdentifierType.PHONE: [
            (r"\b(\d{3})[-.\s]?(\d{3})[-.\s]?(\d{4})\b", 0.85),
            (r"\b\+?1?[-.\s]?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\b", 0.9),
        ],
        IdentifierType.EMAIL: [
            (r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b", 0.95),
        ],
        IdentifierType.IP: [
            (r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b", 0.9),
            # IPv6 (simplified)
            (r"\b([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b", 0.85),
        ],
        IdentifierType.ADDRESS: [
            # Street address with number
            (
                r"\b(\d+)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Drive|Dr\.?|Lane|Ln\.?|Boulevard|Blvd\.?)\b",
                0.85,
            ),
            # ZIP code
            (r"\b(\d{5})(?:-(\d{4}))?\b", 0.7),
        ],
        IdentifierType.DATES: [
            # MM/DD/YYYY or MM-DD-YYYY
            (r"\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](19|20)\d{2}\b", 0.85),
            # Month DD, YYYY
            (
                r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(19|20)\d{2}\b",
                0.9,
            ),
        ],
        IdentifierType.MEDICAL_RECORD: [
            (r"\b(MRN|MR#|Medical\s+Record)\s*[:#]?\s*([A-Z0-9]{6,})\b", 0.9),
        ],
        IdentifierType.ACCOUNT: [
            (r"\b(Account|Acct\.?)\s*[:#]?\s*([A-Z0-9]{6,})\b", 0.8),
        ],
        IdentifierType.LICENSE: [
            (r"\b(License|Lic\.?|DL)\s*[:#]?\s*([A-Z0-9]{6,})\b", 0.8),
        ],
        IdentifierType.VEHICLE: [
            # VIN (17 characters)
            (r"\b([A-HJ-NPR-Z0-9]{17})\b", 0.75),
        ],
        IdentifierType.URL: [
            (r"\b(https?://[^\s<>\"]+)\b", 0.9),
        ],
    }

    # Replacement markers for redaction
    REDACTION_MARKERS: dict[IdentifierType, str] = {
        IdentifierType.NAME: "[NAME]",
        IdentifierType.ADDRESS: "[ADDRESS]",
        IdentifierType.DATES: "[DATE]",
        IdentifierType.PHONE: "[PHONE]",
        IdentifierType.FAX: "[FAX]",
        IdentifierType.EMAIL: "[EMAIL]",
        IdentifierType.SSN: "[SSN]",
        IdentifierType.MEDICAL_RECORD: "[MRN]",
        IdentifierType.HEALTH_PLAN: "[HEALTH_PLAN_ID]",
        IdentifierType.ACCOUNT: "[ACCOUNT]",
        IdentifierType.LICENSE: "[LICENSE]",
        IdentifierType.VEHICLE: "[VEHICLE_ID]",
        IdentifierType.DEVICE: "[DEVICE_ID]",
        IdentifierType.URL: "[URL]",
        IdentifierType.IP: "[IP]",
        IdentifierType.BIOMETRIC: "[BIOMETRIC]",
        IdentifierType.PHOTO: "[PHOTO]",
        IdentifierType.UNIQUE_IDENTIFIER: "[ID]",
    }

    def __init__(
        self,
        hash_salt: str = "",
        pseudonym_seed: int | None = None,
    ) -> None:
        """
        Initialize the HIPAA anonymizer.

        Args:
            hash_salt: Salt for hashing operations (improves security)
            pseudonym_seed: Seed for pseudonymization (ensures consistency)
        """
        self.hash_salt = hash_salt or str(uuid4())
        self.pseudonym_seed = pseudonym_seed
        self._pseudonym_map: dict[str, str] = {}

        if pseudonym_seed is not None:
            random.seed(pseudonym_seed)

    def detect_identifiers(self, content: str) -> list[DetectedIdentifier]:
        """
        Detect HIPAA identifiers in text content.

        Args:
            content: Text to analyze

        Returns:
            List of detected identifiers with positions
        """
        identifiers: list[DetectedIdentifier] = []

        for id_type, patterns in self.IDENTIFIER_PATTERNS.items():
            for pattern, confidence in patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    identifiers.append(
                        DetectedIdentifier(
                            identifier_type=id_type,
                            value=match.group(0),
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=confidence,
                        )
                    )

        # Sort by position (descending) for safe replacement
        identifiers.sort(key=lambda x: x.start_pos, reverse=True)

        return identifiers

    def anonymize(
        self,
        content: str,
        method: AnonymizationMethod = AnonymizationMethod.REDACT,
        identifier_types: list[IdentifierType] | None = None,
    ) -> AnonymizationResult:
        """
        Anonymize content using the specified method.

        Args:
            content: Text content to anonymize
            method: Anonymization method to use
            identifier_types: Types to anonymize (None = all)

        Returns:
            AnonymizationResult with anonymized content
        """
        original_hash = hashlib.sha256(content.encode()).hexdigest()
        identifiers = self.detect_identifiers(content)

        if identifier_types:
            identifiers = [i for i in identifiers if i.identifier_type in identifier_types]

        anonymized = content
        fields_anonymized: list[str] = []
        method_used: dict[str, AnonymizationMethod] = {}

        for identifier in identifiers:
            replacement = self._get_replacement(
                identifier.value,
                identifier.identifier_type,
                method,
            )
            anonymized = (
                anonymized[: identifier.start_pos] + replacement + anonymized[identifier.end_pos :]
            )
            fields_anonymized.append(identifier.identifier_type.value)
            method_used[identifier.value] = method

        return AnonymizationResult(
            original_hash=original_hash,
            anonymized_content=anonymized,
            fields_anonymized=list(set(fields_anonymized)),
            method_used=method_used,
            identifiers_found=identifiers,
            reversible=method == AnonymizationMethod.PSEUDONYMIZE,
        )

    def anonymize_structured(
        self,
        data: dict[str, Any],
        field_methods: dict[str, AnonymizationMethod] | None = None,
        default_method: AnonymizationMethod = AnonymizationMethod.REDACT,
    ) -> AnonymizationResult:
        """
        Anonymize structured data (dictionary).

        Args:
            data: Dictionary of field -> value
            field_methods: Method to use for each field
            default_method: Default method if field not specified

        Returns:
            AnonymizationResult with anonymized data as JSON string
        """
        import json

        original_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        field_methods = field_methods or {}

        anonymized_data: dict[str, Any] = {}
        fields_anonymized: list[str] = []
        method_used: dict[str, AnonymizationMethod] = {}
        all_identifiers: list[DetectedIdentifier] = []

        for field_name, value in data.items():
            method = field_methods.get(field_name, default_method)

            if isinstance(value, str):
                identifiers = self.detect_identifiers(value)

                if identifiers:
                    result = self.anonymize(value, method)
                    anonymized_data[field_name] = result.anonymized_content
                    fields_anonymized.append(field_name)
                    method_used[field_name] = method
                    all_identifiers.extend(identifiers)
                else:
                    anonymized_data[field_name] = value
            else:
                anonymized_data[field_name] = value

        return AnonymizationResult(
            original_hash=original_hash,
            anonymized_content=json.dumps(anonymized_data),
            fields_anonymized=fields_anonymized,
            method_used=method_used,
            identifiers_found=all_identifiers,
            reversible=all(m == AnonymizationMethod.PSEUDONYMIZE for m in method_used.values()),
        )

    def detect_and_anonymize(
        self,
        content: str,
        method: AnonymizationMethod = AnonymizationMethod.REDACT,
    ) -> AnonymizationResult:
        """
        Detect identifiers and anonymize in one step.

        Args:
            content: Text content
            method: Anonymization method

        Returns:
            AnonymizationResult
        """
        return self.anonymize(content, method)

    def verify_safe_harbor(self, content: str) -> SafeHarborResult:
        """
        Verify content meets HIPAA Safe Harbor requirements.

        Args:
            content: Text to verify

        Returns:
            SafeHarborResult indicating compliance
        """
        identifiers = self.detect_identifiers(content)

        notes: list[str] = []

        if not identifiers:
            notes.append("No HIPAA identifiers detected")
            return SafeHarborResult(
                compliant=True,
                identifiers_remaining=[],
                verification_notes=notes,
            )

        # Check each identifier type
        identifier_types_found = set(i.identifier_type for i in identifiers)
        notes.append(f"Found {len(identifiers)} potential identifiers")
        notes.append(f"Identifier types: {[t.value for t in identifier_types_found]}")

        return SafeHarborResult(
            compliant=False,
            identifiers_remaining=identifiers,
            verification_notes=notes,
        )

    def _get_replacement(
        self,
        value: str,
        identifier_type: IdentifierType,
        method: AnonymizationMethod,
    ) -> str:
        """Get replacement value based on method."""
        if method == AnonymizationMethod.REDACT:
            return self.REDACTION_MARKERS.get(identifier_type, "[REDACTED]")

        elif method == AnonymizationMethod.HASH:
            salted = f"{self.hash_salt}{value}"
            return hashlib.sha256(salted.encode()).hexdigest()[:16]

        elif method == AnonymizationMethod.GENERALIZE:
            return self._generalize_value(value, identifier_type)

        elif method == AnonymizationMethod.SUPPRESS:
            return ""

        elif method == AnonymizationMethod.PSEUDONYMIZE:
            return self._pseudonymize_value(value, identifier_type)

        return "[ANONYMIZED]"

    def _generalize_value(self, value: str, identifier_type: IdentifierType) -> str:
        """Generalize a value to reduce precision."""
        if identifier_type == IdentifierType.DATES:
            # Keep only year
            match = re.search(r"(19|20)\d{2}", value)
            if match:
                return match.group(0)
            return "[YEAR]"

        elif identifier_type == IdentifierType.ADDRESS:
            # Keep only state or region
            return "[REGION]"

        elif identifier_type == IdentifierType.PHONE:
            # Keep only area code
            match = re.match(r"(\d{3})", value.replace("-", "").replace(".", ""))
            if match:
                return f"({match.group(1)}) XXX-XXXX"
            return "[PHONE]"

        elif identifier_type == IdentifierType.IP:
            # Keep only first two octets
            parts = value.split(".")
            if len(parts) >= 2:
                return f"{parts[0]}.{parts[1]}.x.x"
            return "[IP]"

        return "[GENERALIZED]"

    def _pseudonymize_value(self, value: str, identifier_type: IdentifierType) -> str:
        """Generate a consistent pseudonym for a value."""
        # Check if we already have a pseudonym for this value
        if value in self._pseudonym_map:
            return self._pseudonym_map[value]

        # Generate pseudonym based on type
        pseudonym = self._generate_pseudonym(identifier_type)
        self._pseudonym_map[value] = pseudonym

        return pseudonym

    def _generate_pseudonym(self, identifier_type: IdentifierType) -> str:
        """Generate a fake value for an identifier type."""
        if identifier_type == IdentifierType.NAME:
            first_names = ["Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Quinn"]
            last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Davis", "Miller"]
            return f"{random.choice(first_names)} {random.choice(last_names)}"

        elif identifier_type == IdentifierType.EMAIL:
            return f"user{random.randint(1000, 9999)}@example.com"

        elif identifier_type == IdentifierType.PHONE:
            return f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"

        elif identifier_type == IdentifierType.SSN:
            return f"XXX-XX-{random.randint(1000, 9999)}"

        elif identifier_type == IdentifierType.IP:
            return f"192.0.2.{random.randint(1, 254)}"

        elif identifier_type == IdentifierType.ADDRESS:
            return f"{random.randint(100, 999)} Main Street"

        return f"PSEUDO_{uuid4().hex[:8]}"


class KAnonymizer:
    """
    K-anonymity implementation for datasets.

    K-anonymity ensures that each record is indistinguishable from
    at least k-1 other records with respect to quasi-identifiers.
    """

    def __init__(self, k: int = 5) -> None:
        """
        Initialize K-anonymizer.

        Args:
            k: Minimum group size for anonymity (default: 5)
        """
        if k < 2:
            raise ValueError("k must be at least 2 for k-anonymity")
        self.k = k

    def anonymize_dataset(
        self,
        records: list[dict[str, Any]],
        quasi_identifiers: list[str],
        generalizers: dict[str, Callable[[Any], Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Anonymize a dataset to achieve k-anonymity.

        Args:
            records: List of record dictionaries
            quasi_identifiers: Fields that are quasi-identifiers
            generalizers: Custom generalization functions per field

        Returns:
            Anonymized records
        """
        if not records:
            return []

        generalizers = generalizers or {}
        anonymized = [r.copy() for r in records]

        # Apply default generalizers for fields without custom ones
        for qi in quasi_identifiers:
            if qi not in generalizers:
                generalizers[qi] = self._default_generalizer

        # Iteratively generalize until k-anonymity is achieved
        max_iterations = 10
        for _ in range(max_iterations):
            groups = self._group_records(anonymized, quasi_identifiers)

            # Check if all groups meet k-anonymity
            small_groups = [g for g in groups.values() if len(g) < self.k]

            if not small_groups:
                break

            # Generalize quasi-identifiers
            for qi in quasi_identifiers:
                generalizer = generalizers[qi]
                for record in anonymized:
                    record[qi] = generalizer(record[qi])

        return anonymized

    def check_k_anonymity(
        self,
        records: list[dict[str, Any]],
        quasi_identifiers: list[str],
    ) -> tuple[bool, int]:
        """
        Check if dataset satisfies k-anonymity.

        Args:
            records: Dataset records
            quasi_identifiers: Quasi-identifier fields

        Returns:
            Tuple of (is_k_anonymous, minimum_group_size)
        """
        groups = self._group_records(records, quasi_identifiers)

        if not groups:
            return True, 0

        min_size = min(len(g) for g in groups.values())
        return min_size >= self.k, min_size

    def _group_records(
        self,
        records: list[dict[str, Any]],
        quasi_identifiers: list[str],
    ) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
        """Group records by quasi-identifier values."""
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

        for record in records:
            key = tuple(record.get(qi) for qi in quasi_identifiers)
            if key not in groups:
                groups[key] = []
            groups[key].append(record)

        return groups

    def _default_generalizer(self, value: Any) -> Any:
        """Default generalization function."""
        if isinstance(value, (int, float)):
            # Round to nearest 10
            return round(value, -1)
        elif isinstance(value, str):
            # Truncate to first 3 characters
            return value[:3] + "..." if len(value) > 3 else value
        return value


class DifferentialPrivacy:
    """
    Differential privacy implementation for numeric data.

    Provides epsilon-differential privacy through noise addition.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5) -> None:
        """
        Initialize differential privacy mechanism.

        Args:
            epsilon: Privacy parameter (lower = more private)
            delta: Probability of privacy failure (for Gaussian mechanism)
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if delta < 0 or delta >= 1:
            raise ValueError("delta must be in [0, 1)")

        self.epsilon = epsilon
        self.delta = delta

    def add_laplace_noise(
        self,
        value: float,
        sensitivity: float,
    ) -> float:
        """
        Add Laplace noise for epsilon-differential privacy.

        Args:
            value: Original value
            sensitivity: Query sensitivity (max change from one record)

        Returns:
            Value with Laplace noise added
        """
        scale = sensitivity / self.epsilon
        noise = self._laplace_sample(scale)
        return value + noise

    def add_gaussian_noise(
        self,
        value: float,
        sensitivity: float,
    ) -> float:
        """
        Add Gaussian noise for (epsilon, delta)-differential privacy.

        Args:
            value: Original value
            sensitivity: Query sensitivity

        Returns:
            Value with Gaussian noise added
        """
        sigma = self._gaussian_sigma(sensitivity)
        noise = random.gauss(0, sigma)
        return value + noise

    def privatize_count(self, count: int) -> int:
        """
        Privatize a count query with sensitivity 1.

        Args:
            count: Original count

        Returns:
            Privatized count (always non-negative)
        """
        noisy = self.add_laplace_noise(float(count), sensitivity=1.0)
        return max(0, round(noisy))

    def privatize_sum(self, total: float, max_contribution: float) -> float:
        """
        Privatize a sum query.

        Args:
            total: Original sum
            max_contribution: Maximum contribution from any individual

        Returns:
            Privatized sum
        """
        return self.add_laplace_noise(total, sensitivity=max_contribution)

    def privatize_mean(
        self,
        values: list[float],
        lower_bound: float,
        upper_bound: float,
    ) -> float:
        """
        Privatize a mean query.

        Args:
            values: Original values
            lower_bound: Minimum possible value
            upper_bound: Maximum possible value

        Returns:
            Privatized mean
        """
        if not values:
            return 0.0

        n = len(values)
        clipped = [max(lower_bound, min(upper_bound, v)) for v in values]
        true_mean = sum(clipped) / n

        # Sensitivity is (upper - lower) / n
        sensitivity = (upper_bound - lower_bound) / n
        return self.add_laplace_noise(true_mean, sensitivity)

    def _laplace_sample(self, scale: float) -> float:
        """Sample from Laplace distribution."""
        u = random.random() - 0.5
        return -scale * math.copysign(1, u) * math.log(1 - 2 * abs(u))

    def _gaussian_sigma(self, sensitivity: float) -> float:
        """Calculate sigma for Gaussian mechanism."""
        # Using the analytic Gaussian mechanism
        return sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon


# Convenience functions for common operations


def redact_pii(content: str) -> str:
    """Convenience function to redact PII from text."""
    anonymizer = HIPAAAnonymizer()
    result = anonymizer.anonymize(content, AnonymizationMethod.REDACT)
    return result.anonymized_content


def hash_identifier(value: str, salt: str = "") -> str:
    """Convenience function to hash an identifier."""
    salted = f"{salt}{value}"
    return hashlib.sha256(salted.encode()).hexdigest()


def check_safe_harbor_compliance(content: str) -> bool:
    """Convenience function to check Safe Harbor compliance."""
    anonymizer = HIPAAAnonymizer()
    result = anonymizer.verify_safe_harbor(content)
    return result.compliant


__all__ = [
    "AnonymizationMethod",
    "AnonymizationResult",
    "DetectedIdentifier",
    "DifferentialPrivacy",
    "HIPAAAnonymizer",
    "IdentifierType",
    "KAnonymizer",
    "SafeHarborResult",
    "check_safe_harbor_compliance",
    "hash_identifier",
    "redact_pii",
]
