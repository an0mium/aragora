"""Validated structured-identifier detectors for PHI/PII.

These are deterministic, *format-validated* detectors for well-defined
structured identifiers (SSN, NPI, ICD-10, MRN, date of birth, email, phone).
They are NOT semantic classifiers -- each identifier has a precise, published
format, so format detection (with check-digit / range validation where one
exists) is the industry-standard correct tool here (cf. Microsoft Presidio).

Each detector is a callable ``(content: str) -> list[DetectorMatch]`` returning
the matched substring plus its start offset, so the framework can attach a line
number and build a :class:`~aragora.compliance.framework.ComplianceIssue`.

The framework wires these in through the new ``validators`` field on
``ComplianceRule`` so they share the existing finding model and severity/scoring
path rather than forming a parallel system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Callable


@dataclass(frozen=True)
class DetectorMatch:
    """A single structured-identifier match within content."""

    text: str
    start: int


# A named detector: takes content, returns the matches it found.
Detector = Callable[[str], list[DetectorMatch]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _luhn_is_valid(digits: str) -> bool:
    """Return True if ``digits`` passes the Luhn (mod-10) check."""
    total = 0
    reverse = digits[::-1]
    for idx, char in enumerate(reverse):
        digit = ord(char) - ord("0")
        if idx % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


# ---------------------------------------------------------------------------
# SSN
# ---------------------------------------------------------------------------

# US SSN: AAA-GG-SSSS, optionally unseparated when explicitly labeled.
_SSN_RE = re.compile(r"\b(\d{3})([- ]?)(\d{2})([- ]?)(\d{4})\b")
_SSN_CONTEXT_RE = re.compile(r"\b(?:SSN|social\s+security)\b", re.IGNORECASE)


def detect_ssn(content: str) -> list[DetectorMatch]:
    """Detect US Social Security Numbers with structural validation.

    Rejects allocations that the SSA never issues: area 000/666/900-999,
    group 00, serial 0000.
    """
    matches: list[DetectorMatch] = []
    for m in _SSN_RE.finditer(content):
        if not (m.group(2) or m.group(4)):
            context = content[max(0, m.start() - 24) : m.start()]
            if not _SSN_CONTEXT_RE.search(context):
                continue
        area, group, serial = m.group(1), m.group(3), m.group(5)
        if area in ("000", "666") or area[0] == "9":
            continue
        if group == "00" or serial == "0000":
            continue
        matches.append(DetectorMatch(text=m.group(0), start=m.start()))
    return matches


# ---------------------------------------------------------------------------
# NPI (National Provider Identifier)
# ---------------------------------------------------------------------------

# 10-digit number, labeled with "NPI" to avoid false positives on arbitrary
# account IDs, timestamps, or phone-like values that happen to pass Luhn.
_NPI_RE = re.compile(r"\b(\d{10})\b")
_NPI_CONTEXT_RE = re.compile(r"\bNPI\b", re.IGNORECASE)


def detect_npi(content: str) -> list[DetectorMatch]:
    """Detect NPI numbers (10 digits, Luhn-valid with 80840 prefix).

    Per CMS, NPI check-digit validation prepends the constant ``80840`` before
    running the Luhn algorithm on all 15 digits.
    """
    matches: list[DetectorMatch] = []
    for m in _NPI_RE.finditer(content):
        before = content[max(0, m.start() - 24) : m.start()]
        after = content[m.end() : min(len(content), m.end() + 24)]
        if not (_NPI_CONTEXT_RE.search(before) or _NPI_CONTEXT_RE.search(after)):
            continue
        candidate = m.group(1)
        if _luhn_is_valid("80840" + candidate):
            matches.append(DetectorMatch(text=m.group(0), start=m.start()))
    return matches


# ---------------------------------------------------------------------------
# ICD-10 diagnosis codes
# ---------------------------------------------------------------------------

# ICD-10-CM: letter, 2 digits, optional decimal + up to 4 alnum chars.
# Context is required below because this format overlaps with technical tokens.
_ICD10_RE = re.compile(r"\b([A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?)\b")
_ICD10_CONTEXT_RE = re.compile(r"\b(?:ICD-?10|diagnos(?:is|ed)|dx|condition)\b", re.IGNORECASE)


def detect_icd10(content: str) -> list[DetectorMatch]:
    """Detect ICD-10-CM diagnosis codes by format."""
    matches: list[DetectorMatch] = []
    for m in _ICD10_RE.finditer(content):
        context = content[max(0, m.start() - 40) : min(len(content), m.end() + 40)]
        if not _ICD10_CONTEXT_RE.search(context):
            continue
        matches.append(DetectorMatch(text=m.group(0), start=m.start()))
    return matches


# ---------------------------------------------------------------------------
# MRN (Medical Record Number)
# ---------------------------------------------------------------------------

# MRNs have no universal format, so detection is label-anchored to avoid
# false positives on arbitrary numbers.
_MRN_RE = re.compile(
    r"\b(?:MRN|medical\s+record\s+(?:number|no\.?|#))\b\s*[:#]?\s*([A-Z0-9-]{5,12})",
    re.IGNORECASE,
)


def detect_mrn(content: str) -> list[DetectorMatch]:
    """Detect labeled Medical Record Numbers."""
    matches: list[DetectorMatch] = []
    for m in _MRN_RE.finditer(content):
        matches.append(DetectorMatch(text=m.group(1), start=m.start(1)))
    return matches


# ---------------------------------------------------------------------------
# Date of birth
# ---------------------------------------------------------------------------

# ISO (YYYY-MM-DD) or US (MM/DD/YYYY) numeric dates, optionally DOB-labeled.
_DATE_RE = re.compile(r"\b(?:(\d{4})-(\d{2})-(\d{2})|(\d{1,2})/(\d{1,2})/(\d{4}))\b")
_DOB_CONTEXT_RE = re.compile(r"\b(?:DOB|date\s+of\s+birth|birth\s+date|born)\b", re.IGNORECASE)


def _valid_date_parts(year: int, month: int, day: int) -> bool:
    if not 1900 <= year <= 2100:
        return False
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


def detect_dob(content: str) -> list[DetectorMatch]:
    """Detect dates of birth (ISO or US numeric formats) with range checks."""
    matches: list[DetectorMatch] = []
    for m in _DATE_RE.finditer(content):
        context = content[max(0, m.start() - 40) : min(len(content), m.end() + 40)]
        if not _DOB_CONTEXT_RE.search(context):
            continue
        if m.group(1):  # ISO
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        else:  # US
            month, day, year = int(m.group(4)), int(m.group(5)), int(m.group(6))
        if _valid_date_parts(year, month, day):
            matches.append(DetectorMatch(text=m.group(0), start=m.start()))
    return matches


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def detect_email(content: str) -> list[DetectorMatch]:
    """Detect email addresses."""
    return [DetectorMatch(text=m.group(0), start=m.start()) for m in _EMAIL_RE.finditer(content)]


# ---------------------------------------------------------------------------
# Phone
# ---------------------------------------------------------------------------

# US/NANP phone numbers with common separators, optional +1 / parens.
_PHONE_RE = re.compile(
    r"(?<![\d-])(?:\+?1[-.\s]?)?(?:(?:\(\d{3}\)[-.\s]?)|(?:\d{3}[-.\s]))\d{3}[-.\s]\d{4}(?![\d-])"
)


def detect_phone(content: str) -> list[DetectorMatch]:
    """Detect telephone numbers in common North American formats."""
    return [DetectorMatch(text=m.group(0), start=m.start()) for m in _PHONE_RE.finditer(content)]


# Registry of named detectors, referenced by ComplianceRule.validators.
PHI_DETECTORS: dict[str, Detector] = {
    "ssn": detect_ssn,
    "npi": detect_npi,
    "icd10": detect_icd10,
    "mrn": detect_mrn,
    "dob": detect_dob,
    "email": detect_email,
    "phone": detect_phone,
}


__all__ = [
    "DetectorMatch",
    "Detector",
    "PHI_DETECTORS",
    "detect_ssn",
    "detect_npi",
    "detect_icd10",
    "detect_mrn",
    "detect_dob",
    "detect_email",
    "detect_phone",
]
