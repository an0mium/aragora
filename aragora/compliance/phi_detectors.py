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
#
# The trailing boundary after the label alternation is a negative lookahead for
# a word character (``(?![A-Za-z0-9])``) rather than ``\b``. ``\b`` fails when a
# label variant ends in a non-word char (``#`` in "Medical Record #" or ``.`` in
# "Medical Record no.") followed by whitespace, because there is no word/non-word
# transition there -- which silently dropped those common labels. The lookahead
# accepts the end of any variant (letter, ``.``, or ``#``) as long as the label
# token is not glued to additional word chars.
#
# The captured identifier must contain at least one digit
# (``(?=[A-Z0-9-]{0,11}\d)``). Without this constraint the ``5..12``-char token
# greedily swallowed any ordinary word that happened to follow the label text
# (e.g. "MRN status", "medical record number REDACTED"), producing false
# positives. Real MRNs are numeric or alphanumeric, never plain English words,
# so requiring a digit removes those matches while preserving genuine MRNs. All
# quantifiers remain bounded (the digit lookahead spans at most 12 chars) so the
# pattern stays linear (no ReDoS).
_MRN_RE = re.compile(
    r"\b(?:MRN|medical\s+record\s+(?:number|no\.?|#))(?![A-Za-z0-9])\s*[:#]?\s*"
    r"(?=[A-Z0-9-]{0,11}\d)([A-Z0-9-]{5,12})",
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

# Both the local part and the domain are matched as sequences of dot-separated
# labels whose label class deliberately *excludes* ``.``. Because no character
# class overlaps the literal ``.`` separator, a given dot in the input can only
# be consumed one way -- there is no ambiguous split for the engine to backtrack
# through. Every quantifier is additionally *bounded* to RFC-aligned limits
# (local/domain labels up to 64/63 chars, at most a handful of dot-segments,
# TLD 2-24 letters), so the per-position matching cost is a constant. Together
# these make matching linear in the length of the input and immune to the
# catastrophic backtracking (ReDoS) of an unbounded ``[A-Za-z0-9.-]+`` middle
# class on adversarial input such as ``x@`` + ``a.`` * N or ``x`` + ``.x`` * N.
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9_%+-]{1,64}(?:\.[A-Za-z0-9_%+-]{1,64}){0,8}"
    r"@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){0,8}\.[A-Za-z]{2,24}\b"
)

# Defense-in-depth: a single contiguous run of email-ish characters longer than
# this cannot be a real address (RFC limits a whole address to ~320 chars). The
# detector skips over such runs rather than feeding them to the engine, which
# bounds worst-case work even if the pattern above were ever weakened.
_MAX_EMAIL_TOKEN_LEN = 512
_EMAIL_TOKEN_RE = re.compile(r"[A-Za-z0-9._%+@-]{1," + str(_MAX_EMAIL_TOKEN_LEN) + r"}")


def detect_email(content: str) -> list[DetectorMatch]:
    """Detect email addresses."""
    matches: list[DetectorMatch] = []
    # Scan in bounded email-ish tokens so no single regex application ever sees
    # an unbounded adversarial run (defense-in-depth for the linear pattern).
    for token in _EMAIL_TOKEN_RE.finditer(content):
        for m in _EMAIL_RE.finditer(token.group(0)):
            matches.append(DetectorMatch(text=m.group(0), start=token.start() + m.start()))
    return matches


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
