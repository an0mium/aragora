"""
Data Extraction Skill.

Provides capabilities to extract structured data from unstructured text.
Supports various extraction patterns including entities, dates, emails, URLs, etc.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..base import (
    Skill,
    SkillContext,
    SkillManifest,
    SkillResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of a single extraction."""

    type: str
    value: str
    start: int
    end: int
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "value": self.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class DataExtractionSkill(Skill):
    """
    Skill for extracting structured data from unstructured text.

    Supports extraction of:
    - Email addresses
    - URLs and links
    - Phone numbers
    - Dates and times
    - Currency/money values
    - Numbers and percentages
    - Named entities (basic)
    - Custom patterns (regex)
    - JSON from text
    - Key-value pairs
    """

    @property
    def manifest(self) -> SkillManifest:
        return SkillManifest(
            name="data_extraction",
            version="1.0.0",
            description="Extract structured data from unstructured text",
            capabilities=[],
            input_schema={
                "text": {
                    "type": "string",
                    "description": "Text to extract data from",
                    "required": True,
                },
                "extract_types": {
                    "type": "array",
                    "description": "Types to extract: email, url, phone, date, money, number, percent, entity, json, key_value, all",
                    "default": ["all"],
                },
                "custom_patterns": {
                    "type": "object",
                    "description": "Custom regex patterns: {name: pattern}",
                },
                "include_positions": {
                    "type": "boolean",
                    "description": "Include start/end positions in results",
                    "default": False,
                },
            },
            tags=["extraction", "nlp", "parsing"],
            debate_compatible=True,
            max_execution_time_seconds=30.0,
        )

    async def execute(
        self,
        input_data: dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """Execute data extraction."""
        text = input_data.get("text", "")
        if not text:
            return SkillResult.create_failure(
                "Text is required",
                error_code="missing_text",
            )

        extract_types = input_data.get("extract_types", ["all"])
        custom_patterns = input_data.get("custom_patterns", {})
        include_positions = input_data.get("include_positions", False)

        # Normalize extract_types
        if "all" in extract_types:
            extract_types = [
                "email",
                "url",
                "phone",
                "date",
                "money",
                "number",
                "percent",
                "entity",
                "json",
                "key_value",
            ]

        try:
            extractions: dict[str, list[dict[str, Any]]] = {}

            # Run each extractor
            if "email" in extract_types:
                extractions["emails"] = self._format_results(
                    self._extract_emails(text), include_positions
                )

            if "url" in extract_types:
                extractions["urls"] = self._format_results(
                    self._extract_urls(text), include_positions
                )

            if "phone" in extract_types:
                extractions["phone_numbers"] = self._format_results(
                    self._extract_phones(text), include_positions
                )

            if "date" in extract_types:
                extractions["dates"] = self._format_results(
                    self._extract_dates(text), include_positions
                )

            if "money" in extract_types:
                extractions["money"] = self._format_results(
                    self._extract_money(text), include_positions
                )

            if "number" in extract_types:
                extractions["numbers"] = self._format_results(
                    self._extract_numbers(text), include_positions
                )

            if "percent" in extract_types:
                extractions["percentages"] = self._format_results(
                    self._extract_percentages(text), include_positions
                )

            if "entity" in extract_types:
                extractions["entities"] = self._format_results(
                    self._extract_entities(text), include_positions
                )

            if "json" in extract_types:
                extractions["json_objects"] = self._format_results(
                    self._extract_json(text), include_positions
                )

            if "key_value" in extract_types:
                extractions["key_values"] = self._format_results(
                    self._extract_key_values(text), include_positions
                )

            # Extract custom patterns
            if custom_patterns:
                for name, pattern in custom_patterns.items():
                    try:
                        extractions[f"custom_{name}"] = self._format_results(
                            self._extract_custom(text, pattern, name), include_positions
                        )
                    except re.error as e:
                        logger.warning("Invalid regex pattern '%s': %s", name, e)

            # Calculate summary
            total_extractions = sum(len(v) for v in extractions.values())

            return SkillResult.create_success(
                {
                    "extractions": extractions,
                    "total_extractions": total_extractions,
                    "types_found": [k for k, v in extractions.items() if v],
                }
            )

        except (RuntimeError, ValueError, TypeError, KeyError) as e:
            logger.exception("Data extraction failed: %s", e)
            return SkillResult.create_failure(f"Extraction failed: {e}")

    def _format_results(
        self, results: list[ExtractionResult], include_positions: bool
    ) -> list[dict[str, Any]]:
        """Format extraction results."""
        formatted = []
        for r in results:
            if include_positions:
                formatted.append(r.to_dict())
            else:
                formatted.append(
                    {
                        "value": r.value,
                        "confidence": r.confidence,
                        **(r.metadata if r.metadata else {}),
                    }
                )
        return formatted

    def _extract_emails(self, text: str) -> list[ExtractionResult]:
        """Extract email addresses."""
        pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        results = []
        for match in re.finditer(pattern, text):
            results.append(
                ExtractionResult(
                    type="email",
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                )
            )
        return results

    def _extract_urls(self, text: str) -> list[ExtractionResult]:
        """Extract URLs."""
        pattern = (
            r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(:\d+)?(/[-\w._~:/?#\[\]@!$&\'()*+,;=%]*)?"
        )
        results = []
        for match in re.finditer(pattern, text):
            url = match.group()
            # Determine URL type
            domain = re.search(r"https?://([^/]+)", url)
            metadata = {}
            if domain:
                metadata["domain"] = domain.group(1)

            results.append(
                ExtractionResult(
                    type="url",
                    value=url,
                    start=match.start(),
                    end=match.end(),
                    metadata=metadata,
                )
            )
        return results

    def _extract_phones(self, text: str) -> list[ExtractionResult]:
        """Extract phone numbers."""
        # Various phone formats
        patterns = [
            r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",  # US format
            r"\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",  # International
        ]
        results = []
        seen = set()

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                value = match.group()
                # Normalize for deduplication
                normalized = re.sub(r"[-.\s()]", "", value)
                if normalized not in seen and len(normalized) >= 10:
                    seen.add(normalized)
                    results.append(
                        ExtractionResult(
                            type="phone",
                            value=value,
                            start=match.start(),
                            end=match.end(),
                            metadata={"normalized": normalized},
                        )
                    )
        return results

    def _extract_dates(self, text: str) -> list[ExtractionResult]:
        """Extract dates and times."""
        patterns = [
            # ISO format
            (r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2})?", "iso"),
            # US format (MM/DD/YYYY)
            (r"\d{1,2}/\d{1,2}/\d{2,4}", "us"),
            # European format (DD.MM.YYYY)
            (r"\d{1,2}\.\d{1,2}\.\d{2,4}", "eu"),
            # Written format
            (
                r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
                r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
                r"Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{2,4}",
                "written",
            ),
            # Time
            (r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?", "time"),
        ]

        results = []
        for pattern, date_format in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                results.append(
                    ExtractionResult(
                        type="date",
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        metadata={"format": date_format},
                    )
                )
        return results

    def _extract_money(self, text: str) -> list[ExtractionResult]:
        """Extract currency/money values."""
        patterns = [
            (r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?", "USD"),
            (r"USD\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?", "USD"),
            (r"EUR\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?", "EUR"),
            (r"\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD)", "USD"),
            (r"\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:euros?|EUR)", "EUR"),
            (r"GBP\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?", "GBP"),
            (r"\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:pounds?|GBP)", "GBP"),
        ]

        results = []
        for pattern, currency in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value = match.group()
                # Extract numeric value
                numeric = re.sub(r"[^\d.]", "", value)
                try:
                    amount = float(numeric)
                except ValueError:
                    amount = None

                results.append(
                    ExtractionResult(
                        type="money",
                        value=value,
                        start=match.start(),
                        end=match.end(),
                        metadata={"currency": currency, "amount": amount},
                    )
                )
        return results

    def _extract_numbers(self, text: str) -> list[ExtractionResult]:
        """Extract numbers (not already captured as money/dates/phones)."""
        # Match standalone numbers
        pattern = r"(?<![/$@#])\b-?\d{1,3}(?:,\d{3})*(?:\.\d+)?\b"
        results = []

        for match in re.finditer(pattern, text):
            value = match.group()
            # Skip if likely part of date, phone, or money
            context = text[max(0, match.start() - 5) : min(len(text), match.end() + 5)]
            if re.search(r"[/$@]|AM|PM|:|/\d|\.com", context, re.IGNORECASE):
                continue

            # Parse numeric value
            try:
                numeric = float(value.replace(",", ""))
            except ValueError:
                numeric = None

            results.append(
                ExtractionResult(
                    type="number",
                    value=value,
                    start=match.start(),
                    end=match.end(),
                    metadata={"numeric_value": numeric},
                )
            )
        return results

    def _extract_percentages(self, text: str) -> list[ExtractionResult]:
        """Extract percentage values."""
        pattern = r"-?\d+(?:\.\d+)?%"
        results = []

        for match in re.finditer(pattern, text):
            value = match.group()
            try:
                numeric = float(value.rstrip("%"))
            except ValueError:
                numeric = None

            results.append(
                ExtractionResult(
                    type="percent",
                    value=value,
                    start=match.start(),
                    end=match.end(),
                    metadata={"numeric_value": numeric},
                )
            )
        return results

    def _extract_entities(self, text: str) -> list[ExtractionResult]:
        """Extract named entities using simple heuristics."""
        results = []

        # Capitalized sequences (potential names/organizations)
        # Matches sequences of capitalized words
        name_pattern = r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
        for match in re.finditer(name_pattern, text):
            value = match.group()
            # Skip common false positives
            if value.lower() in {"the", "this", "that", "these", "those"}:
                continue
            results.append(
                ExtractionResult(
                    type="entity",
                    value=value,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,
                    metadata={"entity_type": "possible_name"},
                )
            )

        # All-caps acronyms
        acronym_pattern = r"\b[A-Z]{2,6}\b"
        for match in re.finditer(acronym_pattern, text):
            value = match.group()
            # Skip common words
            if value in {"THE", "AND", "FOR", "NOT", "ARE", "WAS", "BUT", "HAS", "HAD"}:
                continue
            results.append(
                ExtractionResult(
                    type="entity",
                    value=value,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.6,
                    metadata={"entity_type": "acronym"},
                )
            )

        return results

    def _extract_json(self, text: str) -> list[ExtractionResult]:
        """Extract JSON objects/arrays from text."""
        results = []

        # Find potential JSON objects
        brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        for match in re.finditer(brace_pattern, text):
            value = match.group()
            try:
                parsed = json.loads(value)
                results.append(
                    ExtractionResult(
                        type="json",
                        value=value,
                        start=match.start(),
                        end=match.end(),
                        metadata={"parsed": parsed, "json_type": "object"},
                    )
                )
            except json.JSONDecodeError as e:
                logger.debug("Failed to parse JSON data: %s", e)

        # Find potential JSON arrays
        bracket_pattern = r"\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]"
        for match in re.finditer(bracket_pattern, text):
            value = match.group()
            try:
                parsed = json.loads(value)
                results.append(
                    ExtractionResult(
                        type="json",
                        value=value,
                        start=match.start(),
                        end=match.end(),
                        metadata={"parsed": parsed, "json_type": "array"},
                    )
                )
            except json.JSONDecodeError as e:
                logger.debug("Failed to parse JSON data: %s", e)

        return results

    def _extract_key_values(self, text: str) -> list[ExtractionResult]:
        """Extract key-value pairs from text."""
        results = []

        # Common key-value patterns
        patterns = [
            r"(\w+(?:\s+\w+)?)\s*[:=]\s*([^\n,;]+)",  # key: value or key = value
            r"(\w+)\s*-\s*([^\n,;]+)",  # key - value
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text):
                key = match.group(1).strip()
                value = match.group(2).strip()

                # Skip if key is too long or looks like a sentence
                if len(key) > 50 or " " in key and len(key.split()) > 3:
                    continue

                results.append(
                    ExtractionResult(
                        type="key_value",
                        value=f"{key}: {value}",
                        start=match.start(),
                        end=match.end(),
                        metadata={"key": key, "value": value},
                    )
                )

        return results

    def _extract_custom(self, text: str, pattern: str, name: str) -> list[ExtractionResult]:
        """Extract using custom regex pattern."""
        results = []
        for match in re.finditer(pattern, text):
            results.append(
                ExtractionResult(
                    type=f"custom_{name}",
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    metadata={"groups": match.groups()} if match.groups() else {},
                )
            )
        return results


# Skill instance for registration
SKILLS = [DataExtractionSkill()]
