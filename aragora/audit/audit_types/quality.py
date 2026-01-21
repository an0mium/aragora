"""
Documentation quality analysis for document auditing.

Detects:
- Ambiguous requirements
- Missing documentation
- Incomplete specifications
- Style/formatting inconsistencies
- Readability issues
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from aragora.audit.document_auditor import (
    AuditFinding,
    AuditSession,
    AuditType,
    FindingSeverity,
)

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for a document."""

    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    readability_score: float = 0.0  # 0-100, higher is easier
    passive_voice_count: int = 0
    hedging_words_count: int = 0
    undefined_terms_count: int = 0
    todo_count: int = 0
    placeholder_count: int = 0


class QualityAuditor:
    """
    Audits documents for quality and completeness issues.

    Analyzes documentation quality, identifies gaps, and
    flags potential issues with clarity and completeness.
    """

    # Hedging words that indicate uncertainty
    HEDGING_WORDS = [
        "may",
        "might",
        "could",
        "possibly",
        "perhaps",
        "probably",
        "generally",
        "typically",
        "usually",
        "often",
        "sometimes",
        "should",
        "approximately",
        "roughly",
        "around",
        "about",
        "likely",
        "unlikely",
        "seems",
        "appears",
        "suggests",
    ]

    # Patterns indicating incomplete content
    INCOMPLETE_PATTERNS = [
        (
            re.compile(r"(?i)\b(TODO|FIXME|XXX|HACK|BUG)\b[:\s]*(.*)"),
            "todo_marker",
            FindingSeverity.MEDIUM,
        ),
        (re.compile(r"(?i)\b(TBD|TBA|TBC)\b"), "to_be_determined", FindingSeverity.MEDIUM),
        (re.compile(r"\[[\s]*\]"), "empty_brackets", FindingSeverity.LOW),
        (
            re.compile(r"(?i)(insert|add|fill in|placeholder)[\s]+here"),
            "placeholder",
            FindingSeverity.MEDIUM,
        ),
        (re.compile(r"(?i)lorem ipsum"), "lorem_ipsum", FindingSeverity.LOW),
        (re.compile(r"\.\.\.$|…$"), "trailing_ellipsis", FindingSeverity.LOW),
        (
            re.compile(r"(?i)(coming soon|under construction|work in progress)"),
            "incomplete_section",
            FindingSeverity.MEDIUM,
        ),
        (re.compile(r"\?\?\?|\*\*\*"), "question_marks", FindingSeverity.MEDIUM),
    ]

    # Ambiguous language patterns
    AMBIGUITY_PATTERNS = [
        (
            re.compile(r"(?i)\b(etc\.?|and so on|and more)\b"),
            "vague_continuation",
            FindingSeverity.LOW,
        ),
        (
            re.compile(r"(?i)\b(some|several|many|few|various|numerous)\s+\w+"),
            "vague_quantity",
            FindingSeverity.LOW,
        ),
        (
            re.compile(r"(?i)\b(appropriate|adequate|sufficient|reasonable|suitable)\b"),
            "vague_qualifier",
            FindingSeverity.LOW,
        ),
        (
            re.compile(r"(?i)\b(as needed|when necessary|if required|as appropriate)\b"),
            "vague_condition",
            FindingSeverity.LOW,
        ),
        (
            re.compile(r"(?i)\b(similar|comparable|equivalent|related)\b"),
            "vague_comparison",
            FindingSeverity.LOW,
        ),
        (
            re.compile(r"(?i)\b(soon|shortly|in the near future|later)\b"),
            "vague_timeline",
            FindingSeverity.MEDIUM,
        ),
    ]

    # Passive voice patterns
    PASSIVE_PATTERNS = [
        re.compile(
            r"(?i)\b(is|are|was|were|be|been|being)\s+(being\s+)?(\w+ed|written|done|made|taken|given)\b"
        ),
    ]

    def __init__(self):
        """Initialize quality auditor."""
        pass

    async def audit(
        self,
        chunks: list[dict[str, Any]],
        session: AuditSession,
    ) -> list[AuditFinding]:
        """
        Audit document chunks for quality issues.

        Args:
            chunks: Document chunks to analyze
            session: Audit session context

        Returns:
            List of quality findings
        """
        findings = []

        full_content = "\n".join(c.get("content", "") for c in chunks)

        # Calculate quality metrics
        metrics = self._calculate_metrics(full_content)

        # Check for incomplete content
        incomplete_findings = self._check_incomplete_content(chunks, session.id)
        findings.extend(incomplete_findings)

        # Check for ambiguous language
        ambiguity_findings = self._check_ambiguity(chunks, session.id)
        findings.extend(ambiguity_findings)

        # Check readability
        readability_findings = self._check_readability(metrics, session.id, chunks)
        findings.extend(readability_findings)

        # Check for missing sections
        structure_findings = self._check_document_structure(full_content, session.id, chunks)
        findings.extend(structure_findings)

        # LLM-based quality analysis
        llm_findings = await self._llm_quality_analysis(
            full_content, session.id, chunks, session.model
        )
        findings.extend(llm_findings)

        return findings

    def _calculate_metrics(self, content: str) -> QualityMetrics:
        """Calculate quality metrics for content."""
        # Word and sentence counts
        words = content.split()
        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if s.strip()]

        word_count = len(words)
        sentence_count = len(sentences) if sentences else 1

        # Average sentence length
        avg_sentence_length = word_count / sentence_count

        # Readability (Flesch Reading Ease approximation)
        # Higher = easier to read (0-100 scale)
        syllable_count = sum(self._count_syllables(w) for w in words)
        if word_count > 0 and sentence_count > 0:
            readability = (
                206.835
                - (1.015 * (word_count / sentence_count))
                - (84.6 * (syllable_count / word_count))
            )
            readability = max(0, min(100, readability))
        else:
            readability = 50.0

        # Passive voice
        passive_count = sum(len(pattern.findall(content)) for pattern in self.PASSIVE_PATTERNS)

        # Hedging words
        hedging_count = sum(
            len(re.findall(rf"\b{word}\b", content, re.IGNORECASE)) for word in self.HEDGING_WORDS
        )

        # TODO markers
        todo_count = len(re.findall(r"(?i)\b(TODO|FIXME|XXX)\b", content))

        # Placeholders
        placeholder_count = len(re.findall(r"\[[\s]*\]|\bTBD\b|\bTBA\b", content, re.IGNORECASE))

        return QualityMetrics(
            word_count=word_count,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            readability_score=readability,
            passive_voice_count=passive_count,
            hedging_words_count=hedging_count,
            todo_count=todo_count,
            placeholder_count=placeholder_count,
        )

    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count for a word."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        prev_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # Adjust for silent e
        if word.endswith("e"):
            count -= 1

        return max(1, count)

    def _check_incomplete_content(
        self,
        chunks: list[dict[str, Any]],
        session_id: str,
    ) -> list[AuditFinding]:
        """Check for incomplete or placeholder content."""
        findings = []

        for chunk in chunks:
            content = chunk.get("content", "")
            chunk_id = chunk.get("id", "")
            document_id = chunk.get("document_id", "")

            for pattern, category, severity in self.INCOMPLETE_PATTERNS:
                matches = pattern.finditer(content)
                for match in matches:
                    # Get context
                    start = max(0, match.start() - 30)
                    end = min(len(content), match.end() + 30)
                    context = content[start:end]

                    finding = AuditFinding(
                        session_id=session_id,
                        document_id=document_id,
                        chunk_id=chunk_id,
                        audit_type=AuditType.QUALITY,
                        category=category,
                        severity=severity,
                        confidence=0.95,
                        title=f"Incomplete Content: {category.replace('_', ' ').title()}",
                        description=f"Found {category.replace('_', ' ')} marker indicating incomplete content",
                        evidence_text=context.strip(),
                        evidence_location=f"Position {match.start()}",
                        recommendation="Complete the marked section or remove placeholder",
                        found_by="completeness_checker",
                    )
                    findings.append(finding)

        return findings

    def _check_ambiguity(
        self,
        chunks: list[dict[str, Any]],
        session_id: str,
    ) -> list[AuditFinding]:
        """Check for ambiguous language."""
        findings = []

        # Track ambiguity counts per document
        doc_ambiguity: dict[str, int] = {}

        for chunk in chunks:
            content = chunk.get("content", "")
            document_id = chunk.get("document_id", "")

            for pattern, category, severity in self.AMBIGUITY_PATTERNS:
                matches = pattern.findall(content)
                if matches:
                    if document_id not in doc_ambiguity:
                        doc_ambiguity[document_id] = 0
                    doc_ambiguity[document_id] += len(matches)

        # Report if high ambiguity count
        for doc_id, count in doc_ambiguity.items():
            if count > 10:  # Threshold for reporting
                finding = AuditFinding(
                    session_id=session_id,
                    document_id=doc_id,
                    audit_type=AuditType.QUALITY,
                    category="high_ambiguity",
                    severity=FindingSeverity.MEDIUM,
                    confidence=0.8,
                    title="High Ambiguity Level",
                    description=f"Document contains {count} instances of vague or ambiguous language",
                    evidence_text="Examples include: etc., some, several, as needed, soon",
                    recommendation="Replace vague terms with specific, measurable language",
                    found_by="ambiguity_checker",
                )
                findings.append(finding)

        return findings

    def _check_readability(
        self,
        metrics: QualityMetrics,
        session_id: str,
        chunks: list[dict[str, Any]],
    ) -> list[AuditFinding]:
        """Check readability metrics."""
        findings = []
        document_id = chunks[0].get("document_id", "") if chunks else ""

        # Very difficult to read
        if metrics.readability_score < 30:
            finding = AuditFinding(
                session_id=session_id,
                document_id=document_id,
                audit_type=AuditType.QUALITY,
                category="poor_readability",
                severity=FindingSeverity.MEDIUM,
                confidence=0.85,
                title="Poor Readability",
                description=f"Document has low readability score ({metrics.readability_score:.1f}/100). Average sentence length: {metrics.avg_sentence_length:.1f} words.",
                evidence_text="Based on Flesch Reading Ease analysis",
                recommendation="Shorten sentences and use simpler vocabulary",
                found_by="readability_checker",
            )
            findings.append(finding)

        # Too many passive voice constructions
        if metrics.passive_voice_count > 20:
            finding = AuditFinding(
                session_id=session_id,
                document_id=document_id,
                audit_type=AuditType.QUALITY,
                category="excessive_passive_voice",
                severity=FindingSeverity.LOW,
                confidence=0.75,
                title="Excessive Passive Voice",
                description=f"Document contains {metrics.passive_voice_count} passive voice constructions",
                evidence_text="Passive voice can make requirements unclear",
                recommendation="Use active voice for clearer, more direct statements",
                found_by="style_checker",
            )
            findings.append(finding)

        # Too many hedging words
        if metrics.hedging_words_count > 15:
            finding = AuditFinding(
                session_id=session_id,
                document_id=document_id,
                audit_type=AuditType.QUALITY,
                category="excessive_hedging",
                severity=FindingSeverity.LOW,
                confidence=0.75,
                title="Excessive Hedging Language",
                description=f"Document contains {metrics.hedging_words_count} hedging words (may, might, probably, etc.)",
                evidence_text="Hedging language creates uncertainty in requirements",
                recommendation="Use definitive language for clear requirements",
                found_by="style_checker",
            )
            findings.append(finding)

        return findings

    def _check_document_structure(
        self,
        content: str,
        session_id: str,
        chunks: list[dict[str, Any]],
    ) -> list[AuditFinding]:
        """Check for missing common documentation sections."""
        findings = []
        document_id = chunks[0].get("document_id", "") if chunks else ""
        content_lower = content.lower()

        # Expected sections for technical documentation
        expected_sections = [
            ("introduction", ["introduction", "overview", "about", "summary"]),
            ("requirements", ["requirements", "prerequisites", "dependencies"]),
            ("installation", ["installation", "setup", "getting started"]),
            ("configuration", ["configuration", "settings", "options"]),
            ("usage", ["usage", "how to use", "examples", "tutorial"]),
            ("troubleshooting", ["troubleshooting", "faq", "common issues"]),
        ]

        missing_sections = []
        for section_name, keywords in expected_sections:
            if not any(kw in content_lower for kw in keywords):
                missing_sections.append(section_name)

        # Only report if document is substantial and missing multiple sections
        if len(content) > 1000 and len(missing_sections) >= 3:
            finding = AuditFinding(
                session_id=session_id,
                document_id=document_id,
                audit_type=AuditType.QUALITY,
                category="missing_sections",
                severity=FindingSeverity.LOW,
                confidence=0.7,
                title="Potentially Missing Documentation Sections",
                description=f"Document may be missing: {', '.join(missing_sections)}",
                evidence_text="Based on common documentation structure analysis",
                recommendation="Consider adding missing sections for completeness",
                found_by="structure_checker",
            )
            findings.append(finding)

        return findings

    async def _llm_quality_analysis(
        self,
        content: str,
        session_id: str,
        chunks: list[dict[str, Any]],
        model: str,
    ) -> list[AuditFinding]:
        """Use LLM for deeper quality analysis."""
        findings = []

        try:
            from aragora.agents.api_agents.anthropic import AnthropicAgent  # type: ignore[attr-defined]

            agent = AnthropicAgent(name="quality_analyst", model="claude-3.5-sonnet")

            prompt = f"""Analyze this document for quality issues:

{content[:15000]}

Evaluate:
1. Clarity - Are statements clear and unambiguous?
2. Completeness - Are there gaps or missing information?
3. Consistency - Is terminology used consistently?
4. Specificity - Are requirements specific and measurable?
5. Organization - Is the structure logical?

Only report significant issues with HIGH confidence. Format as JSON array:
[{{"title": "...", "severity": "medium|low", "evidence": "...", "recommendation": "..."}}]

If no significant issues found, respond with empty array: []"""

            response = await agent.generate(prompt)

            # Parse response
            import json
            import re

            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                try:
                    items = json.loads(json_match.group())
                    for item in items:
                        finding = AuditFinding(
                            session_id=session_id,
                            document_id=chunks[0].get("document_id", "") if chunks else "",
                            audit_type=AuditType.QUALITY,
                            category="llm_detected",
                            severity=FindingSeverity(item.get("severity", "medium").lower()),
                            confidence=0.7,
                            title=item.get("title", "Quality Issue"),
                            description=item.get("description", ""),
                            evidence_text=item.get("evidence", ""),
                            recommendation=item.get("recommendation", ""),
                            found_by="quality_analyst",
                        )
                        findings.append(finding)
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            logger.debug(f"LLM quality analysis skipped: {e}")

        return findings


__all__ = ["QualityAuditor"]
