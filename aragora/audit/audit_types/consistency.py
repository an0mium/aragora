"""
Cross-document consistency detection for document auditing.

Detects:
- Contradictory statements across documents
- Outdated references (stale dates, deprecated APIs)
- Version mismatches
- Calculation/formula errors
- Conflicting specifications
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from aragora.audit.document_auditor import (
    AuditFinding,
    AuditSession,
    AuditType,
    FindingSeverity,
)

logger = logging.getLogger(__name__)


@dataclass
class Statement:
    """A factual statement extracted from a document."""

    text: str
    document_id: str
    chunk_id: str
    category: str  # date, number, definition, specification, etc.
    key: str  # Normalized key for comparison
    value: str  # Extracted value
    location: str = ""
    context: str = ""  # Surrounding text for disambiguation


@dataclass
class Contradiction:
    """A detected contradiction between statements."""

    statement1: Statement
    statement2: Statement
    conflict_type: str
    severity: FindingSeverity
    explanation: str


class ConsistencyAuditor:
    """
    Audits documents for internal and cross-document consistency.

    Compares facts, dates, numbers, and specifications
    across the document collection to find contradictions.
    """

    # Patterns for extracting comparable facts
    DATE_PATTERNS = [
        (
            re.compile(
                r"(?i)(effective date|start date|end date|deadline|due date)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
            ),
            "date",
        ),
        (
            re.compile(
                r"(?i)(valid until|expires?|expiration)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
            ),
            "expiration",
        ),
        (re.compile(r"(?i)(?:(\w+)\s+)?(version|v\.?)\s*(\d+(?:\.\d+)*)"), "version"),
        (
            re.compile(r"(?i)(last updated|modified|revised)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"),
            "update_date",
        ),
    ]

    NUMBER_PATTERNS = [
        (re.compile(r"(?i)(price|cost|fee|amount)[:\s]*\$?([\d,]+(?:\.\d{2})?)"), "monetary"),
        (re.compile(r"(?i)(limit|maximum|minimum|threshold)[:\s]*(\d+(?:,\d{3})*)"), "limit"),
        (re.compile(r"(?i)(uptime|availability|SLA)[:\s]*([\d.]+)\s*%"), "percentage"),
        (
            re.compile(r"(?i)(timeout|delay|duration)[:\s]*(\d+)\s*(ms|seconds?|minutes?|hours?)"),
            "duration",
        ),
    ]

    DEFINITION_PATTERNS = [
        (
            re.compile(r'(?i)"([^"]+)"\s+(?:means?|refers? to|is defined as)\s+(.+?)(?:\.|$)'),
            "definition",
        ),
        (re.compile(r"(?i)([A-Z][a-zA-Z]+)\s+shall mean\s+(.+?)(?:\.|$)"), "definition"),
    ]

    def __init__(self):
        """Initialize consistency auditor."""
        self.statements: list[Statement] = []

    async def audit(
        self,
        chunks: list[dict[str, Any]],
        session: AuditSession,
    ) -> list[AuditFinding]:
        """
        Audit document chunks for consistency issues.

        Args:
            chunks: Document chunks to analyze
            session: Audit session context

        Returns:
            List of consistency findings
        """
        findings = []
        self.statements = []

        # Phase 1: Extract statements from all chunks
        for chunk in chunks:
            content = chunk.get("content", "")
            chunk_id = chunk.get("id", "")
            document_id = chunk.get("document_id", "")

            # Extract dates and versions
            for pattern, category in self.DATE_PATTERNS:
                for match in pattern.finditer(content):
                    if category == "version":
                        # Version pattern has 3 groups: (qualifier, version_word, number)
                        qualifier = match.group(1) or ""
                        value = match.group(3)

                        # Build semantic key: "python_version" instead of just "version"
                        if qualifier:
                            key = f"{qualifier.lower()}_version"
                        else:
                            key = "version"

                        # Capture context window (30 chars each side)
                        ctx_start = max(0, match.start() - 30)
                        ctx_end = min(len(content), match.end() + 30)
                        context = content[ctx_start:ctx_end].strip()
                    else:
                        # Standard 2-group patterns (date, expiration, update_date)
                        key = match.group(1).lower().strip()
                        value = match.group(2)
                        context = ""

                    self.statements.append(
                        Statement(
                            text=match.group(0),
                            document_id=document_id,
                            chunk_id=chunk_id,
                            category=category,
                            key=self._normalize_key(key),
                            value=value,
                            location=f"Position {match.start()}",
                            context=context,
                        )
                    )

            # Extract numbers
            for pattern, category in self.NUMBER_PATTERNS:
                for match in pattern.finditer(content):
                    key = match.group(1).lower().strip()
                    value = match.group(2)
                    self.statements.append(
                        Statement(
                            text=match.group(0),
                            document_id=document_id,
                            chunk_id=chunk_id,
                            category=category,
                            key=self._normalize_key(key),
                            value=value,
                            location=f"Position {match.start()}",
                        )
                    )

            # Extract definitions
            for pattern, category in self.DEFINITION_PATTERNS:
                for match in pattern.finditer(content):
                    key = match.group(1).lower().strip()
                    value = match.group(2).strip()
                    self.statements.append(
                        Statement(
                            text=match.group(0),
                            document_id=document_id,
                            chunk_id=chunk_id,
                            category=category,
                            key=self._normalize_key(key),
                            value=value,
                            location=f"Position {match.start()}",
                        )
                    )

        # Phase 2: Compare statements for contradictions
        contradictions = self._find_contradictions()

        # Phase 3: Convert contradictions to findings
        for contradiction in contradictions:
            finding = self._contradiction_to_finding(contradiction, session.id)
            findings.append(finding)

        # Phase 4: Check for outdated references
        outdated_findings = self._check_outdated_references(session.id)
        findings.extend(outdated_findings)

        # Phase 5: LLM-based consistency analysis
        full_content = "\n".join(c.get("content", "") for c in chunks)
        llm_findings = await self._llm_consistency_analysis(
            full_content,
            session.id,
            chunks,
            session.model,
        )
        findings.extend(llm_findings)

        return findings

    def _normalize_key(self, key: str) -> str:
        """Normalize a key for comparison."""
        # Remove common variations
        key = key.lower().strip()
        key = re.sub(r"\s+", "_", key)
        key = re.sub(r"[^\w_]", "", key)

        # Normalize common synonyms
        synonyms = {
            "cost": "price",
            "fee": "price",
            "amount": "price",
            "max": "maximum",
            "min": "minimum",
            "starts": "start_date",
            "begins": "start_date",
            "ends": "end_date",
            "expires": "expiration",
        }
        return synonyms.get(key, key)

    def _contexts_refer_to_same_subject(self, ctx1: str, ctx2: str) -> bool:
        """Check if two contexts refer to the same subject using word overlap.

        Used to disambiguate version mentions that have generic keys (no qualifier).
        Returns True if the contexts likely refer to the same thing.
        """
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "version",
            "v",
            "of",
            "to",
            "for",
        }

        # Extract significant words (3+ characters)
        words1 = set(
            w.lower() for w in re.findall(r"\b\w{3,}\b", ctx1) if w.lower() not in stop_words
        )
        words2 = set(
            w.lower() for w in re.findall(r"\b\w{3,}\b", ctx2) if w.lower() not in stop_words
        )

        if not words1 or not words2:
            return True  # Assume same subject if can't determine

        # Calculate Jaccard similarity
        overlap = len(words1 & words2)
        union = len(words1 | words2)
        similarity = overlap / union if union > 0 else 0

        # Require at least 30% word overlap to consider same subject
        return similarity >= 0.3

    def _find_contradictions(self) -> list[Contradiction]:
        """Find contradictions between statements."""
        contradictions = []

        # Group statements by key
        by_key: dict[str, list[Statement]] = {}
        for stmt in self.statements:
            if stmt.key not in by_key:
                by_key[stmt.key] = []
            by_key[stmt.key].append(stmt)

        # Compare statements with same key
        for key, stmts in by_key.items():
            if len(stmts) < 2:
                continue

            for i, stmt1 in enumerate(stmts):
                for stmt2 in stmts[i + 1 :]:
                    # Skip same document unless explicitly checking internal consistency
                    if stmt1.document_id == stmt2.document_id:
                        continue

                    # Compare values
                    if self._values_contradict(stmt1, stmt2):
                        severity = self._determine_severity(stmt1.category)
                        contradictions.append(
                            Contradiction(
                                statement1=stmt1,
                                statement2=stmt2,
                                conflict_type=f"{stmt1.category}_mismatch",
                                severity=severity,
                                explanation=f"Conflicting values for '{key}': '{stmt1.value}' vs '{stmt2.value}'",
                            )
                        )

        return contradictions

    def _values_contradict(self, stmt1: Statement, stmt2: Statement) -> bool:
        """Check if two statement values contradict."""
        v1, v2 = stmt1.value.strip(), stmt2.value.strip()

        if v1 == v2:
            return False

        # For dates, parse and compare
        if stmt1.category in ("date", "expiration", "update_date"):
            return self._dates_differ(v1, v2)

        # For versions, check if different (with context disambiguation)
        if stmt1.category == "version":
            # If keys are generic (no qualifier), use context to verify same subject
            if stmt1.key == "version" and stmt1.context and stmt2.context:
                if not self._contexts_refer_to_same_subject(stmt1.context, stmt2.context):
                    return False  # Different subjects - not a contradiction
            return v1 != v2

        # For numbers, check if significantly different
        if stmt1.category in ("monetary", "limit", "percentage", "duration"):
            return self._numbers_differ(v1, v2)

        # For definitions, check semantic similarity
        if stmt1.category == "definition":
            return self._definitions_differ(v1, v2)

        return v1 != v2

    def _dates_differ(self, d1: str, d2: str) -> bool:
        """Check if two date strings differ."""
        try:
            # Try multiple formats
            for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y", "%m/%d/%y", "%d/%m/%y"):
                try:
                    dt1 = datetime.strptime(d1, fmt)
                    dt2 = datetime.strptime(d2, fmt)
                    return dt1 != dt2
                except ValueError:
                    continue
        except Exception:  # noqa: BLE001 - Date comparison fallback
            pass
        return d1 != d2

    def _numbers_differ(self, n1: str, n2: str) -> bool:
        """Check if two numbers differ significantly."""
        try:
            # Remove commas and parse
            num1 = float(n1.replace(",", ""))
            num2 = float(n2.replace(",", ""))

            if num1 == num2:
                return False

            # Allow 1% tolerance for floating point
            if num1 != 0:
                diff_pct = abs(num1 - num2) / abs(num1)
                return diff_pct > 0.01

            return True
        except ValueError:
            return n1 != n2

    def _definitions_differ(self, def1: str, def2: str) -> bool:
        """Check if two definitions are semantically different."""
        # Simple word overlap check
        words1 = set(def1.lower().split())
        words2 = set(def2.lower().split())

        if not words1 or not words2:
            return def1 != def2

        overlap = len(words1 & words2)
        total = len(words1 | words2)

        # If less than 50% overlap, consider different
        return (overlap / total) < 0.5

    def _determine_severity(self, category: str) -> FindingSeverity:
        """Determine severity based on category."""
        high_severity = {"monetary", "expiration", "version", "limit"}
        medium_severity = {"date", "percentage", "definition"}

        if category in high_severity:
            return FindingSeverity.HIGH
        elif category in medium_severity:
            return FindingSeverity.MEDIUM
        return FindingSeverity.LOW

    def _contradiction_to_finding(
        self,
        contradiction: Contradiction,
        session_id: str,
    ) -> AuditFinding:
        """Convert a contradiction to an audit finding."""
        return AuditFinding(
            session_id=session_id,
            document_id=contradiction.statement1.document_id,
            chunk_id=contradiction.statement1.chunk_id,
            audit_type=AuditType.CONSISTENCY,
            category=contradiction.conflict_type,
            severity=contradiction.severity,
            confidence=0.9,
            title=f"Contradictory {contradiction.statement1.category.replace('_', ' ').title()}",
            description=contradiction.explanation,
            evidence_text=f"Document 1: {contradiction.statement1.text}\nDocument 2: {contradiction.statement2.text}",
            evidence_location=f"{contradiction.statement1.document_id} vs {contradiction.statement2.document_id}",
            recommendation="Review and reconcile the conflicting values across documents",
            affected_scope="collection",
            found_by="consistency_checker",
        )

    def _check_outdated_references(self, session_id: str) -> list[AuditFinding]:
        """Check for outdated references."""
        findings = []
        current_year = datetime.now().year

        for stmt in self.statements:
            if stmt.category in ("date", "expiration", "update_date"):
                # Check for very old dates
                try:
                    for fmt in (
                        "%m/%d/%Y",
                        "%d/%m/%Y",
                        "%Y-%m-%d",
                        "%m-%d-%Y",
                        "%m/%d/%y",
                        "%d/%m/%y",
                    ):
                        try:
                            dt = datetime.strptime(stmt.value, fmt)
                            years_old = current_year - dt.year

                            if years_old > 2 and stmt.category == "update_date":
                                findings.append(
                                    AuditFinding(
                                        session_id=session_id,
                                        document_id=stmt.document_id,
                                        chunk_id=stmt.chunk_id,
                                        audit_type=AuditType.CONSISTENCY,
                                        category="outdated_reference",
                                        severity=FindingSeverity.MEDIUM,
                                        confidence=0.85,
                                        title="Potentially Outdated Document",
                                        description=f"Document was last updated {years_old} years ago",
                                        evidence_text=stmt.text,
                                        recommendation="Review and update document content",
                                        found_by="outdated_checker",
                                    )
                                )

                            if stmt.category == "expiration" and dt < datetime.now():
                                findings.append(
                                    AuditFinding(
                                        session_id=session_id,
                                        document_id=stmt.document_id,
                                        chunk_id=stmt.chunk_id,
                                        audit_type=AuditType.CONSISTENCY,
                                        category="expired_reference",
                                        severity=FindingSeverity.HIGH,
                                        confidence=0.95,
                                        title="Expired Date Reference",
                                        description=f"Referenced date has passed: {stmt.value}",
                                        evidence_text=stmt.text,
                                        recommendation="Update or remove expired date reference",
                                        found_by="expiration_checker",
                                    )
                                )
                            break
                        except ValueError:
                            continue
                except Exception:  # noqa: BLE001 - Date extraction fallback
                    pass

        return findings

    async def _llm_consistency_analysis(
        self,
        content: str,
        session_id: str,
        chunks: list[dict[str, Any]],
        model: str,
    ) -> list[AuditFinding]:
        """Use LLM for deeper consistency analysis."""
        findings = []

        try:
            from aragora.agents.api_agents.gemini import GeminiAgent

            agent = GeminiAgent(name="consistency_analyst", model=model)

            prompt = f"""Analyze these documents for consistency issues:

{content[:20000]}

Look for:
1. Contradictory statements (same topic, different claims)
2. Outdated information (old dates, deprecated technologies, obsolete references)
3. Version mismatches (conflicting version numbers)
4. Calculation errors (numbers that don't add up)
5. Logical inconsistencies (statements that cannot both be true)

Only report issues with HIGH confidence. Format as JSON array:
[{{"title": "...", "severity": "high|medium|low", "evidence": "...", "explanation": "..."}}]

If no issues found, respond with empty array: []"""

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
                            audit_type=AuditType.CONSISTENCY,
                            category="llm_detected",
                            severity=FindingSeverity(item.get("severity", "medium").lower()),
                            confidence=0.75,
                            title=item.get("title", "Consistency Issue"),
                            description=item.get("explanation", ""),
                            evidence_text=item.get("evidence", ""),
                            recommendation="Review and reconcile the inconsistency",
                            found_by="consistency_analyst",
                        )
                        findings.append(finding)
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            logger.debug(f"LLM consistency analysis skipped: {e}")

        return findings


__all__ = ["ConsistencyAuditor"]
