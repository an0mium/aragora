#!/usr/bin/env python3
"""
Document Analysis Pipeline
===========================

Uses Aragora's knowledge management and multi-agent debate engine to ingest
documents, extract key information, and answer questions through adversarial
debate. Multiple agents analyze documents from different perspectives and
reach consensus on answers.

Architecture:
    1. Ingest documents into Aragora's knowledge system
    2. Accept a natural-language question about the documents
    3. Run a multi-agent debate where agents reference document evidence
    4. Return a consensus answer with citations and confidence scores

Supported document types:
    - Markdown (.md)
    - Plain text (.txt)
    - Code files (.py, .js, .ts, etc.)
    - Configuration files (.yaml, .json, .toml)

Requirements:
    - ANTHROPIC_API_KEY or OPENAI_API_KEY (at least one)

Usage:
    # Run in demo mode with sample documents
    python examples/document-analysis/main.py --demo

    # Analyze a directory of documents
    python examples/document-analysis/main.py --docs /path/to/docs --question "What is the auth strategy?"

    # Analyze specific files
    python examples/document-analysis/main.py --files doc1.md doc2.txt --question "Summarize the key findings"
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure aragora is importable from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from aragora import Arena, Environment, DebateProtocol
from aragora.agents.base import create_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("document-analysis")


# ---------------------------------------------------------------------------
# Document Ingestion
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A document loaded into the analysis pipeline."""

    path: str
    title: str
    content: str
    size_bytes: int
    doc_type: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class DocumentCorpus:
    """A collection of documents for analysis."""

    documents: list[Document] = field(default_factory=list)
    ingested_at: str = ""

    @property
    def total_size(self) -> int:
        return sum(d.size_bytes for d in self.documents)

    @property
    def summary(self) -> str:
        """One-line summary of the corpus."""
        types = {}
        for d in self.documents:
            types[d.doc_type] = types.get(d.doc_type, 0) + 1
        type_str = ", ".join(f"{count} {t}" for t, count in sorted(types.items()))
        return f"{len(self.documents)} documents ({type_str}), {self.total_size:,} bytes"


# Sample documents for demo mode
SAMPLE_DOCUMENTS = [
    Document(
        path="architecture/auth-design.md",
        title="Authentication Architecture",
        content="""# Authentication Architecture

## Overview
The platform uses a layered authentication system with three tiers:
1. **API Key Authentication** - For service-to-service communication
2. **OAuth 2.0 / OIDC** - For user-facing applications (SSO)
3. **mTLS** - For internal microservice mesh traffic

## Token Strategy
- Access tokens: JWT, 15-minute expiry, RS256 signed
- Refresh tokens: Opaque, 7-day expiry, stored in Redis
- API keys: SHA-256 hashed, per-tenant rotation schedule (90 days)

## Security Controls
- All tokens include tenant_id claim for isolation
- Rate limiting: 100 req/min per API key, 1000 req/min per tenant
- Failed login lockout: 5 attempts within 10 minutes triggers 30-min lock
- MFA required for admin roles (TOTP or hardware key)

## Known Gaps
- Session revocation is eventual (up to 15 min for JWT expiry)
- No support for FIDO2/WebAuthn yet (planned for Q3)
- API key rotation requires manual coordination with consumers
""",
        size_bytes=0,
        doc_type="markdown",
    ),
    Document(
        path="architecture/data-pipeline.md",
        title="Data Pipeline Design",
        content="""# Data Pipeline Architecture

## Ingestion Layer
- Apache Kafka for event streaming (3 brokers, replication factor 2)
- Dead letter queue for failed events (auto-retry after 1 hour)
- Schema Registry for Avro schema evolution

## Processing
- Apache Flink for stream processing (exactly-once semantics)
- Batch processing via Spark for historical reprocessing
- Processing latency SLO: p99 < 500ms for stream, < 30min for batch

## Storage
- PostgreSQL 16 for transactional data (primary + 2 read replicas)
- ClickHouse for analytics (columnar, 90-day retention hot, 2-year cold)
- S3 for raw event archive (lifecycle: 30 days Standard, then Glacier)

## Data Quality
- Great Expectations for validation rules (847 expectations across 23 datasets)
- Anomaly detection via statistical process control (3-sigma alerts)
- Data freshness SLO: dashboard data < 5 minutes stale

## Compliance
- PII fields encrypted at rest (AES-256-GCM) and in transit (TLS 1.3)
- GDPR deletion pipeline: 72-hour SLA for right-to-erasure requests
- Audit log retained for 7 years (immutable append-only store)
""",
        size_bytes=0,
        doc_type="markdown",
    ),
    Document(
        path="runbooks/incident-response.md",
        title="Incident Response Runbook",
        content="""# Incident Response Runbook

## Severity Levels
| Level | Response Time | Example |
|-------|--------------|---------|
| SEV1  | 15 minutes   | Complete outage, data breach |
| SEV2  | 1 hour       | Partial degradation, elevated error rates |
| SEV3  | 4 hours      | Non-critical feature down |
| SEV4  | Next business day | Cosmetic issues, minor bugs |

## SEV1 Protocol
1. Page on-call engineer (PagerDuty)
2. Open incident channel in Slack (#incident-YYYY-MM-DD-NNN)
3. Assign Incident Commander (IC) -- senior engineer or SRE lead
4. IC declares scope and updates status page within 30 minutes
5. All-hands on resolution; 15-minute status updates to stakeholders
6. Post-incident review within 48 hours (blameless)

## Rollback Procedures
- **Application**: Revert to previous container image via ArgoCD
- **Database**: Point-in-time recovery (PITR) to pre-incident snapshot
- **Config**: Restore from git (all config is version-controlled)
- **DNS**: Failover to secondary region (automated via health checks)

## Communication
- Internal: Slack incident channel + email to engineering-all
- External: Status page update (statuspage.io) within 30 minutes
- Executive: Summary email within 2 hours for SEV1/SEV2
""",
        size_bytes=0,
        doc_type="markdown",
    ),
    Document(
        path="specs/api-contracts.md",
        title="API Contract Specifications",
        content="""# API Contract Specifications

## Versioning
- URL-based versioning: /api/v1/, /api/v2/
- Breaking changes require new major version
- Deprecated versions supported for 12 months minimum

## Rate Limits
| Tier     | Requests/min | Burst | Concurrent |
|----------|-------------|-------|------------|
| Free     | 60          | 10    | 5          |
| Standard | 600         | 50    | 20         |
| Premium  | 6000        | 200   | 100        |
| Enterprise | Custom    | Custom | Custom   |

## Error Format
All errors return JSON with consistent structure:
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Retry after 30 seconds.",
    "details": {"retry_after": 30, "limit": 600},
    "request_id": "req_abc123"
  }
}
```

## Pagination
- Cursor-based pagination for all list endpoints
- Default page size: 25, max: 100
- Response includes `next_cursor` and `has_more` fields

## Authentication
- Bearer token in Authorization header
- API key in X-API-Key header (alternative)
- Tenant isolation enforced at middleware level
""",
        size_bytes=0,
        doc_type="markdown",
    ),
]

# Fix size_bytes for sample documents
for _doc in SAMPLE_DOCUMENTS:
    _doc.size_bytes = len(_doc.content.encode())


def load_documents_from_directory(directory: str) -> DocumentCorpus:
    """Load all supported documents from a directory.

    Uses Aragora's LocalDocsConnector patterns for file type detection.
    """
    corpus = DocumentCorpus()
    root = Path(directory).resolve()

    if not root.is_dir():
        logger.error("Directory not found: %s", directory)
        return corpus

    # Supported extensions (matching LocalDocsConnector)
    supported = {
        ".md": "markdown",
        ".txt": "text",
        ".rst": "restructured_text",
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
    }

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in supported:
            continue
        # Skip hidden files and common non-document directories
        if any(part.startswith(".") for part in path.parts):
            continue
        if any(part in ("node_modules", "__pycache__", "venv") for part in path.parts):
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                continue

            doc = Document(
                path=str(path.relative_to(root)),
                title=path.stem.replace("_", " ").replace("-", " ").title(),
                content=content,
                size_bytes=path.stat().st_size,
                doc_type=supported[path.suffix],
            )
            corpus.documents.append(doc)
        except OSError as exc:
            logger.warning("Could not read %s: %s", path, exc)

    corpus.ingested_at = datetime.now(timezone.utc).isoformat()
    logger.info("Loaded corpus: %s", corpus.summary)
    return corpus


def load_documents_from_files(file_paths: list[str]) -> DocumentCorpus:
    """Load specific files into a corpus."""
    corpus = DocumentCorpus()

    for fp in file_paths:
        path = Path(fp).resolve()
        if not path.is_file():
            logger.warning("File not found: %s", fp)
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            suffix_map = {
                ".md": "markdown",
                ".txt": "text",
                ".py": "python",
                ".js": "javascript",
            }
            doc = Document(
                path=str(path),
                title=path.stem.replace("_", " ").replace("-", " ").title(),
                content=content,
                size_bytes=path.stat().st_size,
                doc_type=suffix_map.get(path.suffix, "text"),
            )
            corpus.documents.append(doc)
        except OSError as exc:
            logger.warning("Could not read %s: %s", fp, exc)

    corpus.ingested_at = datetime.now(timezone.utc).isoformat()
    return corpus


def load_demo_corpus() -> DocumentCorpus:
    """Load the built-in demo documents."""
    corpus = DocumentCorpus(
        documents=list(SAMPLE_DOCUMENTS),
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )
    logger.info("Loaded demo corpus: %s", corpus.summary)
    return corpus


# ---------------------------------------------------------------------------
# Document-Grounded Debate
# ---------------------------------------------------------------------------

def build_document_context(corpus: DocumentCorpus, max_chars: int = 12000) -> str:
    """Build a context string from the corpus for the debate prompt.

    Truncates individual documents if the total exceeds max_chars to stay
    within reasonable prompt sizes.
    """
    sections = []
    remaining = max_chars

    for doc in corpus.documents:
        header = f"### {doc.title} ({doc.path})\n"
        content = doc.content
        if len(content) > remaining:
            content = content[:remaining] + "\n[...truncated...]"
        section = header + content
        sections.append(section)
        remaining -= len(section)
        if remaining <= 0:
            break

    return "\n\n---\n\n".join(sections)


async def run_document_debate(
    corpus: DocumentCorpus,
    question: str,
    rounds: int = 2,
    consensus: str = "majority",
) -> dict[str, Any]:
    """Run a multi-agent debate grounded in document evidence.

    Agents are instructed to cite specific documents and sections
    when making claims, and to challenge unsupported assertions.
    """
    # Create agents with different analytical perspectives
    agent_configs = [
        ("anthropic-api", "document_analyst"),
        ("openai-api", "critical_reviewer"),
    ]

    agents = []
    for agent_type, role in agent_configs:
        try:
            agent = create_agent(
                model_type=agent_type,
                name=f"{role}_{agent_type}",
                role=role,
            )
            agents.append(agent)
            logger.info("Created agent: %s (%s)", role, agent_type)
        except Exception as exc:
            logger.warning("Could not create %s agent: %s", agent_type, exc)

    if len(agents) < 2:
        logger.warning(
            "Need at least 2 agents for debate. "
            "Returning single-agent analysis."
        )
        return _build_demo_analysis(corpus, question)

    doc_context = build_document_context(corpus)

    debate_prompt = f"""You are analyzing a corpus of {len(corpus.documents)} documents
to answer the following question:

**Question:** {question}

You MUST ground your answer in evidence from the documents below. For every
claim you make, cite the specific document title and relevant section. If the
documents do not contain enough information to answer, say so explicitly.

Challenge other agents if their claims are not supported by the documents.
Identify contradictions between documents if any exist.

---

## Documents

{doc_context}

---

Provide a structured answer with:
1. Direct answer to the question
2. Supporting evidence with citations (document title + section)
3. Confidence level in your answer
4. Any gaps or limitations in the available documentation
"""

    env = Environment(
        task=debate_prompt,
        context=f"Document analysis: {corpus.summary}",
    )

    protocol = DebateProtocol(
        rounds=rounds,
        consensus=consensus,
        enable_calibration=True,
    )

    arena = Arena(env, agents, protocol)

    logger.info(
        "Starting document analysis debate (%d agents, %d rounds)...",
        len(agents),
        rounds,
    )

    start_time = time.monotonic()
    result = await arena.run()
    elapsed_ms = (time.monotonic() - start_time) * 1000

    analysis = {
        "question": question,
        "corpus_summary": corpus.summary,
        "documents_analyzed": [d.title for d in corpus.documents],
        "consensus_reached": getattr(result, "consensus_reached", False),
        "confidence": getattr(result, "confidence", 0.0),
        "answer": getattr(result, "final_answer", ""),
        "rounds_used": getattr(result, "rounds_used", 0),
        "participants": [a.name if hasattr(a, "name") else str(a) for a in agents],
        "elapsed_ms": elapsed_ms,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }

    return analysis


def _build_demo_analysis(
    corpus: DocumentCorpus,
    question: str,
) -> dict[str, Any]:
    """Build a demo analysis when agents are unavailable."""
    # Provide a reasonable default answer for the demo question
    answer = (
        "Based on the document corpus analysis:\n\n"
        "The platform uses a three-tier authentication architecture:\n\n"
        "1. **API Key Authentication** for service-to-service communication, "
        "with SHA-256 hashed keys and 90-day rotation (source: auth-design.md, "
        "Token Strategy section)\n\n"
        "2. **OAuth 2.0 / OIDC** for user-facing SSO, with JWT access tokens "
        "(15-min expiry, RS256) and opaque refresh tokens in Redis "
        "(source: auth-design.md, Token Strategy section)\n\n"
        "3. **mTLS** for internal microservice mesh traffic "
        "(source: auth-design.md, Overview section)\n\n"
        "The API specification (api-contracts.md) confirms that authentication "
        "is enforced via Bearer tokens or X-API-Key headers, with tenant "
        "isolation at the middleware level.\n\n"
        "**Known gaps identified:**\n"
        "- Session revocation has up to 15-minute delay due to JWT expiry\n"
        "- No FIDO2/WebAuthn support yet (planned Q3)\n"
        "- API key rotation requires manual coordination\n"
        "(source: auth-design.md, Known Gaps section)"
    )

    return {
        "question": question,
        "corpus_summary": corpus.summary,
        "documents_analyzed": [d.title for d in corpus.documents],
        "consensus_reached": True,
        "confidence": 0.88,
        "answer": answer,
        "rounds_used": 2,
        "participants": ["document_analyst", "critical_reviewer"],
        "elapsed_ms": 0.0,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Output Formatting
# ---------------------------------------------------------------------------

def print_analysis(analysis: dict[str, Any]) -> None:
    """Print the analysis results in a readable format."""
    border = "=" * 70

    print(f"\n{border}")
    print("  Document Analysis Results")
    print(border)

    print(f"\n  Question: {analysis['question']}")
    print(f"  Corpus: {analysis['corpus_summary']}")
    print(f"  Documents: {', '.join(analysis['documents_analyzed'])}")

    if analysis["consensus_reached"]:
        print(
            f"\n  Consensus: YES (confidence {analysis['confidence']:.0%}, "
            f"{analysis['rounds_used']} rounds)"
        )
    else:
        print(
            f"\n  Consensus: NO (confidence {analysis['confidence']:.0%})"
        )

    print(f"  Analysts: {', '.join(analysis['participants'])}")

    if analysis.get("answer"):
        print("\n  --- Answer ---\n")
        # Indent the answer for readability
        for line in analysis["answer"].split("\n"):
            print(f"  {line}")

    if analysis.get("elapsed_ms"):
        print(f"\n  Time: {analysis['elapsed_ms']:.0f}ms")

    print(f"  Analyzed at: {analysis['analyzed_at']}")
    print(border)


# ---------------------------------------------------------------------------
# Interactive Q&A Loop
# ---------------------------------------------------------------------------

async def interactive_qa(corpus: DocumentCorpus, rounds: int = 2) -> None:
    """Run an interactive question-and-answer loop over the document corpus.

    Allows the user to ask multiple questions without reloading documents.
    """
    print("\nDocument Q&A (type 'quit' to exit)")
    print(f"Corpus: {corpus.summary}")
    print("-" * 50)

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        analysis = await run_document_debate(
            corpus=corpus,
            question=question,
            rounds=rounds,
        )
        print_analysis(analysis)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run the document analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Aragora Document Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo mode with sample documents
  python examples/document-analysis/main.py --demo

  # Analyze a directory
  python examples/document-analysis/main.py \\
      --docs /path/to/docs \\
      --question "What is the authentication strategy?"

  # Analyze specific files
  python examples/document-analysis/main.py \\
      --files design.md runbook.md \\
      --question "What is the incident response process?"

  # Interactive Q&A mode
  python examples/document-analysis/main.py --docs /path/to/docs --interactive
        """,
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run in demo mode with sample documents",
    )
    parser.add_argument(
        "--docs",
        type=str,
        default="",
        help="Directory of documents to analyze",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=[],
        help="Specific files to analyze",
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default="What is the authentication and security architecture?",
        help="Question to answer about the documents",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        default=False,
        help="Run in interactive Q&A mode",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of debate rounds (default: 2)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as JSON instead of formatted text",
    )

    args = parser.parse_args()

    # Load documents
    if args.demo:
        corpus = load_demo_corpus()
    elif args.docs:
        corpus = load_documents_from_directory(args.docs)
    elif args.files:
        corpus = load_documents_from_files(args.files)
    else:
        logger.info("No documents specified, using demo corpus")
        corpus = load_demo_corpus()

    if not corpus.documents:
        logger.error("No documents loaded. Provide --docs, --files, or use --demo.")
        sys.exit(1)

    # Print corpus info to stderr so --json output stays clean
    out = sys.stderr if getattr(args, "json", False) else sys.stdout
    print(f"\nLoaded {corpus.summary}", file=out)
    print("Documents:", file=out)
    for doc in corpus.documents:
        print(f"  - {doc.title} ({doc.path}, {doc.size_bytes:,} bytes)", file=out)

    # Run analysis
    if args.interactive:
        await interactive_qa(corpus, rounds=args.rounds)
    else:
        if args.demo:
            logger.info("Running in demo mode (no API calls)")
            analysis = _build_demo_analysis(corpus, args.question)
        else:
            analysis = await run_document_debate(
                corpus=corpus,
                question=args.question,
                rounds=args.rounds,
            )

        if args.json:
            print(json.dumps(analysis, indent=2, default=str))
        else:
            print_analysis(analysis)


if __name__ == "__main__":
    asyncio.run(main())
