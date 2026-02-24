# Document Analysis Pipeline

A document question-and-answer system powered by Aragora's multi-agent debate engine and knowledge management. Ingest documents, ask questions, and get evidence-grounded answers where multiple AI agents debate and cite specific sources.

## How It Works

```
Documents (Markdown, text, code, config)
        |
        v
+------------------+
|  Ingest & Index  |  Load documents, build context
+------------------+
        |
        v
+------------------+
| User Question    |  "What is the auth strategy?"
+------------------+
        |
        v
+------------------+
| Multi-Agent      |  Document Analyst (Claude)
| Debate           |  Critical Reviewer (GPT)
| (Evidence-Based) |  Each must cite specific documents
+------------------+
        |
        v
+------------------+
| Consensus Answer |  Grounded in evidence + citations
+------------------+
```

## Quick Start (Demo Mode)

No API keys required -- uses built-in sample architecture documents:

```bash
python examples/document-analysis/main.py --demo
```

## Full Setup

### 1. Install Dependencies

```bash
pip install -r examples/document-analysis/requirements.txt
```

### 2. Set Environment Variables

```bash
# At least one AI provider key
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

### 3. Analyze Documents

```bash
# Analyze a directory of docs
python examples/document-analysis/main.py \
    --docs /path/to/architecture/docs \
    --question "What is the authentication and security architecture?"

# Analyze specific files
python examples/document-analysis/main.py \
    --files design.md runbook.md api-spec.md \
    --question "What is the incident response process?"

# Interactive Q&A mode (ask multiple questions)
python examples/document-analysis/main.py \
    --docs /path/to/docs \
    --interactive

# More thorough analysis with extra debate rounds
python examples/document-analysis/main.py \
    --docs /path/to/docs \
    --question "Are there any security gaps?" \
    --rounds 3
```

## Features

- **Evidence-grounded answers**: Agents must cite specific documents and sections
- **Multi-perspective analysis**: Different agents bring different analytical lenses
- **Contradiction detection**: Agents identify inconsistencies across documents
- **Gap identification**: Explicitly flags when documentation is insufficient
- **Interactive Q&A**: Ask multiple questions without reloading documents
- **Broad format support**: Markdown, text, Python, JavaScript, YAML, JSON, TOML

## Supported Document Types

| Extension | Type |
|-----------|------|
| `.md` | Markdown |
| `.txt` | Plain text |
| `.rst` | reStructuredText |
| `.py` | Python source |
| `.js`, `.ts` | JavaScript / TypeScript |
| `.yaml`, `.yml` | YAML |
| `.json` | JSON |
| `.toml` | TOML |

## Architecture

The pipeline uses three key Aragora components:

1. **`LocalDocsConnector`** -- File ingestion with type detection and security checks (symlink protection, path traversal prevention)
2. **`Arena`** -- Multi-agent debate engine with evidence-grounding instructions
3. **`DebateProtocol`** -- Configurable consensus rules (majority, unanimous) with calibration

## Output Example

```
======================================================================
  Document Analysis Results
======================================================================

  Question: What is the authentication and security architecture?
  Corpus: 4 documents (4 markdown), 4,231 bytes
  Documents: Authentication Architecture, Data Pipeline Design, ...

  Consensus: YES (confidence 88%, 2 rounds)
  Analysts: document_analyst, critical_reviewer

  --- Answer ---

  Based on the document corpus analysis:

  The platform uses a three-tier authentication architecture:

  1. **API Key Authentication** for service-to-service communication,
     with SHA-256 hashed keys and 90-day rotation
     (source: auth-design.md, Token Strategy section)

  2. **OAuth 2.0 / OIDC** for user-facing SSO, with JWT access tokens
     (15-min expiry, RS256) and opaque refresh tokens in Redis
     (source: auth-design.md, Token Strategy section)

  ...

  Analyzed at: 2026-02-23T10:30:00+00:00
======================================================================
```
