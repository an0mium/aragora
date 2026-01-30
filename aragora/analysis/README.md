# aragora.analysis

Document querying, codebase security scanning, and content analysis. Provides
natural language document search with citations, static security analysis (SAST,
secrets, dependencies, SBOM), and communication-level detectors (spam, email priority).

## Modules

| File | Purpose |
|------|---------|
| `nl_query.py` | Natural language document querying with AI-powered answer synthesis and citations |
| `call_graph.py` | Inter-procedural call graph construction, dead code and circular dependency detection |
| `code_intelligence.py` | Tree-sitter AST parsing for semantic symbol extraction (Python, JS/TS, Go, Rust, Java) |
| `spam_detector.py` | Multi-signal spam/phishing detection (Bayesian classification, link analysis, impersonation) |
| `email_priority.py` | Email importance scoring using RLM and ContinuumMemory learned preferences |
| **`codebase/`** | |
| `codebase/models.py` | Shared dataclasses for findings, dependencies, metrics, and scan results |
| `codebase/scanner.py` | Dependency vulnerability scanner (npm, pip, Go, Rust, Ruby lock files) |
| `codebase/sast_scanner.py` | SAST scanner with Semgrep integration, OWASP Top 10 and CWE mapping |
| `codebase/secrets_scanner.py` | Hardcoded credential and API key detection via pattern matching and entropy analysis |
| `codebase/cve_client.py` | Async CVE database client (NVD, OSV, GitHub Advisories) with caching |
| `codebase/bug_detector.py` | Static bug pattern detection (null derefs, resource leaks, race conditions) |
| `codebase/sbom_generator.py` | SBOM generation in CycloneDX 1.5 and SPDX 2.3 formats |
| `codebase/metrics.py` | Code quality metrics: cyclomatic/cognitive complexity, duplication detection |

## Key Classes

- **`DocumentQueryEngine`** -- NL query engine with streaming, citations, and configurable `QueryMode`
- **`SASTScanner`** -- Security scanning with OWASP/CWE mapping; falls back to local patterns without Semgrep
- **`SecretsScanner`** -- Entropy + regex credential detection across repositories
- **`DependencyScanner`** -- Parses lock files and queries CVE databases for known vulnerabilities
- **`SBOMGenerator`** -- Produces CycloneDX/SPDX bills of materials for supply chain compliance
- **`CodeMetricsAnalyzer`** -- Complexity and duplication analysis with per-function breakdowns
- **`CodeIntelligence`** -- Tree-sitter-based semantic code analysis across five languages

## Usage

```python
from aragora.analysis import DocumentQueryEngine, query_documents
from aragora.analysis.codebase import SASTScanner, SecretsScanner

# Quick document query
result = await query_documents(
    question="What are the key terms in this contract?",
    document_ids=["doc_123"],
)
print(result.answer)

# SAST scan
scanner = SASTScanner()
await scanner.initialize()
result = await scanner.scan_repository("/path/to/repo")
print(f"Found {len(result.findings)} issues")

# Secrets scan
secrets = SecretsScanner()
result = await secrets.scan("/path/to/repo")
```
