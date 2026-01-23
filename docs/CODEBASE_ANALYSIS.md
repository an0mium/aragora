# Codebase Analysis & Security

Aragora includes a codebase analysis module for dependency vulnerability
scanning and code quality metrics. It powers API endpoints and UI dashboards for
repository health.

## Overview

The codebase analysis stack includes:

- **CVE Client**: Queries NVD, OSV, and GitHub Advisory databases.
- **Dependency Scanner**: Parses lock files and identifies vulnerable
  dependencies.
- **Metrics Analyzer**: Computes complexity, maintainability, hotspots, and
  duplication signals.
- **Dependency Intelligence**: SBOM generation and license compatibility checks.
- **Secrets Scanner**: Detects exposed secrets and emits security events.
- **SAST Scanner**: Semgrep-backed static analysis with OWASP/CWE mapping.
- **Code Intelligence**: Tree-sitter parsing and call-graph construction.

Core modules live under `aragora/analysis/codebase/` with HTTP handlers in
`aragora/server/handlers/codebase/`.

## API Endpoints

### Security Scanning

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/codebase/{repo}/scan` | Trigger dependency vulnerability scan |
| GET | `/api/v1/codebase/{repo}/scan/latest` | Latest scan result |
| GET | `/api/v1/codebase/{repo}/scan/{scan_id}` | Scan result by ID |
| GET | `/api/v1/codebase/{repo}/scans` | Scan history |
| GET | `/api/v1/codebase/{repo}/vulnerabilities` | Vulnerabilities list |
| GET | `/api/v1/codebase/package/{ecosystem}/{package}/vulnerabilities` | Package advisories |
| GET | `/api/v1/cve/{cve_id}` | CVE details |
| POST | `/api/v1/codebase/{repo}/scan/sast` | Trigger SAST scan |
| GET | `/api/v1/codebase/{repo}/scan/sast/{scan_id}` | SAST scan status |
| GET | `/api/v1/codebase/{repo}/sast/findings` | SAST findings |
| GET | `/api/v1/codebase/{repo}/sast/owasp-summary` | OWASP summary |

### Secrets Scanning

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/codebase/{repo}/scan/secrets` | Trigger secrets scan |
| GET | `/api/v1/codebase/{repo}/scan/secrets/latest` | Latest secrets scan |
| GET | `/api/v1/codebase/{repo}/scan/secrets/{scan_id}` | Secrets scan by ID |
| GET | `/api/v1/codebase/{repo}/secrets` | Secrets list |
| GET | `/api/v1/codebase/{repo}/scans/secrets` | Secrets scan history |

### Dependency Intelligence

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/codebase/analyze-dependencies` | Analyze dependency graph |
| POST | `/api/v1/codebase/scan-vulnerabilities` | Scan repo for CVEs |
| POST | `/api/v1/codebase/check-licenses` | License compatibility |
| POST | `/api/v1/codebase/sbom` | Generate SBOM |

### Code Intelligence

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/codebase/{repo}/analyze` | Analyze codebase structure |
| GET | `/api/v1/codebase/{repo}/symbols` | List symbols |
| GET | `/api/v1/codebase/{repo}/callgraph` | Fetch call graph |
| GET | `/api/v1/codebase/{repo}/deadcode` | Find dead code |
| POST | `/api/v1/codebase/{repo}/impact` | Impact analysis |
| POST | `/api/v1/codebase/{repo}/understand` | Answer code questions |
| POST | `/api/v1/codebase/{repo}/audit` | Run comprehensive audit |
| GET | `/api/v1/codebase/{repo}/audit/{audit_id}` | Audit status/result |

Aliases exist at `/api/codebase/*` for UI-driven flows.

### Quick Scan

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/codebase/quick-scan` | Run quick security scan |
| GET | `/api/codebase/quick-scan/{scan_id}` | Quick scan result |
| GET | `/api/codebase/quick-scans` | Quick scan history |

## UI Surfaces

The codebase analysis UI is available under `/security-scan` with a dependency
panel for SBOM and license checks.

Key components:

- `aragora/live/src/app/(app)/security-scan/page.tsx`
- `aragora/live/src/components/codebase/DependencySecurityPanel.tsx`

Example:

```http
POST /api/v1/codebase/aragora/scan
Content-Type: application/json

{
  "repo_path": "/Users/armand/Development/aragora",
  "branch": "main",
  "commit_sha": "abc123"
}
```

### Quick Scan Example

```http
POST /api/codebase/quick-scan
Content-Type: application/json

{
  "repo_path": "/Users/armand/Development/aragora",
  "severity_threshold": "medium",
  "include_secrets": true
}
```

### Secrets Scan Example

```http
POST /api/v1/codebase/aragora/scan/secrets
Content-Type: application/json

{
  "repo_path": "/Users/armand/Development/aragora",
  "branch": "main"
}
```

### Metrics & Hotspots

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/codebase/{repo}/metrics/analyze` | Run metrics analysis |
| GET | `/api/v1/codebase/{repo}/metrics` | Latest metrics report |
| GET | `/api/v1/codebase/{repo}/metrics/{analysis_id}` | Metrics report by ID |
| GET | `/api/v1/codebase/{repo}/metrics/history` | Metrics history |
| GET | `/api/v1/codebase/{repo}/hotspots` | Complexity hotspots |
| GET | `/api/v1/codebase/{repo}/duplicates` | Code duplication summary |
| GET | `/api/v1/codebase/{repo}/metrics/file/{file_path}` | File-level metrics |

Example:

```http
POST /api/v1/codebase/aragora/metrics/analyze
Content-Type: application/json

{
  "repo_path": "/Users/armand/Development/aragora",
  "include_patterns": ["aragora/**/*.py"],
  "exclude_patterns": ["**/tests/**"],
  "complexity_warning": 10,
  "complexity_error": 20
}
```

## Python API

### Vulnerability Scanning

```python
from aragora.analysis.codebase import DependencyScanner

scanner = DependencyScanner()
result = await scanner.scan_repository("/path/to/repo")
print(result.critical_count, result.high_count)
```

### Dependency Intelligence Example

```http
POST /api/v1/codebase/analyze-dependencies
Content-Type: application/json

{
  "repo_path": "/Users/armand/Development/aragora",
  "include_dev": true
}
```

### CVE Lookup

```python
from aragora.analysis.codebase import CVEClient

client = CVEClient()
finding = await client.get_cve("CVE-2023-12345")
```

### Metrics Analysis

```python
from aragora.analysis.codebase import CodeMetricsAnalyzer

analyzer = CodeMetricsAnalyzer()
report = analyzer.analyze_repository("/path/to/repo", scan_id="metrics_001")
print(report.avg_complexity)
```

## Security Event Debates

Codebase security scans emit security events that can trigger a remediation
debate when critical findings are detected.

Relevant modules:

- `aragora/events/security_events.py`
- `aragora/server/handlers/codebase/security.py`

## Code Intelligence & Call Graph

Tree-sitter based code intelligence and call-graph analysis live under
`aragora/analysis/code_intelligence.py` and `aragora/analysis/call_graph.py`.

```python
from aragora.analysis.code_intelligence import CodeIntelligence
from aragora.analysis.call_graph import CallGraphBuilder

intel = CodeIntelligence()
analysis = intel.analyze_file("aragora/core/decision.py")
print(len(analysis.functions), len(analysis.classes))

builder = CallGraphBuilder(code_intel=intel)
graph = builder.build_from_directory("aragora")
print(graph.node_count, graph.edge_count)
```

## Configuration

- `NVD_API_KEY` - Optional NVD API key (higher rate limits)
- `GITHUB_TOKEN` - Optional GitHub advisory token

## Persistence Notes

- Scan results and metrics reports are stored in memory by default.
- For production, replace the in-memory stores with a durable database.
