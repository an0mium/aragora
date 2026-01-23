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

## Configuration

- `NVD_API_KEY` - Optional NVD API key (higher rate limits)
- `GITHUB_TOKEN` - Optional GitHub advisory token

## Persistence Notes

- Scan results and metrics reports are stored in memory by default.
- For production, replace the in-memory stores with a durable database.
