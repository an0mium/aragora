#!/usr/bin/env python3
"""Extract all TS SDK drift items for analysis."""

import re
import json
from pathlib import Path
from collections import defaultdict

repo = Path(__file__).resolve().parent.parent

# Load OpenAPI specs
spec_path = repo / "docs/api/openapi.json"
spec = json.loads(spec_path.read_text())
gen_path = repo / "docs/api/openapi_generated.json"
gen_spec = json.loads(gen_path.read_text()) if gen_path.exists() else {}

HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}


def _normalize(path):
    path = path.split("?", 1)[0]
    path = re.sub(r"\$\{[^}]+\}", "{param}", path)
    path = re.sub(r"\{[^}]+\}", "{param}", path)
    path = re.sub(r"^/api/v\d+/", "/api/", path)
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return path


openapi_eps = set()
for s in [spec, gen_spec]:
    for path, ops in s.get("paths", {}).items():
        for method in ops:
            if method.lower() in HTTP_METHODS:
                openapi_eps.add((method.lower(), _normalize(path)))

TS_REQUEST_RE = re.compile(
    r"this\.client\.request\(\s*['\"](?P<method>[A-Z]+)['\"]\s*,"
    r"\s*(?P<path>`[^`]+`|'[^']+'|\"[^\"]+\")"
)
TS_DIRECT_RE = re.compile(
    r"this\.client\.(?P<method>get|post|put|delete|patch)\("
    r"\s*(?P<path>`[^`]+`|'[^']+'|\"[^\"]+\")"
)

ts_dir = repo / "sdk/typescript/src/namespaces"
ts_drift = []
ts_all_eps = set()

for ts_file in sorted(ts_dir.glob("*.ts")):
    if ts_file.name.startswith("_") or ts_file.name == "index.ts" or ts_file.name == "CLAUDE.md":
        continue
    ns = ts_file.stem
    content = ts_file.read_text()
    eps = set()
    for m in TS_REQUEST_RE.finditer(content):
        eps.add((m.group("method").lower(), _normalize(m.group("path")[1:-1])))
    for m in TS_DIRECT_RE.finditer(content):
        eps.add((m.group("method").lower(), _normalize(m.group("path")[1:-1])))
    ts_all_eps |= eps
    for ep in sorted(eps - openapi_eps):
        ts_drift.append((ns, ep[0], ep[1]))

print(f"Total TS drift: {len(ts_drift)}")
print(f"Total TS endpoints: {len(ts_all_eps)}")
print(f"OpenAPI endpoints: {len(openapi_eps)}")

# Group by normalized path to see unique paths we need to add
unique_entries = set()
by_path = defaultdict(set)
for ns, method, path in ts_drift:
    unique_entries.add((method, path))
    by_path[path].add(method)

print(f"Unique method+path combos: {len(unique_entries)}")
print(f"Unique paths: {len(by_path)}")
print()

# Print all drift entries grouped by namespace
current_ns = None
for ns, method, path in ts_drift:
    if ns != current_ns:
        print(f"\n--- {ns} ---")
        current_ns = ns
    print(f"  {method.upper()} {path}")

# Output JSON for use by the stub generator
output = []
for method, path in sorted(unique_entries):
    output.append({"method": method, "path": path})

output_path = repo / "scripts" / "_ts_drift_items.json"
output_path.write_text(json.dumps(output, indent=2))
print(f"\n\nWrote {len(output)} drift items to {output_path}")
