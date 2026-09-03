#!/usr/bin/env python3
"""Receipt-First mission scoreboard: ten exit metrics and six guardrails as baseline/now/delta.

Network rows (1, 4, 5, 7, 8, 9) are cached so offline runs still render. Every network call goes
through ``curl`` or ``gh`` so the proxy environment applies uniformly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

Row = dict[str, Any]
REPO = "synaptent/aragora"
BASELINE_REF = "23909906e8"
BASELINE_DATE = "2026-09-02"
STATUSES = ("ok", "fail", "unavailable", "pending-operator")
NETWORK_ROWS = (1, 4, 5, 7, 8, 9)
DEFAULT_CACHE = Path.home() / ".aragora" / "receipt-first-scoreboard-cache.json"
ATLAS_DIR = Path.home() / ".aragora" / "receipt-first-atlas"
API_PROBE = "https://api.aragora.ai/readyz"
CANARY_PROBE = "https://api-canary.aragora.ai/readyz"
CODE_SEARCH_QUERY = "synaptent/aragora@ -repo:synaptent/aragora"
VECTORS = "tests/verify/test_odr_vectors.py"
PIN_RE = re.compile(r"synaptent/aragora@([0-9a-f]{7,40})")
README_PIN_RE = re.compile(r"uses: synaptent/aragora@([0-9a-f]{7,40})")
STATUS_LINE_RE = re.compile(r"^\*\*Status:\*\* (\w+) (v[0-9.]+)")
LEDGER_RE = re.compile(r"^- \[metric (\d+)\] .*#(\d+) head ([0-9a-f]{40}|n/a)\s*$")
METRIC_ROW_RE = re.compile(r"^\| *(10|[1-9]) *\|")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
PARITY_GAP_KEYS = "missing_from_python_sdk missing_from_typescript_sdk missing_from_both_sdks"
PARITY_GAP_KEYS += " stale_python_sdk_paths stale_typescript_sdk_paths"
MYPY_ERROR_LIMIT, MYPY_BASELINE_LIMIT = 1744, 3115

# (id, name, baseline value, baseline cell as printed); frozen for the whole mission.
METRICS: list[tuple[int, str, Any, str]] = [
    (1, "PyPI `aragora` newest release age", 58, "58 d (2.9.0, 2026-07-06)"),
    (2, "README Action pin age", 57, "57 d (8b600a3a, 2026-07-07)"),
    (3, "ODR spec status", "Draft v0.1", "Draft v0.1, no vectors"),
    (4, "`aragora-verify` on PyPI", "0.1.1", "0.1.1 (2026-07-04)"),
    (5, "Disagreement Atlas", 1659, "1659 records, #9951 draft, no atlas-v1 release"),
    (6, "Dissent visibility", 53, "53 CHANGES-REQUESTED / 1659 records"),
    (7, "Hosted API", "000", "000 (canary 200)"),
    (8, "Verifiable published receipts", 0, "0"),
    (9, "External repos using the Action", 0, "0 (demo repo absent)"),
    (10, "Contract-drift units", 398, "398 (target 84, fail)"),
]
MEASUREMENTS = {
    1: "curl https://pypi.org/pypi/aragora/json → max(upload_time_iso_8601 over releases)",
    2: "uses: synaptent/aragora@<sha> in README.md; git log -1 --format=%cI <sha>",
    3: "sed -n 3p docs/specs/OPEN_DECISION_RECEIPT.md; test -f tests/verify/test_odr_vectors.py",
    4: "curl https://pypi.org/pypi/aragora-verify/json; ^version in aragora-verify/pyproject.toml",
    5: "docs/atlas/manifest.json .dataset.record_count; gh release view atlas-v1; gh pr view 9951",
    6: "Atlas JSONL: verdict==changes_requested, posted_to_thread, distinct (pr, head_sha) rounds",
    7: f"curl -s -o /dev/null -w %{{http_code}} --max-time 15 {API_PROBE} (one probe per run)",
    8: "gh release list --limit 100 → receipts-* tags → *.odr.json assets; first-hour run",
    9: "gh api search/code (paths under .github/workflows/); gh repo view aragora-receipt-demo",
    10: "check_contract_drift_ratchet.py --mode program --ref <40-hex> --json .current.total_items",
}
GUARDRAILS: list[tuple[str, str, int, int]] = [
    ("import_cycles", "Mutual import cycles", 140, 144),
    ("handlers_flat_root", "Handlers flat-root files", 187, 187),
    ("ci_workflows", "Workflows", 97, 97),
    ("doc_files", "Docs pages", 1119, 1119),
    ("top_level_modules", "Top-level packages", 145, 145),
    ("openapi_operations", "OpenAPI operations", 3205, 3205),
]


class NetworkUnavailable(Exception):
    """A network-backed measurement could not be completed; fall back to the cache."""


class Cmd(NamedTuple):
    rc: int
    out: str
    err: str

    ok = property(lambda self: self.rc == 0)


def run_cmd(argv: list[str], *, timeout: float = 60, cwd: Any = None, env: Any = None) -> Cmd:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)
    except subprocess.TimeoutExpired:
        return Cmd(124, "", f"timeout after {timeout}s: {' '.join(argv)}")
    except OSError as exc:
        return Cmd(127, "", str(exc))
    return Cmd(p.returncode, p.stdout, p.stderr)


def json_cmd(argv: list[str], *, timeout: float = 60, cwd: Path | None = None) -> Any:
    c = run_cmd(argv, timeout=timeout, cwd=cwd)
    if not c.ok or not c.out.strip():
        raise NetworkUnavailable(c.err.strip() or f"empty output from {argv[0]}")
    return json.loads(c.out)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_date(value: str) -> date:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).date()


def age_days(value: str) -> int:
    return (datetime.now(timezone.utc).date() - utc_date(value)).days


def read_text(p: Path) -> str:
    return p.read_text(errors="replace") if p.is_file() else ""


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compute_delta(baseline: Any, now: Any) -> Any:
    if is_number(baseline) and is_number(now):
        return now - baseline
    return f"{'null' if baseline is None else baseline} → {'null' if now is None else now}"


def semver(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", text)[:3])


def curl_code(url: str, max_time: int) -> str:
    argv = ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", str(max_time), url]
    return run_cmd(argv, timeout=max_time + 5).out.strip() or "000"


def pypi_newest(package: str) -> tuple[str, str]:
    argv = ["curl", "-s", "--max-time", "30", f"https://pypi.org/pypi/{package}/json"]
    d = json_cmd(argv, timeout=35)
    newest = max(f["upload_time_iso_8601"] for files in d["releases"].values() for f in files)
    return d["info"]["version"], newest


def releases() -> list[Row]:
    return json_cmd(f"gh release list -R {REPO} --limit 100 --json tagName,publishedAt".split())


def release_assets(tag: str) -> list[str]:
    c = run_cmd(["gh", "release", "view", tag, "-R", REPO, "--json", "assets"])
    if c.ok:
        return [a["name"] for a in json.loads(c.out).get("assets", [])]
    if "not found" in c.err.lower():
        return []
    raise NetworkUnavailable(c.err.strip())


@dataclass
class Ctx:
    root: Path
    offline: bool
    ref: str
    quorum_runs: int | None
    run_vectors: bool
    network_ok: bool = True
    atlas_v1_assets: list[str] | None = None


def metric_1(ctx: Ctx) -> Row:
    version, uploaded = pypi_newest("aragora")
    age = age_days(uploaded)
    r: Row = {"now": age, "version": version, "upload_date": utc_date(uploaded).isoformat()}
    return r | {"status": "ok" if age <= 14 else "fail"}


def metric_2(ctx: Ctx) -> Row:
    m = README_PIN_RE.search(read_text(ctx.root / "README.md"))
    sha = m.group(1) if m else None
    scope = ["README.md", "docs", "docs-site", "examples", ".github", "action.yml"]
    tracked = run_cmd(["git", "ls-files", *scope], cwd=ctx.root).out.split()
    pins = [p for rel in tracked for p in PIN_RE.findall(read_text(ctx.root / rel))]
    distinct = sorted(set(pins))
    r: Row = {"sha": sha, "pin_count": len(pins), "distinct_shas": distinct, "now": None}
    r.update(commit_date=None, status="fail")
    if len(distinct) > 1:
        r["warning"] = f"{len(distinct)} distinct pinned SHAs: {[s[:10] for s in distinct]}"
    if not sha:
        return r | {"note": "no README pin"}
    argv = ["git", "log", "-1", "--format=%cI", sha]
    if not run_cmd(argv, cwd=ctx.root).ok and not ctx.offline:
        run_cmd(["git", "fetch", "origin", sha, "--quiet"], cwd=ctx.root, timeout=120)
    log = run_cmd(argv, cwd=ctx.root)
    if not (log.ok and log.out.strip()):
        return r | {"note": "pinned sha not in local history"}
    age = age_days(log.out.strip())
    date_ = utc_date(log.out.strip()).isoformat()
    return r | {"now": age, "commit_date": date_, "status": "ok" if age <= 14 else "fail"}


def metric_3(ctx: Ctx) -> Row:
    lines = read_text(ctx.root / "docs/specs/OPEN_DECISION_RECEIPT.md").splitlines()
    m = STATUS_LINE_RE.match(lines[2]) if len(lines) >= 3 else None
    status, version = (m.group(1), m.group(2)) if m else (None, None)
    vectors = (ctx.root / VECTORS).is_file()
    vpass: bool | None = None
    if ctx.run_vectors and vectors:
        argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", VECTORS]
        vpass = run_cmd(argv, cwd=ctx.root, timeout=600).ok
    ok = bool(version) and semver(version or "") >= (0, 2) and vectors and vpass is not False
    r: Row = {"now": f"{status} {version}" if m else None, "spec_status": status}
    r.update(version=version, vectors_present=vectors, vectors_pass=vpass)
    return r | {"status": "ok" if ok else "fail"}


def metric_4(ctx: Ctx) -> Row:
    version, uploaded = pypi_newest("aragora-verify")
    ok = semver(version) >= (0, 2)
    date_ = utc_date(uploaded).isoformat()
    return {"now": version, "upload_date": date_, "status": "ok" if ok else "fail"}


def metric_5(ctx: Ctx) -> Row:
    tags = [x["tagName"] for x in releases()]
    assets = release_assets("atlas-v1") if "atlas-v1" in tags else []
    ctx.atlas_v1_assets = assets
    pr = json_cmd(["gh", "pr", "view", "9951", "-R", REPO, "--json", "state"])
    merged = pr.get("state") == "MERGED"
    manifest = read_text(ctx.root / "docs/atlas/manifest.json")
    if not manifest:
        argv = "git show rf/pr9951-inspect:docs/atlas/manifest.json".split()
        manifest = run_cmd(argv, cwd=ctx.root).out
    try:
        count: int | None = int(json.loads(manifest)["dataset"]["record_count"])
    except (ValueError, KeyError, TypeError):
        count = None
    r: Row = {"now": count, "record_count": count, "pr_9951_merged": merged}
    r.update(atlas_v1_assets=assets, atlas_release_count=sum(t.startswith("atlas-") for t in tags))
    return r | {"status": "ok" if merged and assets else "fail"}


def metric_6(ctx: Ctx) -> Row:
    r: Row = {"now": None, "posted": None, "rounds": None, "records": None, "upper_bound": True}
    r.update(quorum_runs=ctx.quorum_runs, status="unavailable")
    if not ctx.offline and ctx.atlas_v1_assets and not list(ATLAS_DIR.glob("*.jsonl")):
        ATLAS_DIR.mkdir(parents=True, exist_ok=True)
        argv = f"gh release download atlas-v1 -R {REPO} -p *.jsonl --clobber".split()
        run_cmd(argv + ["-D", str(ATLAS_DIR)], timeout=120)
    candidates = [ctx.root / "docs/atlas/atlas-v1.jsonl", *sorted(ATLAS_DIR.glob("*.jsonl"))]
    jsonl = next((p for p in candidates if p.is_file()), None)
    if jsonl is None:
        return r | {"reason": "no Atlas JSONL (docs/atlas/atlas-v1.jsonl or atlas-v1 asset)"}
    recs = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    cr = sum(x.get("verdict") == "changes_requested" for x in recs)
    posted = sum(bool(x.get("posted_to_thread")) for x in recs)
    rounds = len({(x.get("pr"), x.get("head_sha")) for x in recs})
    r.update(now=cr, posted=posted, rounds=rounds, records=len(recs), source=str(jsonl))
    return r | {"ratio": round(posted / rounds, 3) if rounds else None, "status": "fail"}


def metric_7(ctx: Ctx) -> Row:
    code, canary = curl_code(API_PROBE, 15), curl_code(CANARY_PROBE, 15)
    r: Row = {"now": code, "canary_code": canary, "probe": API_PROBE}
    return r | {"status": "ok" if code == "200" else "fail"}


def first_hour_run_ok(after: str) -> bool:
    argv = f"gh run list -R {REPO} --workflow metrics-drift.yml --status success".split()
    c = run_cmd(argv + ["--limit", "10", "--json", "databaseId,createdAt"])
    runs = json.loads(c.out) if c.ok and c.out.strip() else []
    jq = '[.jobs[]|select(.name=="receipt-first-hour" and .conclusion=="success")]|length'
    for run in [x for x in runs if x.get("createdAt", "") >= after][:3]:
        url = f"repos/{REPO}/actions/runs/{run['databaseId']}/jobs"
        j = run_cmd(["gh", "api", url, "--jq", jq])
        if j.ok and j.out.strip().isdigit() and int(j.out) > 0:
            return True
    return False


def metric_8(ctx: Ctx) -> Row:
    rel = [x for x in releases() if x["tagName"].startswith("receipts-")]
    tags = [x["tagName"] for x in rel]
    per_tag = {t: sum(n.endswith(".odr.json") for n in release_assets(t)) for t in tags}
    total = sum(per_tag.values())
    run_ok = total >= 3 and first_hour_run_ok(max(x.get("publishedAt", "") for x in rel))
    r: Row = {"now": total, "receipts_releases": per_tag, "first_hour_run_ok": run_ok}
    return r | {"status": "ok" if total >= 3 and run_ok else "fail"}


def metric_9(ctx: Ctx) -> Row:
    search = json_cmd(["gh", "api", "search/code?q=synaptent/aragora@+-repo:synaptent/aragora"])
    items = search.get("items", [])
    count = sum(str(i.get("path", "")).startswith(".github/workflows/") for i in items)
    repo = run_cmd(["gh", "repo", "view", "synaptent/aragora-receipt-demo", "--json", "name"])
    if not repo.ok and "could not resolve" not in repo.err.lower():
        raise NetworkUnavailable(repo.err.strip())
    r: Row = {"now": count, "code_search_count": count, "demo_repo_exists": repo.ok}
    return r | {"query": CODE_SEARCH_QUERY, "status": "ok" if count >= 1 or repo.ok else "fail"}


def metric_10(ctx: Ctx) -> Row:
    argv = [sys.executable, "scripts/check_contract_drift_ratchet.py", "--mode", "program"]
    out = run_cmd(argv + ["--ref", ctx.ref, "--json"], cwd=ctx.root, timeout=300).out
    d = json.loads(out) if out.strip() else {}
    total = d.get("current", {}).get("total_items")
    argv = [sys.executable, "scripts/check_sdk_parity.py", "--json"]
    parity = run_cmd(argv, cwd=ctx.root, timeout=300).out
    g = json.loads(parity).get("gaps", {}) if parity.strip() else None
    gaps = sum(len(g.get(k, [])) for k in PARITY_GAP_KEYS.split()) if g is not None else None
    inv = read_text(ctx.root / "scripts/baselines/contract_drift_inventory.json")
    items = json.loads(inv).get("items", []) if inv else []
    r: Row = {"now": total, "total_items": total, "ref": ctx.ref, "sdk_parity_gaps": gaps}
    r.update(target=d.get("target", {}).get("max_open_items"), ratchet_status=d.get("status"))
    r["inventory_open"] = sum(i.get("status") == "open" for i in items) if inv else None
    r["informational"] = ["inventory_open", "sdk_parity_gaps"]
    ok = d.get("status") == "pass" and is_number(total) and total <= 398
    return r | {"status": "ok" if ok else "fail"}


def local_verify_version(ctx: Ctx, row: Row) -> None:
    text = read_text(ctx.root / "aragora-verify/pyproject.toml")
    m = re.search(r'^version = "([^"]+)"', text, re.M)
    local, now = (m.group(1) if m else None), row.get("now")
    row["local_version"] = local
    row["local_ahead_of_pypi"] = bool(local and now and semver(local) > semver(str(now)))


def load_cache(path: Path) -> Row:
    try:
        d = json.loads(path.read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def measure_guardrails(root: Path) -> Row:
    graph = [sys.executable, "scripts/ci/measure_import_graph.py", "--json"]
    regen = [sys.executable, "scripts/regenerate_metrics.py", "--json"]
    g, m = json_cmd(graph, cwd=root, timeout=180), json_cmd(regen, cwd=root, timeout=180)
    vals = {x["key"]: x["value"] for x in m["metrics"]} if "metrics" in m else m
    r: Row = {"import_cycles": g.get("mutual_import_cycles")}
    r["handlers_flat_root"] = g.get("handlers_flat_root")
    return r | {gid: vals.get(gid) for gid, _, _, _ in GUARDRAILS[2:]}


def guardrail_rows(values: Row) -> list[Row]:
    rows = []
    for gid, name, ceiling, baseline in GUARDRAILS:
        now = values.get(gid)
        status = "ok" if is_number(now) and now <= ceiling else "over"
        row = {"id": gid, "name": name, "ceiling": ceiling, "baseline": baseline}
        rows.append(row | {"now": now, "status": status})
    return rows


def read_parked(path: Path | None) -> tuple[dict[int, str], str]:
    """Return {metric: '#N'} from the settlement ledger and the verbatim ``## Parked`` block."""
    pending: dict[int, str] = {}
    parked: list[str] = []
    section = None
    for line in read_text(path).splitlines() if path else []:
        if line.startswith("## "):
            section = line[3:].strip()
        elif section == "Awaiting operator settlement" and (m := LEDGER_RE.match(line)):
            pending.setdefault(int(m.group(1)), f"#{m.group(2)}")
        elif section == "Parked":
            parked.append(line)
    return pending, "\n".join(parked).strip("\n")


def origin_main_cycles(root: Path) -> int | None:
    tmp = tempfile.mkdtemp(prefix="rf-scoreboard-main-")
    try:
        unpack = ["sh", "-c", f"git archive origin/main | tar -x -C '{tmp}'"]
        if not run_cmd(unpack, cwd=root, timeout=300).ok:
            return None
        argv = [sys.executable, "scripts/ci/measure_import_graph.py", "--json"]
        g = run_cmd(argv, cwd=tmp, timeout=180, env={**os.environ, "PYTHONPATH": tmp})
        return json.loads(g.out).get("mutual_import_cycles") if g.ok and g.out.strip() else None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_guardrails(root: Path, offline: bool) -> int:
    if not offline:
        run_cmd(["git", "fetch", "origin", "main", "--quiet"], cwd=root, timeout=120)
    sha = run_cmd(["git", "rev-parse", "origin/main"], cwd=root).out.strip()
    main_cycles = origin_main_cycles(root)
    shown = "unknown" if main_cycles is None else main_cycles
    print(f"origin/main mutual_import_cycles: {shown} ({sha[:10]})")
    values = measure_guardrails(root)
    mypy = ["sh", "-c", "mypy aragora/ --ignore-missing-imports | tail -1"]
    tail = run_cmd(mypy, cwd=root, timeout=540).out.strip()
    m = re.search(r"Found (\d+) errors?", tail)
    wc = run_cmd(["sh", "-c", "wc -l < .mypy-baseline"], cwd=root).out.strip()
    print(f"mypy: {tail or 'no output'}")
    limit = max(140, main_cycles) if is_number(main_cycles) else 140
    checks = [("mutual_import_cycles", values["import_cycles"], limit)]
    checks += [(gid, values[gid], ceiling) for gid, _, ceiling, _ in GUARDRAILS[1:]]
    checks += [("mypy_errors", int(m.group(1)) if m else 0, MYPY_ERROR_LIMIT)]
    checks += [("mypy_baseline_lines", int(wc) if wc.isdigit() else None, MYPY_BASELINE_LIMIT)]
    bad = []
    for name, value, cap in checks:
        ok = is_number(value) and value <= cap
        print(f"{name}: {value} (limit {cap}) {'ok' if ok else 'OVER'}")
        bad += [] if ok else [name]
    print(f"FAIL: guardrail regression: {', '.join(bad)}" if bad else "guardrails ok")
    return 1 if bad else 0


def build_rows(ctx: Ctx, cache: Row, pending: dict[int, str]) -> tuple[list[Row], Row]:
    cached = cache.get("metrics", {})
    measured: dict[int, Row] = {}
    failed: dict[int, str] = {}

    def attempt(i: int) -> None:
        if i in NETWORK_ROWS and (ctx.offline or not ctx.network_ok):
            failed[i] = "offline" if ctx.offline else "pypi.org pre-probe failed"
            return
        try:
            measured[i] = globals()[f"metric_{i}"](ctx)
        except NetworkUnavailable as exc:
            failed[i] = str(exc)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            failed[i] = f"{type(exc).__name__}: {exc}"

    attempt(1)  # doubles as the reachability pre-probe for the other network rows
    ctx.network_ok = 1 in measured
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(attempt, (4, 5, 7, 8, 9)))
    for i in (2, 3, 6, 10):
        attempt(i)
    rows, new_cache = [], dict(cached)
    for mid, name, baseline, cell in METRICS:
        row: Row = {"id": mid, "name": name, "baseline": baseline, "baseline_cell": cell}
        old = cached.get(str(mid)) if mid in NETWORK_ROWS else None
        if mid in measured:
            row.update(measured[mid])
            if mid in NETWORK_ROWS:
                new_cache[str(mid)] = {k: v for k, v in measured[mid].items() if k != "status"}
        else:
            reason = failed.get(mid, "not measured")
            row.update(old or {"now": None}, status="unavailable", reason=reason)
            row.update({"cached_at": cache.get("cached_at")} if old else {})
        if mid == 4:
            local_verify_version(ctx, row)
        row.update(measurement=MEASUREMENTS[mid], delta=compute_delta(baseline, row.get("now")))
        if row["status"] == "fail" and mid in pending:
            row.update(status="pending-operator", pending_ref=pending[mid])
        rows.append(row)
    if not any(i in measured for i in NETWORK_ROWS):
        return rows, {}
    return rows, {"cached_at": utc_stamp(), "metrics": new_cache}


def now_cell(r: Row) -> str:
    now = r.get("now")
    if now is None:
        return "null"
    extra = {
        1: f"d ({r.get('version')}, {r.get('upload_date')})",
        2: f"d ({str(r.get('sha'))[:8]}, {r.get('commit_date')})",
        4: f"({r.get('upload_date')}; local {r.get('local_version')})",
        5: f"records, #9951 {'merged' if r.get('pr_9951_merged') else 'open'}",
        6: f"CHANGES-REQUESTED / {r.get('records')} records",
        7: f"(canary {r.get('canary_code')})",
        9: f"(demo repo {'present' if r.get('demo_repo_exists') else 'absent'})",
        10: f"(target {r.get('target')}, {r.get('ratchet_status')})",
    }
    return f"{now} {extra[r['id']]}" if r["id"] in extra else str(now)


def render_markdown(doc: Row, parked_given: bool) -> str:
    out = ["# Receipt-First scoreboard", ""]
    out.append(f"baseline_ref: {doc['baseline_ref']} ({doc['baseline_date']})")
    out.append(f"measured ref: {doc['ref']}")
    out += [f"generated_at: {doc['generated_at']}{' (offline)' if doc['offline'] else ''}", ""]
    out += ["| # | Metric | Baseline | Now | Delta | Status |", "|---|---|---|---|---|---|"]
    for r in doc["metrics"]:
        status = r["status"] + (f" {r['pending_ref']}" if r.get("pending_ref") else "")
        if r["status"] == "unavailable" and r.get("cached_at"):
            status += f" (cached {r['cached_at']})"
        cells = (r["id"], r["name"], r["baseline_cell"], now_cell(r), r["delta"], status)
        out.append("| " + " | ".join(str(c) for c in cells) + " |")
    out += ["", "## Guardrails", "", "| Guardrail | Ceiling | Baseline | Now | Status |"]
    out.append("|---|---|---|---|---|")
    for g in doc["guardrails"]:
        cells = (g["name"], g["ceiling"], g["baseline"], g["now"], g["status"])
        out.append("| " + " | ".join(str(c) for c in cells) + " |")
    if parked_given:
        out += ["", "## Parked", doc["parked"] or "(none yet)"]
    return "\n".join(out) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Receipt-First scoreboard (10 metrics, 6 guardrails)")
    add = p.add_argument
    add("--json", action="store_true", help="print one JSON object")
    add("--markdown", action="store_true", help="print Markdown tables (default)")
    add("--cache", type=Path, default=DEFAULT_CACHE, help="cache file for the network rows")
    add("--offline", action="store_true", help="no network; network rows come from the cache")
    add("--post", action="store_true", help="post the Markdown as one NEW comment on --epic")
    add("--epic", type=int, help="tracking epic issue number (required by --post)")
    add("--parked-file", type=Path, help="ledger file: sole source of pending-operator rows")
    add("--ref", help="40-hex ref for row 10 (default: HEAD)")
    add("--quorum-runs", type=int, help="row 6 denominator (informational)")
    add("--run-vectors", action="store_true", help=f"run pytest {VECTORS} for row 3")
    add("--check-guardrails", action="store_true", help="compare guardrails; exit 1 on regression")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.post and args.epic is None:
        parser.error("--post requires --epic N (nothing posted)")
    if args.post and args.epic <= 0:
        parser.error(f"--epic must be a positive issue number, got epic {args.epic}")
    root = repo_root()
    if args.check_guardrails:
        return check_guardrails(root, args.offline)
    ref = args.ref if args.ref and HEX40.match(args.ref) else None
    ref = ref or run_cmd(["git", "rev-parse", args.ref or "HEAD"], cwd=root).out.strip()
    if not HEX40.match(ref):
        raise SystemExit(f"error: could not resolve --ref {args.ref or 'HEAD'!r} to a 40-hex SHA")
    pending, parked_text = read_parked(args.parked_file)
    cache = load_cache(args.cache)
    ctx = Ctx(root, args.offline, ref, args.quorum_runs, args.run_vectors)
    rows, new_cache = build_rows(ctx, cache, pending)
    if new_cache:
        try:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            args.cache.write_text(json.dumps(new_cache, indent=2, sort_keys=True) + "\n")
        except OSError as exc:
            print(f"warning: could not write cache {args.cache}: {exc}", file=sys.stderr)
    try:
        guardrails = guardrail_rows(measure_guardrails(root))
    except (NetworkUnavailable, ValueError, KeyError) as exc:
        print(f"warning: guardrail measurement failed: {exc}", file=sys.stderr)
        guardrails = guardrail_rows({})
    doc: Row = {"baseline_ref": BASELINE_REF, "baseline_date": BASELINE_DATE}
    doc.update(generated_at=utc_stamp(), ref=ref, offline=args.offline)
    doc.update(network_ok=ctx.network_ok, cache_path=str(args.cache))
    doc.update(cached_at=(new_cache or cache).get("cached_at"), metrics=rows)
    doc.update(guardrails=guardrails, parked=parked_text)
    if args.post:
        body = Path(tempfile.mkdtemp(prefix="rf-scoreboard-")) / "scoreboard.md"
        body.write_text(render_markdown(doc, args.parked_file is not None))
        argv = ["gh", "issue", "comment", str(args.epic), "-R", REPO, "--body-file", str(body)]
        c = run_cmd(argv, timeout=120)
        if not c.ok:
            print(f"error: posting to epic #{args.epic} failed: {c.err.strip()}", file=sys.stderr)
            return 1
        print(c.out.strip())
    elif args.json:
        print(json.dumps(doc, indent=2, ensure_ascii=False))
    else:
        print(render_markdown(doc, args.parked_file is not None), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
