# Three Initiatives Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix CI runner recovery, harden dependency security with lockfiles, and build the full developer onramp.

**Architecture:** Three independent tracks that can be executed in parallel. Track A (runner fix) is pure YAML edits. Track B (vulnerability sweep) is dependency management. Track C (developer onramp) is docs + frontend + SDK.

**Tech Stack:** GitHub Actions YAML, uv/pip-compile, npm, Next.js 16 (App Router), React 19, TypeScript, Python 3.10+

---

## Track A: Runner Checkout Recovery Fix

### Task A1: Fix the canonical recovery guard

**Files:**
- Modify: `.github/workflows/lint.yml` (lines ~59-74, ~185-200)
- Modify: `.github/workflows/test.yml` (lines ~80, ~262, ~328)
- Modify: `.github/workflows/smoke.yml` (all 4 jobs)
- Modify: `.github/workflows/smoke-offline.yml`
- Modify: `.github/workflows/deploy-frontend.yml`

**Step 1: Create a reusable YAML anchor file for the guard**

Since 27+ workflows need the same guard, create a composite action to avoid duplication.

Create `.github/actions/checkout-integrity/action.yml`:
```yaml
name: "Verify checkout integrity"
description: "Ensures checkout is complete, recovers from sparse-checkout corruption"
runs:
  using: "composite"
  steps:
    - name: Verify checkout integrity
      shell: bash
      run: |
        if [[ ! -f pyproject.toml ]]; then
          echo "::warning::pyproject.toml missing after checkout; attempting recovery"
          git sparse-checkout disable || true
          git fetch --no-tags origin "${GITHUB_SHA:-HEAD}"
          git reset --hard FETCH_HEAD
          git clean -ffd || true
        fi
        if [[ ! -f pyproject.toml ]]; then
          echo "::error::Repository checkout is incomplete (pyproject.toml missing)"
          echo "PWD=$(pwd)"
          ls -la
          exit 1
        fi
```

**Step 2: Update the 5 existing workflows to use the composite action**

Replace each inline "Verify checkout integrity" step with:
```yaml
- uses: ./.github/actions/checkout-integrity
```

In each of: `lint.yml`, `test.yml`, `smoke.yml`, `smoke-offline.yml`, `deploy-frontend.yml`.

**Step 3: Add the guard to all workflows that have `actions/checkout` but lack the guard**

Add `- uses: ./.github/actions/checkout-integrity` immediately after every `actions/checkout@v4` step in:

- `core-suites.yml` (2 jobs)
- `integration.yml` (4 jobs)
- `integration-gate.yml` (4 jobs)
- `migration-tests.yml` (1 job)
- `coverage.yml` (2 jobs)
- `quality-smoke.yml` (1 job: quality-smoke-run)
- `backup-verification.yml` (3 jobs)
- `benchmark.yml` (5 jobs)
- `benchmarks.yml` (1 job)
- `nightly-full-matrix.yml` (3 jobs)
- `build.yml` (2 jobs)
- `docker.yml` (4 jobs)
- `deploy-lightsail.yml` (1 job)
- `deploy-secure.yml` (4 jobs)
- `autopilot-worktree-e2e.yml` (2 jobs)
- `publish-aragora.yml` (2 jobs)
- `publish-aragora-debate.yml` (2 jobs)
- `publish-sdk-python.yml` (2 jobs)
- `sdk-parity.yml` (1 job: sdk-parity-run)
- `release.yml` (10 jobs)
- `release-readiness.yml` (1 job)
- `security.yml` (7 jobs)
- `security-gate.yml` (2 jobs)
- Template: `.github/workflow-templates/aragora-review-template.yml` (1 job)

**Step 4: Run lint on the YAML files**

```bash
python -c "import yaml; [yaml.safe_load(open(f)) for f in __import__('glob').glob('.github/workflows/*.yml')]"
```

**Step 5: Commit**

```bash
git add .github/actions/checkout-integrity/action.yml .github/workflows/*.yml .github/workflow-templates/*.yml
git commit -m "ci: add git-fetch recovery to checkout integrity guard across all workflows"
```

---

## Track B: Full Lockfile + Vulnerability Sweep

### Task B1: Generate Python lockfile with uv

**Files:**
- Create: `uv.lock`
- Modify: `.github/workflows/security-gate.yml` (add lockfile freshness check)

**Step 1: Generate the lockfile**

```bash
uv lock
```

**Step 2: Verify the lockfile is valid**

```bash
uv lock --check
```

**Step 3: Add CI validation step to security-gate.yml**

Add after the existing python-security job:
```yaml
- name: Verify lockfile is up-to-date
  run: |
    pip install uv
    uv lock --check || { echo "::error::uv.lock is stale. Run 'uv lock' locally."; exit 1; }
```

**Step 4: Commit**

```bash
git add uv.lock .github/workflows/security-gate.yml
git commit -m "security: add Python lockfile via uv lock with CI freshness check"
```

### Task B2: Upgrade JS dependencies

**Files:**
- Modify: `aragora/live/package.json`
- Modify: `aragora/live/package-lock.json`

**Step 1: Run npm update in the frontend**

```bash
cd aragora/live && npm update && npm audit fix
```

**Step 2: Verify build still works**

```bash
cd aragora/live && npm run build:ci
```

**Step 3: Check remaining audit issues**

```bash
cd aragora/live && npm audit --omit=dev --audit-level=high
```

**Step 4: Commit**

```bash
git add aragora/live/package.json aragora/live/package-lock.json
git commit -m "security: upgrade JS dependencies to resolve known vulnerabilities"
```

### Task B3: Pin vulnerable Python transitives

**Step 1: Run pip-audit to identify high-severity issues**

```bash
pip-audit --strict --ignore-vuln CVE-2025-14009 2>&1 | head -50
```

**Step 2: For each high-severity finding, add explicit floor pin to pyproject.toml**

Follow existing pattern:
```toml
"package>=X.Y.Z",  # CVE-YYYY-NNNNN
```

**Step 3: Regenerate lockfile**

```bash
uv lock
```

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "security: pin vulnerable Python transitives with CVE annotations"
```

### Task B4: Update SECURITY.md and harden SBOM gate

**Files:**
- Modify: `SECURITY.md` (version table)
- Modify: `.github/workflows/sbom.yml` (remove `|| true`)

**Step 1: Update SECURITY.md supported versions**

Change `2.6.x` → `2.8.x` as current supported version. Shift others down.

**Step 2: Remove `|| true` from Grype scan in sbom.yml**

Change:
```yaml
grype sbom:$sbom --fail-on critical || true
```
To:
```yaml
grype sbom:$sbom --fail-on critical
```

**Step 3: Commit**

```bash
git add SECURITY.md .github/workflows/sbom.yml
git commit -m "security: update supported versions and make SBOM critical gate blocking"
```

---

## Track C: Full Developer Onramp

### Task C1: Create canonical docs/quickstart.md

**Files:**
- Create: `docs/quickstart.md`
- Modify: `README.md` (update quickstart link)

**Step 1: Write docs/quickstart.md**

Consolidate the best content from `docs/START_HERE.md`, `docs/QUICKSTART_CLI.md`,
`docs/guides/QUICKSTART.md`, `docs/SDK_QUICKSTART.md` into one canonical guide.

Structure:
1. Install (pip install aragora — 30 seconds)
2. Zero-key demo (aragora demo — no API keys needed)
3. Three-line debate (Python):
   ```python
   from aragora import Arena, Environment, DebateProtocol
   result = await Arena(Environment(task="..."), agents, DebateProtocol()).run()
   print(result.summary)
   ```
4. API key setup (ANTHROPIC_API_KEY or OPENAI_API_KEY)
5. Real debate with live agents
6. Next steps: CLI guide, SDK guide, self-hosting

**Step 2: Update README.md quickstart link**

Point the "Get Started" section to `docs/quickstart.md`.

**Step 3: Commit**

```bash
git add docs/quickstart.md README.md
git commit -m "docs: create canonical quickstart.md consolidating onramp guides"
```

### Task C2: Add examples/quickstart.ts

**Files:**
- Create: `examples/quickstart.ts`

**Step 1: Write the TypeScript quickstart**

Match the pattern of `examples/quickstart.py` but using `@aragora/sdk`:

```typescript
import { AragoraClient } from "@aragora/sdk";

const client = new AragoraClient({ baseUrl: "http://localhost:8080" });
const result = await client.debates.create({
  task: "Should we use microservices or a monolith?",
  agents: ["claude", "openai"],
  rounds: 3,
});
console.log(result.summary);
```

**Step 2: Commit**

```bash
git add examples/quickstart.ts
git commit -m "docs: add TypeScript quickstart example"
```

### Task C3: Verify and fix /docs and /redoc endpoints

**Files:**
- Modify: `aragora/server/unified_server.py` (if needed)

**Step 1: Check if FastAPI /docs and /redoc are enabled**

```bash
grep -n "docs_url\|redoc_url\|openapi_url" aragora/server/unified_server.py
```

**Step 2: If disabled, re-enable them**

FastAPI defaults: `docs_url="/docs"`, `redoc_url="/redoc"`. If explicitly set to None, change back.

**Step 3: Verify by starting server and checking endpoints**

```bash
aragora serve --api-port 8080 &
sleep 5
curl -s http://localhost:8080/docs | head -5
curl -s http://localhost:8080/redoc | head -5
kill %1
```

**Step 4: Commit if changes needed**

```bash
git add aragora/server/unified_server.py
git commit -m "fix(server): enable /docs and /redoc API documentation endpoints"
```

### Task C4: Add /docs page to frontend (Redoc embed)

**Files:**
- Create: `aragora/live/src/app/(standalone)/docs/page.tsx`

**Step 1: Create the docs page**

```tsx
"use client";

import dynamic from "next/dynamic";

const RedocStandalone = dynamic(
  () => import("redoc").then((mod) => mod.RedocStandalone),
  { ssr: false, loading: () => <div>Loading API docs...</div> }
);

export default function DocsPage() {
  const specUrl = process.env.NEXT_PUBLIC_API_URL
    ? `${process.env.NEXT_PUBLIC_API_URL}/openapi.json`
    : "/api/openapi.json";

  return <RedocStandalone specUrl={specUrl} />;
}
```

**Step 2: Install redoc dependency**

```bash
cd aragora/live && npm install redoc styled-components
```

**Step 3: Verify build**

```bash
cd aragora/live && npm run build:ci
```

**Step 4: Commit**

```bash
git add aragora/live/src/app/\(standalone\)/docs/page.tsx aragora/live/package.json aragora/live/package-lock.json
git commit -m "feat(live): add /docs page with embedded Redoc API documentation"
```

### Task C5: Build Interactive API Playground

**Files:**
- Create: `aragora/live/src/components/playground/RequestBuilder.tsx`
- Create: `aragora/live/src/components/playground/ResponseViewer.tsx`
- Create: `aragora/live/src/components/playground/WebSocketViewer.tsx`
- Create: `aragora/live/src/components/playground/EndpointSelector.tsx`
- Create: `aragora/live/src/components/playground/index.ts`
- Modify: `aragora/live/src/app/(standalone)/playground/page.tsx` (already exists — enhance)

**Step 1: Check existing playground page**

Read `aragora/live/src/app/(standalone)/playground/page.tsx` and
`aragora/live/src/components/Playground.tsx` (standalone component) to understand current state.

**Step 2: Build RequestBuilder component**

Features: method selector (GET/POST/PUT/DELETE), URL input with endpoint autocomplete,
headers editor, JSON body editor with syntax highlighting, "Send" button.

**Step 3: Build ResponseViewer component**

Features: status code badge, response time, headers accordion, JSON body with
syntax highlighting and collapsible sections.

**Step 4: Build WebSocketViewer component**

Features: connect/disconnect toggle, real-time event log with timestamps,
filter by event type, auto-scroll.

**Step 5: Build EndpointSelector component**

Features: grouped by API domain (debates, agents, knowledge, etc.), search/filter,
shows method + path + description, click to populate RequestBuilder.

**Step 6: Wire into playground page**

Update `aragora/live/src/app/(standalone)/playground/page.tsx` to use the new components
in a split-pane layout: left = EndpointSelector, top-right = RequestBuilder,
bottom-right = ResponseViewer/WebSocketViewer (tabbed).

**Step 7: Add rate limiting to sandbox requests**

In the playground page, add client-side rate limiting (max 10 req/min) with visual
feedback. Server-side rate limiting already exists via the unified server.

**Step 8: Verify build + write tests**

```bash
cd aragora/live && npm run build:ci && npm test -- --testPathPattern=playground
```

**Step 9: Commit**

```bash
git add aragora/live/src/components/playground/ aragora/live/src/app/\(standalone\)/playground/
git commit -m "feat(live): interactive API playground with request builder and WS viewer"
```

### Task C6: SDK Parity Push

**Files:**
- Modify: `sdk/typescript/src/` (add missing namespaces)
- Modify: `sdk/python/aragora_sdk/namespaces/` (add missing methods)

**Step 1: Generate current parity report**

```bash
python scripts/generate_sdk_types.py --check 2>&1 | head -100
```

Or run the SDK parity workflow locally:
```bash
python scripts/sdk_parity_check.py 2>&1 | tail -50
```

**Step 2: Identify core capability gaps**

Focus on: debates, agents, knowledge, memory, analytics, audit, billing, compliance,
workflows — the "core" namespaces that must be at 100% parity.

**Step 3: Add missing TypeScript namespace methods**

For each missing method, add to the appropriate namespace in `sdk/typescript/src/namespaces/`.

**Step 4: Add missing Python SDK methods**

For each missing method, add to `sdk/python/aragora_sdk/namespaces/`.

**Step 5: Run parity check to verify**

```bash
python scripts/sdk_parity_check.py 2>&1 | grep -E "missing|stale"
```

**Step 6: Commit**

```bash
git add sdk/typescript/src/ sdk/python/aragora_sdk/
git commit -m "feat(sdk): close core capability parity gaps in Python and TypeScript SDKs"
```

### Task C7: Integration Testing CI Workflow

**Files:**
- Create: `.github/workflows/onramp-integration.yml`

**Step 1: Write the workflow**

```yaml
name: Developer Onramp Integration
on:
  push:
    branches: [main]
  pull_request:
    paths:
      - "docs/quickstart.md"
      - "examples/quickstart.*"
      - "aragora/live/src/app/**/playground/**"
      - "aragora/live/src/app/**/docs/**"
      - "sdk/**"

jobs:
  time-to-first-debate:
    runs-on: aragora
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/checkout-integrity
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install and run quickstart
        run: |
          pip install -e .
          timeout 300 python examples/quickstart.py
        env:
          ARAGORA_OFFLINE: "1"

  playground-response:
    runs-on: aragora
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/checkout-integrity
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
      - name: Build and test playground
        run: |
          cd aragora/live
          npm ci
          npm run build:ci

  docs-page-load:
    runs-on: aragora
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/checkout-integrity
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Verify /docs endpoint
        run: |
          pip install -e .
          aragora serve --api-port 8080 &
          sleep 10
          START=$(date +%s%N)
          curl -sf http://localhost:8080/docs > /dev/null
          END=$(date +%s%N)
          ELAPSED=$(( (END - START) / 1000000 ))
          echo "Docs load time: ${ELAPSED}ms"
          if [ "$ELAPSED" -gt 3000 ]; then
            echo "::error::Docs page load exceeded 3s p95 target (${ELAPSED}ms)"
            exit 1
          fi
          kill %1
```

**Step 2: Commit**

```bash
git add .github/workflows/onramp-integration.yml
git commit -m "ci: add developer onramp integration tests (time-to-debate, docs load)"
```

---

## Execution Order

**Parallel tracks (can be run simultaneously):**
- Track A (A1): Runner fix — 1 task, ~30 min
- Track B (B1-B4): Vulnerability sweep — 4 tasks, ~1 hour
- Track C (C1-C7): Developer onramp — 7 tasks, ~2-3 hours

**Dependencies:**
- C7 depends on A1 (uses the composite action)
- B3 depends on B1 (needs lockfile to exist)
- C5 has no external dependencies (existing playground page)
- C6 is independent (SDK work)

**Recommended execution:** Run A1 first (unblocks everything), then B1-B4 and C1-C7 in parallel.
