# Public Utility Mission Gated-File Audit

**Audit status:** **FAIL CLOSED under the strict file-level rule.**

This audit found one merged mission pull request whose GitHub file manifest includes a gated
file: [PR #8957](https://github.com/synaptent/aragora/pull/8957) includes
`docs/CANONICAL_GOALS.md`. The patch only refreshes generated metric values, and the eventual
squash merge is an effective no-op because equivalent content had already reached `main`, but
the requested rule is file-level: no mission pull request may modify the gated file. The audit
therefore does not report a clean result or silently convert the rule into a doctrine-prose-only
check.

All other strict path checks pass. The README top-line claim is byte-identical before and after
the mission and across both merged mission pull requests that touched `README.md`. The effective
squash-commit union has an empty intersection with the gated path set.

## 1. Snapshot and scope

| Item | Value |
|---|---|
| Audit date | 2026-07-11 |
| Pre-mission `origin/main` | `d780bd489808698ea20836f7b540f9301011f3c1` |
| Post-mission `origin/main` snapshot | `258fb97b821344e0f1e4fd597436108503e61759` |
| Post-mission merge | PR #9204, merged `2026-07-11T17:51:30Z` |
| Included pull requests | Every merged PR with a `factory/pum-` head through the post snapshot |
| Included PR count | 26 |
| Excluded work | Open or closed-unmerged mission PRs; unrelated PRs interleaved on `main` |

The pre-mission SHA is the baseline recorded by the mission's first committed baseline artifact,
[`PUBLIC_UTILITY_MISSION_BASELINE.md`](PUBLIC_UTILITY_MISSION_BASELINE.md). The post SHA is the
live `origin/main` snapshot at audit start. The deterministic PR inventory is bounded by the post
merge timestamp so rerunning the commands after this audit PR merges does not silently expand the
audited set.

### Gated paths

The full-file gates are:

```text
docs/THESIS.md
docs/CANONICAL_GOALS.md
docs/RECEIPT_CONTRACT.md
.github/workflows/**
aragora/gauntlet/odr_schema.json
aragora-verify/src/aragora_verify/odr_schema.json
```

`README.md` is not gated as a whole. Its gated surface is this top-line claim:

```markdown
**Aragora is an auditable execution control plane for AI-assisted decisions:
multi-model review in, a verifiable Decision Receipt out.**
```

## 2. Results

| Check | Result | Evidence |
|---|---|---|
| Per-PR GitHub file manifest, full-file gates | **FAIL** | 25/26 pass; PR #8957 lists `docs/CANONICAL_GOALS.md` |
| Union of GitHub PR file manifests | **FAIL** | Intersection is exactly `docs/CANONICAL_GOALS.md` |
| Per-PR effective squash-commit deltas | **PASS** | 26/26 effective deltas have no gated path |
| Union of effective squash-commit deltas | **PASS** | Empty intersection |
| README top-line claim, per README-touching PR | **PASS** | PRs #8955 and #8991 preserve the bytes |
| README top-line claim, pre/post snapshots | **PASS** | SHA-256 is `667fea0f1c04a36c5053fe521b0e77a36e43db54e8107a5944b1646294ddd122` at both endpoints |
| Both ODR schema copies, pre/post snapshots | **PASS** | Both remain byte-identical with SHA-256 `0d2a934981464d5835eb7d6651f8e6c21780e388100d0cd48828f0f75bb49411` |
| Strict overall result | **FAIL CLOSED** | A single-PR file-level violation is sufficient to fail the requested gate |

### Why the PR manifest and effective merge results differ

PR #8957 was explicitly prepared as an operator-reviewed mechanical metrics regeneration. Its
GitHub patch changed four generated counts inside the metrics block in
`docs/CANONICAL_GOALS.md`:

- Python files: `4,260` to `4,262`
- lines of code: `1,975,628` to `1,976,521`
- test functions: `223,079` to `223,198`
- test files: `5,433` to `5,437`

The PR carried the `operator-review-required` label and two independent PASS comments. Before it
merged, equivalent content had already reached `main`. Consequently:

```text
PR head docs/CANONICAL_GOALS.md SHA-256:
bf963fde7c7abb7c4af2133990c6cbffebd679a5326453ddc1cb1cf4faf05a77

Merge parent docs/CANONICAL_GOALS.md SHA-256:
bf963fde7c7abb7c4af2133990c6cbffebd679a5326453ddc1cb1cf4faf05a77

Squash merge docs/CANONICAL_GOALS.md SHA-256:
bf963fde7c7abb7c4af2133990c6cbffebd679a5326453ddc1cb1cf4faf05a77
```

The squash commit `b60637ae2a3984bbeb7a9c93507c3a5527bd9151` therefore has an empty effective
tree delta. That protects the final repository state, but it does not erase the PR-level gated-file
touch required by the single-PR audit.

### Endpoint hashes

| Gated artifact | Pre SHA-256 | Post SHA-256 | Equal |
|---|---|---|---|
| `docs/THESIS.md` | `c355f57ca164b965a9ecd6061adbe36586bb2bdc0de4fa2ddba96177cfa2bc68` | `c355f57ca164b965a9ecd6061adbe36586bb2bdc0de4fa2ddba96177cfa2bc68` | Yes |
| `docs/CANONICAL_GOALS.md` | `a12c57e30b79cb444920bffd0b87fcd6ea741a06e15ca750c8635ff05aef006b` | `af93890b8728ee812741211c7f11ec26906d107ec0d055b23b0902b674eb8f1c` | No |
| `docs/RECEIPT_CONTRACT.md` | `fc6e069a8eb5eb111ccaca4490ad50815a21f44b846ed3431f6437d4fb540ef3` | `fc6e069a8eb5eb111ccaca4490ad50815a21f44b846ed3431f6437d4fb540ef3` | Yes |
| `aragora/gauntlet/odr_schema.json` | `0d2a934981464d5835eb7d6651f8e6c21780e388100d0cd48828f0f75bb49411` | `0d2a934981464d5835eb7d6651f8e6c21780e388100d0cd48828f0f75bb49411` | Yes |
| `aragora-verify/src/aragora_verify/odr_schema.json` | `0d2a934981464d5835eb7d6651f8e6c21780e388100d0cd48828f0f75bb49411` | `0d2a934981464d5835eb7d6651f8e6c21780e388100d0cd48828f0f75bb49411` | Yes |

The `CANONICAL_GOALS.md` endpoint mismatch alone is not a mission attribution test. Unrelated PRs
were merged between the two snapshots, as were workflow changes. The mission-specific result comes
from the enumerated `factory/pum-*` PR manifests and effective commit deltas.

## 3. Per-PR ledger

`Manifest` checks the GitHub PR file list. `Effective` checks the final squash commit's tree delta.
`Top-line` is shown only for PRs that touched `README.md`.

| PR | Branch | Squash commit | Manifest | Effective | Top-line |
|---:|---|---|---|---|---|
| #8820 | `factory/pum-m2-receipt-reconciliation-doc` | `f2ba66b62499` | PASS | PASS | N/A |
| #8806 | `factory/pum-m0-baseline-map` | `19a85255c2f9` | PASS | PASS | N/A |
| #8829 | `factory/pum-m2-reconciliation-doc-pypi-claim-fix` | `2c6bfeda070d` | PASS | PASS | N/A |
| #8826 | `factory/pum-m2-odr-signed-and-chain-fixtures` | `d6158df3337e` | PASS | PASS | N/A |
| #8814 | `factory/pum-m1-pyproject-tagline-reframe` | `9e5aceec1596` | PASS | PASS | N/A |
| #8822 | `factory/pum-m2-odr-unsigned-state-fixtures` | `bfbbde4fa375` | PASS | PASS | N/A |
| #8832 | `factory/pum-m3-verifier-doc` | `d5ce2bb40178` | PASS | PASS | N/A |
| #8854 | `factory/pum-m3-verifier-hardening-tests` | `9001596cfc4c` | PASS | PASS | N/A |
| #8833 | `factory/pum-m3-pypi-status-corrections` | `535943bcf74d` | PASS | PASS | N/A |
| #8871 | `factory/pum-misc-odr-verify-docstring-honesty` | `5c37afdec1dc` | PASS | PASS | N/A |
| #8857 | `factory/pum-m3-receipt-verify-help-text` | `b014a6d52d1d` | PASS | PASS | N/A |
| #8870 | `factory/pum-misc-docsite-disaster-recovery-links` | `8531dfd43745` | PASS | PASS | N/A |
| #8953 | `factory/pum-misc-docsite-specs-mirror-boundary` | `8b600a3a8dbf` | PASS | PASS | N/A |
| #8955 | `factory/pum-m4-action-wedge-doc` | `7d4aebf68914` | PASS | PASS | PASS |
| #8957 | `factory/pum-misc-doc-stats-drift-fix` | `b60637ae2a39` | **FAIL: `docs/CANONICAL_GOALS.md`** | PASS, empty commit delta | N/A |
| #8967 | `factory/pum-m5-install-matrix-doc` | `92af0c8df50f` | PASS | PASS | N/A |
| #8958 | `factory/pum-misc-specs-mirror-test-globbing` | `06c7f0f1b56e` | PASS | PASS | N/A |
| #8964 | `factory/pum-misc-action-doc-parity` | `5821f54592e3` | PASS | PASS | N/A |
| #8985 | `factory/pum-m5-install-matrix-pypi-regression-fix` | `b0797e2bf3c2` | PASS | PASS | N/A |
| #8991 | `factory/pum-m6-canonical-verbs-and-landing` | `77301f104408` | PASS | PASS | PASS |
| #9003 | `factory/pum-m6-canonical-consistency-sweep` | `2543ae8b42f6` | PASS | PASS | N/A |
| #9001 | `factory/pum-m6-docs-archive-collapse` | `2caed5d61fa0` | PASS | PASS | N/A |
| #9059 | `factory/pum-m6-user-testing-fixes` | `99a7a5cdd7e3` | PASS | PASS | N/A |
| #9067 | `factory/pum-misc-verify-tests-ruff-format` | `b67c139db339` | PASS | PASS | N/A |
| #9193 | `factory/pum-misc-mirror-guard-source-parity` | `a0f0c4179162` | PASS | PASS | N/A |
| #9204 | `factory/pum-m8-dogfood-gate` | `258fb97b8213` | PASS | PASS | N/A |

## 4. Reproduction commands

Run from a checkout containing the post snapshot. These commands require authenticated `gh`, `git`,
`jq`, `awk`, `sed`, `cmp`, `sort`, and `shasum`.

### 4.1 Inventory the exact audited PR set

```bash
POST_MERGED_AT=2026-07-11T17:51:30Z

gh api --paginate \
  '/repos/synaptent/aragora/pulls?state=closed&per_page=100' \
  --jq '.[] |
    select(
      .merged_at != null and
      (.head.ref | startswith("factory/pum-")) and
      .merged_at <= "'"$POST_MERGED_AT"'"
    ) |
    [.merged_at, .number, .head.ref, .merge_commit_sha] | @tsv' |
  sort
```

Expected: 26 rows, from PR #8820 through PR #9204 by merge time.

### 4.2 Strict per-PR and aggregate manifest check

```bash
POST_MERGED_AT=2026-07-11T17:51:30Z
PR_LIST=$(
  gh api --paginate \
    '/repos/synaptent/aragora/pulls?state=closed&per_page=100' \
    --jq '.[] |
      select(
        .merged_at != null and
        (.head.ref | startswith("factory/pum-")) and
        .merged_at <= "'"$POST_MERGED_AT"'"
      ) |
      .number'
)

is_gated='
  $0 == "docs/THESIS.md" ||
  $0 == "docs/CANONICAL_GOALS.md" ||
  $0 == "docs/RECEIPT_CONTRACT.md" ||
  $0 == "aragora/gauntlet/odr_schema.json" ||
  $0 == "aragora-verify/src/aragora_verify/odr_schema.json" ||
  index($0, ".github/workflows/") == 1
'

for pr in $PR_LIST; do
  hits=$(
    gh api --paginate \
      "/repos/synaptent/aragora/pulls/$pr/files?per_page=100" \
      --jq '.[].filename' |
      awk "$is_gated"
  )
  if [ -z "$hits" ]; then
    printf 'PASS\tPR #%s\n' "$pr"
  else
    printf 'FAIL\tPR #%s\t%s\n' "$pr" "$(printf '%s' "$hits" | paste -sd, -)"
  fi
done

for pr in $PR_LIST; do
  gh api --paginate \
    "/repos/synaptent/aragora/pulls/$pr/files?per_page=100" \
    --jq '.[].filename'
done |
  sort -u |
  awk "$is_gated"
```

Expected aggregate output:

```text
docs/CANONICAL_GOALS.md
```

### 4.3 Effective squash-commit union check

```bash
POST_MERGED_AT=2026-07-11T17:51:30Z

gh api --paginate \
  '/repos/synaptent/aragora/pulls?state=closed&per_page=100' \
  --jq '.[] |
    select(
      .merged_at != null and
      (.head.ref | startswith("factory/pum-")) and
      .merged_at <= "'"$POST_MERGED_AT"'"
    ) |
    .merge_commit_sha' |
while read -r sha; do
  git diff-tree --no-commit-id --name-only -r "$sha"
done |
  sort -u |
  awk '
    $0 == "docs/THESIS.md" ||
    $0 == "docs/CANONICAL_GOALS.md" ||
    $0 == "docs/RECEIPT_CONTRACT.md" ||
    $0 == "aragora/gauntlet/odr_schema.json" ||
    $0 == "aragora-verify/src/aragora_verify/odr_schema.json" ||
    index($0, ".github/workflows/") == 1
  '
```

Expected: no output.

### 4.4 README top-line claim check

```bash
POST_MERGED_AT=2026-07-11T17:51:30Z

gh pr list \
  --repo synaptent/aragora \
  --state merged \
  --search 'head:factory/pum-' \
  --limit 400 \
  --json number,headRefName,mergedAt,mergeCommit,files |
jq -r \
  --arg cutoff "$POST_MERGED_AT" '
    .[] |
    select(.mergedAt <= $cutoff) |
    select(any(.files[]; .path == "README.md")) |
    [.number, .headRefName, .mergeCommit.oid] | @tsv
  ' |
while IFS=$'\t' read -r pr branch sha; do
  if cmp -s \
    <(git show "$sha^:README.md" | sed -n '3,4p') \
    <(git show "$sha:README.md" | sed -n '3,4p'); then
    printf 'PASS\tPR #%s\t%s\n' "$pr" "$branch"
  else
    printf 'FAIL\tPR #%s\t%s\n' "$pr" "$branch"
  fi
done

for ref in \
  d780bd489808698ea20836f7b540f9301011f3c1 \
  258fb97b821344e0f1e4fd597436108503e61759
do
  git show "$ref:README.md" | sed -n '3,4p' | shasum -a 256
done
```

Expected: PRs #8955 and #8991 pass; both endpoint hashes are
`667fea0f1c04a36c5053fe521b0e77a36e43db54e8107a5944b1646294ddd122`.

### 4.5 Raw pre/post range, diagnostic only

```bash
git diff --name-only \
  d780bd489808698ea20836f7b540f9301011f3c1..\
258fb97b821344e0f1e4fd597436108503e61759 |
awk '
  $0 == "docs/THESIS.md" ||
  $0 == "docs/CANONICAL_GOALS.md" ||
  $0 == "docs/RECEIPT_CONTRACT.md" ||
  $0 == "aragora/gauntlet/odr_schema.json" ||
  $0 == "aragora-verify/src/aragora_verify/odr_schema.json" ||
  index($0, ".github/workflows/") == 1
'
```

This intentionally reports interleaved, unrelated changes:

```text
.github/workflows/aragora-merge-quorum.yml
.github/workflows/deploy-frontend.yml
.github/workflows/deploy-secure.yml
.github/workflows/lint.yml
.github/workflows/openapi.yml
.github/workflows/sdk-test.yml
docs/CANONICAL_GOALS.md
```

The raw range is useful as a drift diagnostic, but it cannot attribute those paths to this mission.
The PR manifest and enumerated squash-commit checks provide that attribution.

## 5. Contract disposition

| Assertion | Disposition |
|---|---|
| No mission-merged PR modifies a gated file | **Not satisfied**, PR #8957 lists `docs/CANONICAL_GOALS.md` |
| Full merged mission PR manifest union intersects empty | **Not satisfied**, intersection contains `docs/CANONICAL_GOALS.md` |
| Effective merged squash-commit union intersects empty | Satisfied |
| README top-line claim unchanged | Satisfied |
| Audit artifact records commands and results | Satisfied by this document |
| Audit deliverable is docs-only | Satisfied by this audit branch |

Because the violating PR is already merged and its effective commit is empty, no repository edit can
retroactively make the strict historical assertion true. The operator can either adjudicate the
mechanical metrics-block touch as an explicit exception, or leave the strict assertion failed. A
future pre-merge guard should check the GitHub PR file manifest, not only the effective commit delta,
so an eventually-empty squash cannot hide a protected-path touch.
