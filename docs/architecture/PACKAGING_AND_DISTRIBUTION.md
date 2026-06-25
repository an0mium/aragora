# Packaging and Distribution Design (P3)

Authoritative design record for the P3 packaging milestone (HEALTH-6,
[#8263](https://github.com/synaptent/aragora/issues/8263)). This is a
**post-hoc** design doc: the decision-gated rename already shipped on `main` via
PR #8517, so this records what shipped, why, the traced dependency audit, and
the remaining packaging follow-ups. The decision receipt for the dogfood gate is
[`../audits/2026-06-22-p3-packaging-decision-receipt.md`](../audits/2026-06-22-p3-packaging-decision-receipt.md).

## 1. Decision summary

The root distribution is **Option A** from #8263: one `aragora` distribution
shipping the full package and the console entry points, while keeping the
standalone wedges (`aragora-debate/`, and ODR-3's `aragora-verify`) separable.
See the decision receipt for the dogfood debate that gated this (verdict PASS,
heterogeneous quorum grok/mistral-api/deepseek).

## 2. Distribution rename: `aragora-debate` → `aragora`

| Aspect | Before (baseline `a5cf5fc70b`) | After (shipped, #8517) |
|---|---|---|
| `[project].name` | `aragora-debate` | `aragora` |
| Console scripts | none | `aragora = aragora.cli.main:main` |
| Packaged modules | `aragora`, `aragora.core`, `aragora.debate` (3) | full tree via `include = ["aragora*"]` |
| Wedge | `aragora-debate/` standalone | unchanged standalone wedge |

The single name resolves the prior three-name contradiction (`aragora-debate`
build / `aragora` README badge / `aragora-sdk` SDK) called out in #8263.

## 3. Package list strategy: auto-discovery

The build uses setuptools **auto-discovery**, not an explicit list:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["aragora*"]
exclude = ["aragora-debate*", "tests*", "docs*"]
```

Rationale: the tree has ~141 top-level subpackages (`aragora/*/__init__.py`);
an explicit list rots on every new subpackage. `include = ["aragora*"]` captures
`aragora.cli` and `aragora.server` (excluded before, which is precisely why the
documented `aragora serve` path was impossible pre-P3). The `exclude` keeps the
`aragora-debate/` wedge, `tests/`, and `docs/` out of the wheel.

## 4. Dependency audit (traced from actual imports)

Method: count distinct files under `aragora/` with an **anchored import** of
each module's import name, using `rg -l '^\s*(import|from)\s+<mod>\b' aragora/`
(here `<mod>` is the import name, e.g. `yaml` for PyYAML). Anchoring to the start
of an `import` / `from` statement counts genuine import sites and excludes
incidental mentions in comments or strings, so a broad `rg -l <mod> aragora/`
reports materially higher (e.g. ~144 for aiohttp, ~15 for websockets). The counts
below are a point-in-time anchored-import trace (re-running the command above
reproduces them; they drift as the tree changes), cross-referenced against the
empirically bootable runtime list in
`scripts/ci_install_project.sh` (`LEGACY_CONTROL_PLANE_BASE_DEPS`, the floor both
Dockerfiles boot from).

| Dependency | Anchored import sites under `aragora/` | Genuinely needed by | In base `[project.dependencies]`? | Status |
|---|---:|---|---|---|
| `pydantic` | 33 | config, core, server, cli | yes (`>=2.13.4,<3.0`) | floor ✓ |
| `PyYAML` (`yaml`) | 49 | config, hooks, templates | yes (`>=6.0.3,<7.0`) | floor ✓ |
| `aiohttp` | 98 | server handlers, agents, connectors | yes (`>=3.14.1,<4.0`, #8554) | floor ✓ |
| `websockets` | 7 | `aragora/server/stream/*` | yes (`>=13.0,<15.1`, #8554) | floor ✓ |
| `pydantic-settings` | 1 | config | yes (`>=2.14.2,<3.0`) | floor ✓ + CVE floor |
| `click` / `typer` / `httpx` / `python-dateutil` | n/a | cli entry surface | yes | CLI floor ✓ |

**Audited floor = {aiohttp, websockets, pyyaml, pydantic}** plus the CLI
essentials (click, typer, httpx, python-dateutil) and `pydantic-settings`.

### Finding (coordination, not fixed here)

`#8554` folded `aiohttp` (`>=3.14.1,<4.0`) and `websockets` (`>=13.0,<15.1`) into
the base `[project.dependencies]` floor, so the four audited deps (aiohttp +
websockets + pyyaml + pydantic) now satisfy the VAL-P3-003 expectation that they
appear in the declared dependencies. A wider gap remains: the declared base floor
is still narrower than the empirically bootable `LEGACY_CONTROL_PLANE_BASE_DEPS`
set in `scripts/ci_install_project.sh` (the authoritative boot list), so a base
install can still miss runtime deps that only the extras or that script provide.

Recommendation for the pyproject-owning P3 follow-ups (`p3-verification-suite` /
`p3-deploy-finalize`): reconcile the declared base floor against
`LEGACY_CONTROL_PLANE_BASE_DEPS` (or a dedicated `[server]` extra) so a documented
install is bootable without pulling `[all]`.

> This design doc does not edit `pyproject.toml`; that file is owned by other
> P3 features and is path-frozen by several open PRs.

## 5. Extras layout (shipped)

| Extra | Purpose | Key members |
|---|---|---|
| `test` | acceptance install | build, pytest, pytest-asyncio, twine, pydantic(-settings), PyYAML |
| `gateway` | HTTP/API surface | fastapi, uvicorn |
| `blockchain` | ERC-8004 identity | aiohttp, web3 |
| `enterprise` | SSO / managed PG | asyncpg, python3-saml, supabase |
| `connectors` | streaming ingest | aiokafka, aio-pika |
| `experimental` | browser automation | playwright |
| `dev` | type/lint tooling | mypy, mypy-baseline, types-*, bandit |
| `all` | union of runtime extras | gateway ∪ blockchain ∪ enterprise ∪ connectors ∪ experimental |

## 6. Console entry point

```toml
[project.scripts]
aragora = "aragora.cli.main:main"
```

After `pip install -e ".[test]"`, `aragora --help` exposes the full subcommand
surface (`ask`, `serve`, `quickstart`, `gauntlet`, `receipt`, ...).

## 7. Wedges and ODR-3 coordination

- **`aragora-debate/`** — unchanged standalone wedge; its `[project].name`
  stays `aragora-debate`. The name collision is resolved by renaming only the
  root distribution.
- **`aragora-verify` (ODR-3, #8226)** — a separate near-zero-dependency PyPI
  package (`packages/aragora-verify/`) for offline receipt verification. The
  root `aragora` distribution must **not** absorb it: the verifier's value is
  that it is trustworthy *outside* an Aragora install. ODR-3 leads on
  receipt-surface naming. The dependency audit above is shared on #8226.

## 8. Security floors (already on main)

- `pydantic-settings>=2.14.2` closes **GHSA-4xgf-cpjx-pc3j** (the sole open CVE in
  a directly-declared dependency; the vulnerable code path has 0 in-tree usages).
  Declared in base + `[test]`.
- `cryptography>=48.0.1` and `starlette>=1.3.1` are floored via
  `[tool.uv].constraint-dependencies`. Already remediated — do not re-fix.

## 9. Definition of done (from #8263) — status

- [x] Decision receipt attached (dogfood gate) — see the receipt doc.
- [x] Option implemented — Option A shipped via #8517.
- [ ] Clean-machine `pip install` → CLI → zero-key receipt (owned by `p3-verification-suite`).
- [ ] `INSTALL.md`/README/quickstart agree with reality (HEALTH-5 / P2 + P3 follow-ups).
