# Bench-Readiness Pass

Purpose: validate that the minimum prerequisites for step 2 (the thesis
benchmark of multi-agent debate vs. single-model baselines) are in place
before investing in harness code.

This directory is intentionally self-contained. It does not modify any
source files outside `benchmarks/bench_readiness/`. Rerun after any
material change to the debate engine or CLI to catch regressions.

## Artifacts in this directory

| File | Role |
|---|---|
| `flag_ablation_smoke.py` | Probes each `enable_*` flag on the full Arena with demo agents. Expected to fail with timeouts on the current build — see `flag_ablation_smoke.json`. |
| `flag_ablation_smoke.json` | Output of the above. Records which flags timed out / crashed / toggled. |
| `standalone_debate_smoke.py` | Probes flags on the standalone `aragora-debate` package (the path the playground uses). |
| `standalone_debate_smoke.json` | Output of the above. Confirms `enable_trickster` and `enable_convergence` toggle observable behavior in milliseconds. |
| `cli_e2e_smoke.json` | Manual e2e CLI probe record: `demo`, `demo --offline`, `quickstart --demo`, `gauntlet --local`. |
| `write_manifest.py` | Generates the reproducibility manifest. |
| `manifest.json` | Reproducibility snapshot: git SHA, pinned packages, platform, model pins, verdict. |

## How to run

From the repo root with the rebuilt `.venv`:

```bash
.venv/bin/python -m benchmarks.bench_readiness.flag_ablation_smoke
.venv/bin/python benchmarks/bench_readiness/standalone_debate_smoke.py
.venv/bin/python benchmarks/bench_readiness/write_manifest.py
```

## Current verdict (2026-04-17 on `dda5581b`)

- **Tier-1 ablation (mechanics of debate):** ready against the standalone
  `aragora-debate` package. Flag toggles produce observable, sub-10-ms diffs
  in mock debate output.
- **Tier-2 end-to-end (full platform):** blocked. Six concrete blockers are
  listed in `manifest.json`, including stale API keys, a missing `google-generativeai` package, silent failure of `aragora gauntlet --local`, and the fact that the full Arena cannot run a deterministic 3-demo-agent debate in under 30 seconds (it pulls in network calls, pipeline stages, and an execution gate).

## Implication for the benchmark

The Tier-1 harness for the thesis benchmark (debate vs. single-model baseline on a bounded-software-task corpus) should build on `aragora_debate.Debate`, not the full Arena. The Arena can be reserved for Tier-2 end-to-end probes once the blockers are resolved.

This matches the ROADMAP's own "first booster" strategy: prove the minimal core first, then expand.
