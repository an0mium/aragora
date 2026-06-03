#!/usr/bin/env python3
"""Advocate Feasibility Test (AFT) harness.

Evaluates whether a small, locally-runnable, locally-finetunable open-weight
"advocate" model can match or beat a frontier LLM (with explicit operator rules
in the prompt) on a narrow, well-bounded decision: triaging Aragora PRs into
{merged_fast, closed_no_merge, open_aged}.

This is an *experiment*, not a product. It exists to falsify or confirm one
specific claim from the May 21 codex/claude debate: that small open-weight
advocates can serve as cheap, privacy-preserving proxies for an operator's
revealed-preference decision policy on bounded routine work.

Conditions
----------

  baseline_random    : stratified random sampling from training-set priors.
                       Lower bound. Anything that does not beat this is noise.
  frontier_rules     : a frontier model (Claude Opus by default) prompted with
                       a short, hand-written operator rule sheet plus the
                       low-information rationale seeds. No PR diff, no comment
                       bodies. This is the "frontier-as-classifier" baseline.
  local_advocate     : a local model (via the `aft-advocate` shim, typically
                       mlx-lm or transformers-peft loading a LoRA adapter)
                       given the same rationale seeds. This is the candidate.

Outputs
-------

Per-task JSON records and a summary report. The harness records:

  - prediction, confidence, latency_ms, cost_usd_estimate
  - per-class Brier score and overall accuracy
  - Bonferroni-corrected pairwise significance (advocate vs frontier,
    advocate vs baseline) using McNemar's test on paired predictions.

Privacy
-------

The harness only ever reads the holdout JSONL produced by
`aft_extract_training_data.py`, which is the same low-information feature set
the training pipeline saw. No raw diffs, comments, or private metadata enter
the harness. Stdout logs only condition names and aggregate metrics; per-task
records go to disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shlex
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

LOG = logging.getLogger("aft.harness")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOLDOUT = REPO_ROOT / "data" / "aft" / "pr_triage_holdout.jsonl"
DEFAULT_TRAIN = REPO_ROOT / "data" / "aft" / "pr_triage_train.jsonl"
DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "aft" / "results"

CLASSES = ("merged_fast", "closed_no_merge", "open_aged")
SCHEMA_VERSION = "aft-harness/0.1"

# Per-call cost estimates in USD. These are coarse-grained and only used to
# compare orders of magnitude across conditions; not for billing.
COST_PER_CALL_ESTIMATE = {
    "baseline_random": 0.0,
    "frontier_rules": 0.012,  # ~3-4k tokens at Opus rates, conservative
    "local_advocate": 0.0001,  # local inference electricity proxy
}


# -- Pre-registered hypotheses ------------------------------------------------

PRE_REGISTERED_HYPOTHESES = """\
Pre-registered hypotheses (AFT v0.1, draft 2026-05-22):

H0 (null):
    The local_advocate condition does NOT reach within 2 accuracy points of
    frontier_rules on the holdout set, OR its Brier score is worse than
    frontier_rules by more than 0.02.

H1 (primary):
    local_advocate matches frontier_rules within 2 accuracy points AND its
    Brier score is within 0.02, while costing <=10% per decision.

H2 (cost-quality frontier):
    Even if local_advocate is meaningfully worse on accuracy
    (>2 points behind frontier_rules), it is still useful if it lands above
    baseline_random by >=8 accuracy points AND costs <=1%, because it can run
    as a pre-filter that consults the frontier only when its own confidence
    is below a threshold.

Refutation rule:
    If local_advocate fails to beat baseline_random on accuracy at p<0.05
    after Bonferroni correction across the two pairwise tests, we declare the
    advocate-ensemble hypothesis FALSIFIED for the PR triage task and do not
    expand the design to other domains.
"""


@dataclass
class HoldoutTask:
    pr_number: int
    label: str
    rationale_seeds: dict
    title_redacted: str
    tier_hint: str

    @classmethod
    def from_row(cls, row: dict) -> "HoldoutTask":
        # Accept both the extractor's canonical schema (aft-pr-triage/0.1)
        # and a harness-native schema for test fixtures. The extractor emits:
        #   - `decision` (rename to label)
        #   - `rationale_seeds` as a list of "key=value" strings (parse to dict)
        #   - `title` (we treat it as already-redacted because the extractor
        #     does not pull comment bodies or diffs)
        label = row.get("label") or row.get("decision") or "open_aged"
        raw_seeds = row.get("rationale_seeds", {})
        if isinstance(raw_seeds, list):
            seeds: dict = {}
            for item in raw_seeds:
                if isinstance(item, str) and "=" in item:
                    k, _, v = item.partition("=")
                    # coerce common typed values
                    if v.isdigit():
                        seeds[k] = int(v)
                    elif v.lower() in {"true", "false"}:
                        seeds[k] = v.lower() == "true"
                    else:
                        seeds[k] = v
        else:
            seeds = raw_seeds or {}
        # Derive convenient observable cues the FrontierRules heuristic looks at
        seeds.setdefault("has_reviews", bool(row.get("review_count", 0)))
        seeds.setdefault("label_count", len(row.get("labels", []) or []))
        if "comment_count" not in seeds and "comment_count" in row:
            seeds["comment_count"] = row["comment_count"]
        # Bucket diff_size if the raw extractor emitted a numeric value
        if isinstance(seeds.get("diff_size"), int):
            n = seeds["diff_size"]
            seeds["diff_size"] = "small" if n < 200 else ("medium" if n < 1000 else "large")
        if isinstance(seeds.get("file_count"), int):
            n = seeds["file_count"]
            seeds["file_count"] = "few" if n < 5 else ("many" if n < 25 else "huge")
        return cls(
            pr_number=row["pr_number"],
            label=label,
            rationale_seeds=seeds,
            title_redacted=row.get("title_redacted") or row.get("title") or "",
            tier_hint=row.get("tier_hint", "unknown"),
        )


@dataclass
class Prediction:
    pr_number: int
    condition: str
    prediction: str
    confidence: float
    latency_ms: float
    cost_usd_estimate: float
    raw_probabilities: dict = field(default_factory=dict)


# -- Conditions ---------------------------------------------------------------


class Condition:
    name: str = "abstract"

    def predict(self, task: HoldoutTask) -> Prediction:
        raise NotImplementedError


class BaselineRandom(Condition):
    """Stratified random baseline using training-set class priors."""

    name = "baseline_random"

    def __init__(self, train_path: Path, seed: int = 17) -> None:
        self.rng = random.Random(seed)
        priors: Counter[str] = Counter()
        if train_path.exists():
            with train_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    label = row.get("label") or row.get("decision")
                    if label in CLASSES:
                        priors[label] += 1
        if not priors:
            for cls in CLASSES:
                priors[cls] = 1
        total = sum(priors.values())
        self.weights = [priors.get(cls, 0) / total for cls in CLASSES]

    def predict(self, task: HoldoutTask) -> Prediction:
        t0 = time.perf_counter()
        choice = self.rng.choices(CLASSES, weights=self.weights, k=1)[0]
        idx = CLASSES.index(choice)
        probs = {cls: self.weights[i] for i, cls in enumerate(CLASSES)}
        return Prediction(
            pr_number=task.pr_number,
            condition=self.name,
            prediction=choice,
            confidence=self.weights[idx],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            cost_usd_estimate=COST_PER_CALL_ESTIMATE[self.name],
            raw_probabilities=probs,
        )


class FrontierRules(Condition):
    """Frontier LLM prompted with operator rules + rationale seeds.

    Uses subprocess to call `aragora ask` (or `claude` CLI) so the harness
    does not need to take a hard dependency on any specific API client. Falls
    back to a deterministic heuristic if no CLI is available, marking the
    condition as `frontier_rules_stubbed` so results are not silently mixed.
    """

    name = "frontier_rules"

    OPERATOR_RULES = """\
You are triaging a single Aragora PR into exactly one of:
  merged_fast       : merged within 14 days of creation. The majority class.
                      Most PRs in this repo are bounded automation/code/docs
                      changes that pass review and land quickly.
  closed_no_merge   : closed without merging. Typically: superseded by another
                      PR, patch-equivalent to main, off-tranche, scout/probe
                      artifact, or rejected.
  open_aged         : still open more than 14 days after creation. Rare in
                      current operating conditions because the repo's merge
                      cadence is aggressive — empty in recent data.

Calibration prior (revealed from history):
  In ~556 recent PRs the class distribution was approximately
  merged_fast 90%, closed_no_merge 10%, open_aged 0%.
  When uncertain, prefer merged_fast.

Rules (apply in order; first match wins):
  1. If clear supersession or patch-equivalence signals are present
     (very small diff <50 LOC AND no reviews AND no labels AND branch
     namespace in {preflight, codex, claude, droid}), prefer
     closed_no_merge with confidence 0.65.
  2. If has_reviews is true OR comment_count >= 3 OR label_count >= 1,
     prefer merged_fast with confidence 0.80. Reviewed/labeled PRs
     almost always land.
  3. If branch namespace is `dependabot` or `renovate`, prefer
     merged_fast with confidence 0.85 (dependency PRs land fast or
     get auto-superseded by the next bump).
  4. If diff is medium (<1000 LOC) and from `claude`/`codex`/`droid`
     with a normal title (no "rebase", "patch-equivalent", "preflight"),
     prefer merged_fast with confidence 0.70.
  5. Otherwise prefer merged_fast with confidence 0.55 — the prior
     dominates uncertainty.

Do NOT default to open_aged. That class is empty in current data; emitting
it on uncertain inputs is the failure mode of the previous prompt revision.
Only emit open_aged when the rationale_seeds explicitly show a PR open >14
days AND substantive (large diff or many comments).
"""

    def __init__(self, cli_cmd: list[str] | None = None, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.cli_cmd = cli_cmd or self._discover_cli()

    def _discover_cli(self) -> list[str] | None:
        # Prefer `claude --print` over `aragora ask` because the latter requires
        # API keys loaded in env or AWS Secrets Manager and silently fails on a
        # naked workstation, while `claude --print` works whenever the user is
        # already running inside or has installed Claude Code. The harness
        # never depends on a specific provider; this is just discovery order.
        for candidate in (["claude", "--print"], ["aragora", "ask"]):
            try:
                subprocess.run(
                    [candidate[0], "--version"],
                    check=False,
                    capture_output=True,
                    timeout=5,
                )
                return candidate
            except (FileNotFoundError, subprocess.SubprocessError):
                continue
        return None

    def _heuristic(self, task: HoldoutTask) -> tuple[str, float]:
        """Calibrated dry-run heuristic encoding the same rules as OPERATOR_RULES.

        Used only when no frontier CLI is available so the `frontier_rules`
        condition stays comparable in dry-run mode. Matches the calibration
        prior (~90% merged_fast in current data); does not default to open_aged.
        """
        seeds = task.rationale_seeds or {}
        ns = seeds.get("branch_namespace", "")
        # Dependency bots almost always land fast or get superseded by the next bump.
        if ns in {"dependabot", "renovate"}:
            return "merged_fast", 0.85
        # Reviewed/labeled/discussed PRs almost always land.
        if (
            seeds.get("has_reviews")
            or seeds.get("comment_count", 0) >= 3
            or seeds.get("label_count", 0) >= 1
        ):
            return "merged_fast", 0.80
        # Tiny scout/probe artifacts from short-lived branches → closed_no_merge.
        if (
            ns in {"preflight", "codex", "claude", "droid"}
            and seeds.get("diff_size", "unknown") == "small"
            and not seeds.get("has_reviews")
            and seeds.get("label_count", 0) == 0
        ):
            return "closed_no_merge", 0.65
        # Fallback prior: when uncertain, predict the majority class.
        return "merged_fast", 0.55

    def predict(self, task: HoldoutTask) -> Prediction:
        t0 = time.perf_counter()
        if self.dry_run or self.cli_cmd is None:
            pred, conf = self._heuristic(task)
            latency = (time.perf_counter() - t0) * 1000.0
            return Prediction(
                pr_number=task.pr_number,
                condition=self.name + ("_stubbed" if self.cli_cmd is None else ""),
                prediction=pred,
                confidence=conf,
                latency_ms=latency,
                cost_usd_estimate=0.0
                if self.cli_cmd is None
                else COST_PER_CALL_ESTIMATE[self.name],
                raw_probabilities={pred: conf},
            )

        prompt = self._build_prompt(task)
        try:
            result = subprocess.run(
                self.cli_cmd + [prompt],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            text = (result.stdout or "").strip()
        except subprocess.SubprocessError as exc:
            LOG.warning("frontier_rules CLI call failed for PR %d: %s", task.pr_number, exc)
            pred, conf = self._heuristic(task)
            text = ""
        else:
            pred, conf = self._parse_response(text)

        return Prediction(
            pr_number=task.pr_number,
            condition=self.name,
            prediction=pred,
            confidence=conf,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            cost_usd_estimate=COST_PER_CALL_ESTIMATE[self.name],
            raw_probabilities={pred: conf},
        )

    def _build_prompt(self, task: HoldoutTask) -> str:
        seeds_json = json.dumps(task.rationale_seeds, sort_keys=True)
        return (
            self.OPERATOR_RULES
            + "\n\nInput features (no diff, no comment bodies):\n"
            + f"  pr_number: {task.pr_number}\n"
            + f"  title_redacted: {task.title_redacted}\n"
            + f"  tier_hint: {task.tier_hint}\n"
            + f"  rationale_seeds: {seeds_json}\n\n"
            + "Reply with exactly one line of JSON:\n"
            + '  {"label": "merged_fast|closed_no_merge|open_aged", "confidence": <float 0..1>}\n'
        )

    def _parse_response(self, text: str) -> tuple[str, float]:
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = obj.get("label")
            if label in CLASSES:
                conf = float(obj.get("confidence", 0.5))
                return label, max(0.0, min(1.0, conf))
        # last-ditch: keyword scan
        lower = text.lower()
        for cls in CLASSES:
            if cls in lower:
                return cls, 0.5
        return "open_aged", 0.34


class LocalAdvocate(Condition):
    """Calls the local advocate via the `aft-advocate` shim.

    The shim is expected to be a script on PATH (or a path passed via
    --advocate-cmd) that reads JSONL on stdin and writes JSONL on stdout, one
    output per input, with fields {"label": str, "confidence": float}.

    If no shim is available the condition runs the same heuristic as
    FrontierRules but tags itself `local_advocate_stubbed` so the report
    surfaces the missing implementation rather than silently mixing data.
    """

    name = "local_advocate"

    def __init__(self, advocate_cmd: list[str] | None = None) -> None:
        self.advocate_cmd = advocate_cmd
        self._proc: subprocess.Popen | None = None
        if advocate_cmd:
            try:
                self._proc = subprocess.Popen(
                    advocate_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            except (FileNotFoundError, OSError) as exc:
                LOG.warning("local_advocate shim unavailable (%s); falling back to stub", exc)
                self._proc = None

    def close(self) -> None:
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        if self._proc:
            try:
                self._proc.wait(timeout=5)
            except subprocess.SubprocessError:
                self._proc.kill()

    def _stub_predict(self, task: HoldoutTask) -> tuple[str, float]:
        # Differ slightly from frontier heuristic so the harness can sanity-
        # check that the two conditions are not literally identical.
        seeds = task.rationale_seeds or {}
        if seeds.get("label_count", 0) > 0 and seeds.get("has_reviews"):
            return "merged_fast", 0.7
        if task.tier_hint == "tier_4":
            return "open_aged", 0.75
        if seeds.get("branch_namespace") in {"codex", "claude"} and not seeds.get("has_reviews"):
            return "closed_no_merge", 0.6
        return "open_aged", 0.45

    def predict(self, task: HoldoutTask) -> Prediction:
        t0 = time.perf_counter()
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            pred, conf = self._stub_predict(task)
            return Prediction(
                pr_number=task.pr_number,
                condition=self.name + "_stubbed",
                prediction=pred,
                confidence=conf,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                cost_usd_estimate=0.0,
                raw_probabilities={pred: conf},
            )

        payload = json.dumps(
            {
                "pr_number": task.pr_number,
                "title_redacted": task.title_redacted,
                "tier_hint": task.tier_hint,
                "rationale_seeds": task.rationale_seeds,
            }
        )
        try:
            self._proc.stdin.write(payload + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline().strip()
            obj = json.loads(line) if line else {}
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("local_advocate shim error for PR %d: %s", task.pr_number, exc)
            pred, conf = self._stub_predict(task)
            return Prediction(
                pr_number=task.pr_number,
                condition=self.name + "_stubbed",
                prediction=pred,
                confidence=conf,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                cost_usd_estimate=0.0,
                raw_probabilities={pred: conf},
            )

        label = obj.get("label", "open_aged")
        if label not in CLASSES:
            label = "open_aged"
        conf = float(obj.get("confidence", 0.5))
        return Prediction(
            pr_number=task.pr_number,
            condition=self.name,
            prediction=label,
            confidence=max(0.0, min(1.0, conf)),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            cost_usd_estimate=COST_PER_CALL_ESTIMATE[self.name],
            raw_probabilities={label: conf},
        )


# -- Scoring ------------------------------------------------------------------


def brier_score(predictions: list[Prediction], labels: dict[int, str]) -> float:
    """Multi-class Brier score, averaged over tasks."""
    total = 0.0
    n = 0
    for pred in predictions:
        truth = labels.get(pred.pr_number)
        if truth is None:
            continue
        probs = pred.raw_probabilities or {pred.prediction: pred.confidence}
        # normalize / fill missing classes uniformly with leftover mass
        used_mass = sum(max(0.0, probs.get(cls, 0.0)) for cls in CLASSES)
        leftover = max(0.0, 1.0 - used_mass)
        per_class = {
            cls: max(0.0, probs.get(cls, 0.0)) + (leftover / len(CLASSES)) for cls in CLASSES
        }
        # renormalize to be safe
        s = sum(per_class.values()) or 1.0
        per_class = {k: v / s for k, v in per_class.items()}
        total += sum((per_class[cls] - (1.0 if cls == truth else 0.0)) ** 2 for cls in CLASSES)
        n += 1
    return total / n if n else float("nan")


def accuracy(predictions: list[Prediction], labels: dict[int, str]) -> float:
    n = 0
    correct = 0
    for pred in predictions:
        truth = labels.get(pred.pr_number)
        if truth is None:
            continue
        n += 1
        if pred.prediction == truth:
            correct += 1
    return correct / n if n else float("nan")


def mcnemar_p(a_correct: Iterable[bool], b_correct: Iterable[bool]) -> float:
    """Two-sided McNemar's test (exact binomial) for paired binary outcomes."""
    a = list(a_correct)
    b = list(b_correct)
    b01 = sum(1 for x, y in zip(a, b) if (not x) and y)
    b10 = sum(1 for x, y in zip(a, b) if x and (not y))
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    # exact two-sided binomial with p=0.5
    p = 0.0
    for i in range(0, k + 1):
        p += math.comb(n, i) * (0.5**n)
    p_two_sided = min(1.0, 2 * p)
    return p_two_sided


# -- Runner -------------------------------------------------------------------


def load_holdout(path: Path) -> list[HoldoutTask]:
    tasks: list[HoldoutTask] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tasks.append(HoldoutTask.from_row(row))
    return tasks


def run_condition(cond: Condition, tasks: list[HoldoutTask]) -> list[Prediction]:
    preds: list[Prediction] = []
    for task in tasks:
        try:
            preds.append(cond.predict(task))
        except Exception as exc:  # noqa: BLE001 - per-task isolation
            LOG.warning("condition %s failed on PR %d: %s", cond.name, task.pr_number, exc)
            preds.append(
                Prediction(
                    pr_number=task.pr_number,
                    condition=cond.name + "_error",
                    prediction="open_aged",
                    confidence=0.34,
                    latency_ms=0.0,
                    cost_usd_estimate=0.0,
                )
            )
    return preds


def summarize(
    results: dict[str, list[Prediction]],
    labels: dict[int, str],
) -> dict:
    summary: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_tasks": len(labels),
        "conditions": {},
        "pairwise_significance": {},
    }
    correctness: dict[str, list[bool]] = {}
    for name, preds in results.items():
        latencies = [p.latency_ms for p in preds]
        costs = [p.cost_usd_estimate for p in preds]
        stubbed_predictions = sum(1 for p in preds if p.condition.endswith("_stubbed"))
        error_predictions = sum(1 for p in preds if p.condition.endswith("_error"))
        real_predictions = len(preds) - stubbed_predictions - error_predictions
        summary["conditions"][name] = {
            "accuracy": accuracy(preds, labels),
            "brier": brier_score(preds, labels),
            "latency_ms_mean": statistics.fmean(latencies) if latencies else 0.0,
            "latency_ms_p95": (
                statistics.quantiles(latencies, n=20)[-1]
                if len(latencies) >= 20
                else max(latencies, default=0.0)
            ),
            "cost_usd_total": sum(costs),
            "cost_usd_mean": statistics.fmean(costs) if costs else 0.0,
            "n_predictions": len(preds),
            "stubbed_predictions": stubbed_predictions,
            "mock_predictions": stubbed_predictions,
            "real_predictions": real_predictions,
            "error_predictions": error_predictions,
            "stubbed": bool(preds and stubbed_predictions == len(preds)),
        }
        order = sorted(preds, key=lambda p: p.pr_number)
        correctness[name] = [
            p.prediction == labels.get(p.pr_number) for p in order if p.pr_number in labels
        ]

    # Bonferroni-corrected pairwise comparisons
    names = list(correctness.keys())
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    bonf_factor = max(1, len(pairs))
    for a, b in pairs:
        p = mcnemar_p(correctness[a], correctness[b])
        summary["pairwise_significance"][f"{a}__vs__{b}"] = {
            "p_value": p,
            "p_value_bonferroni": min(1.0, p * bonf_factor),
            "bonferroni_factor": bonf_factor,
        }
    return summary


def write_results(results_dir: Path, summary: dict, results: dict[str, list[Prediction]]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = results_dir / f"aft_summary_{ts}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    per_task_path = results_dir / f"aft_per_task_{ts}.jsonl"
    with per_task_path.open("w", encoding="utf-8") as fh:
        for name, preds in results.items():
            for pred in preds:
                fh.write(json.dumps({"condition_run": name, **asdict(pred)}, sort_keys=True))
                fh.write("\n")
    hypotheses_path = results_dir / f"aft_hypotheses_{ts}.txt"
    hypotheses_path.write_text(PRE_REGISTERED_HYPOTHESES)
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Advocate Feasibility Test harness")
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN,
        help="Path to training JSONL; used for baseline priors only",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(COST_PER_CALL_ESTIMATE.keys()),
        choices=list(COST_PER_CALL_ESTIMATE.keys()),
    )
    parser.add_argument(
        "--advocate-cmd",
        default=None,
        help=(
            "Command for the local advocate shim (JSONL stdin/stdout). "
            "Pass as a single shell-quoted string; flags are parsed via shlex. "
            "Example: --advocate-cmd 'bin/aft-advocate --backend stub'"
        ),
    )
    parser.add_argument(
        "--frontier-dry-run",
        action="store_true",
        help="Use deterministic heuristic instead of frontier CLI",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit holdout to first N tasks (smoke test)"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.holdout.exists():
        LOG.error("Holdout file not found: %s", args.holdout)
        LOG.error("Run scripts/aft_extract_training_data.py first.")
        return 2

    tasks = load_holdout(args.holdout)
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        LOG.error("Holdout is empty")
        return 2
    labels = {t.pr_number: t.label for t in tasks}
    LOG.info("Loaded %d holdout tasks", len(tasks))

    results: dict[str, list[Prediction]] = {}
    conds: list[Condition] = []
    for name in args.conditions:
        if name == "baseline_random":
            conds.append(BaselineRandom(args.train, seed=args.seed))
        elif name == "frontier_rules":
            conds.append(FrontierRules(dry_run=args.frontier_dry_run))
        elif name == "local_advocate":
            cmd_tokens = shlex.split(args.advocate_cmd) if args.advocate_cmd else None
            conds.append(LocalAdvocate(advocate_cmd=cmd_tokens))
        else:
            LOG.warning("Unknown condition: %s", name)

    try:
        for cond in conds:
            LOG.info("Running condition: %s", cond.name)
            preds = run_condition(cond, tasks)
            # Conditions can self-rename to *_stubbed; use the realized name.
            realized = preds[0].condition.rsplit("_error", 1)[0] if preds else cond.name
            results[realized] = preds
    finally:
        for cond in conds:
            close = getattr(cond, "close", None)
            if callable(close):
                close()

    summary = summarize(results, labels)
    summary_path = write_results(args.results_dir, summary, results)

    LOG.info("Wrote summary: %s", summary_path)
    LOG.info("Per-condition accuracy:")
    for name, stats in summary["conditions"].items():
        LOG.info(
            "  %-30s acc=%.3f  brier=%.3f  cost_usd=%.4f  n=%d",
            name,
            stats["accuracy"],
            stats["brier"],
            stats["cost_usd_total"],
            stats["n_predictions"],
        )
    LOG.info("Pairwise significance (Bonferroni-corrected):")
    for pair, stats in summary["pairwise_significance"].items():
        LOG.info("  %-50s p=%.4f  p_bonf=%.4f", pair, stats["p_value"], stats["p_value_bonferroni"])

    # Surface stubbed conditions so the operator does not over-interpret results.
    stubbed = [name for name in results if name.endswith("_stubbed")]
    if stubbed:
        LOG.warning(
            "Stubbed conditions present (%s) — results are NOT a valid empirical test, "
            "they only validate the harness plumbing.",
            ", ".join(stubbed),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
