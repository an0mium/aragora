from __future__ import annotations

from pathlib import Path

from scripts.check_aragora_review_gate_policy import (
    check_repo,
    find_review_gate_policy_violations,
)


def _valid_gate_data() -> dict:
    return {
        "on": {
            "pull_request": {
                "types": ["opened", "synchronize", "reopened", "ready_for_review"],
            },
            "workflow_dispatch": {},
        },
        "jobs": {
            "changes": {},
            "review": {
                "steps": [
                    {
                        "name": "Install Aragora",
                        "run": """
python -m pip install --upgrade pip
python -m pip install "pydantic>=2,<3" pydantic-settings aiohttp defusedxml numpy anthropic openai
python -m pip install -e .
""",
                    },
                    {
                        "name": "Run Aragora Review",
                        "run": """
if [[ ! -s /tmp/pr.diff ]]; then
  echo "skip=true" >> "$GITHUB_OUTPUT"
  exit 0
fi
review_output_dir=/tmp/review-artifacts
mkdir -p "$review_output_dir"
set +e
cat /tmp/pr.diff | python -m aragora.cli.review review --output-format json --output-dir "$review_output_dir"
review_exit=$?
set -e
if [[ "$review_exit" -ne 0 ]]; then
  if [[ -s /tmp/review.stderr ]]; then
    cat /tmp/review.stderr
  fi
  exit "$review_exit"
fi
review_json="$review_output_dir/review.json"
if [[ ! -f "$review_json" ]]; then
  exit 1
fi
python3 - <<'PY2'
import json
from pathlib import Path
data = json.loads(Path('/tmp/review-artifacts/review.json').read_text())
assert isinstance(data.get('critical_issues'), list)
assert isinstance(data.get('high_issues'), list)
PY2
""",
                    },
                ]
            },
            "aragora-review": {
                "if": "always()",
                "needs": ["changes", "review"],
            },
        },
    }


def _valid_gate_text() -> str:
    return """
jobs:
  review:
    steps:
      - name: Run Aragora Review
        run: |
          if [[ ! -s /tmp/pr.diff ]]; then
            echo "skip=true" >> "$GITHUB_OUTPUT"
            exit 0
          fi
          review_output_dir=/tmp/review-artifacts
          mkdir -p "$review_output_dir"
          set +e
          cat /tmp/pr.diff | python -m aragora.cli.review review --output-format json --output-dir "$review_output_dir"
          review_exit=$?
          set -e
          if [[ "$review_exit" -ne 0 ]]; then
            if [[ -s /tmp/review.stderr ]]; then
              cat /tmp/review.stderr
            fi
            exit "$review_exit"
          fi
          review_json="$review_output_dir/review.json"
          if [[ ! -f "$review_json" ]]; then
            exit 1
          fi
          python3 - <<'PY2'
          import json
          from pathlib import Path
          data = json.loads(Path('/tmp/review-artifacts/review.json').read_text())
          assert isinstance(data.get('critical_issues'), list)
          assert isinstance(data.get('high_issues'), list)
          PY2
  aragora-review:
    if: always()
    needs: [changes, review]
    steps:
      - run: |
          if [[ "$SHOULD_REVIEW" != "true" ]]; then
            exit 0
          fi
          if [[ "$REVIEW_RESULT" != "success" ]]; then
            exit 1
          fi
"""


def _valid_manual_data() -> dict:
    return {"on": {"workflow_dispatch": {}}}


def _review_step(gate: dict) -> dict:
    for step in gate["jobs"]["review"]["steps"]:
        if step["name"] == "Run Aragora Review":
            return step
    raise AssertionError("Run Aragora Review step missing from fixture")


def test_policy_accepts_valid_review_gate_configuration() -> None:
    violations = find_review_gate_policy_violations(
        _valid_gate_data(),
        _valid_gate_text(),
        _valid_manual_data(),
    )
    assert violations == []


def test_policy_rejects_pull_request_paths_filter() -> None:
    gate = _valid_gate_data()
    gate["on"]["pull_request"]["paths"] = ["aragora/**"]
    violations = find_review_gate_policy_violations(gate, _valid_gate_text(), _valid_manual_data())
    assert "review gate pull_request trigger must not define paths filters" in violations


def test_policy_rejects_fail_open_review_execution() -> None:
    gate = _valid_gate_data()
    _review_step(gate)["run"] = _review_step(gate)["run"].replace(
        'python -m aragora.cli.review review --output-format json --output-dir "$review_output_dir"',
        'python -m aragora.cli.review review --output-format json --output-dir "$review_output_dir" || true',
    )
    text = _valid_gate_text().replace(
        'python -m aragora.cli.review review --output-format json --output-dir "$review_output_dir"',
        'python -m aragora.cli.review review --output-format json --output-dir "$review_output_dir" || true',
    )
    violations = find_review_gate_policy_violations(gate, text, _valid_manual_data())
    assert "review execution must not use || true" in violations


def test_policy_rejects_missing_requirements_install() -> None:
    gate = _valid_gate_data()
    gate["jobs"]["review"]["steps"][0]["run"] = """
python -m pip install --upgrade pip
python -m pip install -e .
"""
    violations = find_review_gate_policy_violations(
        gate,
        _valid_gate_text(),
        _valid_manual_data(),
    )
    assert (
        'review job install step must install "pydantic>=2,<3" for review CLI runtime' in violations
    )
    assert (
        "review job install step must install pydantic-settings for review CLI runtime"
        in violations
    )
    assert "review job install step must install aiohttp for review CLI runtime" in violations
    assert "review job install step must install defusedxml for review CLI runtime" in violations
    assert "review job install step must install numpy for review CLI runtime" in violations
    assert "review job install step must install anthropic for review CLI runtime" in violations
    assert "review job install step must install openai for review CLI runtime" in violations


def test_policy_rejects_missing_review_exit_capture() -> None:
    gate = _valid_gate_data()
    _review_step(gate)["run"] = _review_step(gate)["run"].replace(
        'review_exit=$?\nset -e\nif [[ "$review_exit" -ne 0 ]]; then\n  if [[ -s /tmp/review.stderr ]]; then\n    cat /tmp/review.stderr\n  fi\n  exit "$review_exit"\nfi\n',
        "",
    )
    text = _valid_gate_text().replace(
        "          review_exit=$?\n"
        "          set -e\n"
        '          if [[ "$review_exit" -ne 0 ]]; then\n'
        "            if [[ -s /tmp/review.stderr ]]; then\n"
        "              cat /tmp/review.stderr\n"
        "            fi\n"
        '            exit "$review_exit"\n'
        "          fi\n",
        "",
    )
    violations = find_review_gate_policy_violations(gate, text, _valid_manual_data())
    assert "review execution must capture the review command exit code explicitly" in violations
    assert "review execution must fail closed after surfacing review command stderr" in violations


def test_policy_rejects_outdated_review_cli_contract() -> None:
    gate = _valid_gate_data()
    _review_step(gate)["run"] = "python -m aragora.cli.review --output /tmp/review.json"
    violations = find_review_gate_policy_violations(
        gate,
        _valid_gate_text(),
        _valid_manual_data(),
    )
    assert "review execution must invoke the review subcommand" in violations
    assert "review execution must request json output" in violations
    assert "review execution must write review artifacts via --output-dir" in violations
    assert "review gate must parse the current json artifact schema" in violations


def test_policy_rejects_manual_workflow_pull_request_trigger() -> None:
    manual = {"on": {"pull_request": {"types": ["opened"]}}}
    violations = find_review_gate_policy_violations(
        _valid_gate_data(),
        _valid_gate_text(),
        manual,
    )
    assert "manual Aragora review workflow must not trigger on pull_request" in violations


def test_policy_rejects_critical_count_blocking_gate() -> None:
    gate_text = (
        _valid_gate_text()
        + """
      - name: Gate on Critical Findings
        if: steps.review.outputs.critical != '' && steps.review.outputs.critical != '0'
        run: |
          echo "::error::Aragora review found ${CRITICAL_COUNT} critical issue(s)"
          exit 1
"""
    )
    violations = find_review_gate_policy_violations(
        _valid_gate_data(),
        gate_text,
        _valid_manual_data(),
    )
    assert "review findings must stay advisory and must not fail on critical-count" in violations


def test_repo_review_gate_policy_passes_for_current_tree() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert check_repo(repo_root) == []
