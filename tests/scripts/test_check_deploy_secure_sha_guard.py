from __future__ import annotations

from pathlib import Path

from scripts.check_deploy_secure_sha_guard import (
    check_repo,
    find_rollback_guard_violations,
    find_sha_verification_violations,
)


def _valid_workflow_text() -> str:
    return """
jobs:
  deploy-ec2-staging:
    steps:
      - name: Determine rollback requirement
        id: rollback_gate
        if: always() && steps.deploy.conclusion != 'skipped'
        run: |
          echo "should_rollback=$SHOULD_ROLLBACK" >> "$GITHUB_OUTPUT"
          echo "rollback_reason=$ROLLBACK_REASON" >> "$GITHUB_OUTPUT"
      - name: Rollback on failure
        if: steps.rollback_gate.outputs.should_rollback == 'true'
        run: |
          echo "::warning::Deployment failed - initiating rollback (${{ steps.rollback_gate.outputs.rollback_reason }})"
          if [ -f /tmp/aragora_deploy_state ]; then source /tmp/aragora_deploy_state; fi
          if [ -n "$PREVIOUS_COMMIT" ]; then sudo -u ec2-user git checkout $PREVIOUS_COMMIT; fi
  deploy-ec2-production:
    steps:
      - name: Post-deploy SHA verification
        run: |
          SHA_CMD_ID=$(aws ssm send-command \\
            --instance-ids "${IDS[@]}" \\
            --document-name "AWS-RunShellScript" \\
            --parameters 'commands=[
              "set -e",
              "sudo -u ec2-user git -C /home/ec2-user/aragora rev-parse HEAD || git -C /home/ec2-user/aragora -c safe.directory=/home/ec2-user/aragora rev-parse HEAD"
            ]' \\
            --timeout-seconds 60 \\
            --query 'Command.CommandId' \\
            --output text)
          echo "::warning::SHA stdout for $INST_ID: $STDOUT"
          echo "::warning::SHA stderr for $INST_ID: $STDERR"
      - name: Determine rollback requirement
        id: rollback_gate
        if: always() && steps.deploy.conclusion != 'skipped'
        run: |
          SHA_CONCLUSION="${{ steps.sha_verify.conclusion }}"
          echo "should_rollback=$SHOULD_ROLLBACK" >> "$GITHUB_OUTPUT"
          echo "rollback_reason=$ROLLBACK_REASON" >> "$GITHUB_OUTPUT"
      - name: Rollback on failure
        if: steps.rollback_gate.outputs.should_rollback == 'true'
        run: |
          echo "::warning::Production deployment failed - initiating rollback (${{ steps.rollback_gate.outputs.rollback_reason }})"
          if [ -f /tmp/aragora_deploy_state ]; then source /tmp/aragora_deploy_state; fi
          if [ -n "$PREVIOUS_COMMIT" ]; then sudo -u ec2-user git checkout $PREVIOUS_COMMIT; fi
  notify:
    steps:
      - name: done
        run: echo done
"""


def test_sha_guard_accepts_valid_step() -> None:
    violations = find_sha_verification_violations(_valid_workflow_text())
    assert violations == []


def test_sha_guard_requires_step() -> None:
    violations = find_sha_verification_violations(
        "jobs:\n  deploy-ec2-production:\n    steps: []\n"
    )
    assert violations
    assert "missing `Post-deploy SHA verification` step" in violations[0]


def test_sha_guard_requires_hardened_command_markers() -> None:
    text = _valid_workflow_text().replace("sudo -u ec2-user ", "")
    violations = find_sha_verification_violations(text)
    assert violations
    assert any("ec2_user_command" in message for message in violations)


def test_rollback_guard_accepts_valid_steps() -> None:
    violations = find_rollback_guard_violations(_valid_workflow_text())
    assert violations == []


def test_rollback_guard_requires_gate_step() -> None:
    violations = find_rollback_guard_violations(
        "jobs:\n  deploy-ec2-staging:\n    steps:\n      - name: Rollback on failure\n"
    )
    assert violations
    assert any("Determine rollback requirement" in message for message in violations)


def test_rollback_guard_requires_hardened_markers() -> None:
    text = _valid_workflow_text().replace("git checkout $PREVIOUS_COMMIT", "echo noop")
    violations = find_rollback_guard_violations(text)
    assert violations
    assert any("rollback_previous_commit" in message for message in violations)


def test_rollback_guard_requires_staging_production_parity() -> None:
    text = _valid_workflow_text().replace(
        """
      - name: Determine rollback requirement
        id: rollback_gate
        if: always() && steps.deploy.conclusion != 'skipped'
        run: |
          SHA_CONCLUSION="${{ steps.sha_verify.conclusion }}"
          echo "should_rollback=$SHOULD_ROLLBACK" >> "$GITHUB_OUTPUT"
          echo "rollback_reason=$ROLLBACK_REASON" >> "$GITHUB_OUTPUT"
      - name: Rollback on failure
        if: steps.rollback_gate.outputs.should_rollback == 'true'
        run: |
          echo "::warning::Production deployment failed - initiating rollback (${{ steps.rollback_gate.outputs.rollback_reason }})"
          if [ -f /tmp/aragora_deploy_state ]; then source /tmp/aragora_deploy_state; fi
          if [ -n "$PREVIOUS_COMMIT" ]; then sudo -u ec2-user git checkout $PREVIOUS_COMMIT; fi
""",
        "",
    )
    violations = find_rollback_guard_violations(text)
    assert violations
    assert any("parity for staging+production" in message for message in violations)


def test_repo_sha_guard_passes_for_current_tree() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    violations = check_repo(repo_root)
    assert violations == []
