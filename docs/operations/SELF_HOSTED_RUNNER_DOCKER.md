# Self-Hosted Runner Docker Provisioning

The `aragora`-labeled self-hosted GitHub Actions runners host VPC-resident
workflows that need Docker — notably:

- `load-tests.yml` — uses `services: redis:7-alpine` container

This document captures how Docker is provisioned on those runners and how to
provision new ones consistently.

## Current fleet (as of 2026-04-24)

AWS EC2, us-east-2, Amazon Linux 2023, IAM profile `aragora-ec2-ssm`:

| Instance ID | Private IP | Runner Name |
|---|---|---|
| `i-07e538fafbe61696d` | 172.31.31.173 | `ip-172-31-31-173` |
| `i-0aae2ccd2f68b94d2` | 172.31.24.39 | `ip-172-31-24-39` |
| `i-092c2d3b4dafc1f24` | 172.31.11.203 | `ip-172-31-11-203` |
| `i-014ecbcb79c4474b6` | 172.31.7.189 | `ip-172-31-7-189` |
| `i-0823e60c7c4b924e1` | 172.31.38.234 | `i-0823e60c7c4b924e1` |

The GitHub Actions runner service runs as `ec2-user` (systemd unit
`actions.runner.synaptent-aragora.<hostname>.service`).

## Required packages

- `docker` (Amazon Linux 2023 package, currently 25.x)
- The runner user (`ec2-user`) must be in the `docker` group
- The runner service must be restarted after group membership changes so
  that new supplementary groups take effect in the runner process

Note: `docker-compose-plugin` is NOT available in the default AL2023 repo.
Workflows that rely on `docker compose` should either install it via the
upstream docker-ce repo or use `services:` containers instead.

## Provisioning (via AWS SSM)

Runners can be provisioned without SSH using AWS Systems Manager
Session Manager (the `aragora-ec2-ssm` instance profile grants
`AmazonSSMManagedInstanceCore`).

### Step 1 — Install Docker

```bash
aws ssm send-command \
  --instance-ids <instance-id> \
  --document-name "AWS-RunShellScript" \
  --comment "Install Docker for aragora self-hosted runner" \
  --parameters 'commands=[
    "set -euo pipefail",
    "sudo dnf install -y docker",
    "sudo systemctl enable --now docker",
    "sudo docker --version",
    "sudo usermod -aG docker ec2-user",
    "id ec2-user"
  ]'
```

### Step 2 — Restart the runner service so group membership takes effect

```bash
aws ssm send-command \
  --instance-ids <instance-id> \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=[
    "sudo systemctl restart actions.runner.*",
    "sleep 3",
    "sudo systemctl status actions.runner.* --no-pager | head -20"
  ]'
```

### Step 3 — Verify

```bash
# Docker daemon reachable by ec2-user without sudo:
aws ssm send-command \
  --instance-ids <instance-id> \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["sudo -u ec2-user docker ps"]'

# Runner online on GitHub:
gh api repos/synaptent/aragora/actions/runners --jq '.runners[] | select(.name == "<runner-name>")'
```

## History

- **2026-04-24** — Docker installed on `i-0aae2ccd2f68b94d2` to restore
  Load Tests to self-hosted (reverts #6554 which had temporarily moved
  Load Tests to `ubuntu-latest` because Docker was missing). See
  `.github/workflows/load-tests.yml`.

## When to add ubuntu-latest fallback

If you spin up a new workflow that needs Docker AND:

- The workflow doesn't need VPC access to private AWS resources
- The workflow doesn't need GPU or high-memory instances

…then `runs-on: ubuntu-latest` is a simpler choice — GitHub-hosted runners
come with Docker pre-installed and are included in your Actions minute
allowance. Use self-hosted only when VPC access or specific hardware
is required.
