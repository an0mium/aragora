---
title: CI/CD Security Guide
description: CI/CD Security Guide
---

# CI/CD Security Guide

This document explains the security improvements in Aragora's CI/CD pipeline, specifically the migration from SSH keys to AWS OIDC authentication.

## Overview

### Previous Approach (SSH Keys) - DEPRECATED

The original `deploy.yml` workflow (now removed) used SSH keys stored as GitHub secrets:

```yaml
# Vulnerable pattern - SSH private key in secrets
- name: Configure SSH
  run: |
    echo "${{ secrets.LIGHTSAIL_SSH_KEY }}" > ~/.ssh/lightsail
    ssh -i ~/.ssh/lightsail ubuntu@${{ secrets.LIGHTSAIL_HOST }} '...'
```

**Problems with this approach:**
- Long-lived credentials that never expire
- If secrets are leaked, full server access is compromised
- No audit trail of which workflow used the credentials
- Difficult to rotate credentials
- Overly broad permissions (full SSH access)

### New Approach (AWS OIDC + SSM)

The `deploy-secure.yml` workflow uses AWS OIDC for authentication:

```yaml
# Secure pattern - OIDC authentication
- name: Configure AWS credentials via OIDC
  uses: aws-actions/configure-aws-credentials@v4
  with:
    role-to-assume: arn:aws:iam::${{ secrets.AWS_ACCOUNT_ID }}:role/GitHubActionsDeployRole
    aws-region: us-east-1

- name: Deploy via SSM
  run: |
    aws ssm send-command --instance-ids "$INSTANCE_ID" ...
```

**Benefits:**
- No long-lived credentials stored anywhere
- Temporary credentials (15 min - 1 hour) via STS
- Full audit trail in AWS CloudTrail
- Fine-grained IAM permissions
- Automatic credential rotation
- Instance-level access control via tags

## Setup Instructions

### 1. Create OIDC Identity Provider in AWS

```bash
# Using AWS CLI
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

Or via AWS Console:
1. Go to IAM > Identity providers > Add provider
2. Select "OpenID Connect"
3. Provider URL: `https://token.actions.githubusercontent.com`
4. Audience: `sts.amazonaws.com`

### 2. Create IAM Role for GitHub Actions

Use the trust policy in `deploy/aws/oidc-trust-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:YOUR_ORG/aragora:*"
        }
      }
    }
  ]
}
```

**Important:** Replace `ACCOUNT_ID` and `YOUR_ORG/aragora` with your values.

```bash
# Create the role
aws iam create-role \
  --role-name GitHubActionsDeployRole \
  --assume-role-policy-document file://deploy/aws/oidc-trust-policy.json

# Attach the permissions policy
aws iam put-role-policy \
  --role-name GitHubActionsDeployRole \
  --policy-name DeployPermissions \
  --policy-document file://deploy/aws/deploy-role-policy.json
```

### 3. Configure EC2 Instances for SSM

Each EC2 instance needs:

1. **SSM Agent installed** (pre-installed on Amazon Linux 2, Ubuntu 18.04+)
2. **IAM instance profile** with SSM permissions
3. **Tags** for identification:
   - `Environment`: `staging` or `production`
   - `Application`: `aragora`

```bash
# Create instance profile for SSM
aws iam create-role \
  --role-name AragoraEC2Role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy \
  --role-name AragoraEC2Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

aws iam create-instance-profile \
  --instance-profile-name AragoraEC2Profile

aws iam add-role-to-instance-profile \
  --instance-profile-name AragoraEC2Profile \
  --role-name AragoraEC2Role

# Attach to instance
aws ec2 associate-iam-instance-profile \
  --instance-id i-1234567890abcdef0 \
  --iam-instance-profile Name=AragoraEC2Profile
```

### 4. Set GitHub Repository Secrets

Required secrets (Settings > Secrets and variables > Actions):

| Secret | Description | Example |
|--------|-------------|---------|
| `AWS_ACCOUNT_ID` | Your AWS account ID | `123456789012` |
| `AWS_DEPLOY_ROLE_NAME` | IAM role name | `GitHubActionsDeployRole` |
| `CLOUDFLARE_API_TOKEN` | Cloudflare API token | (from Cloudflare dashboard) |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account ID | (from Cloudflare dashboard) |

### 5. Enable the Secure Workflow

The secure workflow is at `.github/workflows/deploy-secure.yml` and is now the **only** deployment workflow.

**Migration Complete (2026-01):** The legacy `deploy.yml` workflow has been removed. All deployments now use the secure OIDC-based workflow which:
- Triggers automatically on pushes to `main`
- Supports manual deployments via `workflow_dispatch`
- Uses AWS OIDC authentication (no stored credentials)
- Deploys via AWS SSM (no SSH access required)

## Security Comparison

| Aspect | SSH Keys | OIDC + SSM |
|--------|----------|------------|
| Credential lifetime | Permanent | 15 min - 1 hour |
| Credential storage | GitHub Secrets | None (federated) |
| Audit trail | None | CloudTrail |
| Rotation | Manual | Automatic |
| Scope | Full SSH access | IAM-controlled |
| Revocation | Delete secret + rotate key | Update IAM policy |

## Restricting Access by Branch/Environment

The trust policy can be made more restrictive:

```json
{
  "Condition": {
    "StringEquals": {
      "token.actions.githubusercontent.com:sub": "repo:YOUR_ORG/aragora:ref:refs/heads/main"
    }
  }
}
```

This restricts deployments to only the `main` branch.

For environment-specific access:

```json
{
  "Condition": {
    "StringEquals": {
      "token.actions.githubusercontent.com:sub": "repo:YOUR_ORG/aragora:environment:production"
    }
  }
}
```

## Troubleshooting

### "Could not assume role" errors

1. Verify the trust policy has correct repository name
2. Check the OIDC provider thumbprint
3. Ensure `id-token: write` permission is in workflow

### SSM commands not executing

1. Verify SSM agent is running: `sudo systemctl status amazon-ssm-agent`
2. Check instance has IAM profile with SSM permissions
3. Verify instance is tagged correctly

### CloudTrail audit

Find deployment actions in CloudTrail:

```bash
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventName,AttributeValue=SendCommand \
  --start-time 2024-01-01T00:00:00Z
```

## Migration Checklist

**Migration Status: Complete (2026-01)**

- [x] Create OIDC provider in AWS IAM
- [x] Create IAM role with trust policy
- [x] Attach permissions policy to role
- [x] Configure EC2 instances with SSM agent and instance profile
- [x] Tag instances with Environment and Application
- [x] Add AWS_ACCOUNT_ID and AWS_DEPLOY_ROLE_NAME secrets
- [x] Test deploy-secure.yml workflow manually
- [x] Monitor CloudTrail for deployment events
- [x] Remove legacy deploy.yml workflow (completed 2026-01-24)
- [x] Configure environment protection for production (see below)
- [ ] Delete old SSH key secrets from GitHub (LIGHTSAIL_SSH_KEY, etc.)

## Environment Protection Rules

Production deployments require approval via GitHub Environment Protection Rules.

### Configure Required Reviewers

1. Go to **Settings > Environments > production**
2. Enable **Required reviewers**
3. Add one or more reviewers (team leads, DevOps)
4. Optionally set a wait timer (e.g., 5 minutes for "bake time")

### Configure Deployment Branch Rules

1. In the same environment settings, scroll to **Deployment branches**
2. Select "Selected branches" and add `main`
3. This prevents accidental production deploys from feature branches

### What Happens During Deployment

When a workflow targets the `production` environment:

1. Workflow pauses at the job with `environment: production`
2. Designated reviewers receive notification
3. Reviewer approves or rejects in the workflow run UI
4. If approved, deployment proceeds
5. If rejected or timeout (default 30 days), job fails

### Best Practices

- **Require 2+ reviewers** for critical production systems
- **Set deployment branches** to only allow `main`
- **Enable wait timer** for automatic rollback capability
- **Review staging results** before approving production

## References

- [GitHub OIDC with AWS](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services)
- [AWS SSM Run Command](https://docs.aws.amazon.com/systems-manager/latest/userguide/execute-remote-commands.html)
- [IAM Roles for OIDC](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_providers_create_oidc.html)
