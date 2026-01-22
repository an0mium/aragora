---
title: Secrets Management
description: Secrets Management
---

# Secrets Management

This guide covers secure handling of secrets in Aragora deployments.

## Overview

Aragora requires various secrets for operation:

| Secret | Purpose | Required |
|--------|---------|----------|
| `ANTHROPIC_API_KEY` | Claude API access | Yes (or OPENAI) |
| `OPENAI_API_KEY` | GPT API access | Yes (or ANTHROPIC) |
| `OPENROUTER_API_KEY` | Fallback provider | Recommended |
| `SUPABASE_KEY` | Database access | Production |
| `STRIPE_SECRET_KEY` | Billing | If billing enabled |
| `STRIPE_WEBHOOK_SECRET` | Webhook verification | If billing enabled |
| `ARAGORA_JWT_SECRET` | Token signing | Production |
| `GOOGLE_OAUTH_CLIENT_SECRET` | Google OAuth client secret | If Google OAuth enabled |
| `ARAGORA_SSO_CLIENT_SECRET` | SSO client secret (OIDC/SAML) | If SSO enabled |

## Local Development

### Using .env Files

Create a `.env` file in the project root (never commit this file):

```bash
# .env (gitignored)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=eyJ...
```

Load automatically with:
```bash
aragora serve  # Auto-loads .env
```

### Using direnv

For automatic environment loading per directory:

```bash
# Install direnv
brew install direnv  # macOS
sudo apt install direnv  # Ubuntu

# Create .envrc
echo 'dotenv' > .envrc
direnv allow
```

## Kubernetes Deployment

### Option 1: Kubernetes Secrets (Basic)

Create secrets manually:

```bash
kubectl create secret generic aragora-secrets \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-... \
  --from-literal=OPENAI_API_KEY=sk-... \
  --from-literal=SUPABASE_KEY=eyJ... \
  --from-literal=ARAGORA_JWT_SECRET=$(openssl rand -base64 32)
```

Reference in deployment:

```yaml
# deploy/k8s/deployment.yaml
spec:
  containers:
    - name: aragora
      envFrom:
        - secretRef:
            name: aragora-secrets
```

### Option 2: Sealed Secrets (GitOps)

For storing encrypted secrets in Git:

```bash
# Install kubeseal
brew install kubeseal

# Create sealed secret
kubectl create secret generic aragora-secrets \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-... \
  --dry-run=client -o yaml | \
  kubeseal --format=yaml > deploy/k8s/sealed-secrets.yaml
```

The sealed secret can be safely committed to Git.

### Option 3: External Secrets Operator

For pulling secrets from external providers:

```yaml
# deploy/k8s/external-secret.yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: aragora-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: ClusterSecretStore
  target:
    name: aragora-secrets
  data:
    - secretKey: ANTHROPIC_API_KEY
      remoteRef:
        key: aragora/api-keys
        property: anthropic
    - secretKey: OPENAI_API_KEY
      remoteRef:
        key: aragora/api-keys
        property: openai
```

### Option 4: HashiCorp Vault

Direct Vault integration:

```yaml
# deploy/k8s/deployment.yaml
spec:
  template:
    metadata:
      annotations:
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/role: "aragora"
        vault.hashicorp.com/agent-inject-secret-env: "secret/data/aragora/env"
        vault.hashicorp.com/agent-inject-template-env: |
          {{- with secret "secret/data/aragora/env" -}}
          export ANTHROPIC_API_KEY="{{ .Data.data.anthropic_key }}"
          export OPENAI_API_KEY="{{ .Data.data.openai_key }}"
          {{- end }}
```

## Helm Chart Configuration

Using the Aragora Helm chart:

```yaml
# values.yaml
secrets:
  # Option 1: Inline (not recommended for production)
  anthropicApiKey: ""
  openaiApiKey: ""

  # Option 2: Existing secret reference (recommended)
  existingSecret: "aragora-secrets"

  # Keys in the existing secret
  existingSecretKeys:
    anthropicApiKey: ANTHROPIC_API_KEY
    openaiApiKey: OPENAI_API_KEY
    supabaseKey: SUPABASE_KEY
```

Install with:
```bash
helm install aragora deploy/helm/aragora \
  --set secrets.existingSecret=aragora-secrets
```

## AWS Deployment

### AWS Secrets Manager

```bash
# Create secret
aws secretsmanager create-secret \
  --name aragora/api-keys \
  --secret-string '{"ANTHROPIC_API_KEY":"sk-ant-...","OPENAI_API_KEY":"sk-..."}'

# In ECS task definition
{
  "secrets": [
    {
      "name": "ANTHROPIC_API_KEY",
      "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789:secret:aragora/api-keys:ANTHROPIC_API_KEY::"
    }
  ]
}
```

### AWS Parameter Store

```bash
# Create parameters
aws ssm put-parameter \
  --name /aragora/anthropic-api-key \
  --value "sk-ant-..." \
  --type SecureString

# In ECS task definition
{
  "secrets": [
    {
      "name": "ANTHROPIC_API_KEY",
      "valueFrom": "arn:aws:ssm:us-east-1:123456789:parameter/aragora/anthropic-api-key"
    }
  ]
}
```

## GCP Deployment

### Google Secret Manager

```bash
# Create secret
echo -n "sk-ant-..." | gcloud secrets create anthropic-api-key --data-file=-

# Grant access
gcloud secrets add-iam-policy-binding anthropic-api-key \
  --member="serviceAccount:aragora@project.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

In Cloud Run:
```yaml
spec:
  template:
    spec:
      containers:
        - env:
            - name: ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  key: latest
                  name: anthropic-api-key
```

## Secret Rotation

### Automatic Rotation

For production, implement secret rotation:

```python
# Example rotation script
import boto3
from datetime import datetime, timedelta

def rotate_api_key():
    # 1. Generate new key from provider
    new_key = provider.create_api_key()

    # 2. Update secret store
    client = boto3.client('secretsmanager')
    client.update_secret(
        SecretId='aragora/api-keys',
        SecretString=json.dumps({'ANTHROPIC_API_KEY': new_key})
    )

    # 3. Restart pods to pick up new secret
    # Or use reloader: https://github.com/stakater/Reloader
```

### Using Reloader

Auto-restart pods on secret changes:

```yaml
# Deploy Reloader
helm repo add stakater https://stakater.github.io/stakater-charts
helm install reloader stakater/reloader

# Annotate deployment
metadata:
  annotations:
    reloader.stakater.com/auto: "true"
```

## Security Best Practices

### 1. Principle of Least Privilege

Grant minimal permissions:

```yaml
# Kubernetes RBAC
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: aragora-secrets-reader
rules:
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["aragora-secrets"]
    verbs: ["get"]
```

### 2. Audit Logging

Enable secret access logging:

```bash
# AWS CloudTrail
aws cloudtrail create-trail \
  --name secrets-audit \
  --s3-bucket-name audit-logs

# GCP Audit Logs (automatic)
gcloud logging read "resource.type=secretmanager.googleapis.com"
```

### 3. Network Policies

Restrict secret access by network:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: aragora-secrets-access
spec:
  podSelector:
    matchLabels:
      app: aragora
  policyTypes:
    - Egress
  egress:
    - to:
        - ipBlock:
            cidr: 10.0.0.0/8  # Internal only
```

### 4. Secret Scanning

Prevent accidental commits:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

### 5. Environment Isolation

Separate secrets per environment:

```
secrets/
├── dev/
│   └── aragora-secrets
├── staging/
│   └── aragora-secrets
└── prod/
    └── aragora-secrets
```

## Troubleshooting

### Secret Not Found

```bash
# Check secret exists
kubectl get secret aragora-secrets -o yaml

# Check pod can access
kubectl exec -it aragora-pod -- env | grep API_KEY
```

### Permission Denied

```bash
# Check service account
kubectl get serviceaccount aragora -o yaml

# Check role binding
kubectl get rolebinding -l app=aragora
```

### Secret Not Updating

```bash
# Force pod restart
kubectl rollout restart deployment/aragora

# Or use Reloader (see above)
```

## Quick Reference

| Environment | Recommended Method |
|-------------|-------------------|
| Local dev | `.env` file |
| CI/CD | GitHub/GitLab secrets |
| Kubernetes | External Secrets Operator |
| AWS | Secrets Manager |
| GCP | Secret Manager |
| Production | Vault + rotation |

## Related Documentation

- [Kubernetes Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)
- [External Secrets Operator](https://external-secrets.io/)
- [HashiCorp Vault](https://www.vaultproject.io/docs)
- [AWS Secrets Manager](https://docs.aws.amazon.com/secretsmanager/)
- [GCP Secret Manager](https://cloud.google.com/secret-manager/docs)
