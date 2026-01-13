# Secrets Management Guide

Aragora requires several secrets for production deployment. This guide covers secure secrets management using External Secrets Operator and various cloud backends.

## Quick Start

### Development (Local)

For local development, use `.env` file:

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Production (Kubernetes)

For production, use External Secrets Operator to sync secrets from your cloud provider:

```bash
# 1. Install External Secrets Operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  -n external-secrets --create-namespace

# 2. Configure your secret backend (AWS, Vault, GCP, etc.)
kubectl apply -f deploy/k8s/external-secrets/cluster-secret-store.yaml

# 3. Create ExternalSecret to sync secrets
kubectl apply -f deploy/k8s/external-secrets/aragora-secrets.yaml
```

## Required Secrets

| Secret | Environment Variable | Description |
|--------|---------------------|-------------|
| Database URL | `DATABASE_URL` | PostgreSQL connection string |
| JWT Secret | `ARAGORA_JWT_SECRET` | 32+ char secret for auth tokens |
| Anthropic API Key | `ANTHROPIC_API_KEY` | For Claude models |
| OpenAI API Key | `OPENAI_API_KEY` | For GPT models |

## Optional Secrets

| Secret | Environment Variable | Description |
|--------|---------------------|-------------|
| OpenRouter API Key | `OPENROUTER_API_KEY` | Fallback/multi-model |
| Mistral API Key | `MISTRAL_API_KEY` | Mistral models |
| Stripe Secret Key | `STRIPE_SECRET_KEY` | Billing integration |
| Stripe Webhook Secret | `STRIPE_WEBHOOK_SECRET` | Webhook verification |

## Backend Configuration

### AWS Secrets Manager

1. **Create secrets in AWS**:

```bash
# Create secret with multiple keys
aws secretsmanager create-secret \
  --name aragora/production/api-keys \
  --secret-string '{
    "anthropic": "sk-ant-...",
    "openai": "sk-...",
    "openrouter": "sk-or-..."
  }'

aws secretsmanager create-secret \
  --name aragora/production/database \
  --secret-string '{"url": "postgresql://..."}'

aws secretsmanager create-secret \
  --name aragora/production/auth \
  --secret-string '{"jwt-secret": "your-32-char-secret-here"}'
```

2. **Configure IRSA** (IAM Roles for Service Accounts):

```bash
# Create IAM policy
cat > /tmp/policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": "arn:aws:secretsmanager:*:*:secret:aragora/*"
    }
  ]
}
EOF

aws iam create-policy \
  --policy-name AragoraSecretsAccess \
  --policy-document file:///tmp/policy.json

# Create service account with IRSA
eksctl create iamserviceaccount \
  --name external-secrets \
  --namespace external-secrets \
  --cluster your-cluster \
  --attach-policy-arn arn:aws:iam::ACCOUNT:policy/AragoraSecretsAccess \
  --approve
```

3. **Apply ClusterSecretStore**:

```yaml
# deploy/k8s/external-secrets/cluster-secret-store.yaml
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: aws-secrets-manager
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-west-2
```

### HashiCorp Vault

1. **Configure Vault**:

```bash
# Enable KV secrets engine
vault secrets enable -path=secret kv-v2

# Create secrets
vault kv put secret/aragora/production/api-keys \
  anthropic="sk-ant-..." \
  openai="sk-..." \
  openrouter="sk-or-..."

vault kv put secret/aragora/production/database \
  url="postgresql://..."

vault kv put secret/aragora/production/auth \
  jwt-secret="your-32-char-secret"

# Create policy
vault policy write aragora - << 'EOF'
path "secret/data/aragora/*" {
  capabilities = ["read"]
}
EOF

# Configure Kubernetes auth
vault auth enable kubernetes
vault write auth/kubernetes/config \
  kubernetes_host="https://kubernetes.default.svc"

vault write auth/kubernetes/role/aragora \
  bound_service_account_names=external-secrets \
  bound_service_account_namespaces=external-secrets \
  policies=aragora \
  ttl=1h
```

2. **Apply ClusterSecretStore**:

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: vault
spec:
  provider:
    vault:
      server: "https://vault.example.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "aragora"
          serviceAccountRef:
            name: external-secrets
            namespace: external-secrets
```

### Google Cloud Secret Manager

1. **Create secrets in GCP**:

```bash
# Create secrets
echo -n "sk-ant-..." | gcloud secrets create aragora-anthropic-key --data-file=-
echo -n "sk-..." | gcloud secrets create aragora-openai-key --data-file=-
echo -n "postgresql://..." | gcloud secrets create aragora-database-url --data-file=-

# Grant access to service account
gcloud secrets add-iam-policy-binding aragora-anthropic-key \
  --member="serviceAccount:aragora@project.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

2. **Configure Workload Identity**:

```bash
gcloud iam service-accounts add-iam-policy-binding \
  aragora@PROJECT.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:PROJECT.svc.id.goog[external-secrets/external-secrets]"

kubectl annotate serviceaccount external-secrets \
  -n external-secrets \
  iam.gke.io/gcp-service-account=aragora@PROJECT.iam.gserviceaccount.com
```

## Secret Rotation

### Automatic Rotation

External Secrets Operator syncs secrets based on `refreshInterval`:

```yaml
spec:
  refreshInterval: 1h  # Sync every hour
```

For more frequent rotation:

```yaml
spec:
  refreshInterval: 5m  # Sync every 5 minutes
```

### Manual Rotation

1. Update secret in your backend (AWS SM, Vault, etc.)
2. Force refresh the ExternalSecret:

```bash
# Trigger immediate sync
kubectl annotate externalsecret aragora-secrets \
  -n aragora \
  force-sync=$(date +%s) --overwrite

# Restart pods to pick up new secrets
kubectl -n aragora rollout restart deployment/aragora
```

### Rotation Best Practices

1. **Never rotate JWT secret** without a maintenance window (invalidates all sessions)
2. **API keys**: Can be rotated with zero downtime (add new key, deploy, remove old)
3. **Database credentials**: Use connection pooling and rotate during low traffic

## Verification

### Check ExternalSecret Status

```bash
# View sync status
kubectl -n aragora get externalsecret aragora-secrets

# Detailed status
kubectl -n aragora describe externalsecret aragora-secrets

# Check if Kubernetes Secret was created
kubectl -n aragora get secret aragora-secrets -o yaml
```

### Test Secrets in Pod

```bash
# Exec into pod and verify environment
kubectl -n aragora exec -it deploy/aragora -- env | grep -E "^(ANTHROPIC|OPENAI|DATABASE)"

# Test database connection
kubectl -n aragora exec -it deploy/aragora -- python -c "
from aragora.db import get_database
db = get_database()
print('Database connected:', db.health_check())
"
```

## Troubleshooting

### ExternalSecret Not Syncing

```bash
# Check External Secrets Operator logs
kubectl -n external-secrets logs -l app.kubernetes.io/name=external-secrets

# Check ExternalSecret events
kubectl -n aragora describe externalsecret aragora-secrets
```

**Common Issues**:

1. **Authentication failed**: Check ClusterSecretStore credentials
2. **Secret not found**: Verify remote key path matches backend
3. **Permission denied**: Check IAM/RBAC policies

### Secret Not Available in Pod

1. Verify Secret exists: `kubectl -n aragora get secret aragora-secrets`
2. Check deployment references correct secret name
3. Check env/envFrom configuration in deployment

### Sync Delay

External Secrets only syncs at `refreshInterval`. For immediate updates:

```bash
kubectl annotate externalsecret aragora-secrets -n aragora \
  force-sync=$(date +%s) --overwrite
```

## Security Best Practices

1. **Least privilege**: Only grant access to needed secrets
2. **Audit logging**: Enable audit logs on your secret backend
3. **Encryption at rest**: Ensure secrets are encrypted in backend
4. **Network isolation**: Restrict access to secret management APIs
5. **Regular rotation**: Rotate secrets on a schedule
6. **No secrets in git**: Never commit secrets, even encrypted
7. **Separate environments**: Use different secrets for staging/production
