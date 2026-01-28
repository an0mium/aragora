# ArgoCD GitOps for Aragora

This directory contains ArgoCD configuration for GitOps-based deployment of Aragora.

## Overview

ArgoCD provides:
- **Pull-based GitOps**: Git repository is the single source of truth
- **Automatic sync**: Changes in git are automatically deployed
- **Drift detection**: Detects and fixes configuration drift
- **Multi-region**: Single ApplicationSet manages all regions
- **RBAC**: Fine-grained access control per project/team

## Directory Structure

```
argocd/
├── install/              # ArgoCD installation values
│   └── values.yaml       # Helm values for ArgoCD itself
├── applications/         # Application definitions
│   └── applicationset.yaml  # Multi-region ApplicationSet
├── projects/             # AppProject definitions
│   └── aragora-project.yaml  # Production & staging projects
├── notifications/        # Slack/webhook notifications
└── README.md            # This file
```

## Prerequisites

1. **Kubernetes clusters** configured and accessible
2. **kubectl** configured with cluster access
3. **Helm 3** installed
4. **GitHub access** for repository cloning

## Installation

### Step 1: Install ArgoCD

```bash
# Add ArgoCD Helm repo
helm repo add argo https://argoproj.github.io/argo-helm
helm repo update

# Create namespace
kubectl create namespace argocd

# Install ArgoCD with custom values
helm install argocd argo/argo-cd \
  -n argocd \
  -f deploy/argocd/install/values.yaml

# Wait for rollout
kubectl rollout status deployment/argocd-server -n argocd
```

### Step 2: Access ArgoCD UI

```bash
# Port forward (development)
kubectl port-forward svc/argocd-server -n argocd 8443:443

# Get initial admin password
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d

# Access UI at https://localhost:8443
# Username: admin
# Password: <from above>
```

### Step 3: Configure Repository Access

```bash
# Option 1: Deploy key (recommended)
argocd repo add https://github.com/an0mium/aragora \
  --ssh-private-key-path ~/.ssh/aragora-deploy-key

# Option 2: GitHub App (enterprise)
argocd repo add https://github.com/an0mium/aragora \
  --github-app-id <app-id> \
  --github-app-installation-id <installation-id> \
  --github-app-private-key-path ./github-app.pem

# Option 3: HTTPS with token
argocd repo add https://github.com/an0mium/aragora \
  --username git \
  --password <github-token>
```

### Step 4: Register Clusters (Multi-Region)

```bash
# Add each regional cluster
argocd cluster add aragora-us-east-2 \
  --name aragora-us-east-2 \
  --kubeconfig ~/.kube/config

argocd cluster add aragora-eu-west-1 \
  --name aragora-eu-west-1 \
  --kubeconfig ~/.kube/config

argocd cluster add aragora-ap-south-1 \
  --name aragora-ap-south-1 \
  --kubeconfig ~/.kube/config
```

### Step 5: Create Projects

```bash
kubectl apply -f deploy/argocd/projects/aragora-project.yaml
```

### Step 6: Deploy Applications

```bash
kubectl apply -f deploy/argocd/applications/applicationset.yaml
```

## Configuration

### Environment Values

Each region uses a values file:
- `deploy/helm/aragora/values.yaml` - Base configuration
- `deploy/helm/aragora/values-us-east-2.yaml` - US East overrides
- `deploy/helm/aragora/values-eu-west-1.yaml` - EU West overrides
- `deploy/helm/aragora/values-ap-south-1.yaml` - Asia Pacific overrides

### Secrets Management

Secrets are managed via External Secrets Operator:

```bash
# Install External Secrets Operator
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets \
  -n external-secrets --create-namespace

# Create ClusterSecretStore for AWS Secrets Manager
kubectl apply -f - <<EOF
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: aws-secrets-manager
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-2
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets
            namespace: external-secrets
EOF
```

### Notifications

Configure Slack notifications:

```bash
# Create Slack token secret
kubectl create secret generic argocd-notifications-secret \
  -n argocd \
  --from-literal=slack-token=xoxb-your-token

# Notifications will be sent to configured channels
# - aragora-deployments: Successful syncs
# - aragora-alerts: Failed syncs
```

## Usage

### Trigger Deployment

Deployments are triggered automatically when:
1. Code is merged to `main` branch
2. Helm values are updated
3. Kubernetes manifests change

Manual sync:
```bash
argocd app sync aragora-us-east-2
```

### View Status

```bash
# List all applications
argocd app list

# Get application details
argocd app get aragora-us-east-2

# View sync history
argocd app history aragora-us-east-2
```

### Rollback

```bash
# Rollback to previous version
argocd app rollback aragora-us-east-2

# Rollback to specific revision
argocd app rollback aragora-us-east-2 --revision 5
```

### Diff Changes

```bash
# Preview changes before sync
argocd app diff aragora-us-east-2
```

## Sync Policies

| Environment | Auto-Sync | Self-Heal | Prune | Manual Approval |
|-------------|-----------|-----------|-------|-----------------|
| Staging     | Yes       | Yes       | Yes   | No              |
| Production  | Yes       | Yes       | Yes   | No (sync windows) |

### Sync Windows

Production deployments are allowed during:
- **Business hours**: Mon-Fri 6am-10pm UTC
- **Blocked**: Christmas Day (Dec 25)

Override for emergencies:
```bash
argocd app sync aragora-us-east-2 --force
```

## RBAC

| Role | Permissions |
|------|-------------|
| `admin` | Full access (sync, delete, create) |
| `developer` | View, sync, view logs |
| `viewer` | Read-only access |

Assign roles via GitHub groups:
- `aragora-admins` → admin role
- `aragora-developers` → developer role
- `aragora-viewers` → viewer role

## Troubleshooting

### Application Not Syncing

```bash
# Check sync status
argocd app get aragora-us-east-2

# View sync errors
argocd app sync aragora-us-east-2 --dry-run

# Check repo connectivity
argocd repo get https://github.com/an0mium/aragora
```

### Health Check Failed

```bash
# View application health
argocd app get aragora-us-east-2 -o yaml | grep -A 20 health

# Check pod status
kubectl get pods -n aragora -l app.kubernetes.io/name=aragora
```

### Drift Detected

```bash
# View diff
argocd app diff aragora-us-east-2

# Force sync to correct drift
argocd app sync aragora-us-east-2 --force --prune
```

### Notifications Not Working

```bash
# Check notification controller
kubectl logs -n argocd -l app.kubernetes.io/name=argocd-notifications-controller

# Verify secret
kubectl get secret argocd-notifications-secret -n argocd -o yaml
```

## Migration from GitHub Actions

To migrate from push-based deployment:

1. **Install ArgoCD** (Steps 1-4 above)
2. **Test with staging** first
3. **Disable GitHub Actions** deploy workflows
4. **Monitor** ArgoCD sync for 24-48 hours
5. **Enable production** ApplicationSet

Keep GitHub Actions for:
- Docker image building
- Running tests
- Security scanning

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        GitHub                                │
│  ┌─────────────┐                                            │
│  │ main branch │◄──── Merge PR                              │
│  └──────┬──────┘                                            │
└─────────┼───────────────────────────────────────────────────┘
          │
          │ Poll (3 min)
          ▼
┌─────────────────────────────────────────────────────────────┐
│                       ArgoCD                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              ApplicationSet                           │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │  │ us-east-2   │ │ eu-west-1   │ │ ap-south-1  │    │   │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘    │   │
│  └─────────┼───────────────┼───────────────┼────────────┘   │
└────────────┼───────────────┼───────────────┼────────────────┘
             │               │               │
             ▼               ▼               ▼
      ┌──────────┐    ┌──────────┐    ┌──────────┐
      │ EKS      │    │ EKS      │    │ EKS      │
      │ US-East  │    │ EU-West  │    │ AP-South │
      └──────────┘    └──────────┘    └──────────┘
```

## References

- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)
- [ApplicationSet Controller](https://argocd-applicationset.readthedocs.io/)
- [External Secrets Operator](https://external-secrets.io/)
- [Aragora Helm Chart](../helm/aragora/)
