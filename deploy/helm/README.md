# Aragora Helm Chart

Production-grade Helm chart for self-hosted Aragora deployments on Kubernetes.

## Prerequisites

- Kubernetes 1.25+
- Helm 3.10+
- At least one AI provider API key (Anthropic or OpenAI)
- (Optional) cert-manager for automated TLS
- (Optional) Prometheus Operator for metrics scraping

## Quick Install

```bash
# Add dependencies (if using built-in PostgreSQL or Redis)
helm dependency update deploy/helm/aragora/

# Minimal install with Anthropic key
helm install aragora deploy/helm/aragora/ \
  --namespace aragora --create-namespace \
  --set secrets.apiKeys.anthropicApiKey=sk-ant-YOUR-KEY

# With external PostgreSQL and Redis
helm install aragora deploy/helm/aragora/ \
  --namespace aragora --create-namespace \
  --set secrets.apiKeys.anthropicApiKey=sk-ant-YOUR-KEY \
  --set secrets.database.postgresUrl=postgresql://user:pass@db-host:5432/aragora \
  --set secrets.database.redisUrl=redis://redis-host:6379 \
  --set config.dbBackend=postgres

# With built-in PostgreSQL and Redis subcharts
helm install aragora deploy/helm/aragora/ \
  --namespace aragora --create-namespace \
  --set secrets.apiKeys.anthropicApiKey=sk-ant-YOUR-KEY \
  --set postgresql.enabled=true \
  --set postgresql.auth.password=my-secure-password \
  --set redis.enabled=true \
  --set config.dbBackend=postgres

# Production install with ingress and TLS
helm install aragora deploy/helm/aragora/ \
  --namespace aragora --create-namespace \
  -f my-values.yaml
```

## Configuration

### Core Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `backend.replicaCount` | Number of backend replicas | `2` |
| `backend.image.repository` | Backend container image | `ghcr.io/aragora/aragora` |
| `backend.image.tag` | Backend image tag | Chart `appVersion` |
| `backend.resources.requests.cpu` | Backend CPU request | `250m` |
| `backend.resources.requests.memory` | Backend memory request | `512Mi` |
| `backend.resources.limits.cpu` | Backend CPU limit | `2000m` |
| `backend.resources.limits.memory` | Backend memory limit | `4Gi` |
| `frontend.enabled` | Deploy the Next.js frontend | `true` |
| `frontend.replicaCount` | Number of frontend replicas | `2` |
| `frontend.image.repository` | Frontend container image | `ghcr.io/aragora/aragora-frontend` |

### Secrets

| Parameter | Description | Default |
|-----------|-------------|---------|
| `secrets.create` | Create a Kubernetes Secret | `true` |
| `secrets.existingSecret` | Use an existing secret (overrides create) | `""` |
| `secrets.apiKeys.anthropicApiKey` | Anthropic API key | `""` |
| `secrets.apiKeys.openaiApiKey` | OpenAI API key | `""` |
| `secrets.apiKeys.openrouterApiKey` | OpenRouter API key (fallback) | `""` |
| `secrets.apiKeys.mistralApiKey` | Mistral API key | `""` |
| `secrets.aragoraApiToken` | Aragora API authentication token | `""` |
| `secrets.aragoraSecretKey` | Session signing secret key | `""` |
| `secrets.database.postgresUrl` | Full PostgreSQL connection string | `""` |
| `secrets.database.redisUrl` | Full Redis connection string | `""` |

### Ingress

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ingress.enabled` | Enable ingress | `false` |
| `ingress.className` | Ingress class (nginx, traefik, alb) | `""` |
| `ingress.host` | Hostname | `aragora.example.com` |
| `ingress.tls.enabled` | Enable TLS | `false` |
| `ingress.tls.secretName` | TLS secret name | `aragora-tls` |
| `ingress.annotations` | Ingress annotations | `{}` |

### Storage

| Parameter | Description | Default |
|-----------|-------------|---------|
| `persistence.enabled` | Enable PVC for data directory | `true` |
| `persistence.storageClass` | Storage class (empty = default) | `""` |
| `persistence.size` | Volume size | `10Gi` |
| `persistence.accessModes` | Access modes | `[ReadWriteOnce]` |

### Database and Cache

| Parameter | Description | Default |
|-----------|-------------|---------|
| `postgresql.enabled` | Deploy PostgreSQL subchart | `false` |
| `postgresql.auth.database` | Database name | `aragora` |
| `postgresql.auth.username` | Database user | `aragora` |
| `redis.enabled` | Deploy Redis subchart | `false` |
| `externalDatabase.host` | External PostgreSQL host | `""` |
| `externalRedis.host` | External Redis host | `""` |
| `config.dbBackend` | Database backend (auto/postgres/sqlite) | `auto` |

### Autoscaling

| Parameter | Description | Default |
|-----------|-------------|---------|
| `backend.autoscaling.enabled` | Enable backend HPA | `false` |
| `backend.autoscaling.minReplicas` | Minimum backend replicas | `2` |
| `backend.autoscaling.maxReplicas` | Maximum backend replicas | `10` |
| `backend.autoscaling.targetCPUUtilizationPercentage` | CPU threshold | `70` |
| `frontend.autoscaling.enabled` | Enable frontend HPA | `false` |

### Security

| Parameter | Description | Default |
|-----------|-------------|---------|
| `podSecurityContext.runAsNonRoot` | Enforce non-root containers | `true` |
| `podSecurityContext.runAsUser` | UID for containers | `1000` |
| `securityContext.readOnlyRootFilesystem` | Read-only root FS | `true` |
| `securityContext.allowPrivilegeEscalation` | Block privilege escalation | `false` |
| `networkPolicy.enabled` | Enable network policies | `false` |
| `serviceAccount.create` | Create dedicated service account | `true` |

### Monitoring

| Parameter | Description | Default |
|-----------|-------------|---------|
| `serviceMonitor.enabled` | Create Prometheus ServiceMonitor | `false` |
| `serviceMonitor.interval` | Scrape interval | `15s` |
| `podDisruptionBudget.enabled` | Enable PDB | `false` |
| `podDisruptionBudget.minAvailable` | Min pods during disruption | `1` |

## Upgrade

```bash
# Standard upgrade
helm upgrade aragora deploy/helm/aragora/ \
  --namespace aragora \
  -f my-values.yaml

# Upgrade with rollback on failure
helm upgrade aragora deploy/helm/aragora/ \
  --namespace aragora \
  -f my-values.yaml \
  --atomic --timeout 5m
```

## Using External Secrets

For production deployments, manage secrets outside of Helm values using
External Secrets Operator, Sealed Secrets, or a vault integration:

```yaml
# my-values.yaml
secrets:
  create: false
  existingSecret: aragora-external-secrets
```

The existing secret must contain keys matching the env var names:
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `ARAGORA_API_TOKEN`, `DATABASE_URL`,
`REDIS_URL`, etc.

## Uninstall

```bash
helm uninstall aragora --namespace aragora

# Remove PVCs (data will be lost)
kubectl delete pvc -l app.kubernetes.io/instance=aragora -n aragora
```

## Architecture

```
                     Ingress
                        |
           +------------+------------+
           |                         |
    /api, /ws, /healthz         / (everything else)
           |                         |
  +--------+--------+    +----------+---------+
  | Backend Service  |    | Frontend Service   |
  | (port 8080/8765) |    | (port 3000)        |
  +---------+--------+    +--------------------+
            |
     +------+------+
     |             |
  PostgreSQL     Redis
```

The backend serves the HTTP API on port 8080 and WebSocket connections on
port 8765. The frontend is a Next.js application served on port 3000.
Ingress routes `/api/*`, `/ws/*`, and `/healthz` to the backend; all other
paths go to the frontend.
