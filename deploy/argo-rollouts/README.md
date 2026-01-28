# Argo Rollouts - Canary Deployments for Aragora

This directory contains Argo Rollouts configuration for progressive delivery (canary and blue-green deployments).

## Overview

Argo Rollouts provides:
- **Canary Deployments**: Gradual traffic shifting to new versions
- **Blue-Green Deployments**: Instant traffic switching with easy rollback
- **Automated Analysis**: Metrics-based promotion/rollback decisions
- **Traffic Management**: Integration with nginx, Istio, AWS ALB
- **Notifications**: Slack/webhook alerts for deployment status

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              Ingress (nginx)             │
                    │         api.aragora.ai → /               │
                    └────────────────┬────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │        Traffic Splitting         │
                    │   (controlled by Argo Rollouts)  │
                    └────────────────┬────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                       │
              ▼                      ▼                       │
    ┌─────────────────┐    ┌─────────────────┐              │
    │  Stable Service │    │  Canary Service │              │
    │   (80-100%)     │    │    (0-20%)      │              │
    └────────┬────────┘    └────────┬────────┘              │
             │                      │                        │
             ▼                      ▼                        │
    ┌─────────────────┐    ┌─────────────────┐              │
    │  Stable Pods    │    │  Canary Pods    │◄─────────────┘
    │   (v1.0.0)      │    │   (v1.1.0)      │    Analysis
    │   3 replicas    │    │   1 replica     │    Running
    └─────────────────┘    └─────────────────┘
```

## Directory Structure

```
argo-rollouts/
├── install/              # Argo Rollouts controller installation
│   └── values.yaml       # Helm values for the controller
├── rollout.yaml          # Main Rollout resource with canary strategy
└── README.md             # This file
```

## Prerequisites

1. **Kubernetes cluster** v1.23+
2. **kubectl** and **Helm 3**
3. **nginx-ingress** or other supported ingress controller
4. **Prometheus** (for metrics-based analysis)

## Installation

### Step 1: Install Argo Rollouts Controller

```bash
# Add Argo Helm repo
helm repo add argo https://argoproj.github.io/argo-helm
helm repo update

# Create namespace
kubectl create namespace argo-rollouts

# Install with custom values
helm install argo-rollouts argo/argo-rollouts \
  -n argo-rollouts \
  -f deploy/argo-rollouts/install/values.yaml

# Verify installation
kubectl get pods -n argo-rollouts
```

### Step 2: Install kubectl Plugin (Optional but Recommended)

```bash
# macOS
brew install argoproj/tap/kubectl-argo-rollouts

# Linux
curl -LO https://github.com/argoproj/argo-rollouts/releases/latest/download/kubectl-argo-rollouts-linux-amd64
chmod +x kubectl-argo-rollouts-linux-amd64
sudo mv kubectl-argo-rollouts-linux-amd64 /usr/local/bin/kubectl-argo-rollouts

# Verify
kubectl argo rollouts version
```

### Step 3: Deploy Aragora Rollout

```bash
# Apply rollout and related resources
kubectl apply -f deploy/argo-rollouts/rollout.yaml -n aragora

# Verify rollout status
kubectl argo rollouts get rollout aragora -n aragora
```

## Canary Strategy

The default strategy uses a gradual traffic shift:

| Step | Traffic to Canary | Duration | Analysis |
|------|-------------------|----------|----------|
| 1    | 5%                | 2 min    | ✓        |
| 2    | 20%               | 5 min    | ✓        |
| 3    | 50%               | 5 min    | ✓        |
| 4    | 80%               | 5 min    | ✓        |
| 5    | 100%              | -        | Complete |

**Total rollout time**: ~17 minutes (if all analysis passes)

### Analysis Metrics

The following metrics are evaluated during canary:

1. **Success Rate**: HTTP 2xx responses must be ≥ 99%
2. **P99 Latency**: Must be < 500ms
3. **Error Rate**: HTTP 5xx responses must be < 1%
4. **Pod Restarts**: Must be 0 during rollout

If any metric fails 3 consecutive times, the rollout is automatically aborted.

## Usage

### Trigger a Rollout

```bash
# Update image (triggers canary)
kubectl argo rollouts set image aragora aragora=aragora/aragora:v1.1.0 -n aragora

# Or patch the rollout
kubectl patch rollout aragora -n aragora \
  --type merge \
  -p '{"spec":{"template":{"spec":{"containers":[{"name":"aragora","image":"aragora/aragora:v1.1.0"}]}}}}'
```

### Monitor Rollout

```bash
# Watch rollout status
kubectl argo rollouts get rollout aragora -n aragora --watch

# View in dashboard
kubectl argo rollouts dashboard -n argo-rollouts
# Open http://localhost:3100

# CLI status
kubectl argo rollouts status aragora -n aragora
```

### Manual Promotion

Skip waiting at a pause step:

```bash
# Promote to next step
kubectl argo rollouts promote aragora -n aragora

# Skip all steps and fully promote
kubectl argo rollouts promote aragora -n aragora --full
```

### Abort Rollout

```bash
# Abort and rollback
kubectl argo rollouts abort aragora -n aragora

# After abort, you may need to retry
kubectl argo rollouts retry rollout aragora -n aragora
```

### Rollback

```bash
# Rollback to previous revision
kubectl argo rollouts undo aragora -n aragora

# Rollback to specific revision
kubectl argo rollouts undo aragora -n aragora --to-revision=3

# View revision history
kubectl argo rollouts get rollout aragora -n aragora
```

## Testing Canary

### Force Traffic to Canary

You can force requests to the canary using headers:

```bash
# Request sent to canary (X-Canary: true)
curl -H "X-Canary: true" https://api.aragora.ai/health

# Normal request (goes to stable or canary based on weight)
curl https://api.aragora.ai/health
```

### Simulate Canary Failure

To test automatic rollback:

```bash
# Deploy a broken version
kubectl argo rollouts set image aragora aragora=aragora/aragora:broken -n aragora

# Watch it fail analysis and rollback
kubectl argo rollouts get rollout aragora -n aragora --watch
```

## Blue-Green Strategy

For instant cutover instead of gradual canary:

```yaml
spec:
  strategy:
    blueGreen:
      activeService: aragora-active
      previewService: aragora-preview
      autoPromotionEnabled: false
      scaleDownDelaySeconds: 30
      prePromotionAnalysis:
        templates:
          - templateName: aragora-canary-analysis
```

## Monitoring

### Prometheus Metrics

Argo Rollouts exports these metrics:

```promql
# Rollout info
rollout_info{namespace="aragora", rollout="aragora"}

# Phase distribution
rollout_phase{namespace="aragora"}

# Paused status
rollout_info_replicas_paused{namespace="aragora"}
```

### Grafana Dashboard

Import dashboard ID: `14314` (Argo Rollouts Dashboard)

Or use the custom dashboard:
```bash
kubectl apply -f deploy/monitoring/dashboards/argo-rollouts.json
```

### Alerts

```yaml
groups:
  - name: argo-rollouts
    rules:
      - alert: RolloutAborted
        expr: rollout_phase{phase="Aborted"} == 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Rollout {{ $labels.rollout }} was aborted"

      - alert: RolloutDegraded
        expr: rollout_phase{phase="Degraded"} == 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Rollout {{ $labels.rollout }} is degraded"
```

## Notifications

Configure Slack notifications:

```bash
# Create notification secret
kubectl create secret generic argocd-notifications-secret \
  -n argo-rollouts \
  --from-literal=slack-token=xoxb-your-token

# Notifications are sent on:
# - Rollout completed
# - Rollout aborted
# - Analysis failed
```

## Integration with ArgoCD

Use ArgoCD to manage Rollouts:

```yaml
# ArgoCD Application for rollout
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: aragora-rollout
  namespace: argocd
spec:
  project: aragora
  source:
    repoURL: https://github.com/an0mium/aragora
    targetRevision: main
    path: deploy/argo-rollouts
  destination:
    server: https://kubernetes.default.svc
    namespace: aragora
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

## Troubleshooting

### Rollout Stuck

```bash
# Check rollout events
kubectl describe rollout aragora -n aragora

# Check analysis run
kubectl get analysisrun -n aragora
kubectl describe analysisrun <name> -n aragora

# Force promote if analysis is stuck
kubectl argo rollouts promote aragora -n aragora --full
```

### Analysis Failing

```bash
# Check analysis run details
kubectl get analysisrun -l rollout=aragora -n aragora

# View failed metrics
kubectl describe analysisrun <name> -n aragora

# Check Prometheus connectivity
kubectl exec -n aragora -it deploy/aragora -- \
  curl http://prometheus.monitoring.svc.cluster.local:9090/api/v1/query?query=up
```

### Traffic Not Shifting

```bash
# Verify ingress annotations
kubectl get ingress aragora-stable -n aragora -o yaml

# Check nginx configuration
kubectl exec -n ingress-nginx deploy/ingress-nginx-controller -- \
  cat /etc/nginx/nginx.conf | grep -A 20 "aragora"
```

## Best Practices

1. **Start Conservative**: Begin with low canary percentage (5%)
2. **Use Multiple Metrics**: Don't rely on single metric for promotion
3. **Set Appropriate Timeouts**: Allow enough time for metrics to stabilize
4. **Test Rollback**: Regularly test that rollback works correctly
5. **Monitor Analysis**: Watch analysis runs during deployments
6. **Document Changes**: Include rollout status in deployment PRs

## References

- [Argo Rollouts Documentation](https://argoproj.github.io/argo-rollouts/)
- [Traffic Management](https://argoproj.github.io/argo-rollouts/features/traffic-management/)
- [Analysis & Progressive Delivery](https://argoproj.github.io/argo-rollouts/features/analysis/)
- [ArgoCD Integration](../argocd/README.md)
