# Kyverno Policy Enforcement for Aragora

This directory contains Kyverno policies for enforcing security and operational best practices in the Aragora Kubernetes clusters.

## Overview

Kyverno provides:
- **Policy as Code**: Define policies in native Kubernetes YAML
- **Validation**: Block non-compliant resources at admission
- **Mutation**: Automatically add required labels, annotations, and defaults
- **Generation**: Auto-create resources (PDBs, NetworkPolicies, ResourceQuotas)
- **Background Scanning**: Audit existing resources for compliance

## Directory Structure

```
kyverno/
├── install/              # Kyverno installation values
│   └── values.yaml       # Helm values for Kyverno itself
├── policies/             # Policy definitions
│   ├── security-policies.yaml      # Security enforcement
│   └── operational-policies.yaml   # Automation and best practices
└── README.md            # This file
```

## Prerequisites

1. **Kubernetes cluster** v1.25+
2. **kubectl** configured with cluster access
3. **Helm 3** installed

## Installation

### Step 1: Install Kyverno

```bash
# Add Kyverno Helm repo
helm repo add kyverno https://kyverno.github.io/kyverno/
helm repo update

# Create namespace
kubectl create namespace kyverno

# Install Kyverno with HA configuration
helm install kyverno kyverno/kyverno \
  -n kyverno \
  -f deploy/kyverno/install/values.yaml

# Wait for rollout
kubectl rollout status deployment/kyverno-admission-controller -n kyverno
```

### Step 2: Apply Policies

```bash
# Apply security policies
kubectl apply -f deploy/kyverno/policies/security-policies.yaml

# Apply operational policies
kubectl apply -f deploy/kyverno/policies/operational-policies.yaml

# Verify policies are ready
kubectl get clusterpolicies
```

## Policies

### Security Policies

| Policy | Severity | Action | Description |
|--------|----------|--------|-------------|
| `require-run-as-non-root` | High | Enforce | Containers must run as non-root |
| `disallow-privileged-containers` | High | Enforce | Block privileged containers |
| `require-resource-limits` | Medium | Enforce | Require CPU/memory limits |
| `require-resource-requests` | Medium | Enforce | Require CPU/memory requests |
| `disallow-latest-tag` | Medium | Enforce | Block :latest image tags |
| `require-probes` | Medium | Audit | Require liveness/readiness probes |
| `require-labels` | Low | Audit | Require standard labels |
| `disallow-host-network` | High | Enforce | Block host network access |
| `disallow-host-pid` | High | Enforce | Block host PID namespace |
| `restrict-volume-types` | Medium | Enforce | Block hostPath volumes |
| `require-ro-rootfs` | Medium | Audit | Require read-only root filesystem |

### Operational Policies

| Policy | Category | Type | Description |
|--------|----------|------|-------------|
| `add-default-labels` | Best Practices | Mutate | Add standard labels to resources |
| `generate-pdb` | HA | Generate | Auto-create PodDisruptionBudgets |
| `generate-network-policy` | Security | Generate | Auto-create default deny NetworkPolicy |
| `add-prometheus-annotations` | Observability | Mutate | Add Prometheus scrape annotations |
| `enforce-image-pull-policy` | Best Practices | Mutate | Set imagePullPolicy to Always |
| `add-spot-tolerations` | Cost | Mutate | Add tolerations for spot instances |
| `require-external-annotations` | Documentation | Audit | Require annotations on LoadBalancers |
| `generate-resource-quota` | Resource Mgmt | Generate | Auto-create namespace quotas |
| `generate-limit-range` | Resource Mgmt | Generate | Auto-create container limits |

## Configuration

### Excluded Namespaces

System namespaces are excluded from policy enforcement:
- `kube-system`
- `kube-public`
- `kube-node-lease`
- `kyverno`
- `argocd`
- `cert-manager`
- `external-secrets`
- `monitoring`

### Excluded Groups

The following groups bypass policy validation:
- `system:masters`
- `system:serviceaccounts:kube-system`

## Policy Exceptions

For workloads that legitimately require exception from a policy:

```yaml
apiVersion: kyverno.io/v2beta1
kind: PolicyException
metadata:
  name: allow-privileged-csi-driver
  namespace: kyverno
spec:
  exceptions:
    - policyName: disallow-privileged-containers
      ruleNames:
        - deny-privileged
  match:
    any:
      - resources:
          kinds:
            - Pod
          namespaces:
            - csi-driver
          names:
            - "*-node-*"
```

Apply exception:
```bash
kubectl apply -f my-exception.yaml
```

## Validation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `Enforce` | Block non-compliant resources | Production security policies |
| `Audit` | Allow but report violations | New policies, gradual rollout |

### Switching Modes

To change a policy from Audit to Enforce:

```bash
kubectl patch clusterpolicy require-probes --type merge -p '
spec:
  validationFailureAction: Enforce
'
```

## Monitoring

### View Policy Reports

```bash
# View cluster-wide policy reports
kubectl get clusterpolicyreport

# View namespace-scoped reports
kubectl get policyreport -A

# Detailed report for a namespace
kubectl get policyreport -n aragora -o yaml
```

### Prometheus Metrics

Kyverno exports metrics to Prometheus:

```bash
# Port-forward to metrics endpoint
kubectl port-forward svc/kyverno-svc -n kyverno 8000:443

# View metrics
curl -k https://localhost:8000/metrics
```

Key metrics:
- `kyverno_policy_results_total` - Policy evaluation results
- `kyverno_admission_requests_total` - Admission requests
- `kyverno_policy_execution_duration_seconds` - Policy execution time

### Grafana Dashboard

Import the Kyverno dashboard from Grafana:
- Dashboard ID: `15804`
- Or use the dashboard from `deploy/monitoring/dashboards/kyverno.json`

## Troubleshooting

### Policy Not Enforcing

```bash
# Check policy status
kubectl get clusterpolicy <policy-name> -o yaml

# Check Kyverno logs
kubectl logs -n kyverno -l app.kubernetes.io/component=admission-controller

# Test policy against a resource
kubectl apply -f my-pod.yaml --dry-run=server
```

### Background Scan Issues

```bash
# Check background controller
kubectl logs -n kyverno -l app.kubernetes.io/component=background-controller

# Trigger manual scan
kubectl annotate clusterpolicy <policy-name> \
  policies.kyverno.io/rescan=true
```

### Policy Violations

```bash
# List all violations
kubectl get policyreport -A -o jsonpath='{range .items[*]}{.metadata.namespace}{"\t"}{.results[?(@.result=="fail")].policy}{"\n"}{end}'

# Get details for a specific violation
kubectl describe policyreport -n aragora
```

### Admission Timeouts

If policies cause admission timeouts:

1. Check resource limits on Kyverno pods
2. Reduce policy complexity
3. Increase webhook timeout in values.yaml

```yaml
webhooks:
  timeoutSeconds: 30  # Increase from default 10
```

## Best Practices

1. **Start with Audit mode** - New policies should use `Audit` before `Enforce`
2. **Use background scanning** - Identify existing violations before enforcement
3. **Create exceptions sparingly** - Document why each exception is needed
4. **Monitor policy performance** - Use metrics to identify slow policies
5. **Version control policies** - All policies should be in Git
6. **Test in staging** - Always test policy changes in staging first

## Integration with ArgoCD

Kyverno policies are deployed via ArgoCD:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: kyverno-policies
  namespace: argocd
spec:
  project: aragora
  source:
    repoURL: https://github.com/an0mium/aragora
    targetRevision: main
    path: deploy/kyverno/policies
  destination:
    server: https://kubernetes.default.svc
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

## References

- [Kyverno Documentation](https://kyverno.io/docs/)
- [Policy Library](https://kyverno.io/policies/)
- [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [Aragora ArgoCD Configuration](../argocd/)
