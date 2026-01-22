# Multi-Region Deployment Guide

This directory contains configurations for deploying Aragora across multiple geographic regions.

## Architecture Overview

```
                    ┌──────────────────────────────────────────┐
                    │          Global Load Balancer            │
                    │     (CloudFlare/AWS Route53/GCP GLB)     │
                    └─────────────┬────────────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
    ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
    │   US-EAST-1   │     │   EU-WEST-1   │     │   AP-SOUTH-1  │
    │    Region     │     │    Region     │     │    Region     │
    └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
            │                     │                     │
    ┌───────┴───────┐     ┌───────┴───────┐     ┌───────┴───────┐
    │   K8s Cluster │     │   K8s Cluster │     │   K8s Cluster │
    │   + Istio     │◄───►│   + Istio     │◄───►│   + Istio     │
    └───────────────┘     └───────────────┘     └───────────────┘
            │                     │                     │
    ┌───────┴───────┐     ┌───────┴───────┐     ┌───────┴───────┐
    │  PostgreSQL   │     │  PostgreSQL   │     │  PostgreSQL   │
    │   (Primary)   │────►│   (Replica)   │────►│   (Replica)   │
    └───────────────┘     └───────────────┘     └───────────────┘
            │                     │                     │
    ┌───────┴───────┐     ┌───────┴───────┐     ┌───────┴───────┐
    │ Redis Cluster │◄───►│ Redis Cluster │◄───►│ Redis Cluster │
    └───────────────┘     └───────────────┘     └───────────────┘
```

## Components

| Component | Purpose | Files |
|-----------|---------|-------|
| Global LB | Geographic DNS routing | `cloudflare/`, `terraform/` |
| Kubernetes | Per-region clusters | `kubernetes/` |
| Istio | Service mesh & mTLS | `istio/` |
| PostgreSQL | Primary + read replicas | `postgres/` |
| Redis | Distributed state sync | `redis/` |
| Helm Charts | Templated deployment | `helm/` |
| Monitoring | Multi-region observability | `monitoring/` |

## Quick Start

### 1. Prerequisites

```bash
# Required tools
kubectl >= 1.28
helm >= 3.12
istioctl >= 1.20
terraform >= 1.5
```

### 2. Deploy Primary Region (US-EAST-1)

```bash
# Set context
export ARAGORA_REGION=us-east-2
export ARAGORA_PRIMARY=true

# Deploy infrastructure
cd terraform/aws
terraform init
terraform apply -var="region=us-east-2" -var="primary=true"

# Deploy Kubernetes resources
cd ../../kubernetes
helm upgrade --install aragora ./helm/aragora \
  -f helm/values/us-east-2.yaml \
  --set region.name=us-east-2 \
  --set region.primary=true
```

### 3. Deploy Secondary Regions

```bash
# EU-WEST-1
export ARAGORA_REGION=eu-west-1
helm upgrade --install aragora ./helm/aragora \
  -f helm/values/eu-west-1.yaml \
  --set region.name=eu-west-1 \
  --set region.primary=false \
  --set database.replicaOf=us-east-2

# AP-SOUTH-1
export ARAGORA_REGION=ap-south-1
helm upgrade --install aragora ./helm/aragora \
  -f helm/values/ap-south-1.yaml \
  --set region.name=ap-south-1 \
  --set region.primary=false \
  --set database.replicaOf=us-east-2
```

### 4. Configure Global Load Balancer

```bash
# CloudFlare (recommended)
cd cloudflare
terraform apply

# Or AWS Route53
cd ../terraform/aws
terraform apply -target=module.route53
```

## Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ARAGORA_REGION` | Current region identifier | `us-east-2` |
| `ARAGORA_REGION_ID` | Unique region ID | `region-001` |
| `ARAGORA_PRIMARY` | Is this the primary region | `true` |
| `REDIS_URL` | Regional Redis endpoint | `redis://redis.us-east-2:6379` |
| `DATABASE_URL` | Regional PostgreSQL endpoint | `postgresql://...` |
| `FEDERATION_ENABLED` | Enable cross-region sync | `true` |

### Helm Values

See `helm/values/` for region-specific configurations:

- `base.yaml` - Shared configuration
- `us-east-2.yaml` - US East (Primary)
- `eu-west-1.yaml` - EU West (Secondary)
- `ap-south-1.yaml` - Asia Pacific (Secondary)

## Failover

### Automatic Failover

The system uses health checks to automatically route traffic away from unhealthy regions:

1. **Health Probes**: `/health/live` and `/health/ready` endpoints
2. **Circuit Breakers**: Application-level (see `platform_resilience.py`)
3. **DNS Failover**: CloudFlare/Route53 health checks

### Manual Failover

```bash
# Promote secondary to primary
./scripts/promote-region.sh eu-west-1

# Drain traffic from region
./scripts/drain-region.sh us-east-2
```

## Monitoring

Multi-region dashboards are available in Grafana:

- **Global Overview**: Cross-region traffic, latency, errors
- **Regional Health**: Per-region metrics and SLOs
- **Replication Lag**: Database and Redis sync status
- **Federation Status**: Knowledge sync metrics

## Disaster Recovery

| Metric | Target |
|--------|--------|
| RTO (Recovery Time Objective) | < 5 minutes |
| RPO (Recovery Point Objective) | < 1 minute |
| Backup Frequency | Every 6 hours |
| Cross-Region Backup Replication | Real-time |

See `disaster-recovery/` for runbooks and automation.
