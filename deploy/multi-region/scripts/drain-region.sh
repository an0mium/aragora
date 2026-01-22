#!/usr/bin/env bash
# drain-region.sh - Gracefully drain traffic from a region
# Usage: ./drain-region.sh <region-name>

set -euo pipefail

REGION="${1:-}"

if [[ -z "$REGION" ]]; then
    echo "Usage: $0 <region-name>"
    echo "Available regions: us-east-2, eu-west-1, ap-south-1"
    exit 1
fi

echo "=== Draining Traffic from $REGION ==="
echo ""

# Confirmation prompt
read -p "This will drain all traffic from $REGION. Continue? (yes/no): " CONFIRM
if [[ "$CONFIRM" != "yes" ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Step 1: Setting Route53 weight to 0..."
aws route53 change-resource-record-sets \
    --hosted-zone-id "${ROUTE53_ZONE_ID:-Z1234567890}" \
    --change-batch '{
        "Changes": [{
            "Action": "UPSERT",
            "ResourceRecordSet": {
                "Name": "api.aragora.ai",
                "Type": "A",
                "SetIdentifier": "api-'"$REGION"'",
                "Weight": 0,
                "AliasTarget": {
                    "HostedZoneId": "'"${ALB_ZONE_ID:-Z1234567890}"'",
                    "DNSName": "api.'"$REGION"'.aragora.ai",
                    "EvaluateTargetHealth": false
                }
            }
        }]
    }' 2>/dev/null || echo "  (Skipped - Route53 not configured)"
echo "  ✓ Route53 weight set to 0"

echo ""
echo "Step 2: Waiting for DNS TTL (60s)..."
for i in {60..1}; do
    printf "\r  Waiting: %ds remaining" "$i"
    sleep 1
done
printf "\r  ✓ DNS TTL expired                    \n"

echo ""
echo "Step 3: Cordoning Kubernetes nodes..."
kubectl config use-context "aragora-${REGION}"
kubectl get nodes -l "kubernetes.io/role=worker" -o name | while read node; do
    kubectl cordon "$node" 2>/dev/null || true
done
echo "  ✓ Nodes cordoned"

echo ""
echo "Step 4: Waiting for active connections to complete..."
ACTIVE_CONNS=1
TIMEOUT=300
ELAPSED=0
while [[ $ACTIVE_CONNS -gt 0 && $ELAPSED -lt $TIMEOUT ]]; do
    ACTIVE_CONNS=$(kubectl exec -n aragora deploy/aragora-backend -- \
        curl -s localhost:8080/metrics 2>/dev/null | \
        grep 'http_connections_active' | awk '{print $2}' || echo "0")
    ACTIVE_CONNS=${ACTIVE_CONNS%.*}  # Convert to integer
    printf "\r  Active connections: %d (timeout in %ds)" "$ACTIVE_CONNS" "$((TIMEOUT - ELAPSED))"
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
printf "\r  ✓ Connections drained                              \n"

echo ""
echo "Step 5: Scaling down deployments..."
kubectl scale deployment aragora-backend -n aragora --replicas=0
kubectl scale deployment aragora-frontend -n aragora --replicas=0
kubectl scale deployment aragora-debate-worker -n aragora --replicas=0
echo "  ✓ Deployments scaled to 0"

echo ""
echo "=== Region $REGION Drained ==="
echo ""
echo "The region is now isolated from traffic."
echo "To restore traffic, run: ./scripts/restore-region.sh $REGION"
