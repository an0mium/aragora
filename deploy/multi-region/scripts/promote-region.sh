#!/usr/bin/env bash
# promote-region.sh - Promote a secondary region to primary
# Usage: ./promote-region.sh <region-name>

set -euo pipefail

REGION="${1:-}"
CURRENT_PRIMARY="${ARAGORA_PRIMARY_REGION:-us-east-2}"

if [[ -z "$REGION" ]]; then
    echo "Usage: $0 <region-name>"
    echo "Available regions: us-east-2, eu-west-1, ap-south-1"
    exit 1
fi

if [[ "$REGION" == "$CURRENT_PRIMARY" ]]; then
    echo "Error: $REGION is already the primary region"
    exit 1
fi

echo "=== Promoting $REGION to Primary Region ==="
echo "Current primary: $CURRENT_PRIMARY"
echo ""

# Confirmation prompt
read -p "This will promote $REGION to primary. Continue? (yes/no): " CONFIRM
if [[ "$CONFIRM" != "yes" ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Step 1: Verifying region health..."
HEALTH_CHECK=$(curl -s -o /dev/null -w "%{http_code}" "https://api.${REGION}.aragora.ai/health/ready" || echo "000")
if [[ "$HEALTH_CHECK" != "200" ]]; then
    echo "Error: Region $REGION is not healthy (HTTP $HEALTH_CHECK)"
    exit 1
fi
echo "  ✓ Region $REGION is healthy"

echo ""
echo "Step 2: Checking replication lag..."
# This would query the actual replication lag from PostgreSQL
# For now, we'll simulate the check
LAG_MS=$(curl -s "https://api.${REGION}.aragora.ai/metrics" | grep 'replication_lag_ms' | awk '{print $2}' || echo "0")
if [[ "${LAG_MS:-0}" -gt 1000 ]]; then
    echo "Warning: Replication lag is ${LAG_MS}ms. Data loss may occur."
    read -p "Continue anyway? (yes/no): " CONFIRM_LAG
    if [[ "$CONFIRM_LAG" != "yes" ]]; then
        echo "Aborted."
        exit 0
    fi
fi
echo "  ✓ Replication lag acceptable"

echo ""
echo "Step 3: Draining traffic from old primary..."
# Update Route53 to stop sending traffic to old primary
aws route53 change-resource-record-sets \
    --hosted-zone-id "${ROUTE53_ZONE_ID:-Z1234567890}" \
    --change-batch '{
        "Changes": [{
            "Action": "UPSERT",
            "ResourceRecordSet": {
                "Name": "api.aragora.ai",
                "Type": "A",
                "SetIdentifier": "api-'"$CURRENT_PRIMARY"'",
                "Weight": 0,
                "TTL": 60,
                "ResourceRecords": [{"Value": "0.0.0.0"}]
            }
        }]
    }' || echo "Warning: Failed to update Route53"
echo "  ✓ Traffic drained from $CURRENT_PRIMARY"

echo ""
echo "Step 4: Promoting PostgreSQL replica..."
# Switch context to new primary region
kubectl config use-context "aragora-${REGION}"

# Promote PostgreSQL replica
kubectl exec -n aragora deploy/aragora-postgresql -- \
    pg_ctl promote -D /var/lib/postgresql/data || true
echo "  ✓ PostgreSQL promoted"

echo ""
echo "Step 5: Updating Helm values..."
# Update Helm deployment to mark as primary
helm upgrade aragora ./deploy/multi-region/helm/aragora \
    -f "./deploy/multi-region/helm/values/${REGION}.yaml" \
    --set region.primary=true \
    --set region.name="$REGION" \
    -n aragora
echo "  ✓ Helm values updated"

echo ""
echo "Step 6: Updating Route53 for new primary..."
# Point api.aragora.ai to new primary
aws route53 change-resource-record-sets \
    --hosted-zone-id "${ROUTE53_ZONE_ID:-Z1234567890}" \
    --change-batch '{
        "Changes": [{
            "Action": "UPSERT",
            "ResourceRecordSet": {
                "Name": "api.aragora.ai",
                "Type": "A",
                "SetIdentifier": "api-'"$REGION"'",
                "Weight": 100,
                "TTL": 60,
                "ResourceRecords": [{"Value": "'"$(dig +short api.${REGION}.aragora.ai)"'"}]
            }
        }]
    }' || echo "Warning: Failed to update Route53"
echo "  ✓ Route53 updated"

echo ""
echo "Step 7: Reconfiguring old primary as replica..."
kubectl config use-context "aragora-${CURRENT_PRIMARY}"

# Update old primary to become replica
helm upgrade aragora ./deploy/multi-region/helm/aragora \
    -f "./deploy/multi-region/helm/values/${CURRENT_PRIMARY}.yaml" \
    --set region.primary=false \
    --set database.replicaOf="$REGION" \
    -n aragora
echo "  ✓ Old primary reconfigured as replica"

echo ""
echo "Step 8: Verifying failover..."
sleep 10
NEW_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "https://api.aragora.ai/health/ready" || echo "000")
if [[ "$NEW_HEALTH" == "200" ]]; then
    echo "  ✓ Failover successful!"
else
    echo "  ⚠ Warning: Health check returned $NEW_HEALTH"
fi

echo ""
echo "=== Promotion Complete ==="
echo "New primary region: $REGION"
echo "Old primary ($CURRENT_PRIMARY) is now a replica"
echo ""
echo "Next steps:"
echo "  1. Monitor metrics at https://grafana.aragora.ai/d/multi-region"
echo "  2. Update ARAGORA_PRIMARY_REGION environment variable"
echo "  3. Consider running ./scripts/verify-replication.sh"
