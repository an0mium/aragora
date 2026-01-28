# Aragora Single-Region AWS Infrastructure

Simplified Terraform configuration for deploying Aragora on AWS in a single region.

## Architecture

This creates:
- **VPC** with public, private, and database subnets across 3 AZs
- **EKS** Kubernetes cluster with managed node groups
- **RDS PostgreSQL** with encryption and automated backups
- **ElastiCache Redis** for caching and session storage
- **S3 Bucket** for backup storage with lifecycle policies
- **Security Groups** properly configured for internal communication

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. Terraform >= 1.5.0
3. kubectl installed

## Quick Start

```bash
# 1. Initialize Terraform
terraform init

# 2. Copy and edit variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

# 3. Preview changes
terraform plan

# 4. Apply (this takes ~15-20 minutes)
terraform apply

# 5. Configure kubectl
$(terraform output -raw configure_kubectl)

# 6. Verify cluster access
kubectl get nodes
```

## Cost Estimates

| Size | Monthly Estimate | Concurrent Debates |
|------|-----------------|-------------------|
| Small | ~$200-300 | < 10 |
| Medium | ~$500-800 | 10-100 |
| Large | ~$1,500+ | 100+ |

*Estimates exclude data transfer and vary by region.*

## Configuration Options

### Small Deployment (Default)
```hcl
eks_instance_types = ["t3.medium", "t3.large"]
eks_min_nodes      = 2
db_instance_class  = "db.t3.medium"
redis_node_type    = "cache.t3.small"
```

### Production Deployment
```hcl
eks_instance_types = ["m5.large", "m5.xlarge"]
eks_min_nodes      = 3
db_instance_class  = "db.r6g.large"
redis_node_type    = "cache.r6g.medium"
```

## Post-Deployment

After Terraform completes:

1. **Deploy Aragora via Helm**
   ```bash
   helm install aragora ../helm/aragora \
     --set database.host=$(terraform output -raw rds_endpoint) \
     --set redis.host=$(terraform output -raw redis_endpoint) \
     --set backup.bucket=$(terraform output -raw backup_bucket)
   ```

2. **Get RDS credentials** (stored in AWS Secrets Manager)
   ```bash
   aws secretsmanager get-secret-value \
     --secret-id $(terraform output -raw rds_secret_arn) \
     --query SecretString --output text | jq .
   ```

3. **Configure ingress/TLS** per your requirements

## Cleanup

```bash
# Destroy all resources (irreversible!)
terraform destroy
```

## Remote State (Recommended)

For team collaboration, uncomment the backend configuration in `main.tf` and create the state bucket:

```bash
aws s3 mb s3://your-terraform-state-bucket --region us-east-1
aws dynamodb create-table \
  --table-name terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST
```

## Troubleshooting

### EKS nodes not joining
Check node IAM role has proper permissions and security groups allow communication.

### RDS connection timeout
Verify security group allows traffic from EKS cluster security group.

### Terraform state lock
If state is locked, check if another operation is running or manually release:
```bash
terraform force-unlock LOCK_ID
```
