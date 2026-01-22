# AWS Multi-Region Infrastructure for Aragora
# This creates the foundational infrastructure for multi-region deployment

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "aragora-terraform-state"
    key            = "multi-region/terraform.tfstate"
    region         = "us-east-2"
    encrypt        = true
    dynamodb_table = "aragora-terraform-locks"
  }
}

# Variables
variable "region" {
  description = "AWS region to deploy to"
  type        = string
  default     = "us-east-2"
}

variable "primary" {
  description = "Is this the primary region"
  type        = bool
  default     = false
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "domain" {
  description = "Base domain name"
  type        = string
  default     = "aragora.ai"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# Locals
locals {
  region_config = {
    "us-east-2" = {
      vpc_cidr = "10.0.0.0/16"
      azs      = ["us-east-2a", "us-east-2b", "us-east-2c"]
    }
    "eu-west-1" = {
      vpc_cidr = "10.1.0.0/16"
      azs      = ["eu-west-1a", "eu-west-1b", "eu-west-1c"]
    }
    "ap-south-1" = {
      vpc_cidr = "10.2.0.0/16"
      azs      = ["ap-south-1a", "ap-south-1b", "ap-south-1c"]
    }
  }

  tags = {
    Project     = "aragora"
    Environment = var.environment
    Region      = var.region
    ManagedBy   = "terraform"
  }
}

# Provider configuration
provider "aws" {
  region = var.region

  default_tags {
    tags = local.tags
  }
}

# Provider for us-east-2 (for global resources like Route53)
provider "aws" {
  alias  = "global"
  region = "us-east-2"

  default_tags {
    tags = local.tags
  }
}

# VPC Module
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "aragora-${var.region}"
  cidr = local.region_config[var.region].vpc_cidr

  azs             = local.region_config[var.region].azs
  private_subnets = [for i, az in local.region_config[var.region].azs : cidrsubnet(local.region_config[var.region].vpc_cidr, 4, i)]
  public_subnets  = [for i, az in local.region_config[var.region].azs : cidrsubnet(local.region_config[var.region].vpc_cidr, 4, i + 8)]

  enable_nat_gateway     = true
  single_nat_gateway     = false
  one_nat_gateway_per_az = true

  enable_dns_hostnames = true
  enable_dns_support   = true

  # VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true
  flow_log_max_aggregation_interval    = 60

  tags = {
    "kubernetes.io/cluster/aragora-${var.region}" = "shared"
  }

  public_subnet_tags = {
    "kubernetes.io/cluster/aragora-${var.region}" = "shared"
    "kubernetes.io/role/elb"                      = 1
  }

  private_subnet_tags = {
    "kubernetes.io/cluster/aragora-${var.region}" = "shared"
    "kubernetes.io/role/internal-elb"             = 1
  }
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "aragora-${var.region}"
  cluster_version = "1.28"

  cluster_endpoint_public_access = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Managed node groups
  eks_managed_node_groups = {
    general = {
      name = "general-${var.region}"

      instance_types = ["m6i.xlarge", "m5.xlarge"]
      capacity_type  = "ON_DEMAND"

      min_size     = 3
      max_size     = 20
      desired_size = var.primary ? 5 : 3

      labels = {
        role = "general"
      }

      tags = {
        "k8s.io/cluster-autoscaler/enabled"                  = "true"
        "k8s.io/cluster-autoscaler/aragora-${var.region}" = "owned"
      }
    }

    compute = {
      name = "compute-${var.region}"

      instance_types = ["c6i.2xlarge", "c5.2xlarge"]
      capacity_type  = "SPOT"

      min_size     = 0
      max_size     = 30
      desired_size = var.primary ? 4 : 2

      labels = {
        role = "compute"
      }

      taints = [
        {
          key    = "compute"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }

  # OIDC provider for IAM roles for service accounts
  enable_irsa = true

  # Cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  tags = local.tags
}

# RDS PostgreSQL (Primary or Read Replica)
module "rds" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "aragora-${var.region}"

  engine               = "postgres"
  engine_version       = "15.4"
  family               = "postgres15"
  major_engine_version = "15"
  instance_class       = var.primary ? "db.r6g.xlarge" : "db.r6g.large"

  allocated_storage     = var.primary ? 200 : 100
  max_allocated_storage = var.primary ? 1000 : 500

  db_name  = "aragora"
  username = "aragora"
  port     = 5432

  # Multi-AZ for primary only
  multi_az = var.primary

  # Read replica configuration
  replicate_source_db = var.primary ? null : data.aws_db_instance.primary[0].identifier

  # Subnet group
  db_subnet_group_name   = module.vpc.database_subnet_group
  vpc_security_group_ids = [module.security_group_rds.security_group_id]

  # Backup configuration
  backup_retention_period = var.primary ? 30 : 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"

  # Enable deletion protection in production
  deletion_protection = var.environment == "production"

  # Performance Insights
  performance_insights_enabled          = true
  performance_insights_retention_period = 7

  # Enhanced monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring.arn

  # Parameter group for replication
  parameters = var.primary ? [
    {
      name  = "wal_level"
      value = "replica"
    },
    {
      name  = "max_wal_senders"
      value = "10"
    },
    {
      name  = "max_replication_slots"
      value = "10"
    }
  ] : []

  tags = local.tags
}

# Data source for primary RDS (used by read replicas)
data "aws_db_instance" "primary" {
  count                  = var.primary ? 0 : 1
  db_instance_identifier = "aragora-us-east-2"

  provider = aws.global
}

# ElastiCache Redis Cluster
module "elasticache" {
  source  = "terraform-aws-modules/elasticache/aws"
  version = "~> 1.0"

  cluster_id           = "aragora-${var.region}"
  engine               = "redis"
  engine_version       = "7.0"
  node_type            = var.primary ? "cache.r6g.large" : "cache.r6g.medium"
  num_cache_nodes      = var.primary ? 3 : 2
  parameter_group_name = "default.redis7"
  port                 = 6379

  subnet_ids         = module.vpc.private_subnets
  security_group_ids = [module.security_group_redis.security_group_id]

  # Enable cluster mode
  replication_group_id          = "aragora-${var.region}"
  num_node_groups               = var.primary ? 3 : 2
  replicas_per_node_group       = 2

  # Automatic failover
  automatic_failover_enabled = true

  # Encryption
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true

  # Snapshots
  snapshot_retention_limit = 7
  snapshot_window          = "05:00-06:00"

  tags = local.tags
}

# Security Groups
module "security_group_rds" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "~> 5.0"

  name        = "aragora-rds-${var.region}"
  description = "Security group for RDS PostgreSQL"
  vpc_id      = module.vpc.vpc_id

  ingress_with_source_security_group_id = [
    {
      from_port                = 5432
      to_port                  = 5432
      protocol                 = "tcp"
      source_security_group_id = module.eks.cluster_security_group_id
    }
  ]

  tags = local.tags
}

module "security_group_redis" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "~> 5.0"

  name        = "aragora-redis-${var.region}"
  description = "Security group for ElastiCache Redis"
  vpc_id      = module.vpc.vpc_id

  ingress_with_source_security_group_id = [
    {
      from_port                = 6379
      to_port                  = 6379
      protocol                 = "tcp"
      source_security_group_id = module.eks.cluster_security_group_id
    }
  ]

  tags = local.tags
}

# IAM Role for RDS Enhanced Monitoring
resource "aws_iam_role" "rds_monitoring" {
  name = "aragora-rds-monitoring-${var.region}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# S3 Bucket for backups
resource "aws_s3_bucket" "backup" {
  bucket = "aragora-backup-${var.region}"

  tags = local.tags
}

resource "aws_s3_bucket_versioning" "backup" {
  bucket = aws_s3_bucket.backup.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "backup" {
  bucket = aws_s3_bucket.backup.id

  rule {
    id     = "expire-old-backups"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }
}

# Cross-region replication for primary region
resource "aws_s3_bucket_replication_configuration" "backup" {
  count = var.primary ? 1 : 0

  bucket = aws_s3_bucket.backup.id
  role   = aws_iam_role.replication[0].arn

  rule {
    id     = "replicate-to-eu"
    status = "Enabled"

    destination {
      bucket        = "arn:aws:s3:::aragora-backup-eu-west-1"
      storage_class = "STANDARD_IA"
    }
  }

  rule {
    id     = "replicate-to-ap"
    status = "Enabled"

    destination {
      bucket        = "arn:aws:s3:::aragora-backup-ap-south-1"
      storage_class = "STANDARD_IA"
    }
  }
}

resource "aws_iam_role" "replication" {
  count = var.primary ? 1 : 0

  name = "aragora-s3-replication-${var.region}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
      }
    ]
  })
}

# Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = module.rds.db_instance_endpoint
}

output "redis_endpoint" {
  description = "Redis endpoint"
  value       = module.elasticache.replication_group_primary_endpoint_address
}

output "backup_bucket" {
  description = "S3 backup bucket"
  value       = aws_s3_bucket.backup.bucket
}
