# Aragora Multi-Region EC2 Infrastructure
# Amazon Linux 2023 deployment for api.aragora.ai
#
# Usage:
#   terraform init
#   terraform workspace new us-east-2 || terraform workspace select us-east-2
#   terraform plan -var="role=primary"
#   terraform apply -var="role=primary"

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    http = {
      source  = "hashicorp/http"
      version = "~> 3.0"
    }
  }

  # Uncomment for remote state (recommended for production)
  # backend "s3" {
  #   bucket         = "aragora-terraform-state"
  #   key            = "ec2-multiregion/terraform.tfstate"
  #   region         = "us-east-2"
  #   encrypt        = true
  #   dynamodb_table = "aragora-terraform-locks"
  # }
}

# =============================================================================
# Provider Configuration
# =============================================================================

provider "aws" {
  region = var.region

  default_tags {
    tags = local.tags
  }
}

# =============================================================================
# Locals
# =============================================================================

locals {
  name = "aragora-api-${var.role}"

  tags = {
    Project     = "aragora"
    Environment = var.environment
    Role        = var.role
    Region      = var.region
    ManagedBy   = "terraform"
  }
}

# =============================================================================
# Data Sources
# =============================================================================

# Amazon Linux 2023 AMI (latest)
data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# =============================================================================
# VPC (use default or specify)
# =============================================================================

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "public" {
  filter {
    name   = "vpc-id"
    values = [var.vpc_id != "" ? var.vpc_id : data.aws_vpc.default.id]
  }

  filter {
    name   = "map-public-ip-on-launch"
    values = ["true"]
  }
}

# =============================================================================
# EC2 Instance
# =============================================================================

resource "aws_instance" "aragora" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  key_name               = var.key_name != "" ? var.key_name : null
  vpc_security_group_ids = [aws_security_group.aragora.id]
  iam_instance_profile   = aws_iam_instance_profile.aragora.name
  subnet_id              = var.subnet_id != "" ? var.subnet_id : data.aws_subnets.public.ids[0]

  user_data = templatefile("${path.module}/user-data.sh", {
    environment = var.environment
    role        = var.role
    region      = var.region
  })

  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true

    tags = {
      Name = "${local.name}-root"
    }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"  # IMDSv2 only
    http_put_response_hop_limit = 1
  }

  monitoring = true

  tags = {
    Name = local.name
  }

  lifecycle {
    create_before_destroy = true
  }
}

# =============================================================================
# Elastic IP
# =============================================================================

resource "aws_eip" "aragora" {
  instance = aws_instance.aragora.id
  domain   = "vpc"

  tags = {
    Name = "${local.name}-eip"
  }
}

# =============================================================================
# CloudWatch Alarms
# =============================================================================

resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  alarm_name          = "${local.name}-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "CPU utilization above 80%"
  alarm_actions       = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  dimensions = {
    InstanceId = aws_instance.aragora.id
  }

  tags = local.tags
}

resource "aws_cloudwatch_metric_alarm" "status_check" {
  alarm_name          = "${local.name}-status-check"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "StatusCheckFailed"
  namespace           = "AWS/EC2"
  period              = 60
  statistic           = "Maximum"
  threshold           = 0
  alarm_description   = "Instance status check failed"
  alarm_actions       = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  dimensions = {
    InstanceId = aws_instance.aragora.id
  }

  tags = local.tags
}
