# =============================================================================
# IAM Roles and Policies for Aragora EC2 Instances
# =============================================================================

# =============================================================================
# IAM Role
# =============================================================================

resource "aws_iam_role" "aragora" {
  name = "aragora-ec2-${var.role}-${var.region}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "aragora-ec2-${var.role}-role"
  }
}

# =============================================================================
# Instance Profile
# =============================================================================

resource "aws_iam_instance_profile" "aragora" {
  name = "aragora-ec2-${var.role}-${var.region}"
  role = aws_iam_role.aragora.name

  tags = {
    Name = "aragora-ec2-${var.role}-profile"
  }
}

# =============================================================================
# SSM Managed Instance Core (for Session Manager access)
# =============================================================================

resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.aragora.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# =============================================================================
# CloudWatch Agent (for metrics and logs)
# =============================================================================

resource "aws_iam_role_policy_attachment" "cloudwatch_agent" {
  role       = aws_iam_role.aragora.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

# =============================================================================
# Secrets Manager Access
# =============================================================================

resource "aws_iam_role_policy" "secrets_manager" {
  name = "aragora-secrets-access"
  role = aws_iam_role.aragora.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "GetSecrets"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = [
          "arn:aws:secretsmanager:*:${data.aws_caller_identity.current.account_id}:secret:aragora/*"
        ]
      }
    ]
  })
}

# =============================================================================
# S3 Access for Backups (optional)
# =============================================================================

resource "aws_iam_role_policy" "s3_backups" {
  name = "aragora-s3-backups"
  role = aws_iam_role.aragora.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3BackupAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::aragora-backups-*",
          "arn:aws:s3:::aragora-backups-*/*"
        ]
      }
    ]
  })
}

# =============================================================================
# ECR Access (if using container images)
# =============================================================================

resource "aws_iam_role_policy" "ecr_pull" {
  name = "aragora-ecr-pull"
  role = aws_iam_role.aragora.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECRAuth"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken"
        ]
        Resource = "*"
      },
      {
        Sid    = "ECRPull"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = [
          "arn:aws:ecr:*:${data.aws_caller_identity.current.account_id}:repository/aragora*"
        ]
      }
    ]
  })
}
