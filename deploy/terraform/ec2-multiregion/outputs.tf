# =============================================================================
# Outputs for Aragora EC2 Multi-Region Infrastructure
# =============================================================================

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.aragora.id
}

output "instance_arn" {
  description = "EC2 instance ARN"
  value       = aws_instance.aragora.arn
}

output "public_ip" {
  description = "Elastic IP address (use this for Cloudflare)"
  value       = aws_eip.aragora.public_ip
}

output "private_ip" {
  description = "Private IP address"
  value       = aws_instance.aragora.private_ip
}

output "availability_zone" {
  description = "Availability zone"
  value       = aws_instance.aragora.availability_zone
}

output "ami_id" {
  description = "AMI ID used (Amazon Linux 2023)"
  value       = data.aws_ami.al2023.id
}

output "ami_name" {
  description = "AMI name"
  value       = data.aws_ami.al2023.name
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.aragora.id
}

output "iam_role_arn" {
  description = "IAM role ARN"
  value       = aws_iam_role.aragora.arn
}

output "iam_instance_profile_name" {
  description = "IAM instance profile name"
  value       = aws_iam_instance_profile.aragora.name
}

# =============================================================================
# Connection Information
# =============================================================================

output "ssm_start_session" {
  description = "Command to connect via SSM Session Manager"
  value       = "aws ssm start-session --target ${aws_instance.aragora.id} --region ${var.region}"
}

output "health_check_url" {
  description = "Health check URL (via Elastic IP)"
  value       = "http://${aws_eip.aragora.public_ip}/api/health"
}

output "cloudflare_origin" {
  description = "Origin configuration for Cloudflare load balancer"
  value = {
    name    = "ec2-${var.role}-${var.region}"
    address = aws_eip.aragora.public_ip
    port    = 80
    weight  = var.role == "dr" ? 0.5 : 1.0
    enabled = true
  }
}
