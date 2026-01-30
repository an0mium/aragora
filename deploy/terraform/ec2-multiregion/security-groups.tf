# =============================================================================
# Security Groups for Aragora EC2 Instances
# Restricts HTTP access to Cloudflare IP ranges only
# =============================================================================

# Fetch current Cloudflare IP ranges
data "http" "cloudflare_ips_v4" {
  url = "https://www.cloudflare.com/ips-v4"
}

data "http" "cloudflare_ips_v6" {
  url = "https://www.cloudflare.com/ips-v6"
}

locals {
  cloudflare_ipv4_cidrs = [
    for ip in split("\n", trimspace(data.http.cloudflare_ips_v4.response_body)) :
    ip if ip != ""
  ]

  cloudflare_ipv6_cidrs = [
    for ip in split("\n", trimspace(data.http.cloudflare_ips_v6.response_body)) :
    ip if ip != ""
  ]
}

# =============================================================================
# Main Security Group
# =============================================================================

resource "aws_security_group" "aragora" {
  name        = "aragora-api-${var.role}-${var.region}"
  description = "Security group for Aragora API servers - Cloudflare access only"
  vpc_id      = var.vpc_id != "" ? var.vpc_id : data.aws_vpc.default.id

  tags = {
    Name = "aragora-api-${var.role}-sg"
  }
}

# =============================================================================
# Ingress Rules - Cloudflare IPv4
# =============================================================================

resource "aws_security_group_rule" "cloudflare_http_ipv4" {
  count             = length(local.cloudflare_ipv4_cidrs)
  type              = "ingress"
  from_port         = 80
  to_port           = 80
  protocol          = "tcp"
  cidr_blocks       = [local.cloudflare_ipv4_cidrs[count.index]]
  security_group_id = aws_security_group.aragora.id
  description       = "HTTP from Cloudflare IPv4"
}

# =============================================================================
# Ingress Rules - Cloudflare IPv6
# =============================================================================

resource "aws_security_group_rule" "cloudflare_http_ipv6" {
  count             = length(local.cloudflare_ipv6_cidrs)
  type              = "ingress"
  from_port         = 80
  to_port           = 80
  protocol          = "tcp"
  ipv6_cidr_blocks  = [local.cloudflare_ipv6_cidrs[count.index]]
  security_group_id = aws_security_group.aragora.id
  description       = "HTTP from Cloudflare IPv6"
}

# =============================================================================
# Ingress Rules - SSH (optional, prefer SSM)
# =============================================================================

resource "aws_security_group_rule" "ssh" {
  count             = length(var.admin_cidr_blocks) > 0 ? 1 : 0
  type              = "ingress"
  from_port         = 22
  to_port           = 22
  protocol          = "tcp"
  cidr_blocks       = var.admin_cidr_blocks
  security_group_id = aws_security_group.aragora.id
  description       = "SSH from admin IPs"
}

# =============================================================================
# Egress Rules - All outbound
# =============================================================================

resource "aws_security_group_rule" "egress_all" {
  type              = "egress"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  ipv6_cidr_blocks  = ["::/0"]
  security_group_id = aws_security_group.aragora.id
  description       = "Allow all outbound traffic"
}
