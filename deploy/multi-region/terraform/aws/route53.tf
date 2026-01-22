# Route53 Global DNS Configuration for Multi-Region Routing
# This must be applied from the primary region (us-east-2)

# Hosted Zone (assumes already exists)
data "aws_route53_zone" "main" {
  name         = var.domain
  private_zone = false
}

# Health Checks for each region
resource "aws_route53_health_check" "us_east_1" {
  fqdn              = "api.us-east-2.${var.domain}"
  port              = 443
  type              = "HTTPS"
  resource_path     = "/health/ready"
  failure_threshold = 3
  request_interval  = 10

  tags = {
    Name   = "aragora-health-us-east-2"
    Region = "us-east-2"
  }
}

resource "aws_route53_health_check" "eu_west_1" {
  fqdn              = "api.eu-west-1.${var.domain}"
  port              = 443
  type              = "HTTPS"
  resource_path     = "/health/ready"
  failure_threshold = 3
  request_interval  = 10

  tags = {
    Name   = "aragora-health-eu-west-1"
    Region = "eu-west-1"
  }
}

resource "aws_route53_health_check" "ap_south_1" {
  fqdn              = "api.ap-south-1.${var.domain}"
  port              = 443
  type              = "HTTPS"
  resource_path     = "/health/ready"
  failure_threshold = 3
  request_interval  = 10

  tags = {
    Name   = "aragora-health-ap-south-1"
    Region = "ap-south-1"
  }
}

# Latency-based routing for API
resource "aws_route53_record" "api_us_east_1" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "api.${var.domain}"
  type    = "A"

  alias {
    name                   = data.aws_lb.us_east_1.dns_name
    zone_id                = data.aws_lb.us_east_1.zone_id
    evaluate_target_health = true
  }

  set_identifier  = "api-us-east-2"
  latency_routing_policy {
    region = "us-east-2"
  }
  health_check_id = aws_route53_health_check.us_east_1.id
}

resource "aws_route53_record" "api_eu_west_1" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "api.${var.domain}"
  type    = "A"

  alias {
    name                   = data.aws_lb.eu_west_1.dns_name
    zone_id                = data.aws_lb.eu_west_1.zone_id
    evaluate_target_health = true
  }

  set_identifier  = "api-eu-west-1"
  latency_routing_policy {
    region = "eu-west-1"
  }
  health_check_id = aws_route53_health_check.eu_west_1.id
}

resource "aws_route53_record" "api_ap_south_1" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "api.${var.domain}"
  type    = "A"

  alias {
    name                   = data.aws_lb.ap_south_1.dns_name
    zone_id                = data.aws_lb.ap_south_1.zone_id
    evaluate_target_health = true
  }

  set_identifier  = "api-ap-south-1"
  latency_routing_policy {
    region = "ap-south-1"
  }
  health_check_id = aws_route53_health_check.ap_south_1.id
}

# Regional API records (direct access)
resource "aws_route53_record" "api_regional_us_east_1" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "api.us-east-2.${var.domain}"
  type    = "A"

  alias {
    name                   = data.aws_lb.us_east_1.dns_name
    zone_id                = data.aws_lb.us_east_1.zone_id
    evaluate_target_health = true
  }
}

resource "aws_route53_record" "api_regional_eu_west_1" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "api.eu-west-1.${var.domain}"
  type    = "A"

  alias {
    name                   = data.aws_lb.eu_west_1.dns_name
    zone_id                = data.aws_lb.eu_west_1.zone_id
    evaluate_target_health = true
  }
}

resource "aws_route53_record" "api_regional_ap_south_1" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "api.ap-south-1.${var.domain}"
  type    = "A"

  alias {
    name                   = data.aws_lb.ap_south_1.dns_name
    zone_id                = data.aws_lb.ap_south_1.zone_id
    evaluate_target_health = true
  }
}

# Frontend latency-based routing
resource "aws_route53_record" "frontend_us_east_1" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = var.domain
  type    = "A"

  alias {
    name                   = data.aws_lb.frontend_us_east_1.dns_name
    zone_id                = data.aws_lb.frontend_us_east_1.zone_id
    evaluate_target_health = true
  }

  set_identifier  = "frontend-us-east-2"
  latency_routing_policy {
    region = "us-east-2"
  }
}

resource "aws_route53_record" "frontend_eu_west_1" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = var.domain
  type    = "A"

  alias {
    name                   = data.aws_lb.frontend_eu_west_1.dns_name
    zone_id                = data.aws_lb.frontend_eu_west_1.zone_id
    evaluate_target_health = true
  }

  set_identifier  = "frontend-eu-west-1"
  latency_routing_policy {
    region = "eu-west-1"
  }
}

resource "aws_route53_record" "frontend_ap_south_1" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = var.domain
  type    = "A"

  alias {
    name                   = data.aws_lb.frontend_ap_south_1.dns_name
    zone_id                = data.aws_lb.frontend_ap_south_1.zone_id
    evaluate_target_health = true
  }

  set_identifier  = "frontend-ap-south-1"
  latency_routing_policy {
    region = "ap-south-1"
  }
}

# Failover configuration for critical endpoints
resource "aws_route53_record" "api_primary_failover" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "api-failover.${var.domain}"
  type    = "A"

  alias {
    name                   = data.aws_lb.us_east_1.dns_name
    zone_id                = data.aws_lb.us_east_1.zone_id
    evaluate_target_health = true
  }

  set_identifier = "primary"
  failover_routing_policy {
    type = "PRIMARY"
  }
  health_check_id = aws_route53_health_check.us_east_1.id
}

resource "aws_route53_record" "api_secondary_failover" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "api-failover.${var.domain}"
  type    = "A"

  alias {
    name                   = data.aws_lb.eu_west_1.dns_name
    zone_id                = data.aws_lb.eu_west_1.zone_id
    evaluate_target_health = true
  }

  set_identifier = "secondary"
  failover_routing_policy {
    type = "SECONDARY"
  }
  health_check_id = aws_route53_health_check.eu_west_1.id
}

# Data sources for load balancers (created by EKS ingress)
data "aws_lb" "us_east_1" {
  tags = {
    "kubernetes.io/cluster/aragora-us-east-2" = "owned"
    "kubernetes.io/service-name"              = "istio-system/istio-ingressgateway"
  }
}

data "aws_lb" "eu_west_1" {
  provider = aws.eu_west_1
  tags = {
    "kubernetes.io/cluster/aragora-eu-west-1" = "owned"
    "kubernetes.io/service-name"              = "istio-system/istio-ingressgateway"
  }
}

data "aws_lb" "ap_south_1" {
  provider = aws.ap_south_1
  tags = {
    "kubernetes.io/cluster/aragora-ap-south-1" = "owned"
    "kubernetes.io/service-name"               = "istio-system/istio-ingressgateway"
  }
}

data "aws_lb" "frontend_us_east_1" {
  tags = {
    "kubernetes.io/cluster/aragora-us-east-2" = "owned"
    "kubernetes.io/service-name"              = "aragora/frontend"
  }
}

data "aws_lb" "frontend_eu_west_1" {
  provider = aws.eu_west_1
  tags = {
    "kubernetes.io/cluster/aragora-eu-west-1" = "owned"
    "kubernetes.io/service-name"              = "aragora/frontend"
  }
}

data "aws_lb" "frontend_ap_south_1" {
  provider = aws.ap_south_1
  tags = {
    "kubernetes.io/cluster/aragora-ap-south-1" = "owned"
    "kubernetes.io/service-name"               = "aragora/frontend"
  }
}

# Providers for other regions
provider "aws" {
  alias  = "eu_west_1"
  region = "eu-west-1"
}

provider "aws" {
  alias  = "ap_south_1"
  region = "ap-south-1"
}

# Outputs
output "health_check_ids" {
  description = "Route53 health check IDs"
  value = {
    us_east_1 = aws_route53_health_check.us_east_1.id
    eu_west_1 = aws_route53_health_check.eu_west_1.id
    ap_south_1 = aws_route53_health_check.ap_south_1.id
  }
}
