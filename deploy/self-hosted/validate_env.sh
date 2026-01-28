#!/bin/bash
# Aragora Environment Validation Script
# Validates .env configuration before deployment
#
# Usage: ./validate_env.sh [--strict]
#
# Options:
#   --strict    Fail on warnings (for CI/CD)
#   --quiet     Only show errors and warnings
#   --help      Show this help message

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
STRICT_MODE=false
QUIET_MODE=false
for arg in "$@"; do
    case $arg in
        --strict)
            STRICT_MODE=true
            shift
            ;;
        --quiet|-q)
            QUIET_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./validate_env.sh [--strict] [--quiet]"
            echo ""
            echo "Options:"
            echo "  --strict    Fail on warnings (for CI/CD)"
            echo "  --quiet     Only show errors and warnings"
            echo "  --help      Show this help message"
            exit 0
            ;;
    esac
done

# Counters
ERRORS=0
WARNINGS=0

# Helper functions
log_error() {
    echo -e "${RED}✗ ERROR:${NC} $1"
    ERRORS=$((ERRORS + 1))
}

log_warning() {
    echo -e "${YELLOW}⚠ WARNING:${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

log_success() {
    if [ "$QUIET_MODE" = false ]; then
        echo -e "${GREEN}✓${NC} $1"
    fi
}

log_info() {
    if [ "$QUIET_MODE" = false ]; then
        echo -e "${BLUE}ℹ${NC} $1"
    fi
}

header() {
    if [ "$QUIET_MODE" = false ]; then
        echo ""
        echo -e "${BLUE}$1${NC}"
        echo "─────────────────────────────────────"
    fi
}

# Check if .env file exists
if [ ! -f ".env" ]; then
    log_error ".env file not found. Run ./init.sh first."
    exit 1
fi

# Load .env file
source .env 2>/dev/null || true

if [ "$QUIET_MODE" = false ]; then
    echo "=================================="
    echo "  Aragora Environment Validation"
    echo "=================================="
fi

# =============================================================================
# Required Configuration
# =============================================================================
header "Required Configuration"

# At least one AI provider required
if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    log_error "No AI provider API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
else
    if [ -n "$ANTHROPIC_API_KEY" ]; then
        if [[ "$ANTHROPIC_API_KEY" == sk-ant-* ]]; then
            log_success "Anthropic API key format valid"
        else
            log_warning "Anthropic API key format unexpected (should start with 'sk-ant-')"
        fi
    fi
    if [ -n "$OPENAI_API_KEY" ]; then
        if [[ "$OPENAI_API_KEY" == sk-* ]]; then
            log_success "OpenAI API key format valid"
        else
            log_warning "OpenAI API key format unexpected (should start with 'sk-')"
        fi
    fi
fi

# =============================================================================
# Database Configuration
# =============================================================================
header "Database Configuration"

# PostgreSQL password
if [ -z "$POSTGRES_PASSWORD" ]; then
    log_error "POSTGRES_PASSWORD is not set"
elif [ "$POSTGRES_PASSWORD" = "CHANGE_ME_STRONG_PASSWORD" ]; then
    log_error "POSTGRES_PASSWORD is still set to default value. Run ./init.sh to generate."
elif [ ${#POSTGRES_PASSWORD} -lt 16 ]; then
    log_warning "POSTGRES_PASSWORD should be at least 16 characters (current: ${#POSTGRES_PASSWORD})"
else
    log_success "PostgreSQL password configured (${#POSTGRES_PASSWORD} chars)"
fi

# Database name
if [ -z "$POSTGRES_DB" ]; then
    log_info "POSTGRES_DB not set, will use default 'aragora'"
else
    log_success "Database name: $POSTGRES_DB"
fi

# =============================================================================
# Security Configuration
# =============================================================================
header "Security Configuration"

# JWT Secret
if [ -z "$ARAGORA_JWT_SECRET" ]; then
    log_error "ARAGORA_JWT_SECRET is not set"
elif [ "$ARAGORA_JWT_SECRET" = "CHANGE_ME_GENERATE_RANDOM_SECRET" ]; then
    log_error "ARAGORA_JWT_SECRET is still set to default value. Run ./init.sh to generate."
elif [ ${#ARAGORA_JWT_SECRET} -lt 32 ]; then
    log_warning "ARAGORA_JWT_SECRET should be at least 32 characters (current: ${#ARAGORA_JWT_SECRET})"
else
    log_success "JWT secret configured (${#ARAGORA_JWT_SECRET} chars)"
fi

# Redis password
if [ -z "$REDIS_PASSWORD" ]; then
    log_warning "REDIS_PASSWORD not set - Redis will be accessible without authentication"
elif [ "$REDIS_PASSWORD" = "CHANGE_ME_REDIS_PASSWORD" ]; then
    log_warning "REDIS_PASSWORD is still set to default value"
elif [ ${#REDIS_PASSWORD} -lt 12 ]; then
    log_warning "REDIS_PASSWORD should be at least 12 characters"
else
    log_success "Redis password configured (${#REDIS_PASSWORD} chars)"
fi

# Encryption key (optional but recommended)
if [ -z "$ARAGORA_ENCRYPTION_KEY" ]; then
    log_info "ARAGORA_ENCRYPTION_KEY not set (optional for data-at-rest encryption)"
elif [ ${#ARAGORA_ENCRYPTION_KEY} -ne 64 ]; then
    log_warning "ARAGORA_ENCRYPTION_KEY should be 64 hex characters (32 bytes)"
else
    log_success "Encryption key configured"
fi

# =============================================================================
# Network Configuration
# =============================================================================
header "Network Configuration"

# Port configuration
HTTP_PORT=${ARAGORA_PORT:-8080}
if [ "$HTTP_PORT" -lt 1024 ] && [ "$(id -u)" -ne 0 ]; then
    log_warning "HTTP port $HTTP_PORT requires root privileges"
else
    log_success "HTTP port: $HTTP_PORT"
fi

# CORS
if [ -z "$ARAGORA_ALLOWED_ORIGINS" ]; then
    log_info "ARAGORA_ALLOWED_ORIGINS not set (will allow all origins in dev)"
elif [ "$ARAGORA_ALLOWED_ORIGINS" = "*" ]; then
    log_warning "CORS allows all origins - restrict in production"
else
    log_success "CORS configured: $ARAGORA_ALLOWED_ORIGINS"
fi

# =============================================================================
# Optional Services
# =============================================================================
header "Optional Services"

# Grafana
if [ -n "$GRAFANA_PASSWORD" ]; then
    if [ "$GRAFANA_PASSWORD" = "CHANGE_ME_GRAFANA_PASSWORD" ]; then
        log_warning "GRAFANA_PASSWORD is still set to default value"
    elif [ ${#GRAFANA_PASSWORD} -lt 8 ]; then
        log_warning "GRAFANA_PASSWORD should be at least 8 characters"
    else
        log_success "Grafana password configured"
    fi
fi

# OpenRouter fallback
if [ -n "$OPENROUTER_API_KEY" ]; then
    log_success "OpenRouter fallback configured"
else
    log_info "OPENROUTER_API_KEY not set (recommended for fallback)"
fi

# Slack integration
if [ -n "$SLACK_BOT_TOKEN" ]; then
    if [[ "$SLACK_BOT_TOKEN" == xoxb-* ]]; then
        log_success "Slack bot token configured"
    else
        log_warning "Slack bot token format unexpected (should start with 'xoxb-')"
    fi
fi

# GitHub integration
if [ -n "$GITHUB_TOKEN" ]; then
    if [[ "$GITHUB_TOKEN" == ghp_* ]] || [[ "$GITHUB_TOKEN" == github_pat_* ]]; then
        log_success "GitHub token configured"
    else
        log_warning "GitHub token format unexpected"
    fi
fi

# =============================================================================
# Production Readiness
# =============================================================================
header "Production Readiness"

# Check for sensitive defaults
if grep -q "CHANGE_ME" .env 2>/dev/null; then
    CHANGE_ME_COUNT=$(grep -c "CHANGE_ME" .env)
    log_warning "$CHANGE_ME_COUNT placeholder(s) still need to be configured"
fi

# Environment mode
if [ -z "$ARAGORA_ENV" ]; then
    log_info "ARAGORA_ENV not set (defaults to development)"
elif [ "$ARAGORA_ENV" = "production" ]; then
    log_success "Production mode enabled"
    # Extra checks for production
    if [ -z "$ARAGORA_ENCRYPTION_KEY" ]; then
        log_warning "Production mode: Consider setting ARAGORA_ENCRYPTION_KEY"
    fi
    if [ "$ARAGORA_ALLOWED_ORIGINS" = "*" ]; then
        log_error "Production mode: CORS should not allow all origins"
    fi
fi

# TLS
if [ -z "$ARAGORA_TLS_CERT" ] && [ -z "$ARAGORA_TLS_KEY" ]; then
    log_info "TLS not configured (use reverse proxy for HTTPS)"
elif [ -n "$ARAGORA_TLS_CERT" ] && [ -n "$ARAGORA_TLS_KEY" ]; then
    if [ -f "$ARAGORA_TLS_CERT" ] && [ -f "$ARAGORA_TLS_KEY" ]; then
        log_success "TLS certificate and key configured"
    else
        log_error "TLS files specified but not found"
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=================================="
echo "  Validation Summary"
echo "=================================="
echo ""

if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}Errors: $ERRORS${NC}"
fi
if [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
fi
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
fi

echo ""

# Exit code
if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}Configuration has errors. Please fix before deploying.${NC}"
    exit 1
elif [ $WARNINGS -gt 0 ] && [ "$STRICT_MODE" = true ]; then
    echo -e "${YELLOW}Strict mode: Warnings treated as errors.${NC}"
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}Configuration has warnings. Review before production deployment.${NC}"
    exit 0
else
    echo -e "${GREEN}Configuration is ready for deployment.${NC}"
    exit 0
fi
