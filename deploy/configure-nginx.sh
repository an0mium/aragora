#!/bin/bash
# Cross-distribution nginx configuration for Aragora
# Supports: Ubuntu, Debian, Amazon Linux 2, RHEL, CentOS, Rocky, Alma
#
# Features:
# - Auto-detects Linux distribution and uses appropriate config paths
# - Auto-detects SSL certificates and falls back to HTTP-only if missing
#
# Usage: ./configure-nginx.sh [config-file]
# Example: ./configure-nginx.sh deploy/nginx-aragora.conf
# Example: ./configure-nginx.sh  # Auto-detect SSL config

set -e

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Auto-detect config based on SSL certificate availability
detect_ssl_config() {
    local explicit_config="$1"

    if [ -n "$explicit_config" ] && [ -f "$explicit_config" ]; then
        # Explicit config provided and exists
        echo "$explicit_config"
        return
    fi

    # Check for Cloudflare origin certificates
    if [ -f /etc/ssl/certs/cloudflare-origin.pem ] && [ -f /etc/ssl/private/cloudflare-origin-key.pem ]; then
        echo "Found Cloudflare origin certs, using SSL config" >&2
        echo "${SCRIPT_DIR}/nginx-aragora.conf"
        return
    fi

    # Check for Let's Encrypt certificates
    if [ -f /etc/letsencrypt/live/api.aragora.ai/fullchain.pem ]; then
        echo "Found Let's Encrypt certs, using SSL config" >&2
        echo "${SCRIPT_DIR}/nginx-aragora.conf"
        return
    fi

    # No SSL certs found - use HTTP-only config (Cloudflare handles TLS upstream)
    echo "No SSL certs found, using HTTP-only config (Cloudflare handles TLS)" >&2
    echo "${SCRIPT_DIR}/nginx-aragora-http.conf"
}

CONFIG_SOURCE=$(detect_ssl_config "$1")

# Detect Linux distribution family
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        case "$ID" in
            ubuntu|debian)
                echo "debian"
                ;;
            amzn|rhel|centos|rocky|almalinux|fedora)
                echo "rhel"
                ;;
            *)
                # Check ID_LIKE for derivatives
                case "$ID_LIKE" in
                    *debian*|*ubuntu*)
                        echo "debian"
                        ;;
                    *rhel*|*centos*|*fedora*)
                        echo "rhel"
                        ;;
                    *)
                        echo "unknown"
                        ;;
                esac
                ;;
        esac
    elif [ -f /etc/debian_version ]; then
        echo "debian"
    elif [ -f /etc/redhat-release ]; then
        echo "rhel"
    else
        echo "unknown"
    fi
}

# Install nginx configuration for the detected distribution
install_nginx_config() {
    local distro
    distro=$(detect_distro)

    echo "Detected distribution family: $distro"

    if [ ! -f "$CONFIG_SOURCE" ]; then
        echo "Error: Config source not found: $CONFIG_SOURCE" >&2
        exit 1
    fi

    case "$distro" in
        debian)
            # Ubuntu/Debian use sites-available/sites-enabled pattern
            sudo cp "$CONFIG_SOURCE" /etc/nginx/sites-available/aragora
            sudo ln -sf /etc/nginx/sites-available/aragora /etc/nginx/sites-enabled/aragora
            sudo rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
            echo "Installed to /etc/nginx/sites-available/aragora"
            echo "Enabled via symlink in /etc/nginx/sites-enabled/"
            ;;
        rhel|unknown)
            # RHEL/CentOS/Amazon Linux use conf.d pattern
            # Also use conf.d as fallback for unknown distros (more universal)
            # Remove old conflicting configs
            sudo rm -f /etc/nginx/conf.d/aragora-api.conf 2>/dev/null || true
            sudo rm -f /etc/nginx/conf.d/aragora-ws.conf 2>/dev/null || true
            sudo cp "$CONFIG_SOURCE" /etc/nginx/conf.d/aragora.conf
            echo "Installed to /etc/nginx/conf.d/aragora.conf"
            ;;
    esac

    # Test and reload nginx
    echo "Testing nginx configuration..."
    if sudo nginx -t; then
        echo "Reloading nginx..."
        sudo systemctl reload nginx
        echo "Nginx configured successfully for Aragora"
    else
        echo "Error: nginx configuration test failed" >&2
        exit 1
    fi
}

# Main
install_nginx_config
