#!/bin/bash
# Stripe Configuration Setup Script
# Run this to configure Stripe environment variables

set -e

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           ARAGORA STRIPE SETUP                            ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Detect environment file location
if [ -f "/etc/aragora/.env" ]; then
    ENV_FILE="/etc/aragora/.env"
elif [ -f ".env" ]; then
    ENV_FILE=".env"
else
    ENV_FILE=".env"
    touch "$ENV_FILE"
fi

echo "Environment file: $ENV_FILE"
echo ""

# Function to prompt for value
prompt_value() {
    local var_name=$1
    local prompt_text=$2
    local current_value=$3
    local is_secret=$4

    if [ -n "$current_value" ]; then
        if [ "$is_secret" = "true" ]; then
            display_value="${current_value:0:10}..."
        else
            display_value="$current_value"
        fi
        echo "Current $var_name: $display_value"
        read -p "Enter new value (or press Enter to keep): " new_value
        if [ -z "$new_value" ]; then
            echo "$current_value"
            return
        fi
    else
        read -p "$prompt_text: " new_value
    fi
    echo "$new_value"
}

# Function to update env file
update_env() {
    local key=$1
    local value=$2

    if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
        # Update existing
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
        else
            sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
        fi
    else
        # Add new
        echo "${key}=${value}" >> "$ENV_FILE"
    fi
}

# Load existing values
source "$ENV_FILE" 2>/dev/null || true

echo "═══════════════════════════════════════════════════════════"
echo "Step 1: API Keys"
echo "Get these from: Stripe Dashboard → Developers → API keys"
echo "═══════════════════════════════════════════════════════════"
echo ""

STRIPE_SECRET_KEY=$(prompt_value "STRIPE_SECRET_KEY" "Stripe Secret Key (sk_live_...)" "$STRIPE_SECRET_KEY" "true")
STRIPE_PUBLISHABLE_KEY=$(prompt_value "STRIPE_PUBLISHABLE_KEY" "Stripe Publishable Key (pk_live_...)" "$STRIPE_PUBLISHABLE_KEY" "false")

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Step 2: Webhook Secret"
echo "Get this from: Stripe Dashboard → Developers → Webhooks"
echo "Click your endpoint → Reveal signing secret"
echo "═══════════════════════════════════════════════════════════"
echo ""

STRIPE_WEBHOOK_SECRET=$(prompt_value "STRIPE_WEBHOOK_SECRET" "Webhook Signing Secret (whsec_...)" "$STRIPE_WEBHOOK_SECRET" "true")

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Step 3: Price IDs"
echo "Get these from: Stripe Dashboard → Products → Click each product"
echo "Copy the Price ID (price_...)"
echo "═══════════════════════════════════════════════════════════"
echo ""

STRIPE_PRICE_STARTER=$(prompt_value "STRIPE_PRICE_STARTER" "Starter Plan Price ID (price_...)" "$STRIPE_PRICE_STARTER" "false")
STRIPE_PRICE_PROFESSIONAL=$(prompt_value "STRIPE_PRICE_PROFESSIONAL" "Professional Plan Price ID (price_...)" "$STRIPE_PRICE_PROFESSIONAL" "false")
STRIPE_PRICE_ENTERPRISE=$(prompt_value "STRIPE_PRICE_ENTERPRISE" "Enterprise Plan Price ID (price_...)" "$STRIPE_PRICE_ENTERPRISE" "false")

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Saving configuration..."
echo "═══════════════════════════════════════════════════════════"

# Update environment file
update_env "STRIPE_SECRET_KEY" "$STRIPE_SECRET_KEY"
update_env "STRIPE_PUBLISHABLE_KEY" "$STRIPE_PUBLISHABLE_KEY"
update_env "STRIPE_WEBHOOK_SECRET" "$STRIPE_WEBHOOK_SECRET"
update_env "STRIPE_PRICE_STARTER" "$STRIPE_PRICE_STARTER"
update_env "STRIPE_PRICE_PROFESSIONAL" "$STRIPE_PRICE_PROFESSIONAL"
update_env "STRIPE_PRICE_ENTERPRISE" "$STRIPE_PRICE_ENTERPRISE"

# Secure the file
chmod 600 "$ENV_FILE"

echo ""
echo "Configuration saved to: $ENV_FILE"
echo ""

# Validation
echo "═══════════════════════════════════════════════════════════"
echo "Validating configuration..."
echo "═══════════════════════════════════════════════════════════"

errors=0

if [[ ! "$STRIPE_SECRET_KEY" =~ ^sk_live_ ]]; then
    echo "⚠ WARNING: Secret key doesn't start with sk_live_ (using test mode?)"
    errors=$((errors + 1))
fi

if [[ ! "$STRIPE_PUBLISHABLE_KEY" =~ ^pk_live_ ]]; then
    echo "⚠ WARNING: Publishable key doesn't start with pk_live_ (using test mode?)"
    errors=$((errors + 1))
fi

if [[ ! "$STRIPE_WEBHOOK_SECRET" =~ ^whsec_ ]]; then
    echo "✗ ERROR: Webhook secret should start with whsec_"
    errors=$((errors + 1))
fi

if [[ ! "$STRIPE_PRICE_STARTER" =~ ^price_ ]]; then
    echo "✗ ERROR: Starter price ID should start with price_"
    errors=$((errors + 1))
fi

if [[ ! "$STRIPE_PRICE_PROFESSIONAL" =~ ^price_ ]]; then
    echo "✗ ERROR: Professional price ID should start with price_"
    errors=$((errors + 1))
fi

if [[ ! "$STRIPE_PRICE_ENTERPRISE" =~ ^price_ ]]; then
    echo "✗ ERROR: Enterprise price ID should start with price_"
    errors=$((errors + 1))
fi

echo ""
if [ $errors -eq 0 ]; then
    echo "✓ All configuration values look valid!"
else
    echo "Found $errors issue(s). Please review and fix."
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Next Steps:"
echo "═══════════════════════════════════════════════════════════"
echo "1. Restart the server: sudo systemctl restart aragora"
echo "2. Run verification:   python scripts/verify_stripe.py"
echo "3. Test with real card at https://aragora.ai"
echo ""
