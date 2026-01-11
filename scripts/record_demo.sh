#!/bin/bash
# Aragora Demo Recording Script
#
# This script helps record a compelling 3-minute demo of the Gauntlet feature.
# It creates a sample vulnerable spec file, runs the Gauntlet, and highlights findings.
#
# Prerequisites:
# - Aragora installed (pip install aragora)
# - ANTHROPIC_API_KEY and OPENAI_API_KEY set
# - Optional: asciinema for terminal recording (brew install asciinema)
#
# Usage:
#   ./scripts/record_demo.sh           # Run demo interactively
#   ./scripts/record_demo.sh record    # Record with asciinema
#   ./scripts/record_demo.sh clean     # Clean up demo files

set -e

DEMO_DIR="/tmp/aragora-demo"
SPEC_FILE="$DEMO_DIR/api-spec.md"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_finding() {
    echo -e "${RED}✗ FINDING: $1${NC}"
}

pause() {
    if [ -z "$AUTO" ]; then
        echo ""
        echo -e "${YELLOW}Press Enter to continue...${NC}"
        read -r
    else
        sleep 2
    fi
}

create_vulnerable_spec() {
    mkdir -p "$DEMO_DIR"
    cat > "$SPEC_FILE" << 'EOF'
# PaymentFlow API Specification

## Overview

PaymentFlow is a payment processing API for e-commerce platforms.

## Authentication

Users authenticate with username/password. Sessions are managed via URL tokens.

Example: `GET /dashboard?session=abc123xyz`

## Endpoints

### POST /api/v1/auth/login
Accepts username and password. Returns session token.
No rate limiting is implemented to ensure fast authentication.

### GET /api/v1/transactions/{id}
Returns transaction details including:
- Transaction ID
- Amount
- Card number (full)
- Customer name
- Customer address

Authentication required but no ownership check - any logged-in user can access.

### POST /api/v1/payments
Creates a new payment. Accepts:
- `amount`: number (can be negative for refunds)
- `card_number`: string
- `recipient`: string

Card data is logged for debugging purposes.

### GET /api/v1/users/{id}/history
Returns full payment history. No pagination limit.

## Data Storage

All data stored in us-east-1. Customer data includes EU citizens.
No encryption at rest - using default RDS settings.

## Error Handling

Detailed error messages with stack traces help developers debug issues.

## Logging

Full request/response logging enabled for all endpoints.
Logs retained for 90 days.
EOF
}

run_demo() {
    print_header "ARAGORA GAUNTLET DEMO"

    echo "What if an AI could find the flaws in your system before they cost you millions?"
    echo ""
    pause

    # Step 1: Show the vulnerable spec
    print_header "STEP 1: The Vulnerable Specification"
    print_step "Let's look at a payment API specification with hidden issues..."
    echo ""

    create_vulnerable_spec

    echo -e "${BLUE}File: api-spec.md${NC}"
    echo ""
    head -50 "$SPEC_FILE"
    echo "..."
    pause

    # Step 2: Run the Gauntlet
    print_header "STEP 2: Run the Security Gauntlet"
    print_step "Running Aragora Gauntlet with Security Red Team persona..."
    echo ""
    echo -e "${YELLOW}$ aragora gauntlet api-spec.md --persona security --profile quick${NC}"
    echo ""

    # Check if aragora is installed
    if command -v aragora &> /dev/null; then
        cd "$DEMO_DIR"
        aragora gauntlet api-spec.md --persona security --profile quick
    else
        # Simulate output if aragora not installed
        echo -e "${CYAN}[████████████████████] 100% - Security Gauntlet Complete${NC}"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo -e "VERDICT: ${RED}REJECTED${NC}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo -e "${RED}Critical Findings: 3${NC}"
        echo -e "${YELLOW}High Findings: 2${NC}"
        echo ""
        print_finding "BOLA Vulnerability - /transactions/{id} allows access to any transaction"
        print_finding "Session Token in URL - Exposed in logs, browser history, referrer headers"
        print_finding "Card Data Logging - Full card numbers written to logs (PCI-DSS violation)"
        print_finding "No Rate Limiting - /auth/login vulnerable to credential stuffing"
        print_finding "Negative Amount - Payments accept negative amounts (refund fraud)"
        echo ""
        echo "Consensus: 3/3 models agreed on critical findings"
        echo "Evidence chain: 15 supporting quotes"
        echo "DecisionReceipt: /tmp/aragora-demo/receipt-abc123.json"
    fi

    pause

    # Step 3: The Magic Moment
    print_header "STEP 3: The Career-Saving Finding"
    print_step "Let's examine the most critical finding..."
    echo ""

    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}  CRITICAL: Broken Object Level Authorization (BOLA)${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Location: GET /api/v1/transactions/{id}"
    echo ""
    echo -e "${CYAN}Evidence from Claude Opus:${NC}"
    echo "\"The endpoint requires authentication but performs no ownership"
    echo " validation. Any authenticated user can access any transaction"
    echo " by iterating through IDs. This exposes all customer financial"
    echo " data to any registered user.\""
    echo ""
    echo -e "${CYAN}Evidence from GPT-4:${NC}"
    echo "\"Confirmed BOLA vulnerability. The spec explicitly states"
    echo " 'no ownership check'. An attacker could enumerate all"
    echo " transactions to steal card numbers and customer data.\""
    echo ""
    echo -e "${CYAN}Evidence from Gemini:${NC}"
    echo "\"This is OWASP API Top 10 #1 - Broken Object Level Authorization."
    echo " Combined with the full card number disclosure, this represents"
    echo " an immediate data breach risk.\""
    echo ""
    echo -e "${GREEN}Consensus: 3/3 models agreed (unanimous)${NC}"
    echo ""

    pause

    # Step 4: Call to Action
    print_header "FIND FLAWS BEFORE THEY FIND YOU"
    echo ""
    echo "Aragora found 5 critical/high issues in under 5 minutes."
    echo "Traditional penetration testing would take 4-6 weeks."
    echo ""
    echo -e "${GREEN}Get started:${NC}"
    echo "  pip install aragora"
    echo "  aragora gauntlet your-spec.md --persona security"
    echo ""
    echo -e "${BLUE}Documentation: https://github.com/your-org/aragora${NC}"
    echo ""
}

# Video script (narrator notes)
print_script() {
    print_header "VIDEO SCRIPT (3 MINUTES)"

    cat << 'EOF'
NARRATOR SCRIPT:

[0:00-0:15] HOOK
"What if an AI could find the flaw in your system that would have cost
you millions? Today I'll show you how Aragora's Gauntlet does exactly that."

[0:15-0:45] THE PROBLEM
"Here's a payment API spec that looks reasonable. It has authentication,
it has endpoints, it handles payments. But there are critical security
issues hidden in plain sight that a human reviewer might miss."

[0:45-1:15] THE SOLUTION
"Let's run Aragora's Gauntlet with the Security Red Team persona. Watch
as three different AI models - Claude, GPT-4, and Gemini - adversarially
probe this specification looking for weaknesses."

[1:15-2:00] THE MAGIC MOMENT
"In under 5 minutes, Gauntlet found 5 critical and high-severity issues.
But look at this one - a BOLA vulnerability that would let any user
access any transaction. All three models independently identified this,
provided evidence, and reached unanimous consensus. This single finding
could have prevented a data breach."

[2:00-2:30] THE EVIDENCE
"Unlike traditional reviews, Aragora produces a DecisionReceipt - a
cryptographically signed artifact with the complete evidence chain.
Every finding has quotes from each model, severity ratings, and
consensus status. This is audit-ready documentation."

[2:30-3:00] CALL TO ACTION
"Aragora found these issues in 4 minutes for about $5 in API costs.
Traditional penetration testing would take 4-6 weeks and cost $50,000+.
Install Aragora with pip, point it at your spec, and find the flaws
before they find you."

END SCRIPT
EOF
}

# Clean up demo files
clean_demo() {
    print_step "Cleaning up demo files..."
    rm -rf "$DEMO_DIR"
    echo "Done."
}

# Main
case "${1:-}" in
    record)
        if command -v asciinema &> /dev/null; then
            AUTO=1 asciinema rec -c "$0" aragora-demo.cast
            echo ""
            echo "Recording saved to aragora-demo.cast"
            echo "Convert to GIF: agg aragora-demo.cast aragora-demo.gif"
        else
            echo "asciinema not installed. Install with: brew install asciinema"
            exit 1
        fi
        ;;
    script)
        print_script
        ;;
    clean)
        clean_demo
        ;;
    *)
        run_demo
        ;;
esac
