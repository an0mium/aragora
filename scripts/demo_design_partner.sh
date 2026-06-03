#!/usr/bin/env bash
# Design Partner Demo Script
# Walks through the core Aragora product loop in ~2 minutes.
#
# Usage:
#   ./scripts/demo_design_partner.sh
#   ./scripts/demo_design_partner.sh --question "Your custom question"
#
# Prerequisites:
#   - At least one LLM provider key set (GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)
#   - Python 3.11+ with aragora installed
#
# What this demonstrates:
#   1. Live adversarial debate with structured receipt
#   2. Receipt visible via API and CLI
#   3. Prompt-to-spec pipeline
#   4. EU AI Act compliance bundle from real receipt
set -euo pipefail

QUESTION="${1:---question}"
if [ "$QUESTION" = "--question" ]; then
    QUESTION="${2:-Should we adopt a microservice architecture for our payment processing system?}"
else
    QUESTION="${QUESTION}"
fi

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "${ARAGORA_REPO_ROOT:-$PWD}")"

BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RESET='\033[0m'

echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}  ARAGORA — Decision Integrity Platform${RESET}"
echo -e "${BOLD}  Design Partner Demo${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""

# Step 1: Live debate
echo -e "${CYAN}[Step 1/4] Live adversarial debate${RESET}"
echo -e "  Question: ${QUESTION}"
echo ""

ARAGORA_USER_ID=demo python3 -m aragora.cli.main quickstart \
    --question "$QUESTION" \
    --no-browser

echo ""

# Step 2: Receipt verification
echo -e "${CYAN}[Step 2/4] Receipt verification${RESET}"
RECEIPT_PATH=".aragora/receipts/quickstart-live-receipt.json"
if [ -f "$RECEIPT_PATH" ]; then
    python3 -m aragora.cli.main receipt verify "$RECEIPT_PATH"
    echo ""
    echo -e "  ${GREEN}Receipt is inspectable, verifiable, and persisted to the API.${RESET}"
    echo "  API endpoint: GET /api/v2/receipts"
    echo "  Dashboard: visible in ReceiptsBrowser"
else
    echo "  Receipt file not found at $RECEIPT_PATH"
fi
echo ""

# Step 3: Prompt-to-spec
echo -e "${CYAN}[Step 3/4] Prompt-to-spec pipeline${RESET}"
echo "  Generating specification from vague prompt..."
echo ""

python3 -m aragora.cli.main spec \
    "Design an approval workflow for the payment system" \
    --skip-interrogation --skip-research \
    --output /tmp/demo-spec.json

echo ""

# Step 4: EU AI Act compliance
echo -e "${CYAN}[Step 4/4] EU AI Act compliance bundle${RESET}"
echo "  Generating compliance artifacts from the live receipt..."
echo ""

python3 -m aragora.cli.main compliance \
    --generate-artifacts \
    --receipt "$RECEIPT_PATH"

echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}  Demo Complete${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""
echo "  What you just saw:"
echo "    1. A live multi-agent adversarial debate producing a signed receipt"
echo "    2. That receipt verified and visible on API/dashboard surfaces"
echo "    3. A vague prompt transformed into a structured specification"
echo "    4. EU AI Act compliance artifacts generated from the real receipt"
echo ""
echo "  Next steps for your evaluation:"
echo "    aragora triage auth         # Connect your Gmail for inbox triage"
echo "    aragora triage run --dry-run # Preview AI triage on your inbox"
echo "    aragora decide 'task' --spec /tmp/demo-spec.json  # Debate the spec"
echo ""
