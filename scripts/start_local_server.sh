#!/bin/bash
# Start Aragora local development server

set -e

cd "$(dirname "$0")/.."

# Check for required API keys
if [[ -z "$ANTHROPIC_API_KEY" && -z "$OPENAI_API_KEY" && -z "$OPENROUTER_API_KEY" ]]; then
    echo "ERROR: No API keys found!"
    echo ""
    echo "At least one AI provider key is required. Set one of:"
    echo "  export ANTHROPIC_API_KEY='your-key-here'"
    echo "  export OPENAI_API_KEY='your-key-here'"
    echo "  export OPENROUTER_API_KEY='your-key-here'  # Fallback for multiple models"
    echo ""
    echo "Get keys from:"
    echo "  - Anthropic: https://console.anthropic.com/"
    echo "  - OpenAI: https://platform.openai.com/api-keys"
    echo "  - OpenRouter: https://openrouter.ai/keys (recommended - one key, many models)"
    exit 1
fi

echo "Starting Aragora server..."
echo "API: http://localhost:8080"
echo "WebSocket: ws://localhost:8765"
echo ""

# Start the server
python -m aragora.server.unified_server --port 8080 --host 0.0.0.0
