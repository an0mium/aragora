# Installation Guide

This guide covers different ways to install and run Aragora.

If you want the shortest path to a first debate, start at [docs/START_HERE.md](docs/START_HERE.md).

## Quick Start (Local Development)

### Prerequisites

- Python 3.10 or higher
- Git
- API keys for at least one LLM provider

### Installation

```bash
# Clone the repository
git clone https://github.com/an0mium/aragora.git
cd aragora

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Configuration

Copy the starter environment file and add your API keys:

```bash
cp .env.starter .env
```

Edit `.env` with your API keys (use `.env.example` for the full template):

```
GEMINI_API_KEY=your_gemini_key
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
XAI_API_KEY=your_xai_key
```

Optional but recommended:
```
ARAGORA_DATA_DIR=.nomic
```

### Running

```bash
# Run the nomic loop
python scripts/nomic_loop.py

# Run the API server
aragora serve --host 0.0.0.0 --ws-port 8765 --api-port 8080
```

## Production Deployment (AWS Lightsail)

For production deployments, use the provided setup script.

### 1. Create a Lightsail Instance

- Launch an Ubuntu 22.04 instance
- Recommended: 2 GB RAM or higher
- Open port 8765 in the firewall

### 2. Run Setup Script

SSH into your instance and run:

```bash
curl -sSL https://raw.githubusercontent.com/an0mium/aragora/main/deploy/lightsail-setup.sh | bash
```

Or manually:

```bash
cd /home/ubuntu
git clone https://github.com/an0mium/aragora.git
cd aragora
./deploy/lightsail-setup.sh
```

### 3. Configure Environment

```bash
sudo nano /home/ubuntu/aragora/.env
# Add your API keys
```

Restart the service:

```bash
sudo systemctl restart aragora-api
```

### 4. Monitor

```bash
# View logs
sudo journalctl -u aragora-api -f

# Check service status
sudo systemctl status aragora-api
```

## Docker Deployment

```bash
# Build image
docker build -t aragora .

# Run with environment variables
docker run -d \
  -p 8080:8080 \
  -p 8765:8765 \
  -e GEMINI_API_KEY=your_key \
  -e ANTHROPIC_API_KEY=your_key \
  aragora
```

## Live Dashboard Setup

The live dashboard is a Next.js application in `aragora/live/`.

### Prerequisites

- Node.js 18+
- npm or pnpm

### Installation

```bash
cd aragora/live
npm install
```

### Configuration

Create `.env.local`:

```
NEXT_PUBLIC_WS_URL=ws://localhost:8765/ws
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url (optional)
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key (optional)
```

### Running

```bash
# Development
npm run dev

# Production build
npm run build
npm start
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes* | Google Gemini API key |
| `ANTHROPIC_API_KEY` | Yes* | Anthropic Claude API key |
| `OPENAI_API_KEY` | Yes* | OpenAI API key |
| `XAI_API_KEY` | Yes* | xAI Grok API key |
| `ARAGORA_ALLOWED_ORIGINS` | No | Comma-separated CORS origins |
| `NEXT_PUBLIC_WS_URL` | No | WebSocket server URL for dashboard |
| `NEXT_PUBLIC_SUPABASE_URL` | No | Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | No | Supabase anonymous key |

*At least one LLM API key is required.

## Troubleshooting

### Common Issues

**"No module named aragora"**
```bash
# Ensure you installed in editable mode
pip install -e .
```

**WebSocket connection refused**
```bash
# Check if server is running
sudo systemctl status aragora-api

# Check firewall
sudo ufw allow 8765
```

**API rate limits**
- Reduce `max_concurrent_agents` in config
- Add delays between requests

### Getting Help

- Check existing issues: https://github.com/an0mium/aragora/issues
- Open a new issue with logs and environment details
