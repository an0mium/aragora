# Aragora in 5 Minutes

Get AI agents debating in under 5 minutes.

## 1. Install

```bash
pip install aragora
```

Or install from source:

```bash
git clone https://github.com/an0mium/aragora
cd aragora
pip install -e .
```

## 2. Configure

Copy the minimal config and add your API key:

```bash
cp .env.starter .env
```

Edit `.env` and uncomment your preferred provider:

```bash
# Pick ONE:
ANTHROPIC_API_KEY=sk-ant-...   # Claude (recommended)
# OPENAI_API_KEY=sk-...        # GPT-4
# MISTRAL_API_KEY=...          # Mistral
```

## 3. Run Your First Debate

```bash
aragora ask "Should we use microservices or a monolith?"
```

This will:
- Spin up multiple AI agents
- Have them debate the topic
- Show real-time arguments and critiques
- Produce a consensus answer (or document dissent)

## 4. View the UI

Start the server with the web interface:

```bash
aragora serve
```

Open http://localhost:8080 to see:
- Live debate streaming
- Agent activity panels
- Consensus visualization
- Evidence citations

## 5. Try Different Modes

**Quick debate (demo mode)**
```bash
aragora ask "Is TypeScript worth the overhead?" --demo
```

**Choose specific agents**
```bash
aragora ask "Best database for real-time apps?" --agents claude-sonnet,gpt-4o,deepseek-r1
```

**More debate rounds**
```bash
aragora ask "Should we rewrite in Rust?" --rounds 5
```

**Code review with Gauntlet**
```bash
aragora gauntlet myspec.md --input-type spec
```

## What's Next?

| Guide | Description |
|-------|-------------|
| [QUICKSTART.md](./QUICKSTART.md) | Detailed getting started guide |
| [FEATURES.md](./FEATURES.md) | All features and capabilities |
| [CUSTOM_AGENTS.md](./CUSTOM_AGENTS.md) | Create custom AI agents |
| [API_REFERENCE.md](./API_REFERENCE.md) | Full API documentation |

## Common Issues

**"No API key configured"**
- Make sure you've set at least one provider key in `.env`
- Check the key is valid and has credits

**Rate limited**
- Add `OPENROUTER_API_KEY` for automatic fallback
- Use `--agents` to switch providers

**"Command not found: aragora"**
- Ensure pip installed to your PATH
- Try `python -m aragora ask "test"` instead

## Get Help

- GitHub Issues: https://github.com/an0mium/aragora/issues
- Documentation: https://aragora.ai/docs
