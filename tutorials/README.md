# Aragora Interactive Tutorials

Interactive Jupyter notebooks to learn Aragora step by step.

## Prerequisites

- Python 3.10+
- At least one API key (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
- Jupyter installed (`pip install jupyter`)

## Quick Start

```bash
# Install Aragora
pip install -e .

# Start Jupyter
cd tutorials
jupyter notebook
```

## Tutorials

| # | Tutorial | Time | Topics |
|---|----------|------|--------|
| 1 | [Basic Debate](01_basic_debate.ipynb) | 5 min | Agents, Environment, Protocol, Results |
| 2 | [Custom Agents](02_custom_agents.ipynb) | 10 min | Roles, Personas, Providers, Temperature |
| 3 | [Memory Integration](03_memory_integration.ipynb) | 10 min | CritiqueStore, Continuum, ELO |
| 4 | [Production Deployment](04_production_deployment.ipynb) | 15 min | Config, Server, Monitoring, Caching |
| 5 | [Advanced Features](05_advanced_features.ipynb) | 15 min | Gauntlet, WebSocket, Templates, MCP |

## Learning Path

**Beginner** (30 min)
1. Start with Tutorial 1 to understand the core concepts
2. Move to Tutorial 2 to customize agents

**Intermediate** (1 hour)
3. Complete Tutorial 3 for memory and learning
4. Run through Tutorial 4 for production setup

**Advanced** (30 min)
5. Explore Tutorial 5 for enterprise features

## Running Without Jupyter

You can also run the examples directly:

```bash
# Run the simple debate example
python examples/01_simple_debate.py

# Run a CLI debate
aragora ask "Design a rate limiter" --agents anthropic-api,openai-api
```

## Troubleshooting

**"No API key found"**
```bash
# Set at least one key
export ANTHROPIC_API_KEY=sk-ant-...
# Or add to .env file
```

**"Module not found"**
```bash
# Install Aragora in development mode
pip install -e .
```

**Notebook kernel not found**
```bash
# Install ipykernel
pip install ipykernel
python -m ipykernel install --user --name aragora
```

## Further Resources

- [Getting Started Guide](../docs/GETTING_STARTED.md)
- [API Reference](../docs/API_REFERENCE.md)
- [CLI Reference](../docs/CLI_REFERENCE.md)
- [Examples](../examples/)
