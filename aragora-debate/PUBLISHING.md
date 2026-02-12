# Publishing aragora-debate to PyPI

## Prerequisites

```bash
pip install build twine
```

## Version Management

The version is set in two places that must stay in sync:

| File | Field |
|------|-------|
| `pyproject.toml` | `version = "X.Y.Z"` |
| `src/aragora_debate/__init__.py` | `__version__ = "X.Y.Z"` |

Bump both before publishing.

## Build

```bash
# Clean previous artifacts
rm -rf dist/

# Build sdist and wheel
python -m build
```

This creates:
- `dist/aragora_debate-X.Y.Z.tar.gz` (source distribution)
- `dist/aragora_debate-X.Y.Z-py3-none-any.whl` (wheel)

## Validate

```bash
# Check package metadata and README rendering
twine check dist/*

# Verify wheel contents
python -m zipfile -l dist/aragora_debate-*.whl

# Run tests against the built package
pip install dist/aragora_debate-*.whl
python -m pytest tests/ -v
```

## Test Publish (TestPyPI)

```bash
twine upload --repository testpypi dist/*
```

Verify at https://test.pypi.org/project/aragora-debate/

Test installation from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ aragora-debate
python -c "import aragora_debate; print(aragora_debate.__version__)"
```

## Production Publish

```bash
twine upload dist/*
```

Requires PyPI credentials configured via:
- `~/.pypirc` file, or
- `TWINE_USERNAME` / `TWINE_PASSWORD` environment variables, or
- `TWINE_API_KEY` for token-based auth (recommended)

## Post-Publish Checklist

- [ ] Verify installation: `pip install aragora-debate==X.Y.Z`
- [ ] Verify imports: `python -c "from aragora_debate import Arena, Debate; print('OK')"`
- [ ] Verify optional deps: `pip install aragora-debate[all]`
- [ ] Verify `python -m aragora_debate` demo runs
- [ ] Tag release: `git tag v-debate-X.Y.Z && git push --tags`

## Package Structure

```
aragora-debate/
├── pyproject.toml          # Build config (hatchling backend)
├── LICENSE                 # MIT
├── README.md               # PyPI long description
├── src/aragora_debate/     # Source package
│   ├── __init__.py         # Public API exports
│   ├── __main__.py         # CLI demo entry point
│   ├── py.typed            # PEP 561 type marker
│   ├── arena.py            # Arena debate engine
│   ├── debate.py           # High-level Debate API
│   ├── types.py            # Core types (Agent, Critique, Vote, etc.)
│   ├── receipt.py          # DecisionReceipt builder
│   ├── agents.py           # Provider agents (Claude, OpenAI, Mistral, Gemini)
│   ├── evidence.py         # Evidence quality + hollow consensus detection
│   ├── convergence.py      # Convergence tracking
│   ├── events.py           # Event emitter
│   ├── trickster.py        # Evidence-powered trickster
│   ├── cross_analysis.py   # Cross-proposal analysis
│   ├── _mock.py            # MockAgent (always available)
│   └── styled_mock.py      # Style-aware mock agents
├── tests/                  # 235 tests
└── examples/               # Usage examples
```

## Optional Dependencies

| Extra | Installs | For |
|-------|----------|-----|
| `anthropic` | `anthropic>=0.20.0` | `ClaudeAgent` |
| `openai` | `openai>=1.0.0` | `OpenAIAgent` |
| `mistral` | `mistralai>=1.0.0` | `MistralAgent` |
| `gemini` | `google-genai>=1.0.0` | `GeminiAgent` |
| `all` | All provider SDKs | All agents |
| `dev` | pytest, pytest-asyncio | Development |
