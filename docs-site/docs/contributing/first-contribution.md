---
title: First Contribution Guide
description: First Contribution Guide
---

# First Contribution Guide

Welcome! This guide will help you make your first contribution to Aragora.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/an0mium/aragora.git
cd aragora

# 2. Set up environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -e ".[dev]"

# 3. Set up API keys (at least one)
export ANTHROPIC_API_KEY="your-key"
# or
export OPENAI_API_KEY="your-key"

# 4. Run tests to verify setup
pytest tests/debate/test_orchestrator.py -v

# 5. Start the server
python -m aragora.server.unified_server --port 8080
```

## Project Structure

```
aragora/
├── aragora/              # Main package
│   ├── debate/           # Core debate engine (start here!)
│   ├── agents/           # AI agent implementations
│   ├── server/           # HTTP/WebSocket API
│   ├── memory/           # Learning and persistence
│   └── ...
├── tests/                # Test files mirror source structure
├── docs/                 # Documentation
└── aragora/live/         # Next.js frontend
```

## Suggested First Issues

### Easy (Good First Issue)

1. **Add docstrings to public methods**
   - Files: `aragora/debate/orchestrator.py`, `aragora/core_types.py`
   - Task: Add missing docstrings with Args, Returns, Examples
   - Skills: Python, reading code

2. **Improve error messages**
   - Files: `aragora/server/handlers/*.py`
   - Task: Make error messages more user-friendly
   - Skills: Python, API design

3. **Add missing empty states to UI pages**
   - Files: `aragora/live/src/app/*/page.tsx`
   - Task: Use EmptyState component for no-data scenarios
   - Skills: React, TypeScript

4. **Add accessibility attributes**
   - Files: `aragora/live/src/components/*.tsx`
   - Task: Add aria-labels, roles, keyboard navigation
   - Skills: React, WCAG knowledge

### Medium

5. **Add unit tests for untested modules**
   - Files: `tests/debate/phases/`
   - Task: Write tests for debate phases
   - Skills: Python, pytest, mocking

6. **Implement breadcrumb navigation**
   - Files: `aragora/live/src/components/`
   - Task: Create reusable Breadcrumbs component
   - Skills: React, TypeScript

7. **Add mobile responsiveness**
   - Files: `aragora/live/src/app/*/page.tsx`
   - Task: Make complex pages mobile-friendly
   - Skills: CSS, Tailwind, responsive design

### Challenging

8. **Add a new debate phase**
   - Files: `aragora/debate/phases/`
   - Task: Implement a new phase (e.g., summary phase)
   - Skills: Python, async, debate system understanding

9. **Create a new connector**
   - Files: `aragora/connectors/`
   - Task: Add integration for a new service
   - Skills: Python, APIs, OAuth

## Making Your First PR

### 1. Find an Issue

- Browse [issues labeled "good first issue"](https://github.com/an0mium/aragora/labels/good%20first%20issue)
- Comment that you'd like to work on it
- Wait for assignment to avoid duplicate work

### 2. Create a Branch

```bash
git checkout -b feat/your-feature-name
# or
git checkout -b fix/issue-description
```

### 3. Make Changes

- Follow existing code style
- Add tests for new functionality
- Update documentation if needed

### 4. Test Your Changes

```bash
# Run relevant tests
pytest tests/path/to/test_file.py -v

# Run type checking (optional)
python -m mypy path/to/your_module.py

# For frontend changes
cd aragora/live
npm run lint
npm run build
```

### 5. Commit with Conventional Commits

```bash
git commit -m "feat(debate): add summary phase to debate flow"
git commit -m "fix(api): handle timeout errors gracefully"
git commit -m "docs: add docstrings to Arena class"
git commit -m "test: add unit tests for consensus detection"
```

Prefixes: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`

### 6. Push and Create PR

```bash
git push -u origin feat/your-feature-name
```

Then open a PR on GitHub with:
- Clear description of changes
- Reference to issue (e.g., "Closes #123")
- Screenshots for UI changes

## Code Style

### Python

- Use type hints for function signatures
- Prefer dataclasses for data structures
- Use async/await for I/O operations
- Keep functions focused and small

```python
async def run_debate(
    self,
    task: str,
    agents: list[Agent],
    *,
    rounds: int = 3,
) -> DebateResult:
    """Run a debate on the given task.

    Args:
        task: The question or topic to debate
        agents: List of agents to participate
        rounds: Number of debate rounds

    Returns:
        DebateResult with final answer and confidence
    """
    ...
```

### TypeScript/React

- Use functional components with hooks
- Prefer explicit types over `any`
- Use Tailwind utility classes

```typescript
interface Props {
  title: string;
  onAction?: () => void;
}

export function Component({ title, onAction }: Props) {
  return (
    <div className="p-4 border border-acid-green/30">
      <h2 className="text-lg font-mono">\{title\}</h2>
      {onAction && (
        <button onClick=\{onAction\} aria-label={`Action for $\{title\}`}>
          Click
        </button>
      )}
    </div>
  );
}
```

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/an0mium/aragora/discussions)
- **Bugs**: Open an [Issue](https://github.com/an0mium/aragora/issues)
- **Docs**: Check [INDEX.md](./documentation-index) for doc navigation

## Architecture Decision Tree

When implementing features, consider:

```
Is it a debate mechanic?
├── Yes → aragora/debate/
│   ├── Phase-related? → aragora/debate/phases/
│   ├── Consensus? → aragora/debate/consensus.py
│   └── Memory? → aragora/memory/
│
Is it an API endpoint?
├── Yes → aragora/server/handlers/
│   └── Follow existing handler patterns
│
Is it UI-related?
├── Yes → aragora/live/src/
│   ├── Page? → app/*/page.tsx
│   ├── Component? → components/*.tsx
│   └── Hook? → hooks/*.ts
│
Is it agent-related?
└── Yes → aragora/agents/
    ├── CLI agent? → cli_agents.py
    └── API agent? → api_agents/
```

## Protected Files

**Do not modify without approval:**
- `CLAUDE.md`
- `core_types.py`
- `aragora/__init__.py`
- `scripts/nomic_loop.py`

## Thank You!

Every contribution helps make Aragora better. We appreciate your time and effort!
