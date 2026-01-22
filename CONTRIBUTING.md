# Contributing to Aragora

Thank you for your interest in contributing to Aragora! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend development)
- Docker (optional, for containerized development)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/aragora-ai/aragora.git
cd aragora

# Install development dependencies
make dev

# Run tests to verify setup
make test-fast

# Start the development server
make serve
```

### Using VS Code Dev Container

1. Install the "Dev Containers" extension in VS Code
2. Open the project folder
3. Click "Reopen in Container" when prompted
4. Wait for the container to build and dependencies to install

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run fast tests only (excludes slow, e2e, load tests)
make test-fast

# Run tests with coverage
make test-cov

# Run specific test file
pytest tests/debate/test_orchestrator.py -v
```

### Code Quality

```bash
# Run linter
make lint

# Format code
make format

# Type checking
make typecheck

# Run all checks
make check
```

### Development Server

```bash
# Start API server
make serve

# Interactive debate REPL
make repl

# System health check
make doctor
```

## Code Style

### Python

- Follow PEP 8 with a line length of 88 characters
- Use type hints for all function signatures
- Use `ruff` for linting and formatting
- Use `mypy` for type checking

### Commit Messages

Follow conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(marketplace): add template rating system
fix(debate): handle timeout in consensus phase
docs(api): update authentication guide
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes
4. Run tests and checks: `make check && make test`
5. Commit with a descriptive message
6. Push to your fork: `git push origin feat/my-feature`
7. Open a Pull Request

## Project Structure

```
aragora/
├── aragora/           # Main package
│   ├── agents/        # Agent implementations
│   ├── debate/        # Debate orchestration
│   ├── memory/        # Memory systems
│   ├── server/        # API server
│   ├── cli/           # CLI commands
│   └── marketplace/   # Template marketplace
├── tests/             # Test suite
├── docs/              # Documentation
└── aragora-js/        # TypeScript SDK
```

## Adding New Features

### Adding a New Agent

1. Create agent file in `aragora/agents/api_agents/`
2. Implement the `Agent` protocol
3. Register in `aragora/agents/__init__.py`
4. Add tests in `tests/agents/`

### Adding a CLI Command

1. Create command file in `aragora/cli/`
2. Register in `aragora/cli/main.py`
3. Add tests in `tests/cli/`

### Adding a Server Handler

1. Create handler in `aragora/server/handlers/`
2. Register routes in server startup
3. Add tests in `tests/server/handlers/`

## Getting Help

- Check existing issues for similar problems
- Open a new issue with a clear description
- Join our community discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
