# AI Red Team PR Review Demos

These demos showcase Aragora's multi-agent code review capabilities, demonstrating the "magic moments" where AI agents either agree unanimously or surface interesting disagreements.

## Quick Start

```bash
# Run any demo
cd demos/pr-review/01_sql_injection
./run.sh

# Or pipe the diff directly
cat 01_sql_injection/diff.patch | aragora review
```

## Demo Scenarios

### 01 - SQL Injection Detection
**Magic Moment:** All agents unanimously identify SQL injection vulnerabilities

Shows how multiple AI models agree on obvious security flaws, giving high confidence that these are real issues worth fixing.

```bash
./01_sql_injection/run.sh
```

### 02 - Caching Tradeoff
**Magic Moment:** Agents disagree on whether in-memory caching is appropriate

Demonstrates split opinions - one agent favors caching for performance, another warns about staleness and memory issues. This helps you understand the tradeoffs.

```bash
./02_caching_tradeoff/run.sh
```

### 03 - Async Conversion Issues
**Magic Moment:** Unanimous detection of blocking I/O in async code

A common mistake when converting sync to async code. Both agents flag the use of `requests` (blocking) instead of `aiohttp` (async) inside async functions.

```bash
./03_async_conversion/run.sh
```

### 04 - Error Handling Gaps
**Magic Moment:** Both agents find the same runtime error paths

Shows detection of NoneType errors and SQLAlchemy session scope issues that would cause crashes in production.

```bash
./04_error_handling/run.sh
```

## Expected Output

Each demo includes:
- `diff.patch` - The code changes to review
- `expected_comment.md` - What the GitHub comment should look like
- `run.sh` - Script to run the demo

## Requirements

- Python 3.11+
- Aragora installed: `pip install aragora`
- API keys configured:
  ```bash
  export ANTHROPIC_API_KEY=sk-ant-...
  export OPENAI_API_KEY=sk-...
  ```

## Understanding the Output

### Unanimous Issues
When all agents agree on an issue, it appears in the "Unanimous Issues" section. High confidence - definitely worth addressing.

### Split Opinions
When agents disagree, you see a table showing which agents are for/against. This surfaces tradeoffs for you to decide.

### Agreement Score
- **90-100%**: High consensus, issues are likely real
- **60-89%**: Some disagreement, review the split opinions
- **<60%**: Significant disagreement, may need more context

## Customizing Reviews

```bash
# Focus only on security
cat diff.patch | aragora review --focus security

# Use different agents
cat diff.patch | aragora review --agents anthropic-api,openai-api,gemini

# More debate rounds for complex changes
cat diff.patch | aragora review --rounds 3

# Output as JSON for CI integration
cat diff.patch | aragora review --output-format json
```
