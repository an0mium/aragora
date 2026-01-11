# Nomic Loop Troubleshooting Guide

The Nomic Loop is Aragora's self-improvement system. This guide covers common issues and how to resolve them.

## Understanding the Nomic Loop

The loop runs 5 phases:

| Phase | Name | Purpose |
|-------|------|---------|
| 0 | **Context** | Gather codebase understanding |
| 1 | **Debate** | Agents propose improvements |
| 2 | **Design** | Architecture planning |
| 3 | **Implement** | Code generation |
| 4 | **Verify** | Tests and checks |

After Phase 4, changes are committed (if verified) and the loop restarts.

## Common Issues

### Phase 0: Context Gathering

#### "Failed to read codebase"

**Symptom**: Context phase fails with file access errors.

**Causes**:
- Nomic directory doesn't exist
- Permissions issue
- Corrupted state files

**Fix**:
```bash
# Reset nomic state
rm -rf .nomic/state.json
mkdir -p .nomic

# Restart
python scripts/nomic_loop.py --cycles 1
```

#### "Context too large"

**Symptom**: Memory errors during context gathering.

**Fix**:
```bash
# Limit context size
export ARAGORA_MAX_CONTEXT_FILES=100
export ARAGORA_MAX_FILE_SIZE=50000  # bytes

python scripts/nomic_loop.py
```

### Phase 1: Debate

#### "No agents available"

**Symptom**: Debate fails to start, no agents respond.

**Causes**:
- Missing API keys
- All agents rate-limited
- Network issues

**Fix**:
```bash
# Check API keys
env | grep -E "(ANTHROPIC|OPENAI|GEMINI)_API_KEY"

# Test individual agent
python -c "
from aragora.agents.registry import AgentRegistry, register_all_agents
register_all_agents()
agent = AgentRegistry.create('anthropic-api')
print(f'Agent created: {agent}')
"
```

#### "Debate timeout"

**Symptom**: Phase 1 hangs or times out.

**Fix**:
```bash
# Increase timeout
export ARAGORA_DEBATE_TIMEOUT=600  # 10 minutes

# Or reduce debate complexity
python scripts/nomic_loop.py --debate-rounds 2
```

#### "Hollow consensus detected"

**Symptom**: Trickster warns about false agreement.

This is intentional. The Trickster detected agents agreeing superficially without substantive engagement.

**Fix**: Re-run with more rounds or different agents:
```bash
python scripts/nomic_loop.py --debate-rounds 5 --agents anthropic-api,openai-api,gemini
```

### Phase 2: Design

#### "Design rejected by safety check"

**Symptom**: Design phase produces a plan that's rejected.

**Causes**:
- Plan modifies protected files
- Plan is too broad/risky
- Checksum mismatch

**Fix**:
Check the rejected plan in `.nomic/designs/`:
```bash
cat .nomic/designs/latest.json | jq '.rejected_reason'
```

If the plan is valid, override (with caution):
```bash
python scripts/nomic_loop.py --skip-safety-check
```

#### "No improvements proposed"

**Symptom**: Design phase produces empty plan.

This can be normal - the system found nothing to improve.

**Fix**: Provide specific improvement goals:
```bash
python scripts/nomic_loop.py --goal "Improve error handling in agents/"
```

### Phase 3: Implementation

#### "Codex/Claude refused to implement"

**Symptom**: Implementation agent declines the task.

**Causes**:
- Task violates safety guidelines
- Task is ambiguous
- Agent doesn't have capability

**Fix**:
```bash
# Try different implementation agent
export ARAGORA_IMPL_AGENT=anthropic-api  # or openai-api

# Simplify the task
python scripts/nomic_loop.py --max-changes 5
```

#### "Syntax error in generated code"

**Symptom**: Implementation produces invalid Python.

**Fix**:
```bash
# Enable syntax validation before commit
export ARAGORA_VALIDATE_SYNTAX=true

# Check the failed file
python -m py_compile path/to/file.py
```

The loop should automatically retry with feedback.

#### "Import error after changes"

**Symptom**: New code breaks imports.

**Fix**:
```bash
# Rollback
git checkout HEAD~1 -- path/to/broken/file.py

# Or use nomic rollback
python scripts/nomic_loop.py --rollback
```

### Phase 4: Verification

#### "Tests failing"

**Symptom**: Verification phase fails on tests.

This is expected behavior - the loop won't commit breaking changes.

**Fix**:
```bash
# See what failed
pytest tests/ -v --tb=short 2>&1 | tail -50

# The loop will retry with test feedback
# If stuck, reset:
python scripts/nomic_loop.py --reset-phase 3
```

#### "Verification timeout"

**Symptom**: Tests take too long.

**Fix**:
```bash
# Increase verification timeout
export ARAGORA_VERIFY_TIMEOUT=600  # 10 minutes

# Or run subset of tests
export ARAGORA_TEST_PATTERN="tests/unit/"
```

#### "Protected file modified"

**Symptom**: Verification rejects changes to protected files.

Protected files are defined in `CLAUDE.md`:
- `core.py`
- `aragora/__init__.py`
- `.env`
- `scripts/nomic_loop.py`

**Fix**: These files require manual modification. The loop correctly rejected automated changes.

### Loop-Level Issues

#### "Infinite loop / no progress"

**Symptom**: Loop keeps running but nothing improves.

**Causes**:
- Improvement goal is impossible
- Agents keep proposing same changes
- State corruption

**Fix**:
```bash
# Clear loop state
rm -rf .nomic/state.json .nomic/history/

# Set cycle limit
python scripts/nomic_loop.py --cycles 3 --goal "specific improvement"
```

#### "Memory exhaustion"

**Symptom**: Loop crashes with OOM.

**Fix**:
```bash
# Reduce memory usage
export ARAGORA_MAX_MEMORY_MB=4096
export ARAGORA_CLEANUP_INTERVAL=5  # Clean every 5 cycles

# Use streaming mode
python scripts/run_nomic_with_stream.py run --cycles 3
```

#### "State file corrupted"

**Symptom**: Loop fails to start with JSON errors.

**Fix**:
```bash
# Backup and reset
mv .nomic/state.json .nomic/state.json.bak
python scripts/nomic_loop.py --init
```

## Recovery Procedures

### Full Reset

```bash
# Backup current state
tar -czf nomic_backup_$(date +%Y%m%d).tar.gz .nomic/

# Reset everything
rm -rf .nomic/
mkdir -p .nomic

# Reinitialize
python scripts/nomic_loop.py --init
```

### Rollback Last Change

```bash
# Via nomic
python scripts/nomic_loop.py --rollback

# Via git
git log --oneline -5  # Find the commit
git revert HEAD       # Revert last commit
```

### Skip Failed Phase

```bash
# Skip to next phase (use with caution)
python scripts/nomic_loop.py --skip-phase 3 --start-phase 4
```

## Debug Mode

Enable verbose logging:

```bash
export ARAGORA_LOG_LEVEL=DEBUG
export ARAGORA_NOMIC_DEBUG=true

python scripts/nomic_loop.py --verbose 2>&1 | tee nomic_debug.log
```

Key log locations:
- `.nomic/logs/` - Phase logs
- `.nomic/history/` - Change history
- `.nomic/designs/` - Generated designs

## Monitoring

### Check Loop Status

```bash
# Current state
cat .nomic/state.json | jq '.'

# Recent history
ls -la .nomic/history/ | tail -10
```

### Watch Progress

```bash
# Stream mode (recommended)
python scripts/run_nomic_with_stream.py run --cycles 3

# Or tail logs
tail -f .nomic/logs/nomic_$(date +%Y%m%d).log
```

## Getting Help

If issues persist:

1. Collect debug logs: `tar -czf nomic_debug.tar.gz .nomic/logs/`
2. Note the exact error message
3. Open an issue: [github.com/an0mium/aragora/issues](https://github.com/an0mium/aragora/issues)

---

*The Nomic Loop learns from failure. So should we.*
