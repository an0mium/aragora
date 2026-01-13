# Troubleshooting Guide

Common issues and solutions for Aragora.

## Table of Contents

- [Server Issues](#server-issues)
- [WebSocket Connection Issues](#websocket-connection-issues)
- [Nomic Loop Problems](#nomic-loop-problems)
- [Database Issues](#database-issues)
- [API Key Configuration](#api-key-configuration)
- [Frontend Issues](#frontend-issues)

---

## Server Issues

### Server Won't Start

**Symptoms:** Server fails to start or crashes immediately.

**Solutions:**

1. **Check port availability:**
   ```bash
   lsof -i :8080
   # Kill any existing process:
   kill -9 <PID>
   ```

2. **Verify Python environment:**
   ```bash
   python --version  # Should be 3.11+
   pip list | grep aragora
   ```

3. **Check for import errors:**
   ```bash
   python -c "from aragora.server.unified_server import UnifiedServer; print('OK')"
   ```

### High Memory Usage

**Solutions:**

1. **Reduce cache size:**
   ```bash
   export ARAGORA_CACHE_MAX_ENTRIES=500
   ```

2. **Lower batch sizes for large debates:**
   - Reduce `limit` parameters in API calls
   - Use pagination

---

## WebSocket Connection Issues

### Connection Fails Immediately

**Symptoms:** WebSocket disconnects right after connecting.

**Solutions:**

1. **Check server is running:**
   ```bash
   curl http://localhost:8080/api/health
   ```

2. **Verify CORS settings:**
   ```bash
   export ARAGORA_ALLOWED_ORIGINS="http://localhost:3000,http://localhost:8080"
   ```

3. **Check for proxy/firewall blocking WebSocket upgrade:**
   - WebSocket uses HTTP upgrade handshake
   - Some proxies block or don't support this

### Connection Drops During Debate

**Solutions:**

1. **Check heartbeat settings:**
   - Server sends ping every 30 seconds
   - Client should respond with pong

2. **Network stability:**
   - Check for intermittent network issues
   - Consider adding reconnection logic

3. **Review server logs for errors:**
   ```bash
   # Server logs connection events
   grep "WebSocket" server.log
   ```

---

## Nomic Loop Problems

### Loop Hangs or Doesn't Progress

**Symptoms:** Nomic loop stuck on a phase.

**Solutions:**

1. **Check phase timeouts:**
   ```bash
   cat .nomic/circuit_breaker.json
   ```

   Default timeouts:
   - context: 300s
   - debate: 600s
   - design: 300s
   - implement: 900s
   - verify: 300s

2. **Verify API keys are valid:**
   ```bash
   # Test Anthropic
   python -c "import anthropic; c = anthropic.Anthropic(); print('OK')"

   # Test OpenAI
   python -c "import openai; c = openai.OpenAI(); print('OK')"
   ```

3. **Check for rate limits:**
   - Review logs for 429 errors
   - Reduce concurrent agent count
   - Add delays between API calls

4. **Review replay events:**
   ```bash
   cat .nomic/replays/nomic-cycle-*/events.jsonl | tail -20
   ```

### Phase Failures

**Symptoms:** Phase fails and rolls back.

**Solutions:**

1. **Check the specific error in events:**
   ```bash
   grep "error" .nomic/replays/nomic-cycle-*/events.jsonl
   ```

2. **Verify implementation tests pass:**
   ```bash
   pytest tests/ -x --timeout=60
   ```

3. **Check for protected file modifications:**
   - Review CLAUDE.md for protected files
   - Nomic loop won't modify protected files

### Rollback Issues

**Symptoms:** Rollback fails or corrupts state.

**Solutions:**

1. **Restore from backup:**
   ```bash
   # List available backups
   ls -la .nomic/backups/

   # Restore a specific backup
   cp -r .nomic/backups/backup_YYYYMMDD_HHMMSS/* .nomic/
   ```

2. **Force reset to clean state:**
   ```bash
   # Backup current state first!
   cp -r .nomic .nomic.bak

   # Reset nomic state
   rm -rf .nomic/replays/*
   python -c "from aragora.modes import NomicLoop; NomicLoop().reset()"
   ```

---

## Database Issues

### Database Validation

Run validation to check database health:

```bash
python scripts/migrate_databases.py --validate
```

### Database Corruption

**Symptoms:** SQLite errors, missing data, crashes on read.

**Solutions:**

1. **Stop all processes accessing the database:**
   ```bash
   # Find processes
   lsof *.db
   ```

2. **Create backup before recovery:**
   ```bash
   python scripts/migrate_databases.py --backup
   ```

3. **Try SQLite recovery:**
   ```bash
   sqlite3 corrupted.db ".recover" | sqlite3 recovered.db
   ```

4. **If recovery fails, delete and recreate:**
   ```bash
   # The server recreates empty databases on startup
   rm corrupted.db
   aragora serve
   ```

### Database Locking

**Symptoms:** "database is locked" errors.

**Solutions:**

1. **Find and kill blocking processes:**
   ```bash
   lsof *.db
   kill <PID>
   ```

2. **Increase timeout:**
   ```python
   conn = sqlite3.connect(db_path, timeout=30)
   ```

3. **Use WAL mode (recommended):**
   ```python
   conn.execute("PRAGMA journal_mode=WAL")
   ```

### Running Database Migration

Consolidate multiple databases:

```bash
# 1. Create backup
python scripts/migrate_databases.py --backup

# 2. Preview migration plan
python scripts/migrate_databases.py --dry-run

# 3. Execute migration
python scripts/migrate_databases.py --migrate

# 4. Verify
python scripts/migrate_databases.py --report
```

---

## API Key Configuration

### Required Environment Variables

| Provider | Variable | Test Command |
|----------|----------|--------------|
| Anthropic | `ANTHROPIC_API_KEY` | `python -c "import anthropic; print(anthropic.Anthropic().models.list())"` |
| OpenAI | `OPENAI_API_KEY` | `python -c "import openai; print(openai.OpenAI().models.list())"` |
| Google | `GEMINI_API_KEY` | Check console output on startup |
| xAI | `XAI_API_KEY` | Check console output on startup |

### Setting Keys

```bash
# In .env file
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Or export directly
export ANTHROPIC_API_KEY=sk-ant-...
```

### Verifying Keys Work

```bash
# Quick test
python -c "
from aragora.agents.api_agents import AnthropicAPIAgent
agent = AnthropicAPIAgent(name=\"anthropic-api\")
print('Anthropic API: OK')
"
```

### Rate Limit Errors

**Symptoms:** 429 errors, "rate limit exceeded"

**Solutions:**

1. **Reduce concurrent requests:**
   ```python
   # In debate config
   protocol = DebateProtocol(max_concurrent_agents=2)
   ```

2. **Add delays between calls:**
   ```python
   import time
   time.sleep(1)  # Between API calls
   ```

3. **Use different API tiers:**
   - Consider upgrading API plan
   - Use multiple API keys with rotation

---

## Frontend Issues

### Frontend Not Loading

**Symptoms:** Blank page, loading spinner stuck.

**Solutions:**

1. **Check build succeeded:**
   ```bash
   cd aragora/live
   npm run build
   ```

2. **Verify development server:**
   ```bash
   npm run dev
   # Should be available at http://localhost:3000
   ```

3. **Check browser console for errors:**
   - Open DevTools (F12)
   - Check Console and Network tabs

4. **Verify API backend is running:**
   ```bash
   curl http://localhost:8080/api/health
   ```

### WebSocket Not Connecting from Frontend

**Solutions:**

1. **Check CORS configuration:**
   ```bash
   export ARAGORA_ALLOWED_ORIGINS="http://localhost:3000"
   ```

2. **Verify WebSocket URL in frontend:**
   ```typescript
   // Should match your backend
   const WS_URL = 'ws://localhost:8765/ws';
   ```

3. **Check for HTTPS/WSS mismatch:**
   - HTTP pages should use ws://
   - HTTPS pages must use wss://

### Slow Performance

**Solutions:**

1. **Enable production build:**
   ```bash
   npm run build
   npm start  # Instead of npm run dev
   ```

2. **Check bundle size:**
   ```bash
   npm run analyze
   ```

3. **Verify lazy loading is working:**
   - Check Network tab in DevTools
   - Heavy components should load on demand

---

## Getting Help

1. **Check documentation:**
   - `docs/API_REFERENCE.md` - Complete API reference
   - `docs/ENVIRONMENT.md` - Environment setup
   - `docs/ARCHITECTURE.md` - System overview

2. **Review logs:**
   ```bash
   # Server logs
   tail -f server.log

   # Nomic loop events
   tail -f .nomic/replays/*/events.jsonl
   ```

3. **Report issues:**
   - https://github.com/anthropics/claude-code/issues

---

## Quick Diagnostic Commands

```bash
# Check system status
python -c "
from aragora.server.unified_server import UnifiedServer
print('Import: OK')
"

# Validate databases
python scripts/migrate_databases.py --validate

# Check API health
curl http://localhost:8080/api/health

# List recent debates
curl http://localhost:8080/api/debates?limit=5

# Get agent leaderboard
curl http://localhost:8080/api/leaderboard?limit=10
```
