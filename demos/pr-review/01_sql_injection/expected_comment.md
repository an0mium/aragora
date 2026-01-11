## AI Red Team Code Review

**3 agents reviewed this PR** (Anthropic, OpenAI)

### Unanimous Issues
> All AI models agree - address these first

- **Security** `api/users.py:15` - SQL injection vulnerability in search_users function. User input is directly interpolated into SQL query.
- **Security** `api/users.py:23` - SQL injection vulnerability in get_user function. The user_id parameter is interpolated directly into the query string.

### Critical & High Severity Issues

- **CRITICAL**: SQL injection in `search_users` - attacker can extract entire database with `' OR '1'='1`
- **CRITICAL**: SQL injection in `get_user` - despite being typed as int, the raw f-string construction is unsafe

### Summary

This PR introduces two SQL injection vulnerabilities by replacing parameterized queries with f-string interpolation. Both functions should use parameterized queries with `?` placeholders.

**Recommended fix:**
```python
# Instead of:
sql = f"SELECT * FROM users WHERE id = {user_id}"

# Use:
sql = "SELECT * FROM users WHERE id = ?"
result = await db.execute(sql, (user_id,))
```

---
*Agreement score: 100% | Powered by [Aragora](https://github.com/an0mium/aragora) - AI Red Team*
