# Breaking Change Documentation Template

Use this template when documenting breaking changes in Aragora. Copy the relevant sections to [BREAKING_CHANGES.md](../reference/BREAKING_CHANGES.md) and customize for your specific change.

---

## Template: Version Entry

```markdown
### vX.Y.Z (YYYY-MM-DD)

#### Category Name (e.g., API Changes, SDK Changes, Configuration Changes)

| Category | Change | Migration |
|----------|--------|-----------|
| [Type] | [Description of what changed] | [Brief migration instruction] |

**Migration Steps:**

1. [Step 1]
2. [Step 2]
3. [Step 3]

**Example Migration:**

\`\`\`python
# Before (vX.Y-1)
[old code example]

# After (vX.Y)
[new code example]
\`\`\`

See [MIGRATION_GUIDE.md](../guides/MIGRATION_GUIDE.md) for complete migration guide.
```

---

## Template: Individual Breaking Change

Use this structure for each breaking change entry:

### Change Title

**Version:** vX.Y.Z
**Category:** API / SDK / Configuration / Internal
**Severity:** High / Medium / Low

#### Description

[Clear description of what changed and why]

#### Impact

[Who is affected and how]

- Affects users who use [feature/endpoint/method]
- Does not affect users who [condition]

#### Migration Steps

1. **Identify affected code**
   ```bash
   grep -r "old_pattern" src/
   ```

2. **Update imports/calls**
   ```python
   # Before
   from aragora.old_module import OldClass

   # After
   from aragora.new_module import NewClass
   ```

3. **Update configuration** (if applicable)
   ```yaml
   # Before
   old_config: value

   # After
   new_config: value
   ```

4. **Test the migration**
   ```bash
   pytest tests/ -v
   ```

#### Before/After Examples

**Python:**
```python
# Before (deprecated)
result = client.old_method(arg1, arg2)

# After (recommended)
result = client.new_method(arg1, arg2, new_param=True)
```

**TypeScript:**
```typescript
// Before (deprecated)
const result = await client.oldMethod(arg1, arg2);

// After (recommended)
const result = await client.newMethod(arg1, arg2, { newParam: true });
```

**HTTP API:**
```bash
# Before (deprecated)
curl -X POST https://api.aragora.io/api/v1/old-endpoint

# After (recommended)
curl -X POST https://api.aragora.io/api/v2/new-endpoint
```

#### Deprecation Timeline

| Date | Milestone |
|------|-----------|
| vX.Y.0 | Change introduced, old behavior deprecated |
| vX.Y+1.0 | Deprecation warnings enabled |
| vX.Y+2.0 | Old behavior removed |

#### Common Issues

**Issue:** [Description of common problem]
**Solution:** [How to fix it]

**Issue:** [Another common problem]
**Solution:** [How to fix it]

#### Related Documentation

- [Link to migration guide]
- [Link to API documentation]
- [Link to SDK documentation]

---

## Template: Upcoming Breaking Change Notice

```markdown
### Scheduled for vX.Y.Z

| Item | Type | Current | Replacement | Deprecated Since |
|------|------|---------|-------------|------------------|
| `old_name` | Module/Config/Endpoint | `old usage` | `new usage` | vA.B.C |
```

---

## Template: SDK Breaking Change

```markdown
### SDK Name vX.Y.Z

| Change | Before | After |
|--------|--------|-------|
| [Change description] | `old_code` | `new_code` |

**Migration Example:**

\`\`\`language
# Before (vX.Y-1)
[old code]

# After (vX.Y)
[new code]
\`\`\`
```

---

## Checklist for Breaking Changes

Before merging a breaking change, ensure:

- [ ] Entry added to [BREAKING_CHANGES.md](../reference/BREAKING_CHANGES.md)
- [ ] Entry added to [CHANGELOG.md](../../CHANGELOG.md) under "Breaking Changes" or "Deprecated"
- [ ] Migration guide created (if change is complex)
- [ ] Deprecation warnings added to affected code
- [ ] SDK documentation updated (Python and TypeScript)
- [ ] Tests added for:
  - [ ] Deprecation warnings are emitted
  - [ ] Backward compatibility during grace period
  - [ ] New replacement functionality works
- [ ] API documentation updated (if API change)
- [ ] OpenAPI spec regenerated (if API change)

---

## Categories Reference

Use these standard categories for consistency:

| Category | Description | Examples |
|----------|-------------|----------|
| **API Changes** | REST API endpoint changes | URL changes, request/response format changes |
| **SDK Changes** | Client library changes | Method renames, parameter changes |
| **Configuration Changes** | Environment variables, config files | Variable renames, new required config |
| **Database Changes** | Schema migrations | New required columns, renamed tables |
| **Authentication Changes** | Auth flow changes | New auth requirements, token format changes |
| **Behavioral Changes** | Functional changes | Default value changes, algorithm changes |

---

## Severity Levels

| Level | Definition | Example |
|-------|------------|---------|
| **High** | Requires immediate code changes, will cause errors | Removed endpoint, renamed required field |
| **Medium** | May cause issues, deprecation warnings present | Deprecated method, changed default value |
| **Low** | Minimal impact, mostly cosmetic | Renamed optional parameter with alias support |

---

*Template version: 1.0.0*
*Last updated: 2026-01-31*
