# Compliance & Audit Presets

Aragora ships built-in audit presets that bundle audit types, rules, and
consensus thresholds for common compliance workflows.

## Built-In Presets

Presets live in `aragora/audit/presets/`:

| Preset | File | Use Case |
|--------|------|----------|
| Legal Due Diligence | `legal_due_diligence.yaml` | M&A review, contract analysis |
| Financial Audit | `financial_audit.yaml` | Controls, accounting checks |
| Code Security | `code_security.yaml` | AppSec and secure code review |

## CLI Usage

```bash
# List presets
aragora audit presets

# Show a specific preset
aragora audit preset "Legal Due Diligence"
```

## API Usage

```bash
# List presets
curl -H "Authorization: Bearer $ARAGORA_API_TOKEN" \
  https://your-host/api/audit/presets
```

## Custom Presets

Add YAML files under any of these directories (auto-discovered):

- `aragora/audit/presets/`
- `~/.aragora/presets/`
- `/etc/aragora/presets/`

Then re-run `aragora audit presets` to verify the new configuration.

## Related Docs

- [Gauntlet](GAUNTLET.md)
- [Evidence system](EVIDENCE.md)
- [Compliance](COMPLIANCE.md)
