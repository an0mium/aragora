# Gauntlet - Adversarial Stress-Testing CLI

Gauntlet is a standalone adversarial validation engine that stress-tests high-stakes decisions through multi-agent debate. It actively attacks specifications, architectures, policies, and code to find weaknesses before they become problems.

## Installation

```bash
# Full installation (recommended)
pip install aragora

# Or install from source
pip install -e .
```

## Quick Start

```bash
# Basic stress-test
gauntlet spec.md --input-type spec

# Thorough analysis with HTML receipt
gauntlet architecture.md --profile thorough --output receipt.html

# GDPR compliance audit
gauntlet privacy-policy.md --persona gdpr --format json

# Code review with formal verification
gauntlet main.py --profile code --verify
```

## Usage

```
gauntlet <input-file> [options]

Options:
  --input-type, -t    Type: spec, architecture, policy, code, strategy, contract
  --profile, -p       Profile: quick, default, thorough, code, policy, gdpr, hipaa, ai_act, security, sox
  --persona           Regulatory persona for compliance testing
  --agents, -a        Comma-separated agents (default: anthropic-api,openai-api)
  --output, -o        Output path for Decision Receipt
  --format, -f        Format: html, json, md (default: html)
  --rounds, -r        Deep audit rounds
  --timeout           Max duration in seconds
  --verify            Enable formal verification (Z3/Lean)
  --no-redteam        Disable red-team attacks
  --no-probing        Disable capability probing
  --no-audit          Disable deep audit
  --quiet, -q         Suppress progress output
  --verbose, -v       Enable verbose output
```

## Profiles

| Profile | Duration | Use Case |
|---------|----------|----------|
| `quick` | ~2 min | Fast sanity check |
| `default` | ~5 min | Standard validation |
| `thorough` | ~15 min | Comprehensive analysis |
| `code` | ~10 min | Code-specific review |
| `policy` | ~10 min | Policy document review |
| `gdpr` | ~10 min | GDPR compliance |
| `hipaa` | ~10 min | HIPAA compliance |
| `ai_act` | ~10 min | EU AI Act compliance |
| `security` | ~10 min | Security-focused |
| `sox` | ~10 min | SOX compliance |

## Regulatory Personas

Available compliance personas:
- `gdpr` - EU General Data Protection Regulation
- `hipaa` - US Health Insurance Portability and Accountability Act
- `ai_act` - EU Artificial Intelligence Act
- `security` - General security best practices
- `soc2` - SOC 2 Type II compliance
- `sox` - Sarbanes-Oxley Act
- `pci_dss` - Payment Card Industry Data Security Standard
- `nist_csf` - NIST Cybersecurity Framework

## Exit Codes

- `0` - PASS: Input passed all stress tests
- `1` - REJECTED: Input failed critical tests
- `2` - NEEDS REVIEW: Input requires human review

## Decision Receipts

Gauntlet produces audit-ready Decision Receipts containing:
- Input artifact hash (SHA-256)
- Verdict with confidence score
- All findings with severity levels
- Risk heatmap visualization
- Cryptographic checksums for integrity

## Environment Variables

Set at least one API key:
```bash
export ANTHROPIC_API_KEY=your-key
export OPENAI_API_KEY=your-key
```

Optional additional providers:
```bash
export GEMINI_API_KEY=your-key
export XAI_API_KEY=your-key
export MISTRAL_API_KEY=your-key
export OPENROUTER_API_KEY=your-key  # Fallback provider
```

## Examples

### Validate an API Specification
```bash
gauntlet api-spec.yaml \
  --input-type spec \
  --profile thorough \
  --output api-audit.html
```

### Security Review with Multiple Agents
```bash
gauntlet auth-system.py \
  --input-type code \
  --profile security \
  --agents anthropic-api,openai-api,gemini \
  --verify
```

### GDPR Compliance Audit
```bash
gauntlet privacy-policy.md \
  --persona gdpr \
  --format json \
  --output gdpr-report.json
```

### CI/CD Integration
```bash
# In your CI pipeline
gauntlet ./specs/api.md --profile quick --quiet
if [ $? -ne 0 ]; then
  echo "Gauntlet detected issues"
  exit 1
fi
```

## Programmatic Usage

```python
from aragora.gauntlet import GauntletRunner, GauntletConfig, AttackCategory

config = GauntletConfig(
    attack_categories=[AttackCategory.SECURITY, AttackCategory.COMPLIANCE],
    agents=["anthropic-api", "openai-api"],
)

runner = GauntletRunner(config)
result = await runner.run(spec_content)

print(f"Verdict: {result.verdict}")
print(f"Findings: {len(result.vulnerabilities)}")

# Generate receipt
receipt = result.to_receipt()
receipt.save_html("audit-report.html")
```

## License

MIT License - Part of the Aragora Multi-Agent Framework
