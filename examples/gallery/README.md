# Aragora Example Gallery

Real-world examples demonstrating Aragora's multi-agent debate capabilities.

## Examples

| Example | Description | Use Case |
|---------|-------------|----------|
| [code_review_debate.py](code_review_debate.py) | Multi-agent code review | Security vulnerabilities, best practices |
| [design_decision_debate.py](design_decision_debate.py) | Architecture decisions | Database choice, API design |
| [security_audit_debate.py](security_audit_debate.py) | Security assessment | Threat modeling, vulnerability analysis |
| [research_synthesis_debate.py](research_synthesis_debate.py) | Research synthesis | Literature review, evidence aggregation |
| [incident_postmortem_debate.py](incident_postmortem_debate.py) | Incident analysis | Root cause analysis, prevention |

## Requirements

Set at least one API key:
```bash
export ANTHROPIC_API_KEY=your_key
# or
export OPENAI_API_KEY=your_key
# or
export GEMINI_API_KEY=your_key
```

## Running Examples

```bash
# From project root
python examples/gallery/code_review_debate.py
python examples/gallery/design_decision_debate.py
python examples/gallery/security_audit_debate.py
python examples/gallery/research_synthesis_debate.py
python examples/gallery/incident_postmortem_debate.py
```

## Expected Output

Each example produces:
- Debate transcript with agent arguments
- Consensus result with confidence score
- Final synthesized answer
- Duration and rounds used
