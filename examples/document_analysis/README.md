# Document Analysis

Multi-agent document analysis that produces a structured JSON report with
key findings, risks, and recommendations.

## How it works

1. Loads a text document (txt, md, rst, csv, json, yaml)
2. Runs a 3-round multi-agent debate where agents analyze the document
3. Agents propose findings, critique each other's assessments, and synthesize
4. Outputs a JSON report with:
   - Key findings summary (consensus answer)
   - Confidence score and consensus strength
   - Individual agent proposals for transparency
   - Dissenting views when consensus is not reached

## Setup

```bash
# Install aragora
pip install aragora

# Set at least one API key
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
```

## Usage

```bash
# Analyze a document, output to stdout
python examples/document_analysis/main.py report.md

# Save report to file
python examples/document_analysis/main.py contract.txt --output analysis.json

# Analyze and pipe to jq for filtering
python examples/document_analysis/main.py memo.md | jq '.analysis.summary'
```

## Output format

```json
{
  "document": "report.md",
  "analyzed_at": "2026-02-23T12:00:00+00:00",
  "analysis": {
    "summary": "The document identifies three key areas...",
    "consensus_reached": true,
    "confidence": 0.85,
    "consensus_strength": "strong"
  },
  "debate_metadata": {
    "rounds_used": 3,
    "participants": ["claude", "gpt-4o", "gemini"],
    "winner": "claude",
    "convergence_status": "converged"
  },
  "agent_proposals": {
    "claude": "Key findings include...",
    "gpt-4o": "The document reveals..."
  }
}
```

## Customization

Edit `main.py` to adjust:
- `build_analysis_task()` -- change the analysis prompt or focus areas
- `MAX_DOCUMENT_CHARS` -- increase for larger documents (watch API costs)
- `DebateProtocol(rounds=3)` -- increase rounds for deeper analysis
- Add `consensus_threshold=0.8` to `Environment` for stricter agreement
