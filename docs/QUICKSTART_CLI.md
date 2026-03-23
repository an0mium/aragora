# Quickstart CLI

`aragora quickstart` is the narrowest CLI-first path from a question to a saved debate artifact.
It does one short run, tells you whether that run was `live` or `demo`, and writes the result to disk.

## Prerequisites

- Python 3.11 or later
- Optional for live mode: at least one supported API key such as `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

Install Aragora:

```bash
pip install aragora
```

Verify the install:

```bash
aragora --version
```

## What Quickstart Does

When you run `aragora quickstart`, the command:

1. Loads `.env` from the current directory or its parent if present.
2. Uses `--question`, or prompts interactively for one.
3. Runs a short debate in `live` mode when supported API keys are detected.
4. Falls back to `demo` mode with local mock agents when no supported API keys are found.
5. Saves one result artifact to disk and prints the exact path.
6. Optionally opens an HTML view in the browser unless `--no-browser` is set.

The default saved artifact path is:

```text
.aragora/receipts/quickstart-<live|demo>-receipt.<format>
```

The default format is `json`. Use `--format md` or `--format html` to change it, or `--output` to choose the exact path.

## Demo Run

Use demo mode when you want an offline, no-key first run:

```bash
aragora quickstart --demo --no-browser
```

If you omit `--question` in demo mode, quickstart uses a built-in sample question.

Expected behavior:

- The terminal reports `Run mode: demo`
- The debate uses local mock agents
- A saved artifact is written to `.aragora/receipts/quickstart-demo-receipt.json` by default

## Live Run

Export a supported API key, then run quickstart:

```bash
export OPENAI_API_KEY=sk-...
aragora quickstart --no-browser
```

Or provide one key inline for a one-shot first run and save it into Aragora's secure key store:

```bash
aragora quickstart \
  --provider openai \
  --api-key sk-... \
  --save-key \
  --question "Should we ship this change?" \
  --no-browser
```

Or provide the question directly:

```bash
aragora quickstart --question "Should we rewrite this service in Go?" --no-browser
```

Expected behavior:

- The terminal reports `Run mode: live`
- Quickstart lists the detected live agent providers it will use
- A saved artifact is written to `.aragora/receipts/quickstart-live-receipt.json` by default
- The live artifact includes a structured receipt payload with `consensus_proof`, `dissenting_views`, and `artifact_hash`

Inspect or verify that saved receipt with the existing receipt tooling:

```bash
aragora receipt inspect .aragora/receipts/quickstart-live-receipt.json
aragora receipt verify .aragora/receipts/quickstart-live-receipt.json
```

If no supported API keys are detected and you did not pass `--demo`, quickstart says it is falling back to demo mode and the saved artifact will reflect `mode: "demo"`.

## Browser Behavior

- If the saved artifact is HTML, quickstart opens that saved file in the browser.
- If the saved artifact is JSON or Markdown, quickstart can still open a temporary HTML preview unless `--no-browser` is set.
- The saved artifact path printed in the terminal is the durable result path. A browser preview path, when shown, is separate.

## Common Flags

```bash
aragora quickstart --question "Should we adopt microservices?"
aragora quickstart --demo
aragora quickstart --format md --no-browser
aragora quickstart --output ./my-first-receipt.html
aragora quickstart --rounds 3
```

## Next Steps

After quickstart, use the broader CLI commands for fuller workflows:

```bash
aragora ask "Your question" --agents anthropic-api,openai-api
aragora decide "Your question"
```
