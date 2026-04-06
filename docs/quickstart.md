# Quickstart

This compatibility path now tracks Aragora's CLI-first onboarding.
The canonical current guide lives at [QUICKSTART_CLI.md](QUICKSTART_CLI.md).

Get from zero to a saved decision receipt in under a minute.

---

## 1. Install

```bash
pip install aragora
aragora --version
```

## 2. Zero-Key Demo

No API keys needed:

```bash
aragora quickstart --demo --no-browser
```

Expected behavior:

- The terminal reports `Run mode: demo`
- The debate uses local mock agents
- A saved artifact is written to `.aragora/receipts/quickstart-demo-receipt.json`

## 3. Live Run

Set at least one supported API key:

```bash
export OPENAI_API_KEY="sk-..."
```

Then run quickstart with a real question:

```bash
aragora quickstart \
  --question "Should we rewrite this service in Go?" \
  --no-browser
```

Expected behavior:

- The terminal reports `Run mode: live`
- Quickstart lists the detected providers it will use
- A saved artifact is written to `.aragora/receipts/quickstart-live-receipt.json`

## 4. Inspect The Receipt

Quickstart writes one durable artifact. Inspect or verify it with the receipt CLI:

```bash
aragora receipt inspect .aragora/receipts/quickstart-live-receipt.json
aragora receipt verify .aragora/receipts/quickstart-live-receipt.json
```

If you ran demo mode, swap in `quickstart-demo-receipt.json`.

## 5. Useful Flags

```bash
aragora quickstart --demo
aragora quickstart --format md --no-browser
aragora quickstart --output ./my-first-receipt.html
aragora quickstart --rounds 3
aragora quickstart --provider openai --api-key sk-... --save-key
```

## Next Steps

| Guide | What you'll learn |
|-------|-------------------|
| [QUICKSTART_CLI.md](QUICKSTART_CLI.md) | Full quickstart behavior and flag details |
| [QUICKSTART_DEVELOPER.md](QUICKSTART_DEVELOPER.md) | Local development workflow |
| [START_HERE.md](START_HERE.md) | Product and architecture overview |
