# Golden Path 4: EU AI Act Compliance Check

Run an automated EU AI Act conformity assessment on an AI system decision. Classify risk levels, generate compliance reports, and map decision receipts to regulatory article requirements.

## What it demonstrates

- Risk classification under Article 6 + Annex III (8 high-risk categories)
- Conformity report generation from an Aragora decision receipt
- Article compliance mapping (Art. 9, 12, 13, 14, 15)
- Dissent tracking as part of the compliance record
- Export to JSON (machine-readable) and Markdown (human-readable)

## Run it

```bash
python examples/golden_paths/compliance_check/main.py
```

No API keys required. Uses Aragora's built-in compliance module.

## Expected output

```
================================================================
  Aragora Golden Path: EU AI Act Compliance Check
================================================================

--- Part 1: Risk Classification ---

  [HIGH RISK]   Resume Screening AI
              Annex III Category 4: Employment and worker management
              Keywords: recruitment, screening, hiring
              Obligations: 5 article requirements

  [LIMITED]     Customer Support Chatbot

  [MINIMAL]     Weather Forecast Model

--- Part 2: Conformity Assessment Report ---

  Receipt ID:     RCP-HR-2026-0312
  Risk Level:     HIGH
  Annex III:      Cat. 4 (Employment and worker management)
  Overall Status: PARTIAL
  ...

--- Part 3: Article Compliance Mapping ---

  Article      Requirement                                Status
  ------------ ------------------------------------------ ----------
  Art. 9       Risk management system                     PARTIAL
  Art. 12      Record-keeping and automatic logging       PASS
  Art. 13      Transparency and information to deployers  PARTIAL
  Art. 14      Human oversight measures                   PASS
  Art. 15      Accuracy, robustness, cybersecurity        PARTIAL
  ...
```

## Regulatory context

The EU AI Act (Regulation (EU) 2024/1689) takes effect August 2, 2026 for Annex III high-risk systems. Aragora maps decision receipts to article requirements automatically, covering:

| Article | Requirement |
|---------|-------------|
| Art. 6 | Risk classification (Annex III category matching) |
| Art. 9 | Risk management assessment |
| Art. 12 | Record-keeping and automatic logging |
| Art. 13 | Transparency (deployer instructions, known risks) |
| Art. 14 | Human oversight (override, stop, bias safeguards) |
| Art. 15 | Accuracy, robustness, and cybersecurity |
| Art. 26 | Deployer log retention (6-month minimum) |

## Key APIs used

| Import | Purpose |
|--------|---------|
| `aragora.compliance.eu_ai_act.RiskClassifier` | Classify AI use cases by risk level |
| `aragora.compliance.eu_ai_act.ConformityReportGenerator` | Generate conformity reports |
| `report.to_json()` / `report.to_markdown()` | Export audit-ready artifacts |

## Next steps

- Use `ComplianceArtifactGenerator` for dedicated Art. 12/13/14 artifact bundles
- See `examples/eu_ai_act_compliance.py` for the full compliance demo
- Integrate with `aragora.compliance.monitor` for continuous compliance monitoring
