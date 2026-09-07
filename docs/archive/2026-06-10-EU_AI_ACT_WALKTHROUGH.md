# EU AI Act End-to-End Walkthrough (June 2026)

> **Dated proof artifact** — every output below was produced by running the
> shipped CLI on 2026-06-10 against a *real* DecisionReceipt from Aragora's own
> autonomous development loop (not synthetic demo data). The EU AI Act's
> high-risk obligations begin enforcement **August 2, 2026**. This walkthrough
> is the buyer-facing path from "we use AI for decisions" to "here is our
> audit-ready conformity evidence", in three commands.
>
> Companion reference: [EU_AI_ACT_GUIDE.md](../compliance/EU_AI_ACT_GUIDE.md) (article
> mappings and artifact schemas). Raw outputs for this walkthrough are
> operator-held under `.aragora/run-20260610/euaiact/`.

## Scenario

Your organization uses AI to support consequential decisions — anything from
"should this software change ship?" to "which job applicants advance?". Under
the EU AI Act you must (1) classify each use case's risk level, (2) for
high-risk systems, produce Article 9/12/13/14/15 evidence, and (3) keep
audit-ready records. Aragora generates all three from the decision receipts it
already produces.

## Step 1 — Classify the use case (`aragora compliance classify`)

Decision-support over software changes (Aragora's own loop):

```text
$ aragora compliance classify "Adversarial multi-model review and receipt-backed
  settlement of software changes for a regulated enterprise (audit-trail
  decision support)"

Risk Level: MINIMAL
Rationale:  Use case does not match high-risk or limited-risk categories.
            Minimal obligations apply.
Obligations:
  - Voluntary adoption of codes of conduct encouraged (Article 95).
```

The same command on a hiring use case lands in **Annex III**:

```text
$ aragora compliance classify "AI system that screens and ranks job applicants
  and recommends hiring decisions to recruiters"

Risk Level: HIGH
Rationale:  Use case falls under Annex III category 4: Employment and worker
            management. Recruitment, CV screening, performance evaluation,
            task allocation, termination.
Applicable Articles: 6, 9, 13, 14, 15
```

Classification is keyword/Annex-III–driven and deterministic — the rationale
and article list are printed, not hidden.

## Step 2 — Generate the Article 9–15 bundle from a real receipt

Every Aragora debate produces a cryptographic `DecisionReceipt`. We fed the
generator a receipt from an actual autonomous merge-gate decision made earlier
the same day (receipt `debate-55f975fb…`, verified `VALID (3/3 checks
passed)` by `aragora receipt verify`):

```text
$ aragora compliance eu-ai-act generate <receipt.json> \
    --output ./euaiact-bundle \
    --provider-name "Synaptent (Aragora)" \
    --system-name "Aragora Decision Integrity Platform" \
    --system-version 2.8.0 --format all

Article Compliance Summary:
  Article 9    Risk identification and analysis                [PASS]
  Article 12   Automatic logging of events with traceability   [PARTIAL]
  Article 13   Agent identification, arguments, dissent        [PASS]
  Article 14   Human oversight, ability to override            [PARTIAL]
  Article 15   Accuracy, robustness, cybersecurity             [PASS]

Recommendations:
  1. Ensure all decision events are logged in the provenance chain.
  2. Integrate human-in-the-loop approval before critical decisions are finalized.
```

The bundle directory contains one JSON evidence file per article plus a
human-readable `conformity_report.md` with a report ID, the source receipt ID,
generation timestamp, and an integrity hash:

```text
article_9_risk_management.json    article_12_record_keeping.json
article_13_transparency.json      article_14_human_oversight.json
article_15_accuracy_robustness.json
compliance_bundle.json  conformity_report.{md,json}
```

**Honesty note:** the PARTIAL grades are real, not staged. The input receipt
came from a single-round, single-counted-agent review with no human-settlement
signal attached — exactly the conditions Articles 12 and 14 flag. A receipt
from a full multi-round debate with the `aragora/human-settlement` status
attached grades higher. The generator tells you the truth about your evidence
instead of laundering it.

## Step 3 — Check content against framework controls

The framework checker runs offline across eight control sets:

```text
$ aragora compliance report ./euaiact-bundle/conformity_report.md

  Status: COMPLIANT
  Score:  100%
  Frameworks checked: hipaa, gdpr, sox, owasp, pci_dss, fda_21_cfr,
                      iso_27001, fedramp
  Issues found: 0
```

## What this gives a compliance team

- **Per-decision evidence**: every consequential decision can carry an
  article-mapped bundle bound to a SHA-256–verifiable receipt.
- **A truthful gap report**: PARTIAL grades + concrete recommendations are the
  remediation backlog, generated rather than hand-assembled.
- **Repeatability**: all three steps are offline CLI commands suitable for CI.

## Known gaps (recorded honestly, 2026-06-10)

1. `aragora compliance report` with no argument exits `Error: No content
   provided.` — correct behavior (it reads stdin), but the error does not
   mention the expected usage; a first-run user may read it as a crash.
2. Article 12/14 grading depends on receipt richness; receipts produced by
   `--rounds 1` reviews grade PARTIAL by design. Teams wanting PASS across the
   board should run multi-round debates and record human settlement.
3. Risk classification is keyword-driven; borderline use cases should be
   phrased concretely (the Annex III rationale is printed for verification).

---

*Generated during autonomous run run-20260610; source outputs preserved under
the run directory. Commands verified against `aragora 2.8.0`.*
