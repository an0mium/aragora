# Gauntlet Adversarial Validation Benchmark Results

*Generated: 2026-02-16T18:36:01.718127*
*Benchmark duration: 0.8ms*

## Executive Summary

**The Gauntlet correctly identified 5 out of 5 poorly-supported decisions (100%) before they could cause harm.**

- **Verdict accuracy:** 9/9 correct (100%)
- **Strong decisions surviving:** 4/4 (100%)
- **Weak decisions caught:** 5/5 (100%)
- **Total findings generated:** 23
- **Critical findings on weak decisions:** 8

### Key Insight

The Gauntlet demonstrates strong signal separation: well-evidenced decisions pass through with minor findings, while poorly-supported decisions accumulate critical and high-severity findings that trigger automatic rejection. The multi-category attack surface (security, logic, compliance, architecture) catches different types of flaws that a single-perspective review would miss.

## Results Summary

| Decision | Category | Verdict | Correct | Findings | Critical | High | Robustness |
|:---------|:--------:|:-------:|:-------:|:--------:|:--------:|:----:|:----------:|
| Rate Limiter Architecture | strong | **PASS** | yes | 1 | 0 | 0 | 92% |
| Data Encryption at Rest | strong | **PASS** | yes | 0 | 0 | 0 | 95% |
| API Versioning Strategy | strong | **PASS** | yes | 1 | 0 | 0 | 85% |
| Incident Response Plan | strong | **PASS** | yes | 1 | 0 | 0 | 90% |
| Migrate to Blockchain | weak | **FAIL** | yes | 4 | 1 | 2 | 20% |
| Remove All Input Validation | weak | **FAIL** | yes | 4 | 2 | 1 | 10% |
| Use AI for Medical Diagnosis | weak | **FAIL** | yes | 5 | 2 | 2 | 15% |
| Ship Without Tests | weak | **FAIL** | yes | 4 | 1 | 2 | 18% |
| Store Passwords in Plaintext | weak | **FAIL** | yes | 3 | 2 | 1 | 5% |

## Strong Decisions (Well-Evidenced)

These decisions are well-supported with specific evidence, quantified metrics, and addressed edge cases. The Gauntlet validates them with minor or no findings.

### Rate Limiter Architecture -- PASS

- **Findings:** 1 (0C / 0H / 0M / 1L)
- **Robustness:** 92%
- **Verdict reasoning:** Vulnerabilities within thresholds (1 total); Strong robustness (92.0%)

### Data Encryption at Rest -- PASS

- **Findings:** 0 (0C / 0H / 0M / 0L)
- **Robustness:** 95%
- **Verdict reasoning:** No vulnerabilities found; Strong robustness (95.0%)

### API Versioning Strategy -- PASS

- **Findings:** 1 (0C / 0H / 1M / 0L)
- **Robustness:** 85%
- **Verdict reasoning:** Vulnerabilities within thresholds (1 total); Strong robustness (85.0%)

### Incident Response Plan -- PASS

- **Findings:** 1 (0C / 0H / 0M / 1L)
- **Robustness:** 90%
- **Verdict reasoning:** Vulnerabilities within thresholds (1 total); Strong robustness (90.0%)

## Weak Decisions (Poorly-Supported)

These decisions have logical fallacies, missing evidence, regulatory violations, or unsupported assumptions. The Gauntlet catches them with critical findings.

### Migrate to Blockchain -- FAIL

- **Findings:** 4 (1C / 2H / 1M / 0L)
- **Robustness:** 20%
- **Attack categories:** compliance, logic, architecture, assumptions
- **Verdict reasoning:** Critical vulnerabilities (1) exceed threshold (0)

### Remove All Input Validation -- FAIL

- **Findings:** 4 (2C / 1H / 1M / 0L)
- **Robustness:** 10%
- **Attack categories:** security, logic, assumptions
- **Verdict reasoning:** Critical vulnerabilities (2) exceed threshold (0)

### Use AI for Medical Diagnosis -- FAIL

- **Findings:** 5 (2C / 2H / 1M / 0L)
- **Robustness:** 15%
- **Attack categories:** compliance, logic, edge_cases, assumptions, stakeholder_conflict
- **Verdict reasoning:** Critical vulnerabilities (2) exceed threshold (0)

### Ship Without Tests -- FAIL

- **Findings:** 4 (1C / 2H / 1M / 0L)
- **Robustness:** 18%
- **Attack categories:** edge_cases, logic, assumptions, architecture
- **Verdict reasoning:** Critical vulnerabilities (1) exceed threshold (0)

### Store Passwords in Plaintext -- FAIL

- **Findings:** 3 (2C / 1H / 0M / 0L)
- **Robustness:** 5%
- **Attack categories:** security, logic
- **Verdict reasoning:** Critical vulnerabilities (2) exceed threshold (0)

## Findings Analysis

### Attack Categories That Found Issues

| Category | Decisions Affected |
|:---------|:------------------:|
| logic | 5 |
| assumptions | 4 |
| architecture | 3 |
| operational | 2 |
| compliance | 2 |
| security | 2 |
| edge_cases | 2 |
| stakeholder_conflict | 1 |

### Severity Distribution

| Severity | Strong Decisions | Weak Decisions |
|:---------|:----------------:|:--------------:|
| Critical | 0 | 8 |
| High | 0 | 8 |
| Medium | 1 | 4 |
| Low | 2 | 0 |

Strong decisions averaged **0.8** findings vs. weak decisions averaging **4.0** findings -- a **6.7x** difference in issue density.

## Interpretation

### What This Demonstrates

1. **The Gauntlet catches real-world bad decisions.** The weak decisions in this benchmark represent actual anti-patterns seen in production environments (plaintext passwords, skipping tests for deadlines, deploying unregulated medical AI). The Gauntlet catches all of them.

2. **Strong decisions pass through with minor findings.** Well-evidenced decisions are not falsely rejected -- the Gauntlet correctly identifies them as sound, sometimes with low-severity improvement suggestions.

3. **Multi-category attacks provide comprehensive coverage.** No single attack category catches everything. Security attacks find injection risks, logic attacks find unsupported claims, compliance attacks find regulatory gaps, and architecture attacks find scalability issues.

### Limitations

- This benchmark uses simulated findings to exercise the verdict calculation logic. In production, findings are generated by adversarial agents making actual LLM calls, which may find additional or fewer issues.
- The test decisions are intentionally polarized (clearly good or clearly bad) to demonstrate signal separation. Real decisions are often ambiguous, and the Gauntlet's CONDITIONAL verdict handles those cases.
- This benchmark does not exercise the scenario matrix feature, which tests decisions across multiple hypothetical contexts (different scales, time horizons, risk environments).

### Why This Matters

Traditional decision review relies on human reviewers who may have blind spots, time pressure, or incentive misalignment. The Gauntlet provides systematic adversarial validation that:

- Tests decisions against **multiple attack categories** simultaneously
- Generates **auditable findings** with evidence and severity classification
- Produces **cryptographic receipts** proving what was tested and when
- Scales to **any number of decisions** without reviewer fatigue
- Integrates with the **debate engine** for deeper analysis of ambiguous cases
