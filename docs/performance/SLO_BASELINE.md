# Streaming SLO Baseline

This document defines the four streaming SLOs enforced by the baseline load test,
explains how to run the test, interpret results, and adjust thresholds.

## SLO Definitions

| # | SLO | Target | Unit | Comparison | Env Override |
|---|-----|--------|------|------------|--------------|
| 1 | First Byte Latency p95 | < 500 | ms | lte | `SLO_FIRST_BYTE_P95_MS` |
| 2 | Message Throughput (floor) | >= 10 | messages/sec/debate | gte | `SLO_MSG_THROUGHPUT_MIN` |
| 3 | Reconnection Success Rate | >= 99% | ratio | gte | `SLO_RECONNECT_RATE_MIN` |
| 4 | Debate Completion p99 | < 30 | seconds | lte | `SLO_COMPLETION_P99_S` |

### 1. First Byte Latency (p95 < 500ms)

Measures the time from debate start to the first streamed byte (token) reaching
the client. This is critical for perceived responsiveness -- users should see
activity within half a second of submitting a debate.

**How it is measured:** For each debate, the test records `time.monotonic()` at
debate start and at the first token emitted by any agent. The difference in
milliseconds is the first-byte latency. The p95 across all completed debates is
compared against the 500ms target.

### 2. Message Throughput (>= 10 messages/sec/debate)

Measures sustained message delivery rate per debate. This covers proposal tokens,
critique responses, votes, and system messages. A floor of 10 messages/sec ensures
debates progress at a reasonable pace without stalling.

**How it is measured:** Total messages emitted during a debate divided by the
debate duration in seconds. The 5th percentile (p5) across all debates is used
as the effective floor -- 95% of debates must meet or exceed 10 msg/sec.

### 3. Reconnection Success Rate (>= 99%)

Measures the reliability of WebSocket reconnections during debates. Debates may
experience transient disconnects; the system must reconnect successfully at least
99% of the time.

**How it is measured:** Each debate has a configurable probability of triggering
a reconnection attempt per round. The ratio of successful reconnections to total
attempts across all debates is compared against the 99% target.

### 4. End-to-End Debate Completion (p99 < 30s)

Measures total wall-clock time from debate start to final result. This is the
ultimate latency SLO -- even the slowest 1% of debates must complete within 30
seconds.

**How it is measured:** `time.monotonic()` at debate start and end. The p99
across all completed debates is compared against 30 seconds.

## Running the Baseline Test

### Quick Start

```bash
# Default: 60s duration, 50 concurrent debates
python scripts/load_test_baseline.py
```

### Common Options

```bash
# Custom duration and concurrency
python scripts/load_test_baseline.py --duration 120 --concurrency 100

# Save JSON report to file
python scripts/load_test_baseline.py --output baseline_report.json

# Strict mode: exit non-zero on any SLO failure (for CI)
python scripts/load_test_baseline.py --strict

# JSON-only output (no human-readable report)
python scripts/load_test_baseline.py --json-only

# Verbose logging
python scripts/load_test_baseline.py -v
```

### Running the Unit Tests

```bash
# All SLO baseline tests
pytest tests/performance/test_slo_baseline.py -v

# Quick subset
pytest tests/performance/test_slo_baseline.py -v -k "not Integration"
```

## Interpreting Results

### Human-Readable Report

The test prints a report like:

```
======================================================================
  ARAGORA STREAMING SLO BASELINE REPORT
======================================================================

  Configuration: 50 concurrent, 62.34s duration
  Debates: 2847/2850 completed (99.9% success)

  First Byte Latency (ms):
    p50:  18.42
    p95:  38.91
    p99:  52.17

  Message Throughput (messages/sec/debate):
    mean: 48.12
    min:  22.15
    p5:   28.03

  Reconnection:
    attempts:  412
    successes: 410
    rate:      99.51%

  Debate Completion (s):
    p50:  0.421
    p95:  0.687
    p99:  0.812

----------------------------------------------------------------------
  SLO Validation:
    [PASS] First Byte Latency p95: 38.91 ms (target: 500.0 ms, headroom: 461.09)
    [PASS] Message Throughput: 28.03 messages/sec/debate (target: 10.0, headroom: 18.03)
    [PASS] Reconnection Success Rate: 0.9951 ratio (target: 0.99, headroom: 0.0051)
    [PASS] End-to-End Debate Completion p99: 0.812 seconds (target: 30.0, headroom: 29.19)

  Overall: ALL PASSED
======================================================================
```

### JSON Report Structure

The JSON output has this structure:

```json
{
  "configuration": {
    "concurrency": 50,
    "target_duration_seconds": 60.0,
    "total_debates": 2850
  },
  "timing": {
    "actual_duration_seconds": 62.34,
    "started_at": "2026-02-24T...",
    "completed_at": "2026-02-24T..."
  },
  "results": {
    "completed_debates": 2847,
    "failed_debates": 3,
    "success_rate": 0.9989
  },
  "metrics": {
    "first_byte_latency_ms": { "p50": ..., "p95": ..., "p99": ... },
    "message_throughput_per_sec": { "mean": ..., "min": ..., "p5": ... },
    "reconnection": { "total_attempts": ..., "success_rate": ... },
    "debate_completion_s": { "p50": ..., "p95": ..., "p99": ... }
  },
  "slo_validation": {
    "first_byte_latency": { "passed": true, "actual": ..., "target": ... },
    "message_throughput": { "passed": true, ... },
    "reconnection_success_rate": { "passed": true, ... },
    "debate_completion": { "passed": true, ... }
  },
  "all_slos_passed": true,
  "errors": []
}
```

### Key Metrics to Watch

| Metric | What it tells you | Concern threshold |
|--------|-------------------|-------------------|
| `first_byte_latency_ms.p95` | User-perceived responsiveness | > 300ms (approaching SLO) |
| `message_throughput_per_sec.p5` | Floor of debate progress speed | < 15 msg/s (approaching SLO) |
| `reconnection.success_rate` | Connection reliability | < 0.995 (approaching SLO) |
| `debate_completion_s.p99` | Worst-case debate duration | > 20s (approaching SLO) |
| `results.success_rate` | Overall infrastructure health | < 0.99 |

## Adjusting Thresholds

### Via Environment Variables

Override any SLO target at runtime:

```bash
# Tighten first-byte latency to 250ms
SLO_FIRST_BYTE_P95_MS=250 python scripts/load_test_baseline.py

# Relax completion time for larger debates
SLO_COMPLETION_P99_S=60 python scripts/load_test_baseline.py --rounds 5

# Require higher throughput
SLO_MSG_THROUGHPUT_MIN=20 python scripts/load_test_baseline.py
```

### When to Adjust Thresholds

**Tighten** thresholds when:
- You are consistently passing with large margins (> 50% headroom)
- You have optimized the hot path and want to lock in gains
- SLA commitments to customers require stricter guarantees

**Relax** thresholds when:
- Running on slower CI hardware where mock timings are less predictable
- Testing with more agents or rounds (larger debates take longer)
- Validating a new feature that intentionally adds overhead

### Recommended Profiles

| Profile | Duration | Concurrency | Use Case |
|---------|----------|-------------|----------|
| Smoke | 10s | 5 | Quick local check |
| Default | 60s | 50 | Standard baseline |
| Stress | 300s | 200 | Pre-release validation |
| Soak | 3600s | 50 | Memory leak detection |

## Relationship to Other SLO Systems

This baseline test focuses on **streaming-specific SLOs** during concurrent
debates. Aragora has additional SLO systems:

| System | Location | Scope |
|--------|----------|-------|
| `aragora.observability.slo` | API-level SLOs | Availability, p99 latency, error rate |
| `aragora.observability.debate_slos` | Debate SLOs | TTFT, completion, consensus, dispatch |
| `tests/slo_config.py` | Test SLO config | Per-operation latency targets |
| `scripts/load_test_baseline.py` | **This baseline** | Streaming SLOs under concurrency |
| `tests/load/locustfile.py` | Locust suite | HTTP endpoint load testing |
| `tests/load/k6_api_test.js` | k6 suite | HTTP endpoint load testing |

## CI Integration

To gate builds on SLO compliance, add to your CI pipeline:

```yaml
- name: Run SLO baseline
  run: python scripts/load_test_baseline.py --strict --duration 30 --concurrency 20
```

The `--strict` flag exits with code 1 if any SLO fails, which will fail the CI step.
