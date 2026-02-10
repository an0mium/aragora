# Research Integration Metrics Dashboard Specification

This document specifies the Grafana dashboard for monitoring research integration health.

## Overview

The dashboard provides real-time visibility into:
- Feature enablement status across integration levels
- Performance metrics for each research component
- Error rates and alert conditions
- Historical trends and anomalies

## Dashboard Panels

### 1. System Overview Row

#### 1.1 Integration Level Status
- **Type**: Stat panel
- **Query**: `aragora_research_integration_level`
- **Display**: Current integration level (MINIMAL/STANDARD/FULL/CUSTOM)
- **Thresholds**: Info only

#### 1.2 Enabled Features Count
- **Type**: Gauge
- **Query**: `count(aragora_feature_enabled{enabled="true"})`
- **Display**: Number of enabled features
- **Thresholds**:
  - Green: 6+
  - Yellow: 3-5
  - Red: <3

#### 1.3 Total Debates (24h)
- **Type**: Stat panel
- **Query**: `sum(increase(aragora_debates_total[24h]))`
- **Display**: Total debates in last 24 hours

#### 1.4 Average Debate Latency
- **Type**: Stat panel with sparkline
- **Query**: `avg(aragora_debate_duration_seconds)`
- **Display**: Average debate completion time
- **Thresholds**:
  - Green: <30s
  - Yellow: 30-60s
  - Red: >60s

---

### 2. Adaptive Stopping (Phase 1) Row

#### 2.1 Early Stop Rate
- **Type**: Time series
- **Query**: `rate(aragora_adaptive_stopping_early_stops_total[5m]) / rate(aragora_debates_total[5m])`
- **Display**: Percentage of debates stopped early
- **Target**: 20-40%

#### 2.2 False Early Stop Rate
- **Type**: Gauge
- **Query**: `aragora_adaptive_stopping_false_stops_total / aragora_adaptive_stopping_early_stops_total`
- **Display**: False positive rate
- **Thresholds**:
  - Green: <5%
  - Yellow: 5-10%
  - Red: >10%

#### 2.3 Stability Score Distribution
- **Type**: Histogram
- **Query**: `histogram_quantile(0.5, aragora_stability_score_bucket)`
- **Display**: Distribution of stability scores

#### 2.4 Compute Savings
- **Type**: Stat panel
- **Query**: `1 - (sum(aragora_actual_rounds_total) / sum(aragora_max_rounds_total))`
- **Display**: Percentage of compute saved via early stopping
- **Target**: ≥20%

---

### 3. MUSE Calibration Row

#### 3.1 MUSE Divergence Over Time
- **Type**: Time series
- **Query**: `avg(aragora_muse_divergence)`
- **Display**: Average JSD divergence across debates
- **Alert**: >0.5 sustained for 10m

#### 3.2 Calibration Error
- **Type**: Gauge
- **Query**: `aragora_muse_calibration_error`
- **Display**: Current calibration error
- **Thresholds**:
  - Green: <10%
  - Yellow: 10-20%
  - Red: >20%

#### 3.3 Best Subset Size Distribution
- **Type**: Bar chart
- **Query**: `sum by (subset_size) (aragora_muse_subset_selections_total)`
- **Display**: Distribution of best subset sizes

---

### 4. LaRA Routing Row

#### 4.1 Routing Mode Distribution
- **Type**: Pie chart
- **Query**: `sum by (mode) (aragora_lara_routing_decisions_total)`
- **Display**: Breakdown of routing decisions (RAG/RLM/LONG_CONTEXT/GRAPH/HYBRID)

#### 4.2 Routing Decision Time
- **Type**: Time series
- **Query**: `histogram_quantile(0.95, aragora_lara_routing_latency_seconds_bucket)`
- **Display**: p95 routing decision latency
- **Threshold**: <100ms

#### 4.3 Retrieval Relevance by Mode
- **Type**: Bar chart
- **Query**: `avg by (mode) (aragora_retrieval_relevance_score)`
- **Display**: Average relevance score per routing mode

---

### 5. ASCoT Fragility Row

#### 5.1 Fragility Score by Round
- **Type**: Heatmap
- **Query**: `aragora_ascot_fragility_score`
- **Display**: Fragility scores across debate rounds

#### 5.2 Critical Fragility Events
- **Type**: Stat panel
- **Query**: `sum(increase(aragora_ascot_critical_events_total[24h]))`
- **Display**: Critical fragility events in 24h
- **Alert**: >10 per hour

#### 5.3 Verification Intensity Distribution
- **Type**: Bar chart
- **Query**: `sum by (intensity) (aragora_ascot_verification_intensity_total)`
- **Display**: Distribution of verification intensity (LOW/MEDIUM/HIGH/CRITICAL)

---

### 6. ThinkPRM Verification Row

#### 6.1 Step Verification Results
- **Type**: Pie chart
- **Query**: `sum by (verdict) (aragora_think_prm_verifications_total)`
- **Display**: Distribution of verdicts (CORRECT/INCORRECT/NEEDS_REVISION)

#### 6.2 Verification Confidence
- **Type**: Time series
- **Query**: `avg(aragora_think_prm_confidence)`
- **Display**: Average verification confidence over time

#### 6.3 Critical Errors Detected
- **Type**: Time series
- **Query**: `sum(rate(aragora_think_prm_critical_errors_total[5m]))`
- **Display**: Rate of critical errors in late-stage rounds
- **Alert**: >5 per minute sustained

#### 6.4 Verification Latency
- **Type**: Stat panel
- **Query**: `histogram_quantile(0.95, aragora_think_prm_latency_seconds_bucket)`
- **Display**: p95 verification latency

---

### 7. GraphRAG Retrieval Row

#### 7.1 Hybrid Retrieval Usage
- **Type**: Time series
- **Query**: `rate(aragora_graph_rag_queries_total[5m])`
- **Display**: GraphRAG query rate

#### 7.2 Graph Expansion Depth
- **Type**: Histogram
- **Query**: `aragora_graph_rag_expansion_depth`
- **Display**: Distribution of graph expansion depths

#### 7.3 Community Detection
- **Type**: Stat panel
- **Query**: `avg(aragora_graph_rag_communities_found)`
- **Display**: Average communities found per query

---

### 8. ClaimCheck Verification Row

#### 8.1 Claim Decomposition Stats
- **Type**: Bar chart
- **Query**: `avg(aragora_claim_check_atomic_claims_per_request)`
- **Display**: Average atomic claims per verification request

#### 8.2 Verification Status Distribution
- **Type**: Pie chart
- **Query**: `sum by (status) (aragora_claim_check_verifications_total)`
- **Display**: VERIFIED/PARTIALLY_VERIFIED/UNVERIFIED/CONTRADICTED

#### 8.3 Evidence Coverage
- **Type**: Gauge
- **Query**: `avg(aragora_claim_check_evidence_coverage)`
- **Display**: Average evidence coverage percentage
- **Thresholds**:
  - Green: >80%
  - Yellow: 60-80%
  - Red: <60%

---

### 9. A-HMAD Team Selection Row

#### 9.1 Role Distribution
- **Type**: Pie chart
- **Query**: `sum by (role) (aragora_ahmad_role_assignments_total)`
- **Display**: Distribution of assigned roles

#### 9.2 Diversity Score
- **Type**: Time series
- **Query**: `avg(aragora_ahmad_diversity_score)`
- **Display**: Team diversity score over time
- **Threshold**: ≥0.6 (alert if below)

#### 9.3 Coverage Score
- **Type**: Gauge
- **Query**: `avg(aragora_ahmad_coverage_score)`
- **Display**: Role coverage score
- **Thresholds**:
  - Green: >0.9
  - Yellow: 0.7-0.9
  - Red: <0.7

---

### 10. SICA Self-Improvement Row

#### 10.1 Improvement Cycles
- **Type**: Stat panel
- **Query**: `sum(increase(aragora_sica_cycles_total[24h]))`
- **Display**: Improvement cycles in 24h

#### 10.2 Patch Success Rate
- **Type**: Gauge
- **Query**: `sum(aragora_sica_patches_successful_total) / sum(aragora_sica_patches_applied_total)`
- **Display**: Percentage of successful patches
- **Thresholds**:
  - Green: >80%
  - Yellow: 50-80%
  - Red: <50%

#### 10.3 Opportunities by Type
- **Type**: Bar chart
- **Query**: `sum by (type) (aragora_sica_opportunities_total)`
- **Display**: Distribution by improvement type

#### 10.4 Rollbacks
- **Type**: Stat panel
- **Query**: `sum(increase(aragora_sica_rollbacks_total[24h]))`
- **Display**: Rollbacks in 24h
- **Alert**: >5 per cycle

---

## Alert Rules

### Critical Alerts (P1)

| Alert Name | Condition | Duration | Action |
|------------|-----------|----------|--------|
| `HighFalseEarlyStopRate` | `aragora_adaptive_stopping_false_stops_rate > 0.10` | 10m | Check stability thresholds |
| `ThinkPRMCriticalErrors` | `rate(aragora_think_prm_critical_errors_total[5m]) > 5` | 5m | Review debate quality |
| `LowDiversityScore` | `aragora_ahmad_diversity_score < 0.5` | 15m | Check team composition |
| `SICAHighRollbackRate` | `rate(aragora_sica_rollbacks_total[1h]) > 5` | 15m | Review patch quality |

### Warning Alerts (P2)

| Alert Name | Condition | Duration | Action |
|------------|-----------|----------|--------|
| `ElevatedMUSEDivergence` | `aragora_muse_divergence > 0.4` | 20m | Check agent calibration |
| `SlowLaRARouting` | `aragora_lara_routing_latency_p95 > 0.2` | 10m | Check routing logic |
| `LowEvidenceCoverage` | `aragora_claim_check_evidence_coverage < 0.6` | 30m | Review evidence sources |
| `ReducedComputeSavings` | `aragora_compute_savings < 0.15` | 1h | Review stopping criteria |

### Info Alerts (P3)

| Alert Name | Condition | Duration | Action |
|------------|-----------|----------|--------|
| `IntegrationLevelChanged` | `changes(aragora_research_integration_level[1h]) > 0` | 0m | Log configuration change |
| `NewFeatureEnabled` | `changes(aragora_feature_enabled[1h]) > 0` | 0m | Log feature change |

---

## Metrics Reference

### Counter Metrics

```
aragora_debates_total
aragora_adaptive_stopping_early_stops_total
aragora_adaptive_stopping_false_stops_total
aragora_lara_routing_decisions_total{mode}
aragora_think_prm_verifications_total{verdict}
aragora_think_prm_critical_errors_total
aragora_graph_rag_queries_total
aragora_claim_check_verifications_total{status}
aragora_ahmad_role_assignments_total{role}
aragora_sica_cycles_total
aragora_sica_patches_applied_total
aragora_sica_patches_successful_total
aragora_sica_rollbacks_total
aragora_sica_opportunities_total{type}
```

### Gauge Metrics

```
aragora_research_integration_level
aragora_feature_enabled{feature}
aragora_muse_divergence
aragora_muse_calibration_error
aragora_ascot_fragility_score
aragora_think_prm_confidence
aragora_ahmad_diversity_score
aragora_ahmad_coverage_score
aragora_claim_check_evidence_coverage
```

### Histogram Metrics

```
aragora_debate_duration_seconds_bucket
aragora_stability_score_bucket
aragora_lara_routing_latency_seconds_bucket
aragora_think_prm_latency_seconds_bucket
aragora_graph_rag_expansion_depth_bucket
```

---

## Dashboard JSON

The complete Grafana dashboard JSON can be generated using:

```bash
python scripts/generate_dashboard.py --output grafana/dashboards/research_integration.json
```

This will create a dashboard with all panels configured according to this specification.

---

## Runbook: Common Alerts

### HighFalseEarlyStopRate

**Symptom**: Debates stopping early but consensus was incorrect

**Investigation**:
1. Check recent stability scores: `aragora_stability_score_bucket`
2. Review MUSE divergence at stop time
3. Examine ASCoT fragility gates

**Resolution**:
1. Increase `stability_threshold` from 0.85 to 0.90
2. Lower `muse_disagreement_gate` from 0.4 to 0.3
3. Temporarily disable adaptive stopping if severe

### ElevatedMUSEDivergence

**Symptom**: High disagreement between agent subsets

**Investigation**:
1. Check individual agent calibration scores
2. Review debate topics for controversial subjects
3. Examine agent voting patterns

**Resolution**:
1. Recalibrate poorly performing agents
2. Adjust `muse_weight` downward
3. Review agent selection criteria

### LowDiversityScore

**Symptom**: Team compositions too homogeneous

**Investigation**:
1. Check role assignment distribution
2. Review available agent pool
3. Examine topic analysis results

**Resolution**:
1. Increase `diversity_penalty` in A-HMAD config
2. Ensure diverse agent pool is available
3. Review `min_diversity_score` threshold

### SICAHighRollbackRate

**Symptom**: Many patches being rolled back after tests fail

**Investigation**:
1. Review patch types being generated
2. Check validation pipeline results
3. Examine rollback patterns by file

**Resolution**:
1. Increase `min_confidence` threshold
2. Enable more validation steps
3. Review `auto_approve_threshold`
4. Consider requiring human approval temporarily
