# Belief Network Benchmark Results

*Generated: 2026-02-16T18:37:40.666634*
*Benchmark duration: 2846.4ms*

## Executive Summary

**The Belief Network identified 5 crux claims where changing one belief would flip the entire decision.** Out of 12 claims connected by 17 relationships, the network pinpointed the specific beliefs that are most load-bearing for the debate outcome.

- **Claims analyzed:** 12
- **Relationships mapped:** 17
- **Crux claims identified:** 5
- **Consensus probability:** 58%
- **Contested claims:** 4
- **Convergence barrier:** 0.52
- **Propagation:** converged in 25 iterations

### Key Insight

The Belief Network transforms a debate from an unstructured argument into a probabilistic graph where each claim has a quantified probability of being true. By running belief propagation, the network identifies which claims are *load-bearing* (high centrality) and which are *contested* (high disagreement between agents). The intersection of these -- crux claims -- tells facilitators exactly where to focus to break through deadlocks.

## Methodology

- **Debate topic:** Cloud migration strategy (migrate vs. stay on-premises)
- **Agents:** 5 agents with distinct roles (cloud_architect, security_lead, engineering_manager, cfo, cto)
- **Claims:** 12 structured claims with evidence and confidence scores
- **Relationships:** 17 typed relationships (SUPPORTS, CONTRADICTS, DEPENDS_ON)
- **Propagation:** Loopy belief propagation with damping factor 0.5
- **Crux detection:** Weighted composite of influence, disagreement, uncertainty, and centrality scores

## Network Statistics

| Metric | Value |
|:-------|------:|
| Total claims | 12 |
| Total relationships | 17 |
| Graph density | 0.129 |
| Propagation converged | Yes |
| Propagation iterations | 25 |
| Max belief change | 0.000996 |
| Average uncertainty | 0.487 |
| Convergence barrier | 0.518 |

## Crux Claims Identified

These are the claims where resolving the disagreement would have the largest impact on the overall debate outcome. A high crux score means the claim is simultaneously influential, contested, uncertain, and central to the argument graph.

| Rank | Claim | Author | Crux Score | Influence | Disagreement | Uncertainty | Centrality | Affected |
|:----:|:------|:------:|:----------:|:---------:|:------------:|:-----------:|:----------:|:--------:|
| 1 | Our engineering team lacks cloud-native skills; migration will require 6-month r... | engineering_manager | **0.504** | 0.559 | 0.712 | 0.550 | 0.064 | 1 |
| 2 | CI/CD pipelines in cloud will reduce deployment time from 2 weeks to 2 hours, en... | cloud_architect | **0.503** | 0.683 | 0.616 | 0.526 | 0.038 | 0 |
| 3 | Containerize all workloads before migration to ensure portability and reduce clo... | cto | **0.495** | 0.706 | 0.652 | 0.300 | 0.136 | 3 |
| 4 | Cloud migration will reduce infrastructure costs by 30-40% over 3 years through ... | cloud_architect | **0.489** | 0.747 | 0.752 | 0.160 | 0.038 | 0 |
| 5 | Data sovereignty requirements for EU customers mandate data residency in EU regi... | security_lead | **0.467** | 0.448 | 0.951 | 0.194 | 0.042 | 1 |

## Counterfactual Analysis

For each top crux claim, the Belief Network simulates: "What if this claim were definitively true? What if it were definitively false?" The number of affected claims and the magnitude of belief shifts reveal the claim's true pivotal power.

### `team-skills`
*Our engineering team lacks cloud-native skills; migration will require 6-month r...*

**If TRUE:** 8 claims shift

  - `deployment-velocity`: 0.73 -> 0.49 (-0.24)
  - `containerization-first`: 0.90 -> 0.79 (-0.10)
  - `scalability`: 0.91 -> 0.87 (-0.04)
  - `hidden-costs`: 0.19 -> 0.22 (+0.03)
  - `vendor-lock-in`: 0.02 -> 0.05 (+0.02)

**If FALSE:** 6 claims shift

  - `phased-approach`: 0.93 -> 0.72 (-0.20)
  - `deployment-velocity`: 0.73 -> 0.91 (+0.18)
  - `attack-surface`: 0.24 -> 0.32 (+0.09)
  - `containerization-first`: 0.90 -> 0.95 (+0.06)
  - `vendor-lock-in`: 0.02 -> 0.01 (-0.01)

### `deployment-velocity`
*CI/CD pipelines in cloud will reduce deployment time from 2 weeks to 2 hours, en...*

**If TRUE:** 6 claims shift

  - `team-skills`: 0.71 -> 0.39 (-0.32)
  - `phased-approach`: 0.93 -> 0.89 (-0.04)
  - `containerization-first`: 0.90 -> 0.94 (+0.04)
  - `scalability`: 0.91 -> 0.93 (+0.02)
  - `vendor-lock-in`: 0.02 -> 0.01 (-0.01)

**If FALSE:** 10 claims shift

  - `scalability`: 0.91 -> 0.41 (-0.50)
  - `hidden-costs`: 0.19 -> 0.55 (+0.35)
  - `attack-surface`: 0.24 -> 0.48 (+0.24)
  - `team-skills`: 0.71 -> 0.89 (+0.18)
  - `cost-savings`: 0.96 -> 0.82 (-0.14)

### `containerization-first`
*Containerize all workloads before migration to ensure portability and reduce clo...*

**If TRUE:** 4 claims shift

  - `team-skills`: 0.71 -> 0.63 (-0.08)
  - `deployment-velocity`: 0.73 -> 0.77 (+0.04)
  - `vendor-lock-in`: 0.02 -> 0.02 (-0.01)
  - `phased-approach`: 0.93 -> 0.92 (-0.01)

**If FALSE:** 9 claims shift

  - `deployment-velocity`: 0.73 -> 0.48 (-0.25)
  - `team-skills`: 0.71 -> 0.95 (+0.24)
  - `vendor-lock-in`: 0.02 -> 0.22 (+0.19)
  - `scalability`: 0.91 -> 0.87 (-0.05)
  - `hybrid-strategy`: 0.99 -> 0.95 (-0.04)

## Belief Confidence Distribution

### Most Certain Claims (highest posterior confidence)

| Claim | Author | Confidence | P(True) | Verdict |
|:------|:------:|:----------:|:-------:|:-------:|
| A hybrid cloud strategy keeps sensitive data on-premises whi... | cto | 99% | 0.99 | TRUE |
| Proprietary cloud services create vendor lock-in that will m... | security_lead | 98% | 0.02 | FALSE |
| Cloud migration will reduce infrastructure costs by 30-40% o... | cloud_architect | 96% | 0.96 | TRUE |
| Data sovereignty requirements for EU customers mandate data ... | security_lead | 94% | 0.94 | TRUE |
| A phased migration starting with stateless services reduces ... | engineering_manager | 92% | 0.92 | TRUE |

### Most Uncertain Claims (highest entropy)

| Claim | Author | Entropy | P(True) |
|:------|:------:|:-------:|:-------:|
| Our engineering team lacks cloud-native skills; migration wi... | engineering_manager | 0.873 | 0.71 |
| CI/CD pipelines in cloud will reduce deployment time from 2 ... | cloud_architect | 0.833 | 0.74 |
| Cloud migration increases the attack surface through IAM mis... | security_lead | 0.786 | 0.23 |
| Cloud costs are unpredictable and often exceed projections b... | cfo | 0.700 | 0.19 |
| Shifting from CapEx to OpEx model improves cash flow and pro... | cfo | 0.576 | 0.86 |

### Load-Bearing Claims (highest centrality)

These claims have the most connections to other claims in the graph. If they change, many other beliefs cascade.

| Claim | Author | Centrality |
|:------|:------:|:----------:|
| A phased migration starting with stateless services reduces ... | engineering_manager | 0.2611 |
| Containerize all workloads before migration to ensure portab... | cto | 0.1359 |
| A hybrid cloud strategy keeps sensitive data on-premises whi... | cto | 0.0894 |
| Cloud costs are unpredictable and often exceed projections b... | cfo | 0.0804 |
| Cloud migration increases the attack surface through IAM mis... | security_lead | 0.0756 |

## Consensus Estimation

- **Consensus probability:** 58%
- **Contested claims:** 4 out of 12
- **Convergence barrier:** 0.52

Consensus is possible but not guaranteed. Resolving the top crux claims would significantly increase the consensus probability.

## Interpretation

### What This Demonstrates

1. **Crux detection identifies leverage points.** Instead of debating everything equally, the Belief Network shows exactly which claims matter most. Resolving one crux claim can cascade through the graph and shift multiple dependent beliefs.

2. **Counterfactual analysis quantifies impact.** The what-if analysis shows that changing a single crux claim (e.g., 'hidden costs are real') affects multiple downstream claims about cost savings and scalability. This transforms intuitive arguments into measurable impacts.

3. **Consensus probability provides early warning.** Before a debate even finishes, the network can estimate whether consensus is achievable or whether fundamental disagreements need to be escalated to human decision-makers.

4. **Load-bearing claims reveal structural dependencies.** Some claims are more important than others, not because of their content, but because of their position in the argument graph. The centrality analysis identifies these structural dependencies.

### Limitations

- The benchmark uses hand-crafted claims and relationships. In production, these are extracted automatically from agent debate messages using the ClaimsKernel's fast_extract_claims() and relationship detection.
- Loopy belief propagation does not guarantee exact posteriors on cyclic graphs, but converges reliably with damping. The benchmark verifies convergence before reporting results.
- Crux scores are relative within a single debate. Comparing crux scores across different debates requires normalization.
- In production, the Belief Network integrates with the Knowledge Mound to seed prior beliefs from past debates, improving accuracy for recurring topics.

### Why This Matters

Most multi-agent systems treat debate as a black box: agents argue, and the final answer is selected by voting or averaging. The Belief Network provides an X-ray into the argument structure:

- **Identifies exactly which claims to focus on** (crux detection)
- **Quantifies how much each claim matters** (influence + centrality)
- **Predicts consensus probability** before the debate ends
- **Runs what-if scenarios** to test the impact of resolving specific claims
- **Tracks belief evolution** across debate rounds via the propagation history

This is a fundamentally different approach from simple majority voting or confidence averaging. It treats debate as a Bayesian inference problem, where each agent's claims are evidence that updates a shared belief graph.
