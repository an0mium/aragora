# Trickster A/B Benchmark Results

**Generated:** 2026-02-16 20:11 UTC
**Total duration:** 0.1s
**Test cases:** 15
**Rounds per debate:** 2
**Agents:** Advocate (supportive), Skeptic (critical), Mediator (balanced)

## Detailed Results

| # | Question | Category | Conf (with) | Conf (w/o) | Delta | Interventions | Consensus Change | Similarity (with) | Similarity (w/o) |
|---|----------|----------|-------------|------------|-------|---------------|------------------|-------------------|------------------|
| 1 | Is Python or JavaScript better for web scraping? | clear-answer | 0.73 | 0.73 | 0.000 | 0 | unchanged | 0.03 | 0.03 |
| 2 | Should a startup use PostgreSQL or MongoDB for a CRM? | clear-answer | 0.60 | 0.60 | 0.000 | 0 | unchanged | 0.03 | 0.03 |
| 3 | Should we use TypeScript or JavaScript for a new Rea... | clear-answer | 0.40 | 0.40 | 0.000 | 0 | unchanged | 0.03 | 0.03 |
| 4 | Is Redis or Memcached better for session caching? | clear-answer | 0.74 | 0.74 | 0.000 | 0 | unchanged | 0.03 | 0.03 |
| 5 | Should we use pytest or unittest for our Python test... | clear-answer | 0.72 | 0.72 | 0.000 | 0 | unchanged | 0.03 | 0.03 |
| 6 | Is AI good for society? | ambiguous | 0.60 | 0.60 | 0.000 | 0 | unchanged | 0.03 | 0.03 |
| 7 | Should we adopt microservices? | ambiguous | 0.73 | 0.73 | 0.000 | 0 | unchanged | 0.03 | 0.03 |
| 8 | Is remote work better than in-office for engineering... | ambiguous | 0.40 | 0.40 | 0.000 | 0 | unchanged | 0.02 | 0.02 |
| 9 | Should startups prioritize growth or profitability? | ambiguous | 0.73 | 0.73 | 0.000 | 0 | unchanged | 0.03 | 0.03 |
| 10 | Is open source a better strategy than proprietary fo... | ambiguous | 0.72 | 0.72 | 0.000 | 0 | unchanged | 0.02 | 0.02 |
| 11 | Should we use Kubernetes or ECS for container orches... | domain-specific | 0.61 | 0.61 | 0.000 | 0 | unchanged | 0.02 | 0.02 |
| 12 | Is GraphQL better than REST for our mobile API? | domain-specific | 0.73 | 0.73 | 0.000 | 0 | unchanged | 0.03 | 0.03 |
| 13 | Should we use Kafka or RabbitMQ for our event bus? | domain-specific | 0.40 | 0.40 | 0.000 | 0 | unchanged | 0.04 | 0.04 |
| 14 | Is Terraform or Pulumi better for our IaC? | domain-specific | 0.61 | 0.61 | 0.000 | 0 | unchanged | 0.03 | 0.03 |
| 15 | Should we implement CQRS or keep a simple CRUD archi... | domain-specific | 0.38 | 0.38 | 0.000 | 0 | unchanged | 0.03 | 0.03 |

## Evidence Quality Scores

| # | Question | Evid. Quality (with) | Evid. Quality (w/o) | Avg Proposal Len (with) | Avg Proposal Len (w/o) | Dissents (with) | Dissents (w/o) |
|---|----------|----------------------|---------------------|-------------------------|------------------------|-----------------|----------------|
| 1 | Is Python or JavaScript better for web scraping? | 0.20 | 0.20 | 262 | 262 | 1 | 1 |
| 2 | Should a startup use PostgreSQL or MongoDB for a CRM? | 0.20 | 0.20 | 281 | 281 | 1 | 1 |
| 3 | Should we use TypeScript or JavaScript for a new Rea... | 0.30 | 0.30 | 269 | 269 | 2 | 2 |
| 4 | Is Redis or Memcached better for session caching? | 0.22 | 0.22 | 263 | 263 | 1 | 1 |
| 5 | Should we use pytest or unittest for our Python test... | 0.30 | 0.30 | 240 | 240 | 1 | 1 |
| 6 | Is AI good for society? | 0.30 | 0.30 | 259 | 259 | 1 | 1 |
| 7 | Should we adopt microservices? | 0.20 | 0.20 | 281 | 281 | 1 | 1 |
| 8 | Is remote work better than in-office for engineering... | 0.20 | 0.20 | 272 | 272 | 2 | 2 |
| 9 | Should startups prioritize growth or profitability? | 0.20 | 0.20 | 262 | 262 | 1 | 1 |
| 10 | Is open source a better strategy than proprietary fo... | 0.20 | 0.20 | 272 | 272 | 1 | 1 |
| 11 | Should we use Kubernetes or ECS for container orches... | 0.20 | 0.20 | 291 | 291 | 1 | 1 |
| 12 | Is GraphQL better than REST for our mobile API? | 0.20 | 0.20 | 281 | 281 | 1 | 1 |
| 13 | Should we use Kafka or RabbitMQ for our event bus? | 0.25 | 0.25 | 252 | 252 | 2 | 2 |
| 14 | Is Terraform or Pulumi better for our IaC? | 0.36 | 0.36 | 241 | 241 | 1 | 1 |
| 15 | Should we implement CQRS or keep a simple CRUD archi... | 0.26 | 0.26 | 244 | 244 | 2 | 2 |

## Summary Statistics

| Metric | With Trickster | Without Trickster | Delta |
|--------|----------------|-------------------|-------|
| Avg confidence | 0.61 | 0.61 | 0.000 |
| Consensus rate | 11/15 (73%) | 11/15 (73%) | 0.000 |
| Total interventions | 0 | 0 | +0 |
| Cases with intervention | 0/15 (0%) | 0/0 | -- |
| Avg evidence quality | 0.24 | 0.24 | 0.000 |
| Avg final similarity | 0.03 | 0.03 | 0.000 |
| Avg debate duration | 0.004s | 0.003s | +0.001s |

## Per-Category Breakdown

### Clear Answer (5 questions)

- **Avg confidence with trickster:** 0.64
- **Avg confidence without trickster:** 0.64
- **Confidence delta:** 0.000
- **Consensus rate (with):** 4/5
- **Consensus rate (without):** 4/5
- **Total trickster interventions:** 0

### Ambiguous (5 questions)

- **Avg confidence with trickster:** 0.64
- **Avg confidence without trickster:** 0.64
- **Confidence delta:** 0.000
- **Consensus rate (with):** 4/5
- **Consensus rate (without):** 4/5
- **Total trickster interventions:** 0

### Domain Specific (5 questions)

- **Avg confidence with trickster:** 0.55
- **Avg confidence without trickster:** 0.55
- **Confidence delta:** 0.000
- **Consensus rate (with):** 3/5
- **Consensus rate (without):** 3/5
- **Total trickster interventions:** 0

## Interpretation

The Trickster system is designed to detect and challenge *hollow consensus* -- situations where agents converge on an answer without substantive evidence. Key observations from this benchmark:

- The Trickster did not intervene in any debates. This may indicate that the mock agents' canned responses contain enough specificity to avoid triggering hollow consensus detection, or that the sensitivity threshold needs adjustment.
- Average confidence was nearly identical between conditions, indicating the Trickster did not significantly alter debate outcomes for these mock agent configurations.

---

*This report was generated by `scripts/benchmark_trickster.py` using `StyledMockAgent` configurations (no live LLM calls).*
