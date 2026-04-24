"""Aragora custom decision tasks — the domain the product is actually built for.

10 real decisions that a small-to-medium team would plausibly send through
an aragora debate: architecture choices, vendor tradeoffs, hiring/sequencing
calls, and compliance/risk judgments. No ground-truth answer because these
are genuinely debatable; scoring is rubric-based via the LLM-judge.
"""

from __future__ import annotations

from collections.abc import Iterable

from benchmarks.bench_readiness.tier1.tasks.base import TaskItem

_ITEMS: list[tuple[str, str, str, str]] = [
    (
        "monorepo-vs-polyrepo",
        "A Series-A SaaS with 12 engineers, 3 product surfaces (web app, iOS, "
        "a CLI), and a shared TypeScript SDK is deciding whether to consolidate "
        "into a monorepo (Turborepo/Nx) or stay with three separate repos.",
        "Should they adopt a monorepo? Give a defensible recommendation with "
        "the 2-3 strongest arguments on each side, the specific team signal "
        "that would flip the decision, and concrete first steps if adopted.",
        "Strong answer: (1) takes a position rather than hedging; (2) identifies "
        "concrete tradeoffs — atomic cross-package changes, shared tooling, and "
        "refactor ergonomics on the pro side vs CI complexity, tooling maturity "
        "gaps, and onboarding/partial-checkout friction on the con side; "
        "(3) provides a decision-flip signal (e.g., 'if CI costs exceed $X/mo "
        "or > N mobile-only engineers'); (4) gives 2-4 concrete first steps.",
    ),
    (
        "gdpr-fine-risk-us-expansion",
        "A U.S. B2B SaaS startup is considering expanding to EU customers. They "
        "have moderate privacy practices (SSO, encryption-at-rest) but no DPO, "
        "no DPA template, no SCC template, and no EU representative.",
        "Go/no-go on EU expansion right now, with what mitigations? Quantify "
        "the risk where possible.",
        "Strong answer: (1) takes an explicit go/no-go stance; (2) names the "
        "three hard blockers — Art. 27 EU representative, Art. 28 DPA, Art. 46 "
        "SCCs for cross-border transfer — and calls out they are cheap/fast to "
        "fix; (3) quantifies fine exposure appropriately (Art. 83: up to 4% "
        "annual worldwide turnover or €20M, whichever is greater; but for a "
        "pre-Series-B startup enforcement is more likely via customer DPIA "
        "rejections than regulator fines); (4) provides a sequenced 4-6 week "
        "mitigation plan.",
    ),
    (
        "llm-provider-diversification",
        "A fintech customer-support automation platform built on Anthropic's "
        "Claude has a 3-hour incident when Anthropic's API is down. They are "
        "debating whether to add OpenAI GPT as an active failover, keep a "
        "warm standby, or accept the risk with credits.",
        "What's the right incident-response posture for this team given the "
        "cost/complexity tradeoff? Give a specific recommendation with a "
        "concrete implementation outline.",
        "Strong answer: (1) picks ONE of active failover / warm standby / "
        "accept-risk with explicit reasoning; (2) addresses the correctness "
        "gap — different models behave differently under the same prompt, so "
        "failover has an output-quality cost; (3) recommends prompt/eval parity "
        "(run both models through same eval set) before trusting failover; "
        "(4) concrete outline: circuit breaker on Claude, degraded-mode "
        "playbook (shorter/simpler responses), provider-agnostic prompt "
        "abstraction layer; (5) quantifies cost of 3hr outage × frequency vs "
        "dual-vendor fixed+variable cost.",
    ),
    (
        "oss-license-for-core-product",
        "A profitable open-core startup ($8M ARR, 4 years in) has been shipping "
        "MIT-licensed code. A well-funded competitor has forked the core, "
        "added enterprise features, and is now selling a competing product. "
        "Leadership is debating: relicense to AGPL going forward, relicense to "
        "SSPL/BSL, stay MIT and compete on service, or split the repo "
        "(BSL for core, MIT for SDK).",
        "What's the right move? Acknowledge the community backlash risk explicitly.",
        "Strong answer: (1) takes an explicit position with a defensible rationale; "
        "(2) distinguishes between AGPL (still OSI-approved, forces network-use "
        "source disclosure), SSPL (effectively prevents cloud hosting without "
        "commercial license, not OSI-approved), and BSL (time-delayed release); "
        "(3) acknowledges the reputational / community-trust cost of relicensing "
        "— contrast ElasticSearch/HashiCorp backlash vs MongoDB which retained "
        "most users; (4) recommends action on the CLA/contribution model, not "
        "just the license; (5) provides a rollout plan: grandfathering, "
        "communication, pricing signals.",
    ),
    (
        "ai-feature-hallucination-shipping",
        "A customer-facing product is two days from launching an AI summary "
        "feature. QA found a 3% hallucination rate on internal test data. "
        "The team is split between shipping with a disclaimer, delaying a "
        "sprint to add grounding/citation, or shipping behind a feature flag "
        "to 5% of users.",
        "Ship, delay, or gradual rollout? What specific guardrails or metrics justify each choice?",
        "Strong answer: (1) takes a concrete position; (2) distinguishes "
        "disclaimer (visibility only, no real mitigation) vs grounding (reduces "
        "rate but not to zero) vs gradual rollout (limits blast radius, gives "
        "observation signal); (3) names the specific metric that would flip "
        "the decision — e.g., 'if CSAT impact of hallucinations is < 2% in the "
        "5% rollout, proceed; if > 5%, roll back'; (4) acknowledges the "
        "brand-trust asymmetry — hallucinations in a first launch cost more "
        "than hallucinations in a mature feature; (5) provides a rollback trigger.",
    ),
    (
        "vertical-focus-commercial",
        "A general-purpose multi-agent deliberation platform (like aragora "
        "itself — 43 agent types, 5 verticals, multiple channels) is deciding "
        "between: (a) stay horizontal and sell to anyone, (b) pick one vertical "
        "and own it (legal / healthcare / accounting / software engineering), "
        "or (c) pick two verticals as a 'core + adjacent' strategy.",
        "Which strategy wins, why, and what's the concrete 12-month plan to execute on it?",
        "Strong answer: (1) takes a clear position, with preference for ONE "
        "vertical over multi-vertical; (2) applies standard reasoning about "
        "vertical SaaS motion — concentrated sales motion, targeted marketing, "
        "defensible pricing, more predictable buyer persona; (3) names which "
        "vertical with specific reasoning — legal is the strongest fit for "
        "audit-trail / decision-receipt products because law firms bill for "
        "defensibility and have regulated documentation requirements; "
        "healthcare has HIPAA overhead but also Tier-1 willingness-to-pay; "
        "accounting has SOX; software engineering has most competitors; "
        "(4) gives a concrete 12-month plan: 3 months design partner acquisition "
        "in the chosen vertical, 3 months product focus with partners, 6 months "
        "GTM motion.",
    ),
    (
        "on-call-rotation-small-team",
        "A 6-person backend team has an on-call rotation that runs 1-week "
        "shifts, 24/7, follow-the-sun handoff not possible (all US-based). "
        "Two engineers are threatening to leave citing burnout. Options: "
        "shorten shifts to 3.5 days, hire a dedicated SRE, outsource to a "
        "managed on-call service, invest in runbook automation to reduce "
        "page volume.",
        "What's the right move to reduce burnout without cratering on-call "
        "coverage? What metric proves it worked 90 days later?",
        "Strong answer: (1) takes a concrete position; (2) rejects the single-"
        "lever framing — proposes a multi-intervention plan; (3) correctly "
        "identifies the actual leverage point: reducing PAGE VOLUME is higher-"
        "leverage than splitting a fixed page load across shorter shifts or "
        "more people (the latter just spreads misery); (4) specific "
        "implementations — alert quality review (suppress noise), runbook "
        "automation for top-3 pager causes, SLO-gated releases; (5) success "
        "metric — e.g., pages/week and 'would recommend on-call to a friend' "
        "NPS-style score 90 days in.",
    ),
    (
        "postgres-to-datalake-reporting",
        "A mid-market B2B SaaS (100 customers, 40TB of operational Postgres, "
        "complex reporting dashboards) has been building reports directly from "
        "Postgres read replicas. Queries are slow (30s+ for month-end rollups) "
        "and block customer-facing APIs. Options: add a separate analytics "
        "Postgres, move to ClickHouse, move to Snowflake/BigQuery with CDC, "
        "build an in-memory OLAP layer in the app.",
        "Pick the right architecture for the next 3 years of scale and "
        "explain why the others are wrong FOR THIS TEAM.",
        "Strong answer: (1) picks ONE; (2) distinguishes operational vs analytical "
        "workloads clearly; (3) explicit reasoning about why the others are wrong "
        "for THIS team (team size, skills, cost profile, customer data-residency "
        "implications); (4) addresses CDC latency tradeoffs, cost trajectory at "
        "100→500 customers, and what happens to the choice at 10x scale; "
        "(5) includes a migration sequencing plan that doesn't break existing "
        "dashboards during cutover.",
    ),
    (
        "acquire-vs-build-compliance-tool",
        "A fintech has regulatory reporting obligations in three jurisdictions "
        "(US SEC, UK FCA, EU ESMA). They're debating building their own "
        "compliance-reporting tool ($600K / 8 months / 2 eng + 1 compliance "
        "lead) vs buying one of two enterprise SaaS tools ($180K/yr, 80% "
        "fit) vs buying a startup tool ($40K/yr, 50% fit, but growing fast).",
        "Buy one of the three, or build? Decision must include a threshold "
        "for WHEN to revisit the decision.",
        "Strong answer: (1) takes a position; (2) applies cost comparison over "
        "3 years including ongoing build-vs-buy maintenance — build has $600K "
        "one-time + ~$200K/yr ongoing vs enterprise SaaS $540K/3yr all-in; "
        "(3) weighs integration risk, regulator-audit burden of in-house "
        "systems, and feature velocity; (4) names the flip trigger — e.g., "
        "'revisit if fit drops below 60% due to vendor roadmap divergence, or "
        "if headcount growth absorbs the maintenance cost'; (5) if recommending "
        "startup tool, flags the vendor-risk mitigation (escrow, exit clause).",
    ),
    (
        "incident-postmortem-blame",
        "A production outage caused by a junior engineer pushing an unreviewed "
        "migration to main resulted in 4 hours of downtime and $120K in SLA "
        "credits. Leadership is debating: terminate the engineer (public signal), "
        "formal write-up in file, quiet 1:1 coaching with continued tenure, "
        "or blameless postmortem with no individual consequence.",
        "What's the right response organizationally? Address both the "
        "individual AND the systemic signal.",
        "Strong answer: (1) takes an explicit position favoring blameless or "
        "near-blameless approach; (2) distinguishes INDIVIDUAL accountability "
        "(coaching, pairing, re-review process) from SYSTEMIC root cause "
        "(why was the migration pushable to main unreviewed? why was there no "
        "CI/CD gate? why was there no peer review workflow?); (3) correctly "
        "identifies that terminating the engineer punishes the symptom and "
        "teaches the rest of the team to hide mistakes; (4) concrete remediation "
        "— branch protection, review-required gates, migration-specific approvers, "
        "staged rollouts; (5) acknowledges the ONE exception — deliberate or "
        "grossly-negligent action requires individual consequence.",
    ),
]


def load(limit: int, seed: int = 42) -> Iterable[TaskItem]:
    """Yield up to ``limit`` aragora-custom items in file order."""
    for i, (slug, context, prompt, rubric) in enumerate(_ITEMS):
        if i >= limit:
            break
        yield TaskItem(
            task_id=f"aragora-{slug}",
            domain="aragora_custom",
            prompt=prompt,
            context=context,
            reference_answer=rubric,
            eval_strategy="llm_judge",
            metadata={"decision_type": slug},
        )
