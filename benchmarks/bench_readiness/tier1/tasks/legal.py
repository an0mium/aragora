"""Legal contract review — 10 curated tasks.

CUAD and similar public legal datasets have restrictive licenses for
benchmarking. Curated synthetic clauses + ground-truth rubrics give us
a clean signal without licensing complications.

Each task is: (a) a short contract excerpt, (b) a specific review question
that would plausibly be asked by a lawyer or paralegal, (c) a rubric of
points the correct answer must make. The LLM-judge scores against the
rubric.
"""

from __future__ import annotations

from collections.abc import Iterable

from benchmarks.bench_readiness.tier1.tasks.base import TaskItem

_ITEMS: list[tuple[str, str, str, str]] = [
    (
        "liability-cap-carveouts",
        (
            "Except with respect to (a) a party's indemnification obligations, "
            "(b) breach of confidentiality, (c) gross negligence or willful "
            "misconduct, or (d) infringement of intellectual property rights, "
            "neither party's aggregate liability shall exceed the fees paid by "
            "Customer in the twelve months preceding the claim."
        ),
        "Review this limitation-of-liability clause from the Customer's perspective. "
        "What are the carve-outs (if any), are they customary, and what is the "
        "single biggest risk this clause poses to the Customer?",
        "Must identify: (1) the four specific carve-outs (indemnity, confidentiality, "
        "gross negligence/willful misconduct, IP infringement); (2) confirm all four "
        "are customary; (3) flag the primary risk — 12-month fee cap is a narrow "
        "fees-paid cap (not fees-payable), so early terminations before 12 months "
        "or low-fee pilots leave the Customer exposed.",
    ),
    (
        "data-processing-addendum",
        (
            "Processor shall process Personal Data only on documented instructions "
            "from Controller, including transfers to a third country, unless "
            "required by Union or Member State law; Processor shall inform "
            "Controller of any such legal requirement before processing, unless "
            "that law prohibits such information on important grounds of public "
            "interest."
        ),
        "Does this clause satisfy GDPR Art. 28(3)(a)? What's missing for a fully "
        "compliant Data Processing Addendum?",
        "Must note: (1) clause covers the Art. 28(3)(a) 'documented instructions' "
        "requirement including the cross-border carve-out; (2) flag what's missing "
        "for full Art. 28(3): (b) confidentiality obligations on processor staff, "
        "(c) Art. 32 security measures reference, (d) sub-processor rules, "
        "(e) data subject rights assistance, (f) breach notification timelines, "
        "(g) DPIA assistance, (h) end-of-engagement deletion/return, (i) audit rights.",
    ),
    (
        "non-compete-scope",
        (
            "Executive agrees that for a period of 24 months following "
            "termination of employment for any reason, Executive shall not, "
            "directly or indirectly, engage in any business that competes with "
            "the Company anywhere in the world."
        ),
        "Analyze the enforceability of this non-compete in California and in "
        "New York. Is there a meaningful difference in outcomes?",
        "Must state: (1) California generally refuses to enforce non-competes under "
        "Bus. & Prof. Code § 16600 except narrow sale-of-business / dissolution "
        "carve-outs — this clause would be void as to a CA-based executive; "
        "(2) New York enforces non-competes under a reasonableness test — 24 months "
        "is on the long end, worldwide geographic scope is almost certainly "
        "unreasonable and would likely be blue-penciled or voided; (3) material "
        "difference in outcome exists (CA void, NY likely modified).",
    ),
    (
        "ip-assignment-moral-rights",
        (
            "Contractor hereby assigns to Company all right, title, and interest "
            "in and to the Work Product, including all intellectual property "
            "rights therein, and waives any moral rights to the extent "
            "permitted by applicable law."
        ),
        "Is this IP assignment effective in France and in the United States?",
        "Must note: (1) US — effective; 'work-for-hire' doctrine and broad IP "
        "assignments are routinely enforced; moral-rights waiver is meaningful "
        "only under VARA for visual art. (2) France — moral rights (droit moral) "
        "are INALIENABLE under Art. L. 121-1 CPI; a waiver is legally ineffective "
        "regardless of what the contract says. The 'to the extent permitted' "
        "savings clause does not rescue it for moral rights in France. Company "
        "cannot rely on this waiver for French-authored content.",
    ),
    (
        "payment-terms-offset",
        (
            "Customer shall pay all undisputed fees within 30 days of invoice. "
            "Customer may not withhold, offset, or deduct any amount from "
            "payments owed hereunder, notwithstanding any dispute."
        ),
        "From the Customer's side, what is the practical problem with this "
        "no-offset clause and what should the Customer negotiate?",
        "Must identify: (1) problem — Customer loses its strongest negotiating "
        "lever (payment freeze) when Vendor breaches; forces Customer into "
        "litigation/arbitration to claim anything back; (2) Customer should "
        "negotiate: right to withhold disputed portion only, notice-and-cure "
        "window before offset prohibition applies, or setoff right for judgments "
        "and undisputed amounts due from Vendor.",
    ),
    (
        "auto-renewal-notice",
        (
            "This Agreement shall automatically renew for successive 12-month "
            "periods unless either party provides written notice of "
            "non-renewal at least 90 days prior to the end of the then-current "
            "term. Fees for the renewal term will be the Vendor's then-current "
            "list pricing."
        ),
        "Flag every customer-unfriendly element of this auto-renewal clause.",
        "Must flag: (1) 90-day notice window is long — customer must diarize; "
        "(2) 'then-current list pricing' allows unilateral price increases without "
        "cap; (3) missing — no obligation on Vendor to notify Customer of upcoming "
        "auto-renewal or price change; (4) missing — no customer-favorable cap "
        "(e.g., CPI or 5%) on renewal price. Customer should negotiate: shorter "
        "window, Vendor notification duty, and a price-cap formula.",
    ),
    (
        "sla-credits-sole-remedy",
        (
            "If Vendor fails to meet the Service Level commitments, Customer's "
            "sole and exclusive remedy shall be service credits equal to 5% of "
            "the monthly fees for each full hour of unavailability, not to "
            "exceed 30% of the monthly fees for that month."
        ),
        "Evaluate this SLA remedy from the Customer's perspective.",
        "Must address: (1) service credits as 'sole and exclusive remedy' is "
        "standard but restrictive — blocks damages recovery; (2) 5%/hr with 30% "
        "monthly cap = max 6 hours of outage fully credited — worse than "
        "typical enterprise SLAs (99.9% = ~43 min/month, so 6 hours implies "
        "99.17% which is a weak SLA); (3) Customer should negotiate: (a) "
        "termination right after N consecutive months of SLA misses, (b) carve-out "
        "from 'sole remedy' for persistent or severe outages, (c) higher credit "
        "percentages or uncapped credits.",
    ),
    (
        "indemnity-ip-scope",
        (
            "Vendor shall defend, indemnify, and hold harmless Customer from "
            "and against any third-party claims alleging that the Services, "
            "as delivered by Vendor and used in accordance with this "
            "Agreement, infringe any U.S. patent, copyright, or trademark. "
            "This indemnity shall not apply to (i) modifications made by "
            "Customer or (ii) use of the Services in combination with any "
            "third-party products."
        ),
        "What's wrong with this IP indemnity from the Customer's perspective?",
        "Must identify: (1) geographic scope — only U.S. patents/copyrights/"
        "trademarks; international customers or products are unprotected; "
        "(2) missing trade secret coverage; (3) combination carve-out is broad — "
        "almost any enterprise use involves third-party integrations; (4) "
        "'as delivered by Vendor and used in accordance with this Agreement' "
        "is a strict condition; (5) no remedies specified (e.g., replace/"
        "modify/refund on injunction). Customer should negotiate: worldwide "
        "scope (or at least major jurisdictions), trade secret inclusion, "
        "narrower combination carve-out, and specified remedies.",
    ),
    (
        "termination-for-convenience",
        (
            "Customer may terminate this Agreement for convenience upon 180 "
            "days' prior written notice to Vendor, provided that Customer "
            "shall pay a termination fee equal to 50% of the remaining fees "
            "that would have been due through the end of the then-current term."
        ),
        "Is this 'termination for convenience' genuinely for convenience, "
        "or functionally not? What's a reasonable counter-proposal?",
        "Must say: (1) not genuinely for convenience — the 50% remainder-of-term "
        "payment is economically equivalent to no termination right for most of "
        "a term; (2) if customer is 11 months into a 12-month term, cost is 0.5 "
        "months of fees (fine) — if 2 months into a 36-month term, cost is 17 "
        "months of fees (prohibitive); (3) reasonable counter: shorter notice "
        "(60-90 days), a capped termination fee (e.g., 3-6 months), and/or "
        "termination-for-convenience limited to specified causes or a recurring "
        "annual window.",
    ),
    (
        "ai-model-training-rights",
        (
            "Vendor may use Customer Data (including all inputs and outputs) "
            "to train, fine-tune, and improve Vendor's AI models and services, "
            "on a non-exclusive, perpetual, worldwide, royalty-free basis. "
            "Customer may opt out by written notice to the Vendor."
        ),
        "From an enterprise Customer's perspective, list every red flag.",
        "Must flag: (1) opt-out not opt-in — default consent is contrary to GDPR "
        "Art. 6/7 and many enterprise procurement policies; (2) 'all inputs and "
        "outputs' — sweeps in confidential data, trade secrets, PII, regulated "
        "data; (3) perpetual — survives termination; (4) worldwide — ignores "
        "data-residency commitments to end users; (5) royalty-free — customer "
        "provides training data at zero compensation; (6) no indemnity carve-out "
        "if the trained model emits Customer Data to other customers; (7) no "
        "distinction between user prompts and system outputs. Customer must "
        "negotiate: opt-in default, exclusion of confidential/PII/regulated data, "
        "term-limited license, and explicit confidentiality+indemnity flowdown.",
    ),
]


def load(limit: int, seed: int = 42) -> Iterable[TaskItem]:
    """Yield up to ``limit`` legal review items in file order.

    ``seed`` is accepted for interface consistency but unused — items are
    curated rather than sampled.
    """
    for i, (slug, context, prompt, rubric) in enumerate(_ITEMS):
        if i >= limit:
            break
        yield TaskItem(
            task_id=f"legal-{slug}",
            domain="legal",
            prompt=prompt,
            context=context,
            reference_answer=rubric,
            eval_strategy="llm_judge",
            metadata={"clause_type": slug},
        )
