#!/usr/bin/env python3
"""
Evidence Quality Scoring -- Detect hollow consensus and weak arguments.

Demonstrates:
- Scoring agent responses for evidence quality (citations, specificity, reasoning)
- Detecting hollow consensus (agents agreeing without substantive evidence)
- Using EvidenceQualityAnalyzer and HollowConsensusDetector standalone

No API keys required. Works completely offline.

Usage:
    pip install aragora-debate
    python examples/quickstart/03_evidence_quality.py
"""

from aragora_debate import EvidenceQualityAnalyzer, HollowConsensusDetector


def main():
    analyzer = EvidenceQualityAnalyzer()

    # --- Score a STRONG response (with citations, data, specifics) ---
    strong_response = (
        "According to the 2024 CNCF survey, 73% of enterprises now use Kubernetes "
        "in production (up from 58% in 2022). Container orchestration reduces "
        "deployment times by 40-60% based on measurements from Google's Borg paper. "
        "For a team of 5 engineers handling 10,000 requests/second, the operational "
        "overhead is justified because auto-scaling alone saves approximately $2,400/month "
        "in compute costs. Therefore, the migration cost is recovered within 6 months."
    )

    strong_score = analyzer.analyze(strong_response, agent="analyst", round_num=1)

    print("STRONG RESPONSE")
    print(f"  Overall quality:     {strong_score.overall_quality:.2f}")
    print(f"  Citation density:    {strong_score.citation_density:.2f}")
    print(f"  Specificity:         {strong_score.specificity_score:.2f}")
    print(f"  Evidence diversity:  {strong_score.evidence_diversity:.2f}")
    print(f"  Reasoning chain:     {strong_score.logical_chain_score:.2f}")
    print(f"  Evidence markers:    {len(strong_score.evidence_markers)}")
    print()

    # --- Score a WEAK response (vague, no citations, no data) ---
    weak_response = (
        "I think microservices are generally a good idea. They typically improve "
        "things and are considered best practices in the industry. Various factors "
        "should be considered, but usually the benefits outweigh the costs. "
        "Many organizations have adopted this common approach with significant impact."
    )

    weak_score = analyzer.analyze(weak_response, agent="vague-agent", round_num=1)

    print("WEAK RESPONSE")
    print(f"  Overall quality:     {weak_score.overall_quality:.2f}")
    print(f"  Citation density:    {weak_score.citation_density:.2f}")
    print(f"  Specificity:         {weak_score.specificity_score:.2f}")
    print(f"  Evidence diversity:  {weak_score.evidence_diversity:.2f}")
    print(f"  Reasoning chain:     {weak_score.logical_chain_score:.2f}")
    print(f"  Vague phrases found: {weak_score.vague_phrase_count}")
    print()

    # --- Detect hollow consensus ---
    # When agents agree but lack evidence, that's hollow consensus
    detector = HollowConsensusDetector(min_quality_threshold=0.4)

    alert = detector.check(
        responses={
            "agent-a": weak_response,
            "agent-b": (
                "I agree with the general approach. It makes sense to "
                "follow industry standards and adopt best practices. "
                "The important aspects have been covered and the key "
                "elements align with common patterns."
            ),
        },
        convergence_similarity=0.9,  # high similarity = agents are converging
        round_num=1,
    )

    print("HOLLOW CONSENSUS DETECTION")
    print(f"  Detected:    {alert.detected}")
    print(f"  Severity:    {alert.severity:.2f}")
    print(f"  Reason:      {alert.reason}")
    print(f"  Avg quality: {alert.avg_quality:.2f}")
    if alert.recommended_challenges:
        print(f"  Challenges:")
        for challenge in alert.recommended_challenges:
            print(f"    - {challenge[:100]}...")
    print()

    # --- Compare with non-hollow consensus ---
    non_hollow = detector.check(
        responses={
            "analyst": strong_response,
            "reviewer": (
                "Based on Dreischulte et al. (2015), the 40-60% deployment "
                "time reduction is consistent with measured results. However, "
                "the $2,400/month savings estimate assumes 100% utilization of "
                "auto-scaled resources, which in practice drops to 60-70%. "
                "Therefore, the actual payback period is closer to 9 months."
            ),
        },
        convergence_similarity=0.85,
        round_num=2,
    )

    print("NON-HOLLOW CONSENSUS")
    print(f"  Detected:    {non_hollow.detected}")
    print(f"  Severity:    {non_hollow.severity:.2f}")
    print(f"  Avg quality: {non_hollow.avg_quality:.2f}")


if __name__ == "__main__":
    main()
