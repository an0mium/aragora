from aragora.debate.ml_integration import ConsensusEstimator
from aragora.debate.stability_detector import BetaBinomialStabilityDetector


def test_stability_detector_empty() -> None:
    detector = BetaBinomialStabilityDetector()
    result = detector.calculate_stability([])
    assert result.stability == 0.0
    assert result.total == 0
    assert result.successes == 0


def test_stability_detector_high_agreement() -> None:
    detector = BetaBinomialStabilityDetector(agreement_threshold=0.75)
    result = detector.calculate_stability([0.9, 0.8, 0.85])
    assert result.successes == 3
    assert result.total == 3
    assert abs(result.stability - 0.8) < 1e-6


def test_stability_detector_low_agreement() -> None:
    detector = BetaBinomialStabilityDetector(agreement_threshold=0.75)
    result = detector.calculate_stability([0.1, 0.2])
    assert result.successes == 0
    assert result.total == 2
    assert abs(result.stability - 0.25) < 1e-6


def test_consensus_estimator_stability_path() -> None:
    estimator = ConsensusEstimator(
        early_termination_threshold=0.99,
        min_rounds=1,
        enable_stability_detection=True,
        stability_threshold=0.1,
        stability_min_rounds=1,
    )
    responses = [
        ("agent_a", "The answer is yes"),
        ("agent_b", "The answer is yes"),
    ]
    assert estimator.should_terminate_early(responses, current_round=1, total_rounds=3)
