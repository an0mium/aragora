"""Tests for observability/metrics/evidence.py."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

from aragora.observability.metrics import evidence as mod
from aragora.observability.metrics.base import NoOpMetric


class FakeMetric:
    def __init__(self, name: str, docs: str, labels: tuple[str, ...] = ()) -> None:
        self.name = name
        self.documentation = docs
        self.labelnames = labels
        self.bound_labels: dict[str, str] = {}
        self.children: list[FakeMetric] = []
        self.increments: list[float] = []

    def labels(self, **labels: str) -> FakeMetric:
        child = FakeMetric(self.name, self.documentation, self.labelnames)
        child.bound_labels = labels
        self.children.append(child)
        return child

    def inc(self, amount: float = 1) -> None:
        self.increments.append(amount)


@contextmanager
def fake_counter_metrics():
    created: list[FakeMetric] = []

    def factory(name: str, docs: str, labelnames: list[str] | None = None) -> FakeMetric:
        metric = FakeMetric(name, docs, tuple(labelnames or ()))
        created.append(metric)
        return metric

    with (
        patch.object(mod, "get_metrics_enabled", return_value=True),
        patch.object(mod, "get_or_create_counter", side_effect=factory),
    ):
        yield created


def setup_function() -> None:
    mod._reset_evidence_metrics_for_tests()


def teardown_function() -> None:
    mod._reset_evidence_metrics_for_tests()


def test_init_creates_expected_counter_metrics() -> None:
    with fake_counter_metrics() as created:
        mod.init_evidence_metrics()

    assert [(m.name, m.documentation, m.labelnames) for m in created] == [
        ("aragora_evidence_stored_total", "Evidence items stored in knowledge mound", ()),
        (
            "aragora_evidence_citation_bonuses_total",
            "Evidence citation vote bonuses applied",
            ("agent",),
        ),
        ("aragora_culture_patterns_total", "Culture patterns extracted from debates", ()),
    ]
    assert (mod.EVIDENCE_STORED, mod.EVIDENCE_CITATION_BONUSES, mod.CULTURE_PATTERNS) == tuple(
        created
    )
    assert mod._initialized is True


def test_init_is_idempotent_after_successful_initialization() -> None:
    with fake_counter_metrics() as created:
        mod.init_evidence_metrics()
        first = mod.EVIDENCE_STORED
        mod.init_evidence_metrics()

    assert len(created) == 3
    assert mod.EVIDENCE_STORED is first


def test_init_uses_noop_metrics_when_metrics_are_disabled() -> None:
    with patch.object(mod, "get_metrics_enabled", return_value=False):
        mod.init_evidence_metrics()

    assert isinstance(mod.EVIDENCE_STORED, NoOpMetric)
    assert mod.EVIDENCE_STORED is mod.EVIDENCE_CITATION_BONUSES
    assert mod.CULTURE_PATTERNS is mod.EVIDENCE_STORED


def test_init_falls_back_to_noop_when_counter_creation_fails() -> None:
    with (
        patch.object(mod, "get_metrics_enabled", return_value=True),
        patch.object(mod, "get_or_create_counter", side_effect=ValueError("duplicate")),
    ):
        mod.init_evidence_metrics()

    assert isinstance(mod.EVIDENCE_STORED, NoOpMetric)
    assert isinstance(mod.EVIDENCE_CITATION_BONUSES, NoOpMetric)
    assert isinstance(mod.CULTURE_PATTERNS, NoOpMetric)


def test_record_evidence_stored_tracks_default_and_custom_counts() -> None:
    with fake_counter_metrics() as created:
        mod.record_evidence_stored()
        mod.record_evidence_stored(4)

    assert created[0].increments == [1, 4]


def test_record_evidence_citation_bonus_applies_agent_label() -> None:
    with fake_counter_metrics() as created:
        mod.record_evidence_citation_bonus("codex")

    assert created[1].children[0].bound_labels == {"agent": "codex"}
    assert created[1].children[0].increments == [1]


def test_record_culture_patterns_tracks_default_and_custom_counts() -> None:
    with fake_counter_metrics() as created:
        mod.record_culture_patterns()
        mod.record_culture_patterns(7)

    assert created[2].increments == [1, 7]


def test_reset_clears_module_state_for_isolated_tests() -> None:
    with patch.object(mod, "get_metrics_enabled", return_value=False):
        mod.init_evidence_metrics()

    mod._reset_evidence_metrics_for_tests()

    assert mod.EVIDENCE_STORED is None
    assert mod.EVIDENCE_CITATION_BONUSES is None
    assert mod.CULTURE_PATTERNS is None
    assert mod._initialized is False


def test_public_exports_include_recording_api() -> None:
    assert set(mod.__all__) == {
        "EVIDENCE_STORED",
        "EVIDENCE_CITATION_BONUSES",
        "CULTURE_PATTERNS",
        "init_evidence_metrics",
        "record_evidence_stored",
        "record_evidence_citation_bonus",
        "record_culture_patterns",
    }
