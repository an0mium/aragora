from __future__ import annotations

import importlib

from aragora.knowledge.mound.adapters import factory as adapter_factory


def test_count_km_adapters_matches_factory_defs() -> None:
    doc_stats = importlib.import_module("scripts.doc_stats")

    assert doc_stats._count_km_adapters() == len(adapter_factory._ADAPTER_DEFS) == 42
