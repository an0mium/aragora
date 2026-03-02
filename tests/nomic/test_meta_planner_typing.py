import inspect
from typing import get_type_hints

from aragora.nomic.meta_planner import MetaPlanner


def test_meta_planner_public_methods_have_return_types():
    """
    Verifies that all public methods of MetaPlanner have return type annotations.
    """
    public_methods = [
        (name, member)
        for name, member in inspect.getmembers(MetaPlanner, predicate=inspect.isfunction)
        if not name.startswith("_") or name == "__init__"
    ]

    missing_annotations = []
    for name, method in public_methods:
        signature = inspect.signature(method)
        if signature.return_annotation is inspect.Signature.empty:
            missing_annotations.append(name)

    assert not missing_annotations, (
        f"The following public methods in MetaPlanner are missing return type "
        f"annotations: {', '.join(missing_annotations)}"
    )
