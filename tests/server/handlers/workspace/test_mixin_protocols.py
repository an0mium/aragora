"""Tests for the workspace mixin host protocol.

These tests verify that:

1. ``aragora.server.handlers.workspace._protocols.WorkspaceMixinHost`` is a
   ``Protocol`` type that captures the cross-mixin contract (method names,
   signatures).
2. The concrete :class:`WorkspaceHandler` satisfies the protocol (duck-typed,
   via ``typing.runtime_checkable``-style structural checks).
3. A stub host that *fails* to implement a required method fails the same
   structural check.
4. Importing the protocol module has zero runtime side effects on the
   workspace handler package (no circular imports, no eager evaluation of
   ``TYPE_CHECKING`` imports).

The protocol is a pure type-checking aid, so these are lightweight sanity
tests that catch accidental drift -- e.g., if someone renames
``_get_audit_log`` on the host without updating the protocol.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_callable(obj: Any, name: str) -> bool:
    """Return True iff ``obj`` has a callable attribute named ``name``."""
    value = getattr(obj, name, None)
    return callable(value)


def _method_names(proto_cls: type) -> set[str]:
    """Return the set of method names declared on a Protocol class,
    excluding dunder and Protocol-internal attributes."""
    return {
        name
        for name, member in inspect.getmembers(proto_cls)
        if not name.startswith("_abc_") and not name.startswith("__") and callable(member)
    }


# ---------------------------------------------------------------------------
# Protocol shape tests
# ---------------------------------------------------------------------------


def test_workspace_mixin_host_protocol_is_importable() -> None:
    """The _protocols module must import without touching runtime code paths."""
    mod = importlib.import_module("aragora.server.handlers.workspace._protocols")
    assert hasattr(mod, "WorkspaceMixinHost")


def test_workspace_mixin_host_declares_expected_members() -> None:
    """The protocol must list every cross-mixin method the mixins depend on.

    If this test fails after renaming a helper on ``WorkspaceHandler``, update
    the protocol (and the ``TYPE_CHECKING`` stubs in each mixin) to match.
    """
    from aragora.server.handlers.workspace._protocols import WorkspaceMixinHost

    expected = {
        "_get_user_store",
        "_get_isolation_manager",
        "_get_retention_manager",
        "_get_classifier",
        "_get_audit_log",
        "_run_async",
        "_check_rbac_permission",
        "read_json_body",
    }
    declared = _method_names(WorkspaceMixinHost)
    missing = expected - declared
    assert not missing, f"WorkspaceMixinHost is missing required members: {missing}"


def test_workspace_mixin_host_signatures_use_expected_argument_names() -> None:
    """Spot-check that key argument names haven't drifted in the protocol."""
    from aragora.server.handlers.workspace._protocols import WorkspaceMixinHost

    check = inspect.signature(WorkspaceMixinHost._check_rbac_permission)
    assert list(check.parameters) == [
        "self",
        "handler",
        "permission_key",
        "auth_ctx",
    ]

    run_async = inspect.signature(WorkspaceMixinHost._run_async)
    assert list(run_async.parameters) == ["self", "coro"]

    read_json = inspect.signature(WorkspaceMixinHost.read_json_body)
    assert list(read_json.parameters) == ["self", "handler", "max_size"]


# ---------------------------------------------------------------------------
# Host compliance tests
# ---------------------------------------------------------------------------


def test_workspace_handler_satisfies_mixin_host_protocol() -> None:
    """The concrete WorkspaceHandler must expose every protocol member."""
    from aragora.server.handlers import workspace_module
    from aragora.server.handlers.workspace._protocols import WorkspaceMixinHost

    handler_cls = workspace_module.WorkspaceHandler
    for name in _method_names(WorkspaceMixinHost):
        assert _has_callable(handler_cls, name), (
            f"WorkspaceHandler is missing cross-mixin method {name!r} "
            f"required by WorkspaceMixinHost"
        )


def test_mixin_host_protocol_rejects_incomplete_stub() -> None:
    """A stub that omits required methods must fail the structural check.

    We emulate the check with a plain attribute presence test rather than
    ``isinstance`` so the test works regardless of whether the protocol is
    decorated with ``@runtime_checkable``.
    """
    from aragora.server.handlers.workspace._protocols import WorkspaceMixinHost

    class IncompleteHost:
        # Only implements one of the required members; everything else is
        # deliberately missing so the check should fail.
        def _get_user_store(self) -> Any:
            return None

    required = _method_names(WorkspaceMixinHost)
    missing = [name for name in required if not _has_callable(IncompleteHost, name)]
    assert missing, (
        "Expected IncompleteHost to be missing several protocol members; "
        "got none -- the protocol may have been silently narrowed."
    )


# ---------------------------------------------------------------------------
# Mixin compliance tests
# ---------------------------------------------------------------------------


def test_each_mixin_redeclares_host_members_under_type_checking() -> None:
    """Every mixin's source must reference the host contract members it uses.

    This is a textual smoke test: it protects against regressions where
    someone deletes the ``if TYPE_CHECKING:`` block in a mixin and
    re-introduces the baseline-inflating ``attr-defined`` errors.
    """
    import pathlib

    from aragora.server.handlers import workspace as workspace_pkg

    pkg_dir = pathlib.Path(workspace_pkg.__file__).parent
    for mixin_file, required_methods in (
        (
            "settings.py",
            {
                "_get_user_store",
                "_get_classifier",
                "_get_audit_log",
                "_run_async",
                "_check_rbac_permission",
                "read_json_body",
            },
        ),
        (
            "policies.py",
            {
                "_get_user_store",
                "_get_retention_manager",
                "_get_audit_log",
                "_run_async",
                "_check_rbac_permission",
                "read_json_body",
            },
        ),
        (
            "members.py",
            {
                "_get_user_store",
                "_get_isolation_manager",
                "_get_audit_log",
                "_run_async",
                "_check_rbac_permission",
                "read_json_body",
            },
        ),
    ):
        source = (pkg_dir / mixin_file).read_text()
        assert "if TYPE_CHECKING:" in source, (
            f"{mixin_file} must declare a TYPE_CHECKING host-contract block"
        )
        for method in required_methods:
            assert f"def {method}(" in source, (
                f"{mixin_file} is missing TYPE_CHECKING stub for {method!r}; "
                "this will re-introduce attr-defined mypy errors"
            )


def test_protocol_module_has_no_runtime_side_effects() -> None:
    """Importing the protocol must not force importing privacy subsystems.

    The imports for ``DataIsolationManager``, ``PrivacyAuditLog``, etc. are
    under ``TYPE_CHECKING`` and should not be pulled in when the module is
    imported at runtime.
    """
    import sys

    # Fresh import to observe exactly what the protocol module touches.
    name = "aragora.server.handlers.workspace._protocols"
    sys.modules.pop(name, None)
    mod = importlib.import_module(name)

    # The protocol references these types only in type annotations, not in
    # executable code, so the protocol module itself should expose them as
    # forward references -- not as real imported symbols.
    assert not hasattr(mod, "DataIsolationManager")
    assert not hasattr(mod, "PrivacyAuditLog")
    assert not hasattr(mod, "RetentionPolicyManager")
    assert not hasattr(mod, "SensitivityClassifier")


def test_workspace_mixin_host_annotations_are_strings() -> None:
    """Under ``from __future__ import annotations`` every annotation on the
    protocol is stored as a string (PEP 563). This test makes sure we kept
    that behaviour so the TYPE_CHECKING imports remain lazy -- otherwise the
    privacy subsystems would be imported eagerly.
    """
    from aragora.server.handlers.workspace._protocols import WorkspaceMixinHost

    raw_annotations = WorkspaceMixinHost._check_rbac_permission.__annotations__
    # All annotations should be strings (deferred evaluation).
    for name, ann in raw_annotations.items():
        assert isinstance(ann, str), (
            f"Annotation for {name!r} on _check_rbac_permission is not a "
            f"string ({type(ann).__name__}); this means PEP 563 deferred "
            "evaluation is off and TYPE_CHECKING imports will be eager."
        )
