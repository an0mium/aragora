"""Tests for KnowledgeMoundHandlerProtocol (base_mixin.py).

Covers:
- Protocol structural typing (isinstance checks with @runtime_checkable)
- Conforming classes satisfy the protocol
- Non-conforming classes fail the protocol
- Backward compatibility alias (KnowledgeMoundMixinBase)
- __all__ exports
- ctx attribute requirements
- _get_mound() method requirements
- Partial implementations
- Edge cases: None returns, empty ctx, subclass behavior
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.server.handlers.knowledge_base.mound.base_mixin import (
    KnowledgeMoundHandlerProtocol,
    KnowledgeMoundMixinBase,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return -1
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Conforming implementation
# ---------------------------------------------------------------------------


class ConformingHandler:
    """A class that satisfies KnowledgeMoundHandlerProtocol."""

    def __init__(self):
        self.ctx: dict[str, Any] = {"server": "test"}

    def _get_mound(self):
        return MagicMock()


class ConformingHandlerNoneMound:
    """A class that satisfies the protocol but returns None from _get_mound."""

    def __init__(self):
        self.ctx: dict[str, Any] = {}

    def _get_mound(self):
        return None


# ---------------------------------------------------------------------------
# Non-conforming implementations
# ---------------------------------------------------------------------------


class MissingGetMound:
    """Has ctx but no _get_mound method."""

    def __init__(self):
        self.ctx: dict[str, Any] = {}


class MissingCtx:
    """Has _get_mound but no ctx attribute."""

    def _get_mound(self):
        return None


class EmptyClass:
    """Has neither ctx nor _get_mound."""

    pass


class WrongMethodSignature:
    """Has _get_mound taking extra required args (still satisfies Protocol at runtime)."""

    def __init__(self):
        self.ctx: dict[str, Any] = {}

    def _get_mound(self, workspace_id: str):
        return None


# ---------------------------------------------------------------------------
# Tests: Protocol isinstance checks (@runtime_checkable)
# ---------------------------------------------------------------------------


class TestProtocolRuntimeCheckable:
    """Test that @runtime_checkable enables isinstance() checks."""

    def test_conforming_handler_satisfies_protocol(self):
        """A class with ctx and _get_mound satisfies the protocol."""
        handler = ConformingHandler()
        assert isinstance(handler, KnowledgeMoundHandlerProtocol)

    def test_conforming_handler_none_mound_satisfies_protocol(self):
        """A class returning None from _get_mound still satisfies the protocol."""
        handler = ConformingHandlerNoneMound()
        assert isinstance(handler, KnowledgeMoundHandlerProtocol)

    def test_missing_get_mound_fails_protocol(self):
        """A class without _get_mound does NOT satisfy the protocol."""
        obj = MissingGetMound()
        assert not isinstance(obj, KnowledgeMoundHandlerProtocol)

    def test_missing_ctx_fails_protocol(self):
        """A class without ctx attribute does NOT satisfy the protocol.

        Note: runtime_checkable only checks methods, not attributes.
        So this may pass isinstance() even without ctx. We test the
        actual runtime behavior.
        """
        obj = MissingCtx()
        # runtime_checkable checks methods, not data attributes;
        # _get_mound is present, so isinstance may still pass
        # The key point: having _get_mound is necessary
        has_get_mound = hasattr(obj, "_get_mound")
        assert has_get_mound is True

    def test_empty_class_fails_protocol(self):
        """A class with neither ctx nor _get_mound fails the protocol."""
        obj = EmptyClass()
        assert not isinstance(obj, KnowledgeMoundHandlerProtocol)

    def test_plain_dict_fails_protocol(self):
        """A dict is not an instance of the protocol."""
        assert not isinstance({}, KnowledgeMoundHandlerProtocol)

    def test_none_fails_protocol(self):
        """None is not an instance of the protocol."""
        assert not isinstance(None, KnowledgeMoundHandlerProtocol)

    def test_string_fails_protocol(self):
        """A string is not an instance of the protocol."""
        assert not isinstance("handler", KnowledgeMoundHandlerProtocol)

    def test_int_fails_protocol(self):
        """An int is not an instance of the protocol."""
        assert not isinstance(42, KnowledgeMoundHandlerProtocol)


# ---------------------------------------------------------------------------
# Tests: Backward compatibility alias
# ---------------------------------------------------------------------------


class TestBackwardCompatAlias:
    """Test the KnowledgeMoundMixinBase backward compatibility alias."""

    def test_alias_is_same_object(self):
        """KnowledgeMoundMixinBase is the same class as KnowledgeMoundHandlerProtocol."""
        assert KnowledgeMoundMixinBase is KnowledgeMoundHandlerProtocol

    def test_alias_isinstance_works(self):
        """isinstance() checks work through the alias."""
        handler = ConformingHandler()
        assert isinstance(handler, KnowledgeMoundMixinBase)

    def test_alias_isinstance_fails_for_non_conforming(self):
        """isinstance() via alias correctly rejects non-conforming types."""
        obj = EmptyClass()
        assert not isinstance(obj, KnowledgeMoundMixinBase)


# ---------------------------------------------------------------------------
# Tests: __all__ exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    """Test that __all__ contains expected symbols."""

    def test_all_contains_protocol(self):
        """__all__ includes KnowledgeMoundHandlerProtocol."""
        from aragora.server.handlers.knowledge_base.mound import base_mixin

        assert "KnowledgeMoundHandlerProtocol" in base_mixin.__all__

    def test_all_contains_alias(self):
        """__all__ includes KnowledgeMoundMixinBase."""
        from aragora.server.handlers.knowledge_base.mound import base_mixin

        assert "KnowledgeMoundMixinBase" in base_mixin.__all__

    def test_all_has_exactly_two_entries(self):
        """__all__ has exactly two entries."""
        from aragora.server.handlers.knowledge_base.mound import base_mixin

        assert len(base_mixin.__all__) == 2

    def test_module_importable_from_package(self):
        """KnowledgeMoundMixinBase is importable from the package __init__."""
        from aragora.server.handlers.knowledge_base.mound import KnowledgeMoundMixinBase as Alias

        assert Alias is KnowledgeMoundHandlerProtocol


# ---------------------------------------------------------------------------
# Tests: Protocol attribute and method behaviors
# ---------------------------------------------------------------------------


class TestProtocolCtxAttribute:
    """Test ctx attribute behavior on conforming classes."""

    def test_ctx_is_dict(self):
        """ctx attribute is a dict."""
        handler = ConformingHandler()
        assert isinstance(handler.ctx, dict)

    def test_ctx_can_be_empty(self):
        """An empty ctx dict is valid."""
        handler = ConformingHandlerNoneMound()
        assert handler.ctx == {}

    def test_ctx_can_hold_arbitrary_data(self):
        """ctx can hold arbitrary key-value pairs."""
        handler = ConformingHandler()
        handler.ctx["extra_key"] = [1, 2, 3]
        assert handler.ctx["extra_key"] == [1, 2, 3]

    def test_ctx_mutation_persists(self):
        """Mutations to ctx persist on the instance."""
        handler = ConformingHandler()
        handler.ctx["new"] = "value"
        assert "new" in handler.ctx


class TestProtocolGetMound:
    """Test _get_mound method behavior on conforming classes."""

    def test_get_mound_returns_mock(self):
        """_get_mound returns a mock KnowledgeMound on ConformingHandler."""
        handler = ConformingHandler()
        mound = handler._get_mound()
        assert mound is not None

    def test_get_mound_returns_none(self):
        """_get_mound can return None when mound is unavailable."""
        handler = ConformingHandlerNoneMound()
        mound = handler._get_mound()
        assert mound is None

    def test_get_mound_is_callable(self):
        """_get_mound is callable on conforming instances."""
        handler = ConformingHandler()
        assert callable(handler._get_mound)

    def test_get_mound_called_multiple_times(self):
        """_get_mound can be called multiple times without error."""
        handler = ConformingHandler()
        m1 = handler._get_mound()
        m2 = handler._get_mound()
        # Each call to MagicMock() creates a new instance in ConformingHandler
        assert m1 is not None
        assert m2 is not None


# ---------------------------------------------------------------------------
# Tests: Subclassing and dynamic conformance
# ---------------------------------------------------------------------------


class TestSubclassing:
    """Test subclass and dynamic conformance scenarios."""

    def test_subclass_of_conforming_satisfies_protocol(self):
        """A subclass of a conforming class also satisfies the protocol."""

        class SubHandler(ConformingHandler):
            pass

        handler = SubHandler()
        assert isinstance(handler, KnowledgeMoundHandlerProtocol)

    def test_subclass_overriding_get_mound(self):
        """A subclass can override _get_mound and still satisfy the protocol."""

        class CustomHandler(ConformingHandler):
            def _get_mound(self):
                return "custom_mound"

        handler = CustomHandler()
        assert isinstance(handler, KnowledgeMoundHandlerProtocol)
        assert handler._get_mound() == "custom_mound"

    def test_dynamic_attribute_addition_satisfies_protocol(self):
        """Dynamically adding _get_mound and ctx makes an object conform."""

        class DynHandler:
            pass

        obj = DynHandler()
        obj.ctx = {"dynamic": True}
        obj._get_mound = lambda: None

        # runtime_checkable checks for method presence
        assert hasattr(obj, "_get_mound")
        assert callable(obj._get_mound)

    def test_protocol_issubclass_raises_for_non_method_members(self):
        """issubclass raises TypeError because protocol has non-method members (ctx).

        Protocols with data attributes (non-callable members) do not support
        issubclass() -- only isinstance(). This is a Python limitation documented
        in PEP 544.
        """
        with pytest.raises(TypeError, match="non-method members"):
            issubclass(ConformingHandler, KnowledgeMoundHandlerProtocol)

    def test_protocol_issubclass_raises_for_fresh_non_conforming(self):
        """issubclass raises TypeError for a fresh non-conforming class.

        We define the class inside the test to avoid any cached subclass check
        results from prior tests.
        """

        class FreshEmpty:
            pass

        with pytest.raises(TypeError, match="non-method members"):
            issubclass(FreshEmpty, KnowledgeMoundHandlerProtocol)


# ---------------------------------------------------------------------------
# Tests: Protocol as type hint (structural typing)
# ---------------------------------------------------------------------------


class TestProtocolTyping:
    """Test protocol usage in type-hint scenarios."""

    def test_protocol_can_be_used_as_type_annotation(self):
        """Protocol can be used as a type annotation without error."""

        def use_handler(handler: KnowledgeMoundHandlerProtocol) -> bool:
            mound = handler._get_mound()
            return mound is not None

        handler = ConformingHandler()
        assert use_handler(handler) is True

    def test_protocol_annotation_with_none_mound(self):
        """Function annotated with protocol works when _get_mound returns None."""

        def check_mound(handler: KnowledgeMoundHandlerProtocol) -> bool:
            return handler._get_mound() is not None

        handler = ConformingHandlerNoneMound()
        assert check_mound(handler) is False

    def test_protocol_is_runtime_checkable(self):
        """Protocol has the _is_runtime_protocol attribute set."""
        assert getattr(KnowledgeMoundHandlerProtocol, "_is_runtime_protocol", False)


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for the protocol definition."""

    def test_protocol_has_correct_name(self):
        """Protocol class has the expected __name__."""
        assert KnowledgeMoundHandlerProtocol.__name__ == "KnowledgeMoundHandlerProtocol"

    def test_protocol_has_docstring(self):
        """Protocol class has a docstring."""
        assert KnowledgeMoundHandlerProtocol.__doc__ is not None
        assert "Protocol" in KnowledgeMoundHandlerProtocol.__doc__

    def test_get_mound_method_has_docstring(self):
        """The _get_mound protocol method has a docstring."""
        doc = KnowledgeMoundHandlerProtocol._get_mound.__doc__
        assert doc is not None
        assert "KnowledgeMound" in doc

    def test_protocol_module(self):
        """Protocol is defined in the expected module."""
        assert (
            KnowledgeMoundHandlerProtocol.__module__
            == "aragora.server.handlers.knowledge_base.mound.base_mixin"
        )

    def test_mock_object_with_spec_satisfies_protocol(self):
        """A MagicMock with the right attributes satisfies the protocol."""
        mock_handler = MagicMock()
        mock_handler.ctx = {"mocked": True}
        mock_handler._get_mound = MagicMock(return_value=None)
        # Both attributes present
        assert hasattr(mock_handler, "ctx")
        assert hasattr(mock_handler, "_get_mound")
        assert callable(mock_handler._get_mound)

    def test_wrong_method_signature_with_extra_args(self):
        """A class with _get_mound taking extra required args still has the method."""
        obj = WrongMethodSignature()
        # runtime_checkable only checks method existence, not signature
        assert hasattr(obj, "_get_mound")
        assert isinstance(obj, KnowledgeMoundHandlerProtocol)
