import pytest
from aragora.interrogation.decomposer import InterrogationDecomposer, Dimension


class TestInterrogationDecomposer:
    def test_decompose_vague_prompt_returns_dimensions(self):
        decomposer = InterrogationDecomposer()
        result = decomposer.decompose("Make aragora more powerful")
        assert len(result.dimensions) >= 2
        assert all(isinstance(d, Dimension) for d in result.dimensions)

    def test_decompose_specific_prompt_returns_fewer_dimensions(self):
        decomposer = InterrogationDecomposer()
        result = decomposer.decompose("Fix the login button color")
        assert len(result.dimensions) <= 3

    def test_dimension_has_required_fields(self):
        decomposer = InterrogationDecomposer()
        result = decomposer.decompose("Improve test coverage")
        dim = result.dimensions[0]
        assert dim.name
        assert dim.description
        assert dim.vagueness_score >= 0.0
        assert dim.vagueness_score <= 1.0

    def test_decompose_empty_prompt_raises(self):
        decomposer = InterrogationDecomposer()
        with pytest.raises(ValueError, match="empty"):
            decomposer.decompose("")

    def test_decomposition_includes_original_prompt(self):
        decomposer = InterrogationDecomposer()
        result = decomposer.decompose("Add dark mode")
        assert result.original_prompt == "Add dark mode"
