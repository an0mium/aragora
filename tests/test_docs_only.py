"""Unit tests for docs-only scope helpers."""

from pathlib import Path

from aragora.docs_only import (
    canonical_docs_container_scope,
    infer_docs_safe_hints,
    is_docs_safe_path,
    is_docs_safe_top_level_file,
    normalize_docs_path,
)


def test_normalize_docs_path_normalizes_slashes_and_relative_prefixes():
    assert normalize_docs_path(r" .\docs\guide\intro.md/ ") == "docs/guide/intro.md"


def test_normalize_docs_path_accepts_pathlike_inputs():
    assert normalize_docs_path(Path("./docs-site/reference/")) == "docs-site/reference"


def test_canonical_docs_container_scope_recognizes_docs_variants():
    assert canonical_docs_container_scope("docs") == "docs"
    assert canonical_docs_container_scope("./docs/**/") == "docs"


def test_canonical_docs_container_scope_recognizes_docs_site_variants():
    assert canonical_docs_container_scope("docs-site") == "docs-site"
    assert canonical_docs_container_scope("./docs-site/**") == "docs-site"


def test_canonical_docs_container_scope_rejects_non_container_paths():
    assert canonical_docs_container_scope("docs/guide.md") is None
    assert canonical_docs_container_scope("docs-site-assets") is None


def test_is_docs_safe_top_level_file_accepts_allowlisted_filenames():
    assert is_docs_safe_top_level_file("CHANGELOG.md") is True
    assert is_docs_safe_top_level_file("LICENSE") is True


def test_is_docs_safe_top_level_file_rejects_nested_or_unlisted_files():
    assert is_docs_safe_top_level_file("README.md") is False
    assert is_docs_safe_top_level_file("docs/CHANGELOG.md") is False


def test_is_docs_safe_path_accepts_docs_containers_prefixes_and_top_level_files():
    assert is_docs_safe_path("docs") is True
    assert is_docs_safe_path("./docs-site/guides/getting-started.md") is True
    assert is_docs_safe_path("CONTRIBUTING.md") is True


def test_is_docs_safe_path_rejects_non_docs_paths_and_partial_prefixes():
    assert is_docs_safe_path("") is False
    assert is_docs_safe_path("src/docs_only.py") is False
    assert is_docs_safe_path("docs-site-assets/logo.svg") is False


def test_infer_docs_safe_hints_extracts_normalized_docs_tokens_from_text():
    text = "Touch `./docs/guide.md`, (docs-site/reference/), CHANGELOG.md, and src/app.py."

    assert infer_docs_safe_hints(text) == [
        "docs/guide.md",
        "docs-site/reference",
        "CHANGELOG.md",
    ]


def test_infer_docs_safe_hints_deduplicates_preserving_first_seen_order():
    text = "docs/guide.md `./docs/guide.md` docs-site/reference docs-site/reference/"

    assert infer_docs_safe_hints(text) == ["docs/guide.md", "docs-site/reference"]


def test_infer_docs_safe_hints_ignores_non_docs_tokens():
    text = "src/app.py README.md docs-site-assets/logo.svg"

    assert infer_docs_safe_hints(text) == []
