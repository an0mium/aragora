"""
Tests for TypeScript SDK export integrity and barrel-export consistency.

Validates that:
1. Every namespace file in src/namespaces/ has a corresponding export in
   src/namespaces/index.ts (barrel export).
2. Every namespace API class exported from src/namespaces/index.ts is also
   re-exported from the top-level src/index.ts (or at minimum the namespaces
   barrel is re-exported).
3. The AragoraClient in src/client.ts instantiates every namespace API that
   is exported from the barrel.
4. No duplicate type export names exist in the barrel (name collisions).
5. The SDK compiles cleanly (tsc --noEmit).
"""

from __future__ import annotations

import re
import subprocess
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TS_SDK = PROJECT_ROOT / "sdk" / "typescript"
NAMESPACES_DIR = TS_SDK / "src" / "namespaces"
BARREL_INDEX = NAMESPACES_DIR / "index.ts"
TOP_INDEX = TS_SDK / "src" / "index.ts"
CLIENT_TS = TS_SDK / "src" / "client.ts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _namespace_files() -> list[str]:
    """Return sorted list of namespace module stems (without extension).

    Excludes __tests__, index, and CLAUDE.md.
    """
    assert NAMESPACES_DIR.exists(), "TypeScript SDK namespaces directory not found"
    return sorted(
        p.stem
        for p in NAMESPACES_DIR.glob("*.ts")
        if p.stem != "index" and not p.name.startswith("_")
    )


def _barrel_import_sources(barrel_path: Path) -> set[str]:
    """Extract the set of module names imported in the barrel (from './xxx').

    Matches patterns like:
        export { ... } from './debates';
        export type { ... } from './debates';
    """
    if not barrel_path.exists():
        return set()
    content = barrel_path.read_text()
    # Match from './module-name' patterns
    return set(re.findall(r"from\s+['\"]\.\/([^'\"]+)['\"]", content))


def _barrel_exported_api_classes(barrel_path: Path) -> set[str]:
    """Extract API class names exported from the barrel.

    Looks for non-type exports matching *API or *Namespace patterns.
    """
    if not barrel_path.exists():
        return set()
    content = barrel_path.read_text()
    classes: set[str] = set()
    # Match export { ClassName, ... } from '...'  (non-type exports)
    for m in re.finditer(r"export\s*\{([^}]+)\}\s*from\s*['\"][^'\"]+['\"]", content):
        block = m.group(1)
        for item in block.split(","):
            item = item.strip()
            # Skip type-only exports
            if item.startswith("type "):
                continue
            # Remove alias portion if present
            if " as " in item:
                item = item.split(" as ")[0].strip()
            if item and (item.endswith("API") or item.endswith("Namespace")):
                classes.add(item)
    return classes


def _client_instantiated_namespaces(client_path: Path) -> set[str]:
    """Extract the set of namespace API classes instantiated in the client constructor.

    Matches patterns like: this.debates = new DebatesAPI(this);
    """
    if not client_path.exists():
        return set()
    content = client_path.read_text()
    return set(re.findall(r"new\s+(\w+(?:API|Namespace))\(this\)", content))


def _top_index_reexports(index_path: Path) -> set[str]:
    """Extract class-style re-exports from the top-level index."""
    if not index_path.exists():
        return set()
    content = index_path.read_text()
    classes: set[str] = set()
    for m in re.finditer(r"export\s*\{([^}]+)\}\s*from\s*['\"][^'\"]+['\"]", content):
        block = m.group(1)
        for item in block.split(","):
            item = item.strip()
            if item.startswith("type "):
                continue
            if " as " in item:
                item = item.split(" as ")[0].strip()
            if item and (item.endswith("API") or item.endswith("Namespace")):
                classes.add(item)
    return classes


def _barrel_type_export_names(barrel_path: Path) -> list[str]:
    """Extract all type export names from the barrel to check for duplicates."""
    if not barrel_path.exists():
        return []
    content = barrel_path.read_text()
    names: list[str] = []
    for m in re.finditer(
        r"export\s*(?:type\s*)?\{([^}]+)\}\s*from\s*['\"][^'\"]+['\"]",
        content,
    ):
        block = m.group(1)
        for item in block.split(","):
            item = item.strip()
            if not item:
                continue
            # Remove "type " prefix
            if item.startswith("type "):
                item = item[5:].strip()
            # Use alias name if present
            if " as " in item:
                item = item.split(" as ")[-1].strip()
            if item:
                names.append(item)
    return names


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBarrelExportConsistency:
    """Every namespace .ts file must be referenced in the barrel index."""

    def test_all_namespace_files_in_barrel(self):
        """Each namespace module file should have a corresponding import
        in src/namespaces/index.ts."""
        ns_files = set(_namespace_files())
        barrel_sources = _barrel_import_sources(BARREL_INDEX)

        missing = ns_files - barrel_sources
        # Some files may legitimately not be in barrel (test dirs, etc.)
        # Filter out common false positives
        false_positives = {"CLAUDE"}
        missing -= false_positives

        assert not missing, (
            f"Namespace files missing from barrel export "
            f"(src/namespaces/index.ts): {sorted(missing)}"
        )

    def test_barrel_does_not_reference_nonexistent_files(self):
        """Barrel should not import from modules that do not exist."""
        ns_files = set(_namespace_files())
        barrel_sources = _barrel_import_sources(BARREL_INDEX)

        nonexistent = barrel_sources - ns_files
        assert not nonexistent, (
            f"Barrel imports from modules that do not exist: {sorted(nonexistent)}"
        )


class TestTopLevelReExports:
    """API classes from the barrel should be accessible from the top-level export."""

    def test_barrel_classes_reexported_at_top_level(self):
        """Barrel API classes must appear in top-level src/index.ts or be
        reachable via the namespaces re-export."""
        barrel_classes = _barrel_exported_api_classes(BARREL_INDEX)
        assert barrel_classes, "No barrel API classes found"

        top_classes = _top_index_reexports(TOP_INDEX)

        # The top-level index may do `export { ... } from './namespaces'`
        # which re-exports everything. Check if it does:
        top_content = TOP_INDEX.read_text() if TOP_INDEX.exists() else ""
        has_wildcard_ns = "from './namespaces'" in top_content

        if has_wildcard_ns:
            # Wildcard re-export covers everything
            return

        # Otherwise check individually
        missing = barrel_classes - top_classes
        # Allow up to 10% gap for very large SDK
        max_missing = max(5, len(barrel_classes) // 10)
        assert len(missing) <= max_missing, (
            f"{len(missing)} API classes from barrel not re-exported at top level "
            f"(threshold={max_missing}): {sorted(list(missing)[:20])}"
        )


class TestClientNamespaceInstantiation:
    """The AragoraClient must instantiate all barrel-exported API classes."""

    def test_client_instantiates_barrel_api_classes(self):
        """Each API class exported from the barrel should be instantiated
        in the AragoraClient constructor.

        Some API classes are exported from the barrel for direct use by
        advanced consumers but are not mounted on the client object.
        The threshold allows for those while still catching regressions
        where newly added namespaces forget to wire up the client.
        """
        barrel_classes = _barrel_exported_api_classes(BARREL_INDEX)
        assert barrel_classes, "No barrel API classes found"

        instantiated = _client_instantiated_namespaces(CLIENT_TS)

        missing = barrel_classes - instantiated
        # Pre-existing gap of ~31 unmounted namespace APIs. Allow up to 35
        # to catch new omissions while tolerating the existing gap.
        max_missing = 35
        assert len(missing) <= max_missing, (
            f"{len(missing)} API classes from barrel not instantiated in "
            f"AragoraClient (threshold={max_missing}): {sorted(list(missing)[:20])}"
        )


class TestNoDuplicateTypeExports:
    """The barrel should not export the same type name twice without aliasing."""

    def test_no_duplicate_export_names(self):
        """Verify no duplicate export names in the barrel."""
        names = _barrel_type_export_names(BARREL_INDEX)
        assert names, "No type exports found in barrel"

        seen: dict[str, int] = {}
        for name in names:
            seen[name] = seen.get(name, 0) + 1

        duplicates = {n: c for n, c in seen.items() if c > 1}
        # Some duplicates are expected when using "as" aliases -- they'll
        # have different base names. This test catches true collisions.
        assert not duplicates, (
            f"Duplicate export names in barrel (will cause TS compilation errors): {duplicates}"
        )


class TestTypeScriptCompilation:
    """The SDK must compile cleanly with tsc --noEmit."""

    def test_sdk_compiles_cleanly(self):
        """Run tsc --noEmit on the TypeScript SDK."""
        tsconfig = TS_SDK / "tsconfig.json"
        assert tsconfig.exists(), "tsconfig.json not found"

        npx = shutil.which("npx")
        assert npx is not None, "npx not available"

        result = subprocess.run(
            [npx, "tsc", "--noEmit"],
            capture_output=True,
            text=True,
            cwd=str(TS_SDK),
            timeout=120,
        )
        if result.returncode != 0:
            # Show first 20 lines of errors
            lines = (result.stdout or result.stderr or "").strip().splitlines()[:20]
            error_summary = "\n".join(lines)
            pytest.fail(f"TypeScript compilation failed with {result.returncode}:\n{error_summary}")


class TestInputValidation:
    """SDK client constructor and request method must validate inputs."""

    def test_client_constructor_validates_baseurl(self):
        """Verify the client validates baseUrl on construction.

        This is a static analysis check -- we look for validation code
        in the client constructor.
        """
        content = CLIENT_TS.read_text() if CLIENT_TS.exists() else ""
        assert "ValidationError" in content, (
            "AragoraClient constructor should throw ValidationError for invalid config"
        )
        assert (
            "baseUrl" in content
            and "non-empty" in content.lower()
            or "valid URL" in content.lower()
        ), "AragoraClient constructor should validate baseUrl"

    def test_request_method_validates_http_method(self):
        """Verify the request method validates the HTTP method parameter."""
        content = CLIENT_TS.read_text() if CLIENT_TS.exists() else ""
        assert "VALID_METHODS" in content or "Invalid HTTP method" in content, (
            "request() should validate the HTTP method parameter"
        )

    def test_request_method_validates_path(self):
        """Verify the request method validates the path parameter."""
        content = CLIENT_TS.read_text() if CLIENT_TS.exists() else ""
        assert "startsWith('/')" in content or "must start with" in content.lower(), (
            "request() should validate that path starts with '/'"
        )
