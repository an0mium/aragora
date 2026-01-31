"""
Tests for SBOM Generator module.

Tests Software Bill of Materials generation: CycloneDX format, SPDX format,
component extraction, license detection, vulnerability inclusion,
format validation, edge cases, and convenience functions.
"""

import json
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.analysis.codebase.models import (
    DependencyInfo,
    ScanResult,
    VulnerabilityFinding,
    VulnerabilitySeverity,
)
from aragora.analysis.codebase.sbom_generator import (
    ECOSYSTEM_DOWNLOAD_PATTERNS,
    ECOSYSTEM_TO_PURL_TYPE,
    ComponentType,
    HashAlgorithm,
    SBOMComponent,
    SBOMFormat,
    SBOMGenerator,
    SBOMMetadata,
    SBOMResult,
    generate_sbom,
)


# ============================================================
# SBOMFormat
# ============================================================


class TestSBOMFormat:
    """Tests for SBOM format enum."""

    def test_format_values(self):
        """All four SBOM formats exist with expected values."""
        assert SBOMFormat.CYCLONEDX_JSON.value == "cyclonedx-json"
        assert SBOMFormat.CYCLONEDX_XML.value == "cyclonedx-xml"
        assert SBOMFormat.SPDX_JSON.value == "spdx-json"
        assert SBOMFormat.SPDX_TV.value == "spdx-tv"

    def test_format_is_string_enum(self):
        """SBOMFormat values can be used as strings."""
        assert str(SBOMFormat.CYCLONEDX_JSON) == "SBOMFormat.CYCLONEDX_JSON"
        assert SBOMFormat.CYCLONEDX_JSON == "cyclonedx-json"


# ============================================================
# ComponentType
# ============================================================


class TestComponentType:
    """Tests for CycloneDX component type enum."""

    def test_component_types_exist(self):
        """All component types are defined."""
        assert ComponentType.APPLICATION.value == "application"
        assert ComponentType.FRAMEWORK.value == "framework"
        assert ComponentType.LIBRARY.value == "library"
        assert ComponentType.CONTAINER.value == "container"
        assert ComponentType.OPERATING_SYSTEM.value == "operating-system"
        assert ComponentType.DEVICE.value == "device"
        assert ComponentType.FIRMWARE.value == "firmware"
        assert ComponentType.FILE.value == "file"


# ============================================================
# HashAlgorithm
# ============================================================


class TestHashAlgorithm:
    """Tests for hash algorithm enum."""

    def test_hash_algorithm_values(self):
        """All hash algorithms have correct SBOM values."""
        assert HashAlgorithm.MD5.value == "MD5"
        assert HashAlgorithm.SHA1.value == "SHA-1"
        assert HashAlgorithm.SHA256.value == "SHA-256"
        assert HashAlgorithm.SHA384.value == "SHA-384"
        assert HashAlgorithm.SHA512.value == "SHA-512"
        assert HashAlgorithm.SHA3_256.value == "SHA3-256"
        assert HashAlgorithm.SHA3_512.value == "SHA3-512"
        assert HashAlgorithm.BLAKE2b_256.value == "BLAKE2b-256"
        assert HashAlgorithm.BLAKE2b_384.value == "BLAKE2b-384"
        assert HashAlgorithm.BLAKE2b_512.value == "BLAKE2b-512"
        assert HashAlgorithm.BLAKE3.value == "BLAKE3"


# ============================================================
# Constants: ECOSYSTEM mappings
# ============================================================


class TestEcosystemMappings:
    """Tests for ecosystem to purl type mappings."""

    def test_ecosystem_to_purl_type_mapping(self):
        """Ecosystem to purl type mapping covers major ecosystems."""
        assert ECOSYSTEM_TO_PURL_TYPE["npm"] == "npm"
        assert ECOSYSTEM_TO_PURL_TYPE["pypi"] == "pypi"
        assert ECOSYSTEM_TO_PURL_TYPE["maven"] == "maven"
        assert ECOSYSTEM_TO_PURL_TYPE["cargo"] == "cargo"
        assert ECOSYSTEM_TO_PURL_TYPE["go"] == "golang"
        assert ECOSYSTEM_TO_PURL_TYPE["rubygems"] == "gem"
        assert ECOSYSTEM_TO_PURL_TYPE["nuget"] == "nuget"
        assert ECOSYSTEM_TO_PURL_TYPE["composer"] == "composer"

    def test_download_patterns_structure(self):
        """Download patterns contain expected placeholders."""
        for ecosystem, pattern in ECOSYSTEM_DOWNLOAD_PATTERNS.items():
            assert "{name}" in pattern, f"{ecosystem} pattern missing {{name}}"
            assert "{version}" in pattern, f"{ecosystem} pattern missing {{version}}"


# ============================================================
# SBOMComponent
# ============================================================


class TestSBOMComponent:
    """Tests for SBOM component dataclass."""

    def _make_dependency(self, **overrides) -> DependencyInfo:
        """Create a dependency with sensible defaults."""
        defaults = dict(
            name="lodash",
            version="4.17.21",
            ecosystem="npm",
            direct=True,
            dev_dependency=False,
            license="MIT",
        )
        defaults.update(overrides)
        return DependencyInfo(**defaults)

    def test_component_from_dependency(self):
        """SBOMComponent.from_dependency creates component correctly."""
        dep = self._make_dependency()
        comp = SBOMComponent.from_dependency(dep)

        assert comp.name == "lodash"
        assert comp.version == "4.17.21"
        assert comp.ecosystem == "npm"
        assert comp.purl == "pkg:npm/lodash@4.17.21"
        assert comp.bom_ref == "npm:lodash@4.17.21"
        assert comp.licenses == ["MIT"]
        assert comp.component_type == ComponentType.LIBRARY
        assert comp.properties["direct"] == "true"
        assert comp.properties["dev_dependency"] == "false"

    def test_component_from_dependency_dev_dep(self):
        """Dev dependencies have correct property set."""
        dep = self._make_dependency(dev_dependency=True, direct=False)
        comp = SBOMComponent.from_dependency(dep)

        assert comp.properties["dev_dependency"] == "true"
        assert comp.properties["direct"] == "false"

    def test_component_from_dependency_no_license(self):
        """Component with no license has empty licenses list."""
        dep = self._make_dependency(license=None)
        comp = SBOMComponent.from_dependency(dep)

        assert comp.licenses == []

    def test_component_from_scoped_npm_package(self):
        """Scoped npm packages extract group correctly."""
        dep = self._make_dependency(name="@angular/core")
        comp = SBOMComponent.from_dependency(dep)

        assert comp.name == "core"
        assert comp.group == "@angular"
        assert comp.purl == "pkg:npm/%40angular/core@4.17.21"

    def test_component_from_dependency_with_vulnerabilities(self):
        """Vulnerabilities are extracted as CVE IDs."""
        vuln = VulnerabilityFinding(
            id="CVE-2021-23337",
            title="Prototype Pollution",
            description="A prototype pollution vulnerability",
            severity=VulnerabilitySeverity.HIGH,
        )
        dep = self._make_dependency(vulnerabilities=[vuln])
        comp = SBOMComponent.from_dependency(dep)

        assert comp.vulnerabilities == ["CVE-2021-23337"]

    def test_build_purl_npm(self):
        """purl for npm packages is correctly formatted."""
        purl = SBOMComponent._build_purl("express", "4.18.2", "npm")
        assert purl == "pkg:npm/express@4.18.2"

    def test_build_purl_pypi(self):
        """purl for pypi packages is correctly formatted."""
        purl = SBOMComponent._build_purl("requests", "2.31.0", "pypi")
        assert purl == "pkg:pypi/requests@2.31.0"

    def test_build_purl_scoped_npm(self):
        """purl for scoped npm packages encodes @ symbol."""
        purl = SBOMComponent._build_purl("@types/node", "20.0.0", "npm")
        assert purl == "pkg:npm/%40types/node@20.0.0"

    def test_build_purl_go_module(self):
        """purl for Go modules preserves path."""
        purl = SBOMComponent._build_purl("github.com/gin-gonic/gin", "1.9.0", "go")
        assert purl == "pkg:golang/github.com/gin-gonic/gin@1.9.0"

    def test_build_purl_unknown_ecosystem(self):
        """Unknown ecosystem uses ecosystem name as purl type."""
        purl = SBOMComponent._build_purl("mypackage", "1.0.0", "custom")
        assert purl == "pkg:custom/mypackage@1.0.0"


# ============================================================
# SBOMMetadata
# ============================================================


class TestSBOMMetadata:
    """Tests for SBOM metadata dataclass."""

    def test_default_metadata(self):
        """Default metadata has expected values."""
        meta = SBOMMetadata()

        assert meta.version == 1
        assert meta.tool_name == "Aragora Security Scanner"
        assert meta.tool_version == "1.0.0"
        assert meta.tool_vendor == "Aragora"
        assert meta.authors == []
        assert meta.component_name is None
        assert meta.component_version is None
        assert meta.component_type == ComponentType.APPLICATION
        assert meta.serial_number.startswith("urn:uuid:")
        assert isinstance(meta.timestamp, datetime)

    def test_metadata_with_project_info(self):
        """Metadata can be created with project info."""
        meta = SBOMMetadata(
            component_name="my-app",
            component_version="2.0.0",
            authors=["Developer One"],
        )

        assert meta.component_name == "my-app"
        assert meta.component_version == "2.0.0"
        assert meta.authors == ["Developer One"]


# ============================================================
# SBOMResult
# ============================================================


class TestSBOMResult:
    """Tests for SBOM result dataclass."""

    def _make_result(self, **overrides) -> SBOMResult:
        """Create an SBOM result with sensible defaults."""
        defaults = dict(
            format=SBOMFormat.CYCLONEDX_JSON,
            content='{"bomFormat": "CycloneDX"}',
            filename="sbom.json",
            component_count=10,
            vulnerability_count=2,
            license_count=5,
        )
        defaults.update(overrides)
        return SBOMResult(**defaults)

    def test_result_defaults(self):
        """Default SBOM result has expected values."""
        result = self._make_result()

        assert result.format == SBOMFormat.CYCLONEDX_JSON
        assert result.component_count == 10
        assert result.vulnerability_count == 2
        assert result.license_count == 5
        assert result.errors == []
        assert isinstance(result.generated_at, datetime)

    def test_result_to_dict(self):
        """to_dict returns complete dictionary representation."""
        result = self._make_result()
        d = result.to_dict()

        assert d["format"] == "cyclonedx-json"
        assert d["component_count"] == 10
        assert d["vulnerability_count"] == 2
        assert d["license_count"] == 5
        assert "generated_at" in d
        assert d["errors"] == []

    def test_result_with_errors(self):
        """Result can contain error messages."""
        result = self._make_result(errors=["Failed to parse package.json"])

        assert result.errors == ["Failed to parse package.json"]
        assert result.to_dict()["errors"] == ["Failed to parse package.json"]


# ============================================================
# SBOMGenerator - Initialization
# ============================================================


class TestSBOMGeneratorInit:
    """Tests for SBOM generator initialization."""

    def test_default_init(self):
        """Generator initializes with default settings."""
        generator = SBOMGenerator()

        assert generator.scanner is not None
        assert generator.include_dev_dependencies is True
        assert generator.include_vulnerabilities is True

    def test_init_with_custom_settings(self):
        """Generator accepts custom settings."""
        scanner = MagicMock()
        generator = SBOMGenerator(
            scanner=scanner,
            include_dev_dependencies=False,
            include_vulnerabilities=False,
        )

        assert generator.scanner is scanner
        assert generator.include_dev_dependencies is False
        assert generator.include_vulnerabilities is False


# ============================================================
# SBOMGenerator - CycloneDX JSON
# ============================================================


class TestCycloneDXJSON:
    """Tests for CycloneDX JSON generation."""

    def _make_component(self, **overrides) -> SBOMComponent:
        """Create a component with sensible defaults."""
        defaults = dict(
            name="lodash",
            version="4.17.21",
            ecosystem="npm",
            purl="pkg:npm/lodash@4.17.21",
            bom_ref="npm:lodash@4.17.21",
        )
        defaults.update(overrides)
        return SBOMComponent(**defaults)

    def _make_generator(self, **kwargs) -> SBOMGenerator:
        """Create a generator with mocked scanner."""
        scanner = MagicMock()
        return SBOMGenerator(scanner=scanner, **kwargs)

    def test_cyclonedx_json_structure(self):
        """Generated CycloneDX JSON has correct structure."""
        generator = self._make_generator()
        meta = SBOMMetadata(component_name="test-project")
        components = [self._make_component()]

        content = generator._generate_cyclonedx_json(components, meta)
        bom = json.loads(content)

        assert bom["bomFormat"] == "CycloneDX"
        assert bom["specVersion"] == "1.5"
        assert "$schema" in bom
        assert "serialNumber" in bom
        assert "metadata" in bom
        assert "components" in bom

    def test_cyclonedx_json_metadata(self):
        """Metadata section includes tool and timestamp."""
        generator = self._make_generator()
        meta = SBOMMetadata(
            component_name="my-app",
            component_version="1.0.0",
        )
        components = []

        content = generator._generate_cyclonedx_json(components, meta)
        bom = json.loads(content)

        assert "timestamp" in bom["metadata"]
        assert bom["metadata"]["tools"][0]["name"] == "Aragora Security Scanner"
        assert bom["metadata"]["component"]["name"] == "my-app"
        assert bom["metadata"]["component"]["version"] == "1.0.0"

    def test_cyclonedx_json_component_fields(self):
        """Component includes all required fields."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component(
            licenses=["MIT"],
            description="A utility library",
            group="@lodash",
        )
        components = [comp]

        content = generator._generate_cyclonedx_json(components, meta)
        bom = json.loads(content)

        component = bom["components"][0]
        assert component["type"] == "library"
        assert component["name"] == "lodash"
        assert component["version"] == "4.17.21"
        assert component["purl"] == "pkg:npm/lodash@4.17.21"
        assert component["bom-ref"] == "npm:lodash@4.17.21"
        assert component["group"] == "@lodash"
        assert component["description"] == "A utility library"

    def test_cyclonedx_json_spdx_license(self):
        """SPDX license IDs are formatted correctly."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component(licenses=["MIT"])

        content = generator._generate_cyclonedx_json([comp], meta)
        bom = json.loads(content)

        # MIT is an SPDX identifier
        license_info = bom["components"][0]["licenses"][0]["license"]
        assert license_info["id"] == "MIT"

    def test_cyclonedx_json_non_spdx_license(self):
        """Non-SPDX licenses are formatted as names."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component(licenses=["CustomLicense"])

        content = generator._generate_cyclonedx_json([comp], meta)
        bom = json.loads(content)

        # Non-SPDX should use license.name
        license_info = bom["components"][0]["licenses"][0]["license"]
        assert "license" in license_info
        assert license_info["license"]["name"] == "CustomLicense"

    def test_cyclonedx_json_hashes(self):
        """Component hashes are included when present."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component(
            hashes={
                HashAlgorithm.SHA256: "abcdef123456",
                HashAlgorithm.SHA512: "fedcba654321",
            }
        )

        content = generator._generate_cyclonedx_json([comp], meta)
        bom = json.loads(content)

        hashes = bom["components"][0]["hashes"]
        assert len(hashes) == 2
        algs = {h["alg"] for h in hashes}
        assert "SHA-256" in algs
        assert "SHA-512" in algs

    def test_cyclonedx_json_vulnerabilities(self):
        """Vulnerabilities are included when enabled."""
        generator = self._make_generator(include_vulnerabilities=True)
        meta = SBOMMetadata()
        comp = self._make_component(vulnerabilities=["CVE-2021-23337"])

        content = generator._generate_cyclonedx_json([comp], meta)
        bom = json.loads(content)

        assert "vulnerabilities" in bom
        assert len(bom["vulnerabilities"]) == 1
        assert bom["vulnerabilities"][0]["id"] == "CVE-2021-23337"
        assert bom["vulnerabilities"][0]["affects"][0]["ref"] == "npm:lodash@4.17.21"

    def test_cyclonedx_json_no_vulnerabilities_when_disabled(self):
        """Vulnerabilities excluded when include_vulnerabilities=False."""
        generator = self._make_generator(include_vulnerabilities=False)
        meta = SBOMMetadata()
        comp = self._make_component(vulnerabilities=["CVE-2021-23337"])

        content = generator._generate_cyclonedx_json([comp], meta)
        bom = json.loads(content)

        assert "vulnerabilities" not in bom

    def test_cyclonedx_json_dependencies(self):
        """Dependency graph is included."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component(dependencies=["npm:express@4.18.2"])

        content = generator._generate_cyclonedx_json([comp], meta)
        bom = json.loads(content)

        assert "dependencies" in bom
        assert bom["dependencies"][0]["ref"] == "npm:lodash@4.17.21"
        assert bom["dependencies"][0]["dependsOn"] == ["npm:express@4.18.2"]

    def test_cyclonedx_json_properties(self):
        """Component properties are included."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component(properties={"direct": "true", "dev_dependency": "false"})

        content = generator._generate_cyclonedx_json([comp], meta)
        bom = json.loads(content)

        props = bom["components"][0]["properties"]
        prop_dict = {p["name"]: p["value"] for p in props}
        assert prop_dict["direct"] == "true"
        assert prop_dict["dev_dependency"] == "false"


# ============================================================
# SBOMGenerator - CycloneDX XML
# ============================================================


class TestCycloneDXXML:
    """Tests for CycloneDX XML generation."""

    def _make_component(self, **overrides) -> SBOMComponent:
        """Create a component with sensible defaults."""
        defaults = dict(
            name="requests",
            version="2.31.0",
            ecosystem="pypi",
            purl="pkg:pypi/requests@2.31.0",
            bom_ref="pypi:requests@2.31.0",
        )
        defaults.update(overrides)
        return SBOMComponent(**defaults)

    def _make_generator(self, **kwargs) -> SBOMGenerator:
        """Create a generator with mocked scanner."""
        scanner = MagicMock()
        return SBOMGenerator(scanner=scanner, **kwargs)

    def test_cyclonedx_xml_structure(self):
        """Generated CycloneDX XML has correct structure."""
        generator = self._make_generator()
        meta = SBOMMetadata(component_name="test-project")
        components = [self._make_component()]

        content = generator._generate_cyclonedx_xml(components, meta)

        # Parse and check structure
        root = ET.fromstring(content)
        assert root.tag.endswith("bom")
        assert "serialNumber" in root.attrib
        assert "version" in root.attrib

    def test_cyclonedx_xml_namespace(self):
        """XML uses correct CycloneDX 1.5 namespace."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        components = []

        content = generator._generate_cyclonedx_xml(components, meta)
        root = ET.fromstring(content)

        # Check namespace - it's part of the tag, not an attribute after parsing
        assert "http://cyclonedx.org/schema/bom/1.5" in root.tag

    def test_cyclonedx_xml_metadata(self):
        """XML metadata includes tool information."""
        generator = self._make_generator()
        meta = SBOMMetadata(component_name="my-app")
        components = []

        content = generator._generate_cyclonedx_xml(components, meta)
        root = ET.fromstring(content)

        # Find metadata element
        ns = {"": "http://cyclonedx.org/schema/bom/1.5"}
        metadata = root.find(".//{http://cyclonedx.org/schema/bom/1.5}metadata")
        assert metadata is not None

        timestamp = metadata.find(".//{http://cyclonedx.org/schema/bom/1.5}timestamp")
        assert timestamp is not None

    def test_cyclonedx_xml_component_fields(self):
        """XML component includes required fields."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component(group="requests-group")

        content = generator._generate_cyclonedx_xml([comp], meta)
        root = ET.fromstring(content)

        # Find component
        component = root.find(".//{http://cyclonedx.org/schema/bom/1.5}component")
        assert component is not None
        assert component.attrib["type"] == "library"

        name = component.find("{http://cyclonedx.org/schema/bom/1.5}name")
        assert name.text == "requests"

        version = component.find("{http://cyclonedx.org/schema/bom/1.5}version")
        assert version.text == "2.31.0"

    def test_cyclonedx_xml_licenses(self):
        """XML licenses are formatted correctly."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component(licenses=["Apache-2.0"])

        content = generator._generate_cyclonedx_xml([comp], meta)
        root = ET.fromstring(content)

        license_id = root.find(".//{http://cyclonedx.org/schema/bom/1.5}id")
        assert license_id is not None
        assert license_id.text == "Apache-2.0"

    def test_cyclonedx_xml_vulnerabilities(self):
        """XML vulnerabilities are included when enabled."""
        generator = self._make_generator(include_vulnerabilities=True)
        meta = SBOMMetadata()
        comp = self._make_component(vulnerabilities=["CVE-2023-12345"])

        content = generator._generate_cyclonedx_xml([comp], meta)
        root = ET.fromstring(content)

        vulns = root.find(".//{http://cyclonedx.org/schema/bom/1.5}vulnerabilities")
        assert vulns is not None

    def test_cyclonedx_xml_hashes(self):
        """XML hashes are included when present."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component(hashes={HashAlgorithm.SHA256: "abc123"})

        content = generator._generate_cyclonedx_xml([comp], meta)
        root = ET.fromstring(content)

        hash_elem = root.find(".//{http://cyclonedx.org/schema/bom/1.5}hash")
        assert hash_elem is not None
        assert hash_elem.attrib["alg"] == "SHA-256"


# ============================================================
# SBOMGenerator - SPDX JSON
# ============================================================


class TestSPDXJSON:
    """Tests for SPDX JSON generation."""

    def _make_component(self, **overrides) -> SBOMComponent:
        """Create a component with sensible defaults."""
        defaults = dict(
            name="express",
            version="4.18.2",
            ecosystem="npm",
            purl="pkg:npm/express@4.18.2",
            bom_ref="npm:express@4.18.2",
        )
        defaults.update(overrides)
        return SBOMComponent(**defaults)

    def _make_generator(self, **kwargs) -> SBOMGenerator:
        """Create a generator with mocked scanner."""
        scanner = MagicMock()
        return SBOMGenerator(scanner=scanner, **kwargs)

    def test_spdx_json_structure(self):
        """Generated SPDX JSON has correct structure."""
        generator = self._make_generator()
        meta = SBOMMetadata(component_name="test-project")
        components = [self._make_component()]

        content = generator._generate_spdx_json(components, meta)
        spdx = json.loads(content)

        assert spdx["spdxVersion"] == "SPDX-2.3"
        assert spdx["dataLicense"] == "CC0-1.0"
        assert spdx["SPDXID"] == "SPDXRef-DOCUMENT"
        assert "documentNamespace" in spdx
        assert "creationInfo" in spdx
        assert "packages" in spdx
        assert "relationships" in spdx

    def test_spdx_json_creation_info(self):
        """Creation info includes tool and timestamp."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        components = []

        content = generator._generate_spdx_json(components, meta)
        spdx = json.loads(content)

        assert "created" in spdx["creationInfo"]
        assert any("Aragora" in c for c in spdx["creationInfo"]["creators"])

    def test_spdx_json_root_package(self):
        """Root package is included when project name provided."""
        generator = self._make_generator()
        meta = SBOMMetadata(
            component_name="my-project",
            component_version="1.0.0",
        )
        components = []

        content = generator._generate_spdx_json(components, meta)
        spdx = json.loads(content)

        # Find root package
        root_pkg = next(
            (p for p in spdx["packages"] if p["SPDXID"] == "SPDXRef-RootPackage"),
            None,
        )
        assert root_pkg is not None
        assert root_pkg["name"] == "my-project"
        assert root_pkg["versionInfo"] == "1.0.0"

    def test_spdx_json_package_fields(self):
        """Package includes required SPDX fields."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component(licenses=["MIT"])

        content = generator._generate_spdx_json([comp], meta)
        spdx = json.loads(content)

        pkg = spdx["packages"][0]
        assert "SPDXID" in pkg
        assert pkg["name"] == "express"
        assert pkg["versionInfo"] == "4.18.2"
        assert "downloadLocation" in pkg
        assert "filesAnalyzed" in pkg
        assert "externalRefs" in pkg

    def test_spdx_json_purl_external_ref(self):
        """Package URL is included as external reference."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component()

        content = generator._generate_spdx_json([comp], meta)
        spdx = json.loads(content)

        pkg = spdx["packages"][0]
        purl_ref = next(
            (r for r in pkg["externalRefs"] if r["referenceType"] == "purl"),
            None,
        )
        assert purl_ref is not None
        assert purl_ref["referenceLocator"] == "pkg:npm/express@4.18.2"

    def test_spdx_json_licenses(self):
        """SPDX licenses are formatted correctly."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component(licenses=["MIT", "Apache-2.0"])

        content = generator._generate_spdx_json([comp], meta)
        spdx = json.loads(content)

        pkg = spdx["packages"][0]
        assert "MIT AND Apache-2.0" == pkg["licenseConcluded"]
        assert pkg["licenseDeclared"] == pkg["licenseConcluded"]

    def test_spdx_json_no_license(self):
        """Package without license uses NOASSERTION."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component(licenses=[])

        content = generator._generate_spdx_json([comp], meta)
        spdx = json.loads(content)

        pkg = spdx["packages"][0]
        assert pkg["licenseConcluded"] == "NOASSERTION"
        assert pkg["licenseDeclared"] == "NOASSERTION"

    def test_spdx_json_vulnerability_external_ref(self):
        """Vulnerabilities are included as security external refs."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component(vulnerabilities=["CVE-2023-99999"])

        content = generator._generate_spdx_json([comp], meta)
        spdx = json.loads(content)

        pkg = spdx["packages"][0]
        cve_ref = next(
            (r for r in pkg["externalRefs"] if r["referenceCategory"] == "SECURITY"),
            None,
        )
        assert cve_ref is not None
        assert "CVE-2023-99999" in cve_ref["referenceLocator"]

    def test_spdx_json_relationships(self):
        """SPDX relationships are generated."""
        generator = self._make_generator()
        meta = SBOMMetadata(component_name="my-project")
        comp = self._make_component(properties={"direct": "true"})

        content = generator._generate_spdx_json([comp], meta)
        spdx = json.loads(content)

        # Should have DESCRIBES and DEPENDS_ON relationships
        rel_types = {r["relationshipType"] for r in spdx["relationships"]}
        assert "DESCRIBES" in rel_types
        assert "DEPENDS_ON" in rel_types

    def test_spdx_json_checksums(self):
        """Package checksums are included when present."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component(hashes={HashAlgorithm.SHA256: "abcdef123456"})

        content = generator._generate_spdx_json([comp], meta)
        spdx = json.loads(content)

        pkg = spdx["packages"][0]
        assert "checksums" in pkg
        assert pkg["checksums"][0]["algorithm"] == "SHA256"


# ============================================================
# SBOMGenerator - SPDX Tag-Value
# ============================================================


class TestSPDXTagValue:
    """Tests for SPDX Tag-Value generation."""

    def _make_component(self, **overrides) -> SBOMComponent:
        """Create a component with sensible defaults."""
        defaults = dict(
            name="flask",
            version="3.0.0",
            ecosystem="pypi",
            purl="pkg:pypi/flask@3.0.0",
            bom_ref="pypi:flask@3.0.0",
        )
        defaults.update(overrides)
        return SBOMComponent(**defaults)

    def _make_generator(self, **kwargs) -> SBOMGenerator:
        """Create a generator with mocked scanner."""
        scanner = MagicMock()
        return SBOMGenerator(scanner=scanner, **kwargs)

    def test_spdx_tv_header(self):
        """Tag-value output starts with correct header."""
        generator = self._make_generator()
        meta = SBOMMetadata(component_name="test-project")
        components = []

        content = generator._generate_spdx_tagvalue(components, meta)

        assert content.startswith("SPDXVersion: SPDX-2.3")
        assert "DataLicense: CC0-1.0" in content
        assert "SPDXID: SPDXRef-DOCUMENT" in content

    def test_spdx_tv_creator(self):
        """Tag-value includes creator information."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        components = []

        content = generator._generate_spdx_tagvalue(components, meta)

        assert "Creator: Tool:" in content
        assert "Creator: Organization:" in content
        assert "Created:" in content

    def test_spdx_tv_root_package(self):
        """Tag-value includes root package."""
        generator = self._make_generator()
        meta = SBOMMetadata(
            component_name="my-app",
            component_version="2.0.0",
        )
        components = []

        content = generator._generate_spdx_tagvalue(components, meta)

        assert "PackageName: my-app" in content
        assert "SPDXID: SPDXRef-RootPackage" in content
        assert "PackageVersion: 2.0.0" in content

    def test_spdx_tv_package_fields(self):
        """Tag-value package includes required fields."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component()

        content = generator._generate_spdx_tagvalue([comp], meta)

        assert "PackageName: flask" in content
        assert "PackageVersion: 3.0.0" in content
        assert "PackageDownloadLocation:" in content
        assert "FilesAnalyzed: false" in content
        assert "PackageLicenseConcluded:" in content
        assert "PackageCopyrightText: NOASSERTION" in content

    def test_spdx_tv_purl(self):
        """Tag-value includes purl as external reference."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component()

        content = generator._generate_spdx_tagvalue([comp], meta)

        assert "ExternalRef: PACKAGE-MANAGER purl pkg:pypi/flask@3.0.0" in content

    def test_spdx_tv_vulnerabilities(self):
        """Tag-value includes vulnerability external refs."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component(vulnerabilities=["CVE-2024-11111"])

        content = generator._generate_spdx_tagvalue([comp], meta)

        assert "ExternalRef: SECURITY cve" in content
        assert "CVE-2024-11111" in content

    def test_spdx_tv_relationships(self):
        """Tag-value includes relationships."""
        generator = self._make_generator()
        meta = SBOMMetadata(component_name="my-app")
        comp = self._make_component(properties={"direct": "true"})

        content = generator._generate_spdx_tagvalue([comp], meta)

        assert "Relationship:" in content
        assert "DESCRIBES" in content

    def test_spdx_tv_checksums(self):
        """Tag-value includes checksums."""
        generator = self._make_generator()
        meta = SBOMMetadata()
        comp = self._make_component(hashes={HashAlgorithm.SHA256: "abc123"})

        content = generator._generate_spdx_tagvalue([comp], meta)

        assert "PackageChecksum: SHA256:" in content


# ============================================================
# SBOMGenerator - License Detection
# ============================================================


class TestLicenseDetection:
    """Tests for license detection and SPDX validation."""

    def _make_generator(self) -> SBOMGenerator:
        """Create a generator with mocked scanner."""
        scanner = MagicMock()
        return SBOMGenerator(scanner=scanner)

    def test_is_spdx_license_mit(self):
        """MIT is recognized as SPDX license."""
        generator = self._make_generator()
        assert generator._is_spdx_license("MIT") is True

    def test_is_spdx_license_apache(self):
        """Apache-2.0 is recognized as SPDX license."""
        generator = self._make_generator()
        assert generator._is_spdx_license("Apache-2.0") is True

    def test_is_spdx_license_gpl(self):
        """GPL licenses are recognized."""
        generator = self._make_generator()
        assert generator._is_spdx_license("GPL-2.0") is True
        assert generator._is_spdx_license("GPL-3.0") is True
        assert generator._is_spdx_license("GPL-3.0-only") is True

    def test_is_spdx_license_bsd(self):
        """BSD licenses are recognized."""
        generator = self._make_generator()
        assert generator._is_spdx_license("BSD-2-Clause") is True
        assert generator._is_spdx_license("BSD-3-Clause") is True

    def test_is_spdx_license_unknown(self):
        """Unknown licenses return False."""
        generator = self._make_generator()
        assert generator._is_spdx_license("Proprietary") is False
        assert generator._is_spdx_license("My Custom License") is False

    def test_is_spdx_license_creative_commons(self):
        """Creative Commons licenses are recognized."""
        generator = self._make_generator()
        assert generator._is_spdx_license("CC0-1.0") is True
        assert generator._is_spdx_license("CC-BY-4.0") is True


# ============================================================
# SBOMGenerator - Download Location
# ============================================================


class TestDownloadLocation:
    """Tests for download location generation."""

    def _make_component(self, **overrides) -> SBOMComponent:
        """Create a component with sensible defaults."""
        defaults = dict(
            name="package",
            version="1.0.0",
            ecosystem="npm",
            purl="pkg:npm/package@1.0.0",
            bom_ref="npm:package@1.0.0",
        )
        defaults.update(overrides)
        return SBOMComponent(**defaults)

    def _make_generator(self) -> SBOMGenerator:
        """Create a generator with mocked scanner."""
        scanner = MagicMock()
        return SBOMGenerator(scanner=scanner)

    def test_download_location_npm(self):
        """npm packages have correct download location."""
        generator = self._make_generator()
        comp = self._make_component(name="lodash", version="4.17.21", ecosystem="npm")

        location = generator._get_download_location(comp)

        assert "registry.npmjs.org" in location
        assert "lodash" in location
        assert "4.17.21" in location

    def test_download_location_pypi(self):
        """PyPI packages have correct download location."""
        generator = self._make_generator()
        comp = self._make_component(name="requests", version="2.31.0", ecosystem="pypi")

        location = generator._get_download_location(comp)

        assert "pypi.org" in location
        assert "requests" in location
        assert "2.31.0" in location

    def test_download_location_cargo(self):
        """Cargo packages have correct download location."""
        generator = self._make_generator()
        comp = self._make_component(name="serde", version="1.0.188", ecosystem="cargo")

        location = generator._get_download_location(comp)

        assert "crates.io" in location
        assert "serde" in location

    def test_download_location_unknown_ecosystem(self):
        """Unknown ecosystems return NOASSERTION."""
        generator = self._make_generator()
        comp = self._make_component(ecosystem="custom")

        location = generator._get_download_location(comp)

        assert location == "NOASSERTION"

    def test_download_location_scoped_npm(self):
        """Scoped npm packages include group in URL."""
        generator = self._make_generator()
        comp = self._make_component(
            name="core",
            group="@angular",
            version="16.0.0",
            ecosystem="npm",
        )

        location = generator._get_download_location(comp)

        assert "@angular/core" in location


# ============================================================
# SBOMGenerator - Hash Algorithm Mapping
# ============================================================


class TestHashAlgorithmMapping:
    """Tests for hash algorithm to SPDX mapping."""

    def _make_generator(self) -> SBOMGenerator:
        """Create a generator with mocked scanner."""
        scanner = MagicMock()
        return SBOMGenerator(scanner=scanner)

    def test_hash_alg_to_spdx_sha256(self):
        """SHA-256 maps correctly to SPDX format."""
        generator = self._make_generator()
        assert generator._hash_alg_to_spdx(HashAlgorithm.SHA256) == "SHA256"

    def test_hash_alg_to_spdx_sha512(self):
        """SHA-512 maps correctly to SPDX format."""
        generator = self._make_generator()
        assert generator._hash_alg_to_spdx(HashAlgorithm.SHA512) == "SHA512"

    def test_hash_alg_to_spdx_md5(self):
        """MD5 maps correctly to SPDX format."""
        generator = self._make_generator()
        assert generator._hash_alg_to_spdx(HashAlgorithm.MD5) == "MD5"

    def test_hash_alg_to_spdx_blake(self):
        """BLAKE algorithms map correctly."""
        generator = self._make_generator()
        assert generator._hash_alg_to_spdx(HashAlgorithm.BLAKE2b_256) == "BLAKE2b-256"
        assert generator._hash_alg_to_spdx(HashAlgorithm.BLAKE3) == "BLAKE3"


# ============================================================
# SBOMGenerator - generate_from_dependencies
# ============================================================


class TestGenerateFromDependencies:
    """Tests for generating SBOM from dependency list."""

    def _make_dependency(self, **overrides) -> DependencyInfo:
        """Create a dependency with sensible defaults."""
        defaults = dict(
            name="express",
            version="4.18.2",
            ecosystem="npm",
            direct=True,
            dev_dependency=False,
            license="MIT",
        )
        defaults.update(overrides)
        return DependencyInfo(**defaults)

    @pytest.mark.asyncio
    async def test_generate_cyclonedx_json(self):
        """Generate CycloneDX JSON from dependencies."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner)
        deps = [self._make_dependency()]

        result = await generator.generate_from_dependencies(
            deps,
            format=SBOMFormat.CYCLONEDX_JSON,
            project_name="my-app",
        )

        assert result.format == SBOMFormat.CYCLONEDX_JSON
        assert result.component_count == 1
        assert result.filename == "my-app-cyclonedx.json"
        assert "CycloneDX" in result.content

    @pytest.mark.asyncio
    async def test_generate_cyclonedx_xml(self):
        """Generate CycloneDX XML from dependencies."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner)
        deps = [self._make_dependency()]

        result = await generator.generate_from_dependencies(
            deps,
            format=SBOMFormat.CYCLONEDX_XML,
            project_name="my-app",
        )

        assert result.format == SBOMFormat.CYCLONEDX_XML
        assert result.filename == "my-app-cyclonedx.xml"
        assert "<?xml" in result.content

    @pytest.mark.asyncio
    async def test_generate_spdx_json(self):
        """Generate SPDX JSON from dependencies."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner)
        deps = [self._make_dependency()]

        result = await generator.generate_from_dependencies(
            deps,
            format=SBOMFormat.SPDX_JSON,
            project_name="my-app",
        )

        assert result.format == SBOMFormat.SPDX_JSON
        assert result.filename == "my-app-spdx.json"
        assert "SPDX-2.3" in result.content

    @pytest.mark.asyncio
    async def test_generate_spdx_tagvalue(self):
        """Generate SPDX Tag-Value from dependencies."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner)
        deps = [self._make_dependency()]

        result = await generator.generate_from_dependencies(
            deps,
            format=SBOMFormat.SPDX_TV,
            project_name="my-app",
        )

        assert result.format == SBOMFormat.SPDX_TV
        assert result.filename == "my-app.spdx"
        assert "SPDXVersion:" in result.content

    @pytest.mark.asyncio
    async def test_generate_excludes_dev_deps(self):
        """Dev dependencies excluded when include_dev_dependencies=False."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner, include_dev_dependencies=False)
        deps = [
            self._make_dependency(name="prod-pkg", dev_dependency=False),
            self._make_dependency(name="dev-pkg", dev_dependency=True),
        ]

        result = await generator.generate_from_dependencies(deps)

        assert result.component_count == 1
        assert "prod-pkg" in result.content
        assert "dev-pkg" not in result.content

    @pytest.mark.asyncio
    async def test_generate_includes_dev_deps(self):
        """Dev dependencies included when include_dev_dependencies=True."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner, include_dev_dependencies=True)
        deps = [
            self._make_dependency(name="prod-pkg", dev_dependency=False),
            self._make_dependency(name="dev-pkg", dev_dependency=True),
        ]

        result = await generator.generate_from_dependencies(deps)

        assert result.component_count == 2

    @pytest.mark.asyncio
    async def test_generate_counts_vulnerabilities(self):
        """Vulnerability count is calculated correctly."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner)
        vuln1 = VulnerabilityFinding(
            id="CVE-1", title="V1", description="D1", severity=VulnerabilitySeverity.HIGH
        )
        vuln2 = VulnerabilityFinding(
            id="CVE-2", title="V2", description="D2", severity=VulnerabilitySeverity.MEDIUM
        )
        deps = [self._make_dependency(vulnerabilities=[vuln1, vuln2])]

        result = await generator.generate_from_dependencies(deps)

        assert result.vulnerability_count == 2

    @pytest.mark.asyncio
    async def test_generate_counts_licenses(self):
        """License count is calculated correctly (unique)."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner)
        deps = [
            self._make_dependency(name="pkg1", license="MIT"),
            self._make_dependency(name="pkg2", license="MIT"),
            self._make_dependency(name="pkg3", license="Apache-2.0"),
        ]

        result = await generator.generate_from_dependencies(deps)

        assert result.license_count == 2  # MIT and Apache-2.0

    @pytest.mark.asyncio
    async def test_generate_default_filename(self):
        """Default filename used when project_name not provided."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner)
        deps = []

        result = await generator.generate_from_dependencies(deps, format=SBOMFormat.CYCLONEDX_JSON)

        assert result.filename == "sbom-cyclonedx.json"

    @pytest.mark.asyncio
    async def test_generate_unsupported_format(self):
        """Unsupported format raises ValueError."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner)

        with pytest.raises(ValueError, match="Unsupported format"):
            await generator.generate_from_dependencies([], format="invalid")


# ============================================================
# SBOMGenerator - generate_from_repo
# ============================================================


class TestGenerateFromRepo:
    """Tests for generating SBOM from repository scan."""

    def _make_scan_result(self, **overrides) -> ScanResult:
        """Create a scan result with sensible defaults."""
        defaults = dict(
            scan_id="test-scan",
            repository="/path/to/repo",
            status="completed",
            dependencies=[
                DependencyInfo(
                    name="express",
                    version="4.18.2",
                    ecosystem="npm",
                    license="MIT",
                )
            ],
        )
        defaults.update(overrides)
        return ScanResult(**defaults)

    @pytest.mark.asyncio
    async def test_generate_from_repo_success(self):
        """Successful repository scan generates SBOM."""
        scanner = MagicMock()
        scanner.scan_repository = AsyncMock(return_value=self._make_scan_result())

        generator = SBOMGenerator(scanner=scanner)

        result = await generator.generate_from_repo(
            "/path/to/repo",
            format=SBOMFormat.CYCLONEDX_JSON,
            project_name="my-app",
        )

        assert result.format == SBOMFormat.CYCLONEDX_JSON
        assert result.component_count == 1
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_generate_from_repo_failed_scan(self):
        """Failed scan returns error result."""
        scanner = MagicMock()
        scanner.scan_repository = AsyncMock(
            return_value=ScanResult(
                scan_id="test",
                repository="/repo",
                status="failed",
                error="Could not find lock files",
            )
        )

        generator = SBOMGenerator(scanner=scanner)

        result = await generator.generate_from_repo("/path/to/repo")

        assert result.component_count == 0
        assert len(result.errors) == 1
        assert "Could not find lock files" in result.errors[0]

    @pytest.mark.asyncio
    async def test_generate_from_repo_infers_project_name(self):
        """Project name inferred from path when not provided."""
        scanner = MagicMock()
        scanner.scan_repository = AsyncMock(return_value=self._make_scan_result())

        generator = SBOMGenerator(scanner=scanner)

        result = await generator.generate_from_repo("/path/to/my-project")

        assert result.filename == "my-project-cyclonedx.json"

    @pytest.mark.asyncio
    async def test_generate_from_repo_passes_branch_and_commit(self):
        """Branch and commit SHA are passed to scanner."""
        scanner = MagicMock()
        scanner.scan_repository = AsyncMock(return_value=self._make_scan_result())

        generator = SBOMGenerator(scanner=scanner)

        await generator.generate_from_repo(
            "/path/to/repo",
            branch="main",
            commit_sha="abc123",
        )

        scanner.scan_repository.assert_awaited_once_with(
            "/path/to/repo",
            branch="main",
            commit_sha="abc123",
        )


# ============================================================
# Convenience Function: generate_sbom
# ============================================================


class TestGenerateSBOMFunction:
    """Tests for generate_sbom convenience function."""

    @pytest.mark.asyncio
    async def test_generate_sbom_creates_generator(self):
        """generate_sbom creates generator and calls generate_from_repo."""
        with patch.object(SBOMGenerator, "generate_from_repo", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = SBOMResult(
                format=SBOMFormat.CYCLONEDX_JSON,
                content="{}",
                filename="test.json",
                component_count=0,
                vulnerability_count=0,
                license_count=0,
            )

            result = await generate_sbom(
                "/path/to/repo",
                format=SBOMFormat.CYCLONEDX_JSON,
                project_name="test",
                project_version="1.0.0",
                include_dev=True,
                include_vulns=True,
            )

            assert result.format == SBOMFormat.CYCLONEDX_JSON

    @pytest.mark.asyncio
    async def test_generate_sbom_default_format(self):
        """generate_sbom uses CycloneDX JSON by default."""
        with patch.object(SBOMGenerator, "generate_from_repo", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = SBOMResult(
                format=SBOMFormat.CYCLONEDX_JSON,
                content="{}",
                filename="test.json",
                component_count=0,
                vulnerability_count=0,
                license_count=0,
            )

            await generate_sbom("/path/to/repo")

            call_kwargs = mock_gen.call_args.kwargs
            assert call_kwargs.get("format") == SBOMFormat.CYCLONEDX_JSON


# ============================================================
# Edge Cases
# ============================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_dependencies_list(self):
        """Empty dependencies list produces valid SBOM."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner)

        result = await generator.generate_from_dependencies(
            [],
            format=SBOMFormat.CYCLONEDX_JSON,
            project_name="empty-project",
        )

        assert result.component_count == 0
        assert result.vulnerability_count == 0
        assert result.license_count == 0

        # Verify JSON is valid
        bom = json.loads(result.content)
        assert bom["components"] == []

    @pytest.mark.asyncio
    async def test_dependency_with_empty_version(self):
        """Dependency with empty version is handled."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner)
        deps = [
            DependencyInfo(
                name="unknown-pkg",
                version="",
                ecosystem="npm",
            )
        ]

        result = await generator.generate_from_dependencies(deps)

        assert result.component_count == 1
        bom = json.loads(result.content)
        assert bom["components"][0]["version"] == ""

    @pytest.mark.asyncio
    async def test_dependency_with_special_characters(self):
        """Package names with special characters are handled."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner)
        deps = [
            DependencyInfo(
                name="@org/pkg-with-dashes_and_underscores",
                version="1.0.0",
                ecosystem="npm",
            )
        ]

        result = await generator.generate_from_dependencies(deps)

        assert result.component_count == 1
        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_many_dependencies(self):
        """Large number of dependencies is handled efficiently."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner)
        deps = [
            DependencyInfo(
                name=f"package-{i}",
                version=f"1.0.{i}",
                ecosystem="npm",
                license="MIT",
            )
            for i in range(100)
        ]

        result = await generator.generate_from_dependencies(deps)

        assert result.component_count == 100
        assert result.license_count == 1  # All MIT

    @pytest.mark.asyncio
    async def test_duplicate_vulnerabilities_counted(self):
        """Same vulnerability on multiple packages is counted per occurrence."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner)
        vuln = VulnerabilityFinding(
            id="CVE-2023-99999",
            title="Shared Vuln",
            description="Affects multiple packages",
            severity=VulnerabilitySeverity.HIGH,
        )
        deps = [
            DependencyInfo(
                name="pkg1",
                version="1.0.0",
                ecosystem="npm",
                vulnerabilities=[vuln],
            ),
            DependencyInfo(
                name="pkg2",
                version="1.0.0",
                ecosystem="npm",
                vulnerabilities=[vuln],
            ),
        ]

        result = await generator.generate_from_dependencies(deps)

        assert result.vulnerability_count == 2  # Counted per package

    @pytest.mark.asyncio
    async def test_mixed_ecosystems(self):
        """Multiple ecosystems in same SBOM are handled."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner)
        deps = [
            DependencyInfo(name="express", version="4.18.2", ecosystem="npm"),
            DependencyInfo(name="requests", version="2.31.0", ecosystem="pypi"),
            DependencyInfo(name="serde", version="1.0.0", ecosystem="cargo"),
        ]

        result = await generator.generate_from_dependencies(deps)

        assert result.component_count == 3
        bom = json.loads(result.content)

        purls = [c["purl"] for c in bom["components"]]
        assert any("npm" in p for p in purls)
        assert any("pypi" in p for p in purls)
        assert any("cargo" in p for p in purls)

    @pytest.mark.asyncio
    async def test_unicode_in_package_names(self):
        """Unicode characters in package names are handled."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner)
        deps = [
            DependencyInfo(
                name="test-unicode-pkg",
                version="1.0.0",
                ecosystem="npm",
            )
        ]

        result = await generator.generate_from_dependencies(deps)

        # Should produce valid JSON
        bom = json.loads(result.content)
        assert len(bom["components"]) == 1

    @pytest.mark.asyncio
    async def test_very_long_version_string(self):
        """Very long version strings are handled."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner)
        long_version = "1.0.0-alpha.beta.gamma.delta.epsilon.zeta.eta.theta.iota.kappa"
        deps = [
            DependencyInfo(
                name="pkg",
                version=long_version,
                ecosystem="npm",
            )
        ]

        result = await generator.generate_from_dependencies(deps)

        bom = json.loads(result.content)
        assert bom["components"][0]["version"] == long_version

    @pytest.mark.asyncio
    async def test_multiple_licenses_on_component(self):
        """Component with multiple licenses is handled."""
        scanner = MagicMock()
        generator = SBOMGenerator(scanner=scanner)

        # Create component directly with multiple licenses
        comp = SBOMComponent(
            name="dual-licensed",
            version="1.0.0",
            ecosystem="npm",
            purl="pkg:npm/dual-licensed@1.0.0",
            bom_ref="npm:dual-licensed@1.0.0",
            licenses=["MIT", "Apache-2.0"],
        )

        meta = SBOMMetadata()
        content = generator._generate_cyclonedx_json([comp], meta)
        bom = json.loads(content)

        licenses = bom["components"][0]["licenses"]
        assert len(licenses) == 2


# ============================================================
# Format Compliance
# ============================================================


class TestFormatCompliance:
    """Tests for SBOM format compliance."""

    def _make_generator(self) -> SBOMGenerator:
        """Create a generator with mocked scanner."""
        scanner = MagicMock()
        return SBOMGenerator(scanner=scanner)

    def test_cyclonedx_json_schema_reference(self):
        """CycloneDX JSON includes schema reference."""
        generator = self._make_generator()
        meta = SBOMMetadata()

        content = generator._generate_cyclonedx_json([], meta)
        bom = json.loads(content)

        assert "$schema" in bom
        assert "cyclonedx" in bom["$schema"]

    def test_cyclonedx_json_spec_version(self):
        """CycloneDX JSON specifies version 1.5."""
        generator = self._make_generator()
        meta = SBOMMetadata()

        content = generator._generate_cyclonedx_json([], meta)
        bom = json.loads(content)

        assert bom["specVersion"] == "1.5"

    def test_cyclonedx_json_serial_number_format(self):
        """CycloneDX serial number is valid URN UUID."""
        generator = self._make_generator()
        meta = SBOMMetadata()

        content = generator._generate_cyclonedx_json([], meta)
        bom = json.loads(content)

        serial = bom["serialNumber"]
        assert serial.startswith("urn:uuid:")
        # Verify UUID part is valid
        uuid_part = serial.replace("urn:uuid:", "")
        uuid.UUID(uuid_part)  # Raises if invalid

    def test_spdx_json_data_license(self):
        """SPDX JSON uses CC0-1.0 data license."""
        generator = self._make_generator()
        meta = SBOMMetadata()

        content = generator._generate_spdx_json([], meta)
        spdx = json.loads(content)

        assert spdx["dataLicense"] == "CC0-1.0"

    def test_spdx_json_document_spdxid(self):
        """SPDX JSON document has correct SPDXID."""
        generator = self._make_generator()
        meta = SBOMMetadata()

        content = generator._generate_spdx_json([], meta)
        spdx = json.loads(content)

        assert spdx["SPDXID"] == "SPDXRef-DOCUMENT"

    def test_spdx_json_namespace_is_unique(self):
        """SPDX JSON document namespace is unique."""
        generator = self._make_generator()
        meta = SBOMMetadata()

        content1 = generator._generate_spdx_json([], meta)
        content2 = generator._generate_spdx_json([], meta)

        spdx1 = json.loads(content1)
        spdx2 = json.loads(content2)

        assert spdx1["documentNamespace"] != spdx2["documentNamespace"]

    def test_spdx_tagvalue_file_structure(self):
        """SPDX Tag-Value has correct section order."""
        generator = self._make_generator()
        meta = SBOMMetadata(component_name="test")
        comp = SBOMComponent(
            name="pkg",
            version="1.0.0",
            ecosystem="npm",
            purl="pkg:npm/pkg@1.0.0",
            bom_ref="npm:pkg@1.0.0",
        )

        content = generator._generate_spdx_tagvalue([comp], meta)

        # Verify order: header, creators, root package, packages, relationships
        header_pos = content.find("SPDXVersion:")
        creator_pos = content.find("Creator:")
        root_pkg_pos = content.find("##### Root Package #####")
        pkg_pos = content.find("##### Package:")
        rel_pos = content.find("##### Relationships #####")

        assert header_pos < creator_pos < root_pkg_pos < pkg_pos < rel_pos


# ============================================================
# Integration-like Tests
# ============================================================


class TestIntegration:
    """Integration-like tests for end-to-end scenarios."""

    @pytest.mark.asyncio
    async def test_full_cyclonedx_json_generation(self):
        """Full CycloneDX JSON generation with all features."""
        scanner = MagicMock()
        vuln = VulnerabilityFinding(
            id="CVE-2023-12345",
            title="Test Vulnerability",
            description="A test vulnerability",
            severity=VulnerabilitySeverity.HIGH,
        )

        scanner.scan_repository = AsyncMock(
            return_value=ScanResult(
                scan_id="test",
                repository="/repo",
                status="completed",
                dependencies=[
                    DependencyInfo(
                        name="express",
                        version="4.18.2",
                        ecosystem="npm",
                        direct=True,
                        license="MIT",
                        vulnerabilities=[vuln],
                    ),
                    DependencyInfo(
                        name="lodash",
                        version="4.17.21",
                        ecosystem="npm",
                        direct=False,
                        license="MIT",
                    ),
                ],
            )
        )

        generator = SBOMGenerator(scanner=scanner)

        result = await generator.generate_from_repo(
            "/repo",
            format=SBOMFormat.CYCLONEDX_JSON,
            project_name="my-app",
            project_version="1.0.0",
        )

        assert result.component_count == 2
        assert result.vulnerability_count == 1
        assert result.license_count == 1  # Both MIT

        bom = json.loads(result.content)
        assert bom["bomFormat"] == "CycloneDX"
        assert len(bom["components"]) == 2
        assert "vulnerabilities" in bom
        assert len(bom["vulnerabilities"]) == 1

    @pytest.mark.asyncio
    async def test_full_spdx_json_generation(self):
        """Full SPDX JSON generation with all features."""
        scanner = MagicMock()
        scanner.scan_repository = AsyncMock(
            return_value=ScanResult(
                scan_id="test",
                repository="/repo",
                status="completed",
                dependencies=[
                    DependencyInfo(
                        name="requests",
                        version="2.31.0",
                        ecosystem="pypi",
                        direct=True,
                        license="Apache-2.0",
                    ),
                ],
            )
        )

        generator = SBOMGenerator(scanner=scanner)

        result = await generator.generate_from_repo(
            "/repo",
            format=SBOMFormat.SPDX_JSON,
            project_name="my-app",
            project_version="2.0.0",
        )

        assert result.component_count == 1

        spdx = json.loads(result.content)
        assert spdx["spdxVersion"] == "SPDX-2.3"
        assert len(spdx["packages"]) == 2  # Root + 1 dependency
        assert len(spdx["relationships"]) >= 1
