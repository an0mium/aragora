"""
Software Bill of Materials (SBOM) Generator.

Generates SBOM in CycloneDX and SPDX formats from project dependencies.
Supports:
- CycloneDX 1.4 JSON/XML
- SPDX 2.3 JSON/Tag-Value

Uses the DependencyScanner to enumerate components, then transforms
to standard SBOM formats for supply chain security compliance.
"""

from __future__ import annotations

import json
import logging
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import DependencyInfo
from .scanner import DependencyScanner

logger = logging.getLogger(__name__)


class SBOMFormat(str, Enum):
    """Supported SBOM output formats."""

    CYCLONEDX_JSON = "cyclonedx-json"
    CYCLONEDX_XML = "cyclonedx-xml"
    SPDX_JSON = "spdx-json"
    SPDX_TV = "spdx-tv"  # Tag-value format


class ComponentType(str, Enum):
    """CycloneDX component types."""

    APPLICATION = "application"
    FRAMEWORK = "framework"
    LIBRARY = "library"
    CONTAINER = "container"
    OPERATING_SYSTEM = "operating-system"
    DEVICE = "device"
    FIRMWARE = "firmware"
    FILE = "file"


class HashAlgorithm(str, Enum):
    """Hash algorithms for SBOM verification."""

    MD5 = "MD5"
    SHA1 = "SHA-1"
    SHA256 = "SHA-256"
    SHA384 = "SHA-384"
    SHA512 = "SHA-512"
    SHA3_256 = "SHA3-256"
    SHA3_512 = "SHA3-512"
    BLAKE2b_256 = "BLAKE2b-256"
    BLAKE2b_384 = "BLAKE2b-384"
    BLAKE2b_512 = "BLAKE2b-512"
    BLAKE3 = "BLAKE3"


# Map ecosystems to Package URLs (purl)
ECOSYSTEM_TO_PURL_TYPE = {
    "npm": "npm",
    "pypi": "pypi",
    "maven": "maven",
    "cargo": "cargo",
    "go": "golang",
    "rubygems": "gem",
    "nuget": "nuget",
    "composer": "composer",
    "cocoapods": "cocoapods",
    "swift": "swift",
    "hex": "hex",
    "pub": "pub",
}

# Map ecosystems to SPDX download location patterns
ECOSYSTEM_DOWNLOAD_PATTERNS = {
    "npm": "https://registry.npmjs.org/{name}/-/{name}-{version}.tgz",
    "pypi": "https://pypi.org/packages/source/{initial}/{name}/{name}-{version}.tar.gz",
    "cargo": "https://crates.io/api/v1/crates/{name}/{version}/download",
    "rubygems": "https://rubygems.org/downloads/{name}-{version}.gem",
    "go": "https://proxy.golang.org/{name}/@v/v{version}.zip",
}


@dataclass
class SBOMComponent:
    """A component in the SBOM."""

    name: str
    version: str
    ecosystem: str
    purl: str  # Package URL
    bom_ref: str  # Internal reference ID
    component_type: ComponentType = ComponentType.LIBRARY
    group: Optional[str] = None  # Maven group, npm scope
    description: Optional[str] = None
    licenses: List[str] = field(default_factory=list)
    hashes: Dict[HashAlgorithm, str] = field(default_factory=dict)
    supplier: Optional[str] = None
    author: Optional[str] = None
    cpe: Optional[str] = None  # Common Platform Enumeration
    external_references: List[Dict[str, str]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # bom_ref of deps
    vulnerabilities: List[str] = field(default_factory=list)  # CVE IDs
    properties: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dependency(cls, dep: DependencyInfo) -> "SBOMComponent":
        """Create component from DependencyInfo."""
        purl = cls._build_purl(dep.name, dep.version, dep.ecosystem)
        bom_ref = f"{dep.ecosystem}:{dep.name}@{dep.version}"

        # Extract group/scope for scoped packages
        group = None
        name = dep.name
        if dep.ecosystem == "npm" and dep.name.startswith("@"):
            parts = dep.name.split("/", 1)
            if len(parts) == 2:
                group = parts[0]
                name = parts[1]

        vulnerabilities = [v.id for v in dep.vulnerabilities] if dep.vulnerabilities else []

        return cls(
            name=name,
            version=dep.version,
            ecosystem=dep.ecosystem,
            purl=purl,
            bom_ref=bom_ref,
            group=group,
            licenses=[dep.license] if dep.license else [],
            vulnerabilities=vulnerabilities,
            properties={
                "direct": str(dep.direct).lower(),
                "dev_dependency": str(dep.dev_dependency).lower(),
            },
        )

    @staticmethod
    def _build_purl(name: str, version: str, ecosystem: str) -> str:
        """Build Package URL (purl) for a dependency."""
        purl_type = ECOSYSTEM_TO_PURL_TYPE.get(ecosystem, ecosystem)

        # Handle scoped npm packages
        if ecosystem == "npm" and name.startswith("@"):
            # @scope/name -> pkg:npm/%40scope/name@version
            encoded_name = name.replace("@", "%40")
            return f"pkg:{purl_type}/{encoded_name}@{version}"

        # Handle Go modules with subpaths
        if ecosystem == "go" and "/" in name:
            return f"pkg:{purl_type}/{name}@{version}"

        return f"pkg:{purl_type}/{name}@{version}"


@dataclass
class SBOMMetadata:
    """Metadata about the SBOM itself."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    serial_number: str = field(default_factory=lambda: f"urn:uuid:{uuid.uuid4()}")
    version: int = 1
    tool_name: str = "Aragora Security Scanner"
    tool_version: str = "1.0.0"
    tool_vendor: str = "Aragora"
    authors: List[str] = field(default_factory=list)
    component_name: Optional[str] = None  # Root project name
    component_version: Optional[str] = None
    component_type: ComponentType = ComponentType.APPLICATION
    supplier: Optional[str] = None
    manufacture: Optional[str] = None
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class SBOMResult:
    """Result of SBOM generation."""

    format: SBOMFormat
    content: str
    filename: str
    component_count: int
    vulnerability_count: int
    license_count: int
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Optional[SBOMMetadata] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "format": self.format.value,
            "content": self.content,
            "filename": self.filename,
            "component_count": self.component_count,
            "vulnerability_count": self.vulnerability_count,
            "license_count": self.license_count,
            "generated_at": self.generated_at.isoformat(),
            "errors": self.errors,
        }


class SBOMGenerator:
    """
    Generate Software Bill of Materials from project dependencies.

    Example:
        generator = SBOMGenerator()

        # Generate CycloneDX JSON
        result = await generator.generate_from_repo(
            "/path/to/repo",
            format=SBOMFormat.CYCLONEDX_JSON,
            project_name="MyProject",
            project_version="1.0.0",
        )

        # Save to file
        with open(result.filename, 'w') as f:
            f.write(result.content)

        # Generate from existing scan
        result = await generator.generate_from_dependencies(
            dependencies,
            format=SBOMFormat.SPDX_JSON,
        )
    """

    def __init__(
        self,
        scanner: Optional[DependencyScanner] = None,
        include_dev_dependencies: bool = True,
        include_vulnerabilities: bool = True,
    ):
        """
        Initialize SBOM generator.

        Args:
            scanner: DependencyScanner for parsing dependencies
            include_dev_dependencies: Include dev dependencies in SBOM
            include_vulnerabilities: Include vulnerability info in SBOM
        """
        self.scanner = scanner or DependencyScanner()
        self.include_dev_dependencies = include_dev_dependencies
        self.include_vulnerabilities = include_vulnerabilities

    async def generate_from_repo(
        self,
        repo_path: str,
        format: SBOMFormat = SBOMFormat.CYCLONEDX_JSON,
        project_name: Optional[str] = None,
        project_version: Optional[str] = None,
        branch: Optional[str] = None,
        commit_sha: Optional[str] = None,
    ) -> SBOMResult:
        """
        Generate SBOM from a repository.

        Args:
            repo_path: Path to repository
            format: Output format
            project_name: Name of the project
            project_version: Version of the project
            branch: Git branch
            commit_sha: Git commit SHA

        Returns:
            SBOMResult with generated SBOM
        """
        logger.info(f"[SBOM] Scanning repository: {repo_path}")

        # Use scanner to get dependencies
        scan_result = await self.scanner.scan_repository(
            repo_path, branch=branch, commit_sha=commit_sha
        )

        if scan_result.status == "failed":
            return SBOMResult(
                format=format,
                content="",
                filename="",
                component_count=0,
                vulnerability_count=0,
                license_count=0,
                errors=[scan_result.error or "Scan failed"],
            )

        # Infer project name from path if not provided
        if not project_name:
            project_name = Path(repo_path).name

        return await self.generate_from_dependencies(
            dependencies=scan_result.dependencies,
            format=format,
            project_name=project_name,
            project_version=project_version,
        )

    async def generate_from_dependencies(
        self,
        dependencies: List[DependencyInfo],
        format: SBOMFormat = SBOMFormat.CYCLONEDX_JSON,
        project_name: Optional[str] = None,
        project_version: Optional[str] = None,
    ) -> SBOMResult:
        """
        Generate SBOM from a list of dependencies.

        Args:
            dependencies: List of DependencyInfo objects
            format: Output format
            project_name: Name of the project
            project_version: Version of the project

        Returns:
            SBOMResult with generated SBOM
        """
        # Filter dependencies
        filtered_deps = dependencies
        if not self.include_dev_dependencies:
            filtered_deps = [d for d in dependencies if not d.dev_dependency]

        # Convert to SBOM components
        components = [SBOMComponent.from_dependency(dep) for dep in filtered_deps]

        # Create metadata
        metadata = SBOMMetadata(
            component_name=project_name,
            component_version=project_version,
        )

        # Generate based on format
        if format == SBOMFormat.CYCLONEDX_JSON:
            content = self._generate_cyclonedx_json(components, metadata)
            filename = f"{project_name or 'sbom'}-cyclonedx.json"
        elif format == SBOMFormat.CYCLONEDX_XML:
            content = self._generate_cyclonedx_xml(components, metadata)
            filename = f"{project_name or 'sbom'}-cyclonedx.xml"
        elif format == SBOMFormat.SPDX_JSON:
            content = self._generate_spdx_json(components, metadata)
            filename = f"{project_name or 'sbom'}-spdx.json"
        elif format == SBOMFormat.SPDX_TV:
            content = self._generate_spdx_tagvalue(components, metadata)
            filename = f"{project_name or 'sbom'}.spdx"
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Count vulnerabilities and licenses
        vuln_count = sum(len(c.vulnerabilities) for c in components)
        license_count = len(set(lic for c in components for lic in c.licenses if lic))

        logger.info(
            f"[SBOM] Generated {format.value}: {len(components)} components, "
            f"{vuln_count} vulnerabilities, {license_count} licenses"
        )

        return SBOMResult(
            format=format,
            content=content,
            filename=filename,
            component_count=len(components),
            vulnerability_count=vuln_count,
            license_count=license_count,
            metadata=metadata,
        )

    def _generate_cyclonedx_json(
        self,
        components: List[SBOMComponent],
        metadata: SBOMMetadata,
    ) -> str:
        """Generate CycloneDX 1.4 JSON format."""
        bom: Dict[str, Any] = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "serialNumber": metadata.serial_number,
            "version": metadata.version,
            "metadata": {
                "timestamp": metadata.timestamp.isoformat(),
                "tools": [
                    {
                        "vendor": metadata.tool_vendor,
                        "name": metadata.tool_name,
                        "version": metadata.tool_version,
                    }
                ],
            },
            "components": [],
        }

        # Add component metadata if provided
        if metadata.component_name:
            bom["metadata"]["component"] = {
                "type": metadata.component_type.value,
                "name": metadata.component_name,
            }
            if metadata.component_version:
                bom["metadata"]["component"]["version"] = metadata.component_version

        # Add components
        for comp in components:
            component_json: Dict[str, Any] = {
                "type": comp.component_type.value,
                "bom-ref": comp.bom_ref,
                "name": comp.name,
                "version": comp.version,
                "purl": comp.purl,
            }

            if comp.group:
                component_json["group"] = comp.group

            if comp.description:
                component_json["description"] = comp.description

            if comp.licenses:
                component_json["licenses"] = [
                    {
                        "license": {"id": lic}
                        if self._is_spdx_license(lic)
                        else {"license": {"name": lic}}
                    }
                    for lic in comp.licenses
                ]

            if comp.hashes:
                component_json["hashes"] = [
                    {"alg": alg.value, "content": hash_val} for alg, hash_val in comp.hashes.items()
                ]

            if comp.external_references:
                component_json["externalReferences"] = comp.external_references

            if comp.properties:
                component_json["properties"] = [
                    {"name": k, "value": v} for k, v in comp.properties.items()
                ]

            bom["components"].append(component_json)

        # Add vulnerabilities if enabled
        if self.include_vulnerabilities:
            vulns = []
            for comp in components:
                for vuln_id in comp.vulnerabilities:
                    vulns.append(
                        {
                            "id": vuln_id,
                            "source": {"name": "NVD", "url": "https://nvd.nist.gov/"},
                            "affects": [
                                {
                                    "ref": comp.bom_ref,
                                }
                            ],
                        }
                    )
            if vulns:
                bom["vulnerabilities"] = vulns

        # Add dependency graph
        dependencies = []
        for comp in components:
            if comp.dependencies:
                dependencies.append(
                    {
                        "ref": comp.bom_ref,
                        "dependsOn": comp.dependencies,
                    }
                )
            else:
                dependencies.append({"ref": comp.bom_ref, "dependsOn": []})

        if dependencies:
            bom["dependencies"] = dependencies

        return json.dumps(bom, indent=2)

    def _generate_cyclonedx_xml(
        self,
        components: List[SBOMComponent],
        metadata: SBOMMetadata,
    ) -> str:
        """Generate CycloneDX 1.4 XML format."""
        # Create root element with namespace
        ns = "http://cyclonedx.org/schema/bom/1.4"
        ET.register_namespace("", ns)

        root = ET.Element(
            "bom",
            {
                "xmlns": ns,
                "serialNumber": metadata.serial_number,
                "version": str(metadata.version),
            },
        )

        # Add metadata
        meta_elem = ET.SubElement(root, "metadata")
        ET.SubElement(meta_elem, "timestamp").text = metadata.timestamp.isoformat()

        tools = ET.SubElement(meta_elem, "tools")
        tool = ET.SubElement(tools, "tool")
        ET.SubElement(tool, "vendor").text = metadata.tool_vendor
        ET.SubElement(tool, "name").text = metadata.tool_name
        ET.SubElement(tool, "version").text = metadata.tool_version

        if metadata.component_name:
            component_elem = ET.SubElement(
                meta_elem,
                "component",
                {
                    "type": metadata.component_type.value,
                },
            )
            ET.SubElement(component_elem, "name").text = metadata.component_name
            if metadata.component_version:
                ET.SubElement(component_elem, "version").text = metadata.component_version

        # Add components
        components_elem = ET.SubElement(root, "components")
        for comp in components:
            comp_elem = ET.SubElement(
                components_elem,
                "component",
                {
                    "type": comp.component_type.value,
                    "bom-ref": comp.bom_ref,
                },
            )

            if comp.group:
                ET.SubElement(comp_elem, "group").text = comp.group

            ET.SubElement(comp_elem, "name").text = comp.name
            ET.SubElement(comp_elem, "version").text = comp.version
            ET.SubElement(comp_elem, "purl").text = comp.purl

            if comp.licenses:
                licenses_elem = ET.SubElement(comp_elem, "licenses")
                for lic in comp.licenses:
                    license_elem = ET.SubElement(licenses_elem, "license")
                    if self._is_spdx_license(lic):
                        ET.SubElement(license_elem, "id").text = lic
                    else:
                        ET.SubElement(license_elem, "name").text = lic

            if comp.hashes:
                hashes_elem = ET.SubElement(comp_elem, "hashes")
                for alg, hash_val in comp.hashes.items():
                    hash_elem = ET.SubElement(hashes_elem, "hash", {"alg": alg.value})
                    hash_elem.text = hash_val

        # Add vulnerabilities
        if self.include_vulnerabilities:
            all_vulns = [
                (comp.bom_ref, vuln_id) for comp in components for vuln_id in comp.vulnerabilities
            ]
            if all_vulns:
                vulns_elem = ET.SubElement(root, "vulnerabilities")
                for bom_ref, vuln_id in all_vulns:
                    vuln_elem = ET.SubElement(vulns_elem, "vulnerability", {"ref": vuln_id})
                    ET.SubElement(vuln_elem, "id").text = vuln_id
                    source = ET.SubElement(vuln_elem, "source")
                    ET.SubElement(source, "name").text = "NVD"
                    affects = ET.SubElement(vuln_elem, "affects")
                    target = ET.SubElement(affects, "target")
                    ET.SubElement(target, "ref").text = bom_ref

        # Add dependencies
        deps_elem = ET.SubElement(root, "dependencies")
        for comp in components:
            dep_elem = ET.SubElement(deps_elem, "dependency", {"ref": comp.bom_ref})
            for dep_ref in comp.dependencies:
                ET.SubElement(dep_elem, "dependency", {"ref": dep_ref})

        # Convert to string
        ET.indent(root)
        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def _generate_spdx_json(
        self,
        components: List[SBOMComponent],
        metadata: SBOMMetadata,
    ) -> str:
        """Generate SPDX 2.3 JSON format."""
        # Create document ID
        doc_namespace = f"https://aragora.io/spdx/{uuid.uuid4()}"

        spdx: Dict[str, Any] = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": metadata.component_name or "SBOM",
            "documentNamespace": doc_namespace,
            "creationInfo": {
                "created": metadata.timestamp.isoformat(),
                "creators": [
                    f"Tool: {metadata.tool_name}-{metadata.tool_version}",
                    f"Organization: {metadata.tool_vendor}",
                ],
                "licenseListVersion": "3.19",
            },
            "packages": [],
            "relationships": [],
        }

        # Add root package if project info provided
        if metadata.component_name:
            root_pkg = {
                "SPDXID": "SPDXRef-RootPackage",
                "name": metadata.component_name,
                "versionInfo": metadata.component_version or "NOASSERTION",
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "primaryPackagePurpose": "APPLICATION",
            }
            spdx["packages"].append(root_pkg)

            # Document describes root package
            spdx["relationships"].append(
                {
                    "spdxElementId": "SPDXRef-DOCUMENT",
                    "relatedSpdxElement": "SPDXRef-RootPackage",
                    "relationshipType": "DESCRIBES",
                }
            )

        # Add components as packages
        for i, comp in enumerate(components):
            spdx_id = f"SPDXRef-Package-{i}"

            pkg: Dict[str, Any] = {
                "SPDXID": spdx_id,
                "name": comp.name,
                "versionInfo": comp.version,
                "downloadLocation": self._get_download_location(comp),
                "filesAnalyzed": False,
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": comp.purl,
                    }
                ],
            }

            if comp.group:
                pkg["supplier"] = f"Organization: {comp.group}"

            if comp.licenses:
                # Use SPDX license expression
                pkg["licenseConcluded"] = " AND ".join(
                    lic if self._is_spdx_license(lic) else f"LicenseRef-{lic}"
                    for lic in comp.licenses
                )
                pkg["licenseDeclared"] = pkg["licenseConcluded"]
            else:
                pkg["licenseConcluded"] = "NOASSERTION"
                pkg["licenseDeclared"] = "NOASSERTION"

            pkg["copyrightText"] = "NOASSERTION"

            if comp.hashes:
                pkg["checksums"] = []
                for alg, hash_val in comp.hashes.items():
                    spdx_alg = self._hash_alg_to_spdx(alg)
                    if spdx_alg:
                        pkg["checksums"].append(
                            {
                                "algorithm": spdx_alg,
                                "checksumValue": hash_val,
                            }
                        )

            # Add vulnerability references
            if comp.vulnerabilities:
                for vuln_id in comp.vulnerabilities:
                    pkg["externalRefs"].append(
                        {
                            "referenceCategory": "SECURITY",
                            "referenceType": "cve",
                            "referenceLocator": f"https://nvd.nist.gov/vuln/detail/{vuln_id}",
                        }
                    )

            spdx["packages"].append(pkg)

            # Add relationship to root if it exists
            if metadata.component_name:
                if comp.properties.get("direct") == "true":
                    spdx["relationships"].append(
                        {
                            "spdxElementId": "SPDXRef-RootPackage",
                            "relatedSpdxElement": spdx_id,
                            "relationshipType": "DEPENDS_ON",
                        }
                    )

        return json.dumps(spdx, indent=2)

    def _generate_spdx_tagvalue(
        self,
        components: List[SBOMComponent],
        metadata: SBOMMetadata,
    ) -> str:
        """Generate SPDX 2.3 Tag-Value format."""
        lines = []

        # Document header
        lines.append("SPDXVersion: SPDX-2.3")
        lines.append("DataLicense: CC0-1.0")
        lines.append("SPDXID: SPDXRef-DOCUMENT")
        lines.append(f"DocumentName: {metadata.component_name or 'SBOM'}")
        lines.append(f"DocumentNamespace: https://aragora.io/spdx/{uuid.uuid4()}")
        lines.append("")

        # Creator info
        lines.append(f"Creator: Tool: {metadata.tool_name}-{metadata.tool_version}")
        lines.append(f"Creator: Organization: {metadata.tool_vendor}")
        lines.append(f"Created: {metadata.timestamp.isoformat()}")
        lines.append("")

        # Root package if provided
        if metadata.component_name:
            lines.append("##### Root Package #####")
            lines.append("")
            lines.append(f"PackageName: {metadata.component_name}")
            lines.append("SPDXID: SPDXRef-RootPackage")
            lines.append(f"PackageVersion: {metadata.component_version or 'NOASSERTION'}")
            lines.append("PackageDownloadLocation: NOASSERTION")
            lines.append("FilesAnalyzed: false")
            lines.append("PackageLicenseConcluded: NOASSERTION")
            lines.append("PackageLicenseDeclared: NOASSERTION")
            lines.append("PackageCopyrightText: NOASSERTION")
            lines.append("")

        # Add packages
        for i, comp in enumerate(components):
            spdx_id = f"SPDXRef-Package-{i}"

            lines.append(f"##### Package: {comp.name} #####")
            lines.append("")
            lines.append(f"PackageName: {comp.name}")
            lines.append(f"SPDXID: {spdx_id}")
            lines.append(f"PackageVersion: {comp.version}")
            lines.append(f"PackageDownloadLocation: {self._get_download_location(comp)}")
            lines.append("FilesAnalyzed: false")

            if comp.licenses:
                license_expr = " AND ".join(
                    lic if self._is_spdx_license(lic) else f"LicenseRef-{lic}"
                    for lic in comp.licenses
                )
                lines.append(f"PackageLicenseConcluded: {license_expr}")
                lines.append(f"PackageLicenseDeclared: {license_expr}")
            else:
                lines.append("PackageLicenseConcluded: NOASSERTION")
                lines.append("PackageLicenseDeclared: NOASSERTION")

            lines.append("PackageCopyrightText: NOASSERTION")

            # Add purl as external reference
            lines.append(f"ExternalRef: PACKAGE-MANAGER purl {comp.purl}")

            # Add vulnerabilities
            for vuln_id in comp.vulnerabilities:
                lines.append(
                    f"ExternalRef: SECURITY cve https://nvd.nist.gov/vuln/detail/{vuln_id}"
                )

            if comp.hashes:
                for alg, hash_val in comp.hashes.items():
                    spdx_alg = self._hash_alg_to_spdx(alg)
                    if spdx_alg:
                        lines.append(f"PackageChecksum: {spdx_alg}: {hash_val}")

            lines.append("")

        # Add relationships
        lines.append("##### Relationships #####")
        lines.append("")
        lines.append("Relationship: SPDXRef-DOCUMENT DESCRIBES SPDXRef-RootPackage")

        if metadata.component_name:
            for i, comp in enumerate(components):
                if comp.properties.get("direct") == "true":
                    spdx_id = f"SPDXRef-Package-{i}"
                    lines.append(f"Relationship: SPDXRef-RootPackage DEPENDS_ON {spdx_id}")

        return "\n".join(lines)

    def _is_spdx_license(self, license_id: str) -> bool:
        """Check if license ID is a valid SPDX identifier."""
        # Common SPDX license identifiers
        spdx_licenses = {
            "MIT",
            "Apache-2.0",
            "GPL-2.0",
            "GPL-3.0",
            "BSD-2-Clause",
            "BSD-3-Clause",
            "ISC",
            "MPL-2.0",
            "LGPL-2.1",
            "LGPL-3.0",
            "AGPL-3.0",
            "Unlicense",
            "CC0-1.0",
            "CC-BY-4.0",
            "Zlib",
            "0BSD",
            "BSL-1.0",
            "EPL-1.0",
            "EPL-2.0",
            "GPL-2.0-only",
            "GPL-3.0-only",
            "LGPL-2.1-only",
            "LGPL-3.0-only",
            "Apache-1.1",
            "Artistic-2.0",
            "Python-2.0",
            "Ruby",
        }
        return license_id in spdx_licenses

    def _get_download_location(self, comp: SBOMComponent) -> str:
        """Get download location for a component."""
        pattern = ECOSYSTEM_DOWNLOAD_PATTERNS.get(comp.ecosystem)
        if pattern:
            name = comp.name
            if comp.group and comp.ecosystem == "npm":
                name = f"{comp.group}/{comp.name}"

            return pattern.format(
                name=name,
                version=comp.version,
                initial=name[0].lower() if name else "x",
            )
        return "NOASSERTION"

    def _hash_alg_to_spdx(self, alg: HashAlgorithm) -> Optional[str]:
        """Convert HashAlgorithm to SPDX checksum algorithm."""
        mapping = {
            HashAlgorithm.MD5: "MD5",
            HashAlgorithm.SHA1: "SHA1",
            HashAlgorithm.SHA256: "SHA256",
            HashAlgorithm.SHA384: "SHA384",
            HashAlgorithm.SHA512: "SHA512",
            HashAlgorithm.SHA3_256: "SHA3-256",
            HashAlgorithm.SHA3_512: "SHA3-512",
            HashAlgorithm.BLAKE2b_256: "BLAKE2b-256",
            HashAlgorithm.BLAKE2b_384: "BLAKE2b-384",
            HashAlgorithm.BLAKE2b_512: "BLAKE2b-512",
            HashAlgorithm.BLAKE3: "BLAKE3",
        }
        return mapping.get(alg)


async def generate_sbom(
    repo_path: str,
    format: SBOMFormat = SBOMFormat.CYCLONEDX_JSON,
    project_name: Optional[str] = None,
    project_version: Optional[str] = None,
    include_dev: bool = True,
    include_vulns: bool = True,
) -> SBOMResult:
    """
    Convenience function to generate SBOM from a repository.

    Args:
        repo_path: Path to repository
        format: Output format
        project_name: Project name
        project_version: Project version
        include_dev: Include dev dependencies
        include_vulns: Include vulnerability info

    Returns:
        SBOMResult
    """
    generator = SBOMGenerator(
        include_dev_dependencies=include_dev,
        include_vulnerabilities=include_vulns,
    )
    return await generator.generate_from_repo(
        repo_path,
        format=format,
        project_name=project_name,
        project_version=project_version,
    )


__all__ = [
    "SBOMGenerator",
    "SBOMFormat",
    "SBOMResult",
    "SBOMComponent",
    "SBOMMetadata",
    "ComponentType",
    "HashAlgorithm",
    "generate_sbom",
]
