#!/usr/bin/env python3
"""
SDK Code Generation Pipeline.

Scans handler files for route definitions using AST parsing and docstring
analysis, then generates typed TypeScript namespace and Python client code.

Usage:
    python scripts/sdk_codegen.py --scan                # Report coverage only
    python scripts/sdk_codegen.py --generate --lang ts   # Generate TypeScript
    python scripts/sdk_codegen.py --generate --lang py   # Generate Python
    python scripts/sdk_codegen.py --generate --lang both  # Generate both
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class EndpointDef:
    """Represents a single API endpoint extracted from a handler."""

    method: str  # GET, POST, PUT, PATCH, DELETE
    path: str  # e.g. /api/v2/backups/:backup_id
    params: list[str] = field(default_factory=list)  # path params
    return_type: str = "unknown"
    description: str = ""
    handler_file: str = ""


@dataclass
class CoverageReport:
    """SDK coverage statistics."""

    total_endpoints: int
    covered: int
    missing: list[EndpointDef] = field(default_factory=list)

    @property
    def coverage_percent(self) -> float:
        if self.total_endpoints == 0:
            return 100.0
        return round((self.covered / self.total_endpoints) * 100, 1)


# ---------------------------------------------------------------------------
# Regex patterns for docstring endpoint extraction
# ---------------------------------------------------------------------------

# Pattern 1: "- METHOD /api/path - Description"
_DASH_ENDPOINT_RE = re.compile(
    r"^-\s+(GET|POST|PUT|PATCH|DELETE)\s+(/\S+)\s*(?:-\s*(.*))?$"
)

# Pattern 2: "METHOD  /api/path   - Description" (indented, backup_handler style)
_INDENT_ENDPOINT_RE = re.compile(
    r"^\s+(GET|POST|PUT|PATCH|DELETE)\s+(/\S+)\s*(?:-\s*(.*))?$"
)

# Path parameter patterns: :param_name or {param_name}
_PATH_PARAM_RE = re.compile(r"[:{}]([a-zA-Z_]\w*)")


def _extract_path_params(path: str) -> list[str]:
    """Extract named parameters from a route path."""
    return _PATH_PARAM_RE.findall(path)


def _normalize_path(path: str) -> str:
    """Normalize path by replacing :param / {param} with :param consistently."""
    return re.sub(r"\{(\w+)\}", r":\1", path)


# ---------------------------------------------------------------------------
# SDK Code Generator
# ---------------------------------------------------------------------------


class SDKCodeGenerator:
    """Scans handler files and generates SDK client code."""

    def scan_handlers(self, handler_dir: str | Path) -> list[EndpointDef]:
        """Scan all Python handler files for route definitions.

        Uses AST parsing to extract module docstrings and class-level ROUTES
        lists without importing any handler modules.
        """
        handler_dir = Path(handler_dir)
        if not handler_dir.is_dir():
            raise FileNotFoundError(f"Handler directory not found: {handler_dir}")

        endpoints: list[EndpointDef] = []
        for py_file in sorted(handler_dir.rglob("*.py")):
            if py_file.name.startswith("_") or py_file.name == "__init__.py":
                continue
            try:
                found = self.extract_endpoints(py_file)
                endpoints.extend(found)
            except SyntaxError:
                # Skip files that cannot be parsed
                continue
        return endpoints

    def extract_endpoints(self, handler_file: str | Path) -> list[EndpointDef]:
        """Extract endpoint definitions from a single handler file.

        Strategy:
        1. Parse the module docstring for structured endpoint listings.
        2. Walk the AST for class-level ROUTES list assignments.
        3. Deduplicate by (method, normalized_path).
        """
        handler_file = Path(handler_file)
        source = handler_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(handler_file))

        seen: set[tuple[str, str]] = set()
        endpoints: list[EndpointDef] = []
        rel_path = str(handler_file)

        def _add(method: str, path: str, description: str = "") -> None:
            norm = _normalize_path(path.strip())
            key = (method.upper(), norm)
            if key not in seen:
                seen.add(key)
                endpoints.append(
                    EndpointDef(
                        method=method.upper(),
                        path=norm,
                        params=_extract_path_params(norm),
                        description=description.strip(),
                        handler_file=rel_path,
                    )
                )

        # --- Strategy 1: module docstring ---
        module_doc = ast.get_docstring(tree) or ""
        for raw_line in module_doc.splitlines():
            stripped = raw_line.strip()
            # Try dash style on stripped line, indent style on raw line
            m = _DASH_ENDPOINT_RE.match(stripped) or _INDENT_ENDPOINT_RE.match(raw_line)
            if m:
                _add(m.group(1), m.group(2), m.group(3) or "")

        # --- Strategy 2: AST walk for ROUTES = [...] ---
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if (
                        isinstance(item, ast.Assign)
                        and len(item.targets) == 1
                        and isinstance(item.targets[0], ast.Name)
                        and item.targets[0].id == "ROUTES"
                        and isinstance(item.value, ast.List)
                    ):
                        for elt in item.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(
                                elt.value, str
                            ):
                                route = elt.value
                                if route.startswith("/api/") and "*" not in route:
                                    # ROUTES alone do not indicate HTTP method;
                                    # mark as GET by default (docstring overrides)
                                    _add("GET", route)

        return endpoints

    # ------------------------------------------------------------------
    # TypeScript generation
    # ------------------------------------------------------------------

    def generate_typescript_namespace(
        self, endpoints: list[EndpointDef], namespace_name: str
    ) -> str:
        """Generate a TypeScript namespace class from endpoint definitions."""
        class_name = _to_pascal_case(namespace_name) + "API"
        iface_name = f"{class_name}Client"

        methods: list[str] = []
        for ep in sorted(endpoints, key=lambda e: (e.path, e.method)):
            method_name = _ts_method_name(ep)
            has_body = ep.method in ("POST", "PUT", "PATCH")
            ts_path = _ts_template_path(ep.path)

            # Build parameter list
            fn_params: list[str] = []
            for p in ep.params:
                fn_params.append(f"{_to_camel_case(p)}: string")
            if has_body:
                fn_params.append("data: Record<string, unknown>")

            param_str = ", ".join(fn_params)
            doc = ep.description or f"{ep.method} {ep.path}"

            # Build request call
            opts = ""
            if has_body:
                opts = ", { body: data }"

            method_block = textwrap.dedent(f"""\
                /** {doc} */
                async {method_name}({param_str}): Promise<Record<string, unknown>> {{
                    return this.client.request('{ep.method}', {ts_path}{opts});
                }}""")
            methods.append(textwrap.indent(method_block, "  "))

        methods_str = "\n\n".join(methods)

        return textwrap.dedent(f"""\
            /**
             * {class_name} Namespace API
             *
             * Auto-generated by sdk_codegen.py. Do not edit manually.
             */

            interface {iface_name} {{
              request<T = unknown>(method: string, path: string, options?: Record<string, unknown>): Promise<T>;
            }}

            export class {class_name} {{
              constructor(private client: {iface_name}) {{}}

            {methods_str}
            }}
            """)

    # ------------------------------------------------------------------
    # Python generation
    # ------------------------------------------------------------------

    def generate_python_client(
        self, endpoints: list[EndpointDef], module_name: str
    ) -> str:
        """Generate a Python client module from endpoint definitions."""
        class_name = _to_pascal_case(module_name) + "API"

        methods: list[str] = []
        for ep in sorted(endpoints, key=lambda e: (e.path, e.method)):
            method_name = _py_method_name(ep)
            has_body = ep.method in ("POST", "PUT", "PATCH")
            py_path = _py_format_path(ep.path)

            fn_params = ["self"]
            for p in ep.params:
                fn_params.append(f"{p}: str")
            if has_body:
                fn_params.append("data: dict[str, Any] | None = None")
            param_str = ", ".join(fn_params)

            doc = ep.description or f"{ep.method} {ep.path}"

            body_arg = ""
            if has_body:
                body_arg = ", json=data"

            method_block = textwrap.dedent(f'''\
                def {method_name}({param_str}) -> dict[str, Any]:
                    """{doc}"""
                    return self._client.request("{ep.method}", {py_path}{body_arg})''')
            methods.append(textwrap.indent(method_block, "    "))

        methods_str = "\n\n".join(methods)

        lines = [
            f'"""',
            f'{module_name} namespace - auto-generated by sdk_codegen.py.',
            f'',
            f'Do not edit manually.',
            f'"""',
            f'',
            f'from __future__ import annotations',
            f'',
            f'from typing import TYPE_CHECKING, Any',
            f'',
            f'if TYPE_CHECKING:',
            f'    from ..client import AragoraClient',
            f'',
            f'',
            f'class {class_name}:',
            f'    """Synchronous {module_name} API."""',
            f'',
            f'    def __init__(self, client: AragoraClient) -> None:',
            f'        self._client = client',
            f'',
        ]
        header = "\n".join(lines) + "\n"
        if methods_str:
            return header + methods_str + "\n"
        return header

    # ------------------------------------------------------------------
    # Coverage validation
    # ------------------------------------------------------------------

    def validate_sdk_coverage(
        self, sdk_dir: str | Path, endpoints: list[EndpointDef]
    ) -> CoverageReport:
        """Check which handler endpoints are covered by existing SDK files.

        Scans TypeScript and Python SDK source for path string literals and
        compares against the scanned endpoint set.
        """
        sdk_dir = Path(sdk_dir)
        sdk_paths: set[str] = set()

        # Collect all path literals from SDK source files
        for ext in ("*.ts", "*.py"):
            for f in sdk_dir.rglob(ext):
                if f.name.startswith("_") and f.name != "__init__.py":
                    continue
                try:
                    content = f.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                # Match single/double quoted paths
                for m in re.finditer(r"""['"](/api/[^'"]+)['"]""", content):
                    raw = m.group(1)
                    normalized = re.sub(r"\{(\w+)\}", r":\1", raw)
                    sdk_paths.add(normalized)
                # Match backtick-quoted paths (may contain ${expr})
                for m in re.finditer(r"""`(/api/[^`]+)`""", content):
                    raw = m.group(1)
                    normalized = re.sub(r"\$\{(\w+)\}", r":\1", raw)
                    normalized = re.sub(r"\{(\w+)\}", r":\1", normalized)
                    sdk_paths.add(normalized)

        covered = 0
        missing: list[EndpointDef] = []
        for ep in endpoints:
            norm = _normalize_path(ep.path)
            base = norm.rstrip("/*")
            if base in sdk_paths or norm in sdk_paths:
                covered += 1
            else:
                missing.append(ep)

        return CoverageReport(
            total_endpoints=len(endpoints),
            covered=covered,
            missing=missing,
        )

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def run(
        self,
        handler_dir: str | Path,
        output_dir: str | Path | None = None,
        language: str = "typescript",
    ) -> CoverageReport:
        """Scan handlers and optionally generate SDK code.

        Args:
            handler_dir: Path to the handler directory to scan.
            output_dir: If provided, write generated files here.
            language: 'typescript', 'python', or 'both'.

        Returns:
            Coverage report comparing scanned endpoints to existing SDK.
        """
        endpoints = self.scan_handlers(handler_dir)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Group endpoints by namespace
            groups = _group_by_namespace(endpoints)

            for ns_name, ns_endpoints in sorted(groups.items()):
                if language in ("typescript", "both"):
                    ts_code = self.generate_typescript_namespace(
                        ns_endpoints, ns_name
                    )
                    ts_file = output_dir / "typescript" / f"{ns_name}.ts"
                    ts_file.parent.mkdir(parents=True, exist_ok=True)
                    ts_file.write_text(ts_code, encoding="utf-8")

                if language in ("python", "both"):
                    py_code = self.generate_python_client(ns_endpoints, ns_name)
                    py_name = ns_name.replace("-", "_")
                    py_file = output_dir / "python" / f"{py_name}.py"
                    py_file.parent.mkdir(parents=True, exist_ok=True)
                    py_file.write_text(py_code, encoding="utf-8")

        # Validate coverage against existing SDK
        repo_root = Path(handler_dir).resolve().parent.parent.parent
        sdk_dir = repo_root / "sdk"
        if sdk_dir.is_dir():
            return self.validate_sdk_coverage(sdk_dir, endpoints)

        return CoverageReport(
            total_endpoints=len(endpoints), covered=0, missing=endpoints
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _group_by_namespace(
    endpoints: list[EndpointDef],
) -> dict[str, list[EndpointDef]]:
    """Group endpoints by namespace derived from the URL path."""
    groups: dict[str, list[EndpointDef]] = {}
    for ep in endpoints:
        parts = ep.path.strip("/").split("/")
        idx = 0
        if idx < len(parts) and parts[idx] == "api":
            idx += 1
        if idx < len(parts) and re.match(r"^v\d+$", parts[idx]):
            idx += 1
        ns = parts[idx] if idx < len(parts) else "misc"
        groups.setdefault(ns, []).append(ep)
    return groups


def _to_pascal_case(name: str) -> str:
    """Convert kebab-case or snake_case to PascalCase."""
    return "".join(w.capitalize() for w in re.split(r"[-_]", name))


def _to_camel_case(name: str) -> str:
    """Convert snake_case to camelCase."""
    parts = name.split("_")
    return parts[0] + "".join(w.capitalize() for w in parts[1:])


def _ts_method_name(ep: EndpointDef) -> str:
    """Derive a TypeScript method name from an endpoint."""
    parts = ep.path.strip("/").split("/")
    meaningful = [
        p
        for p in parts
        if p not in ("api",)
        and not re.match(r"^v\d+$", p)
        and not p.startswith(":")
    ]
    if not meaningful:
        meaningful = ["root"]
    name_parts = meaningful[-2:] if len(meaningful) > 1 else meaningful
    prefix_map = {
        "GET": "get",
        "POST": "create",
        "PUT": "update",
        "PATCH": "patch",
        "DELETE": "delete",
    }
    prefix = prefix_map.get(ep.method, ep.method.lower())
    camel = "".join(w.capitalize() for w in name_parts)
    return prefix + camel


def _py_method_name(ep: EndpointDef) -> str:
    """Derive a Python method name from an endpoint."""
    parts = ep.path.strip("/").split("/")
    meaningful = [
        p
        for p in parts
        if p not in ("api",)
        and not re.match(r"^v\d+$", p)
        and not p.startswith(":")
    ]
    if not meaningful:
        meaningful = ["root"]
    name_parts = meaningful[-2:] if len(meaningful) > 1 else meaningful
    prefix_map = {
        "GET": "get",
        "POST": "create",
        "PUT": "update",
        "PATCH": "patch",
        "DELETE": "delete",
    }
    prefix = prefix_map.get(ep.method, ep.method.lower())
    slug = "_".join(p.replace("-", "_") for p in name_parts)
    return f"{prefix}_{slug}"


def _ts_template_path(path: str) -> str:
    """Convert a path to a TypeScript template literal if it has params."""
    if ":" not in path:
        return f"'{path}'"

    def _repl(m: re.Match) -> str:
        return "${" + _to_camel_case(m.group(1)) + "}"

    converted = re.sub(r":(\w+)", _repl, path)
    return f"`{converted}`"


def _py_format_path(path: str) -> str:
    """Convert a path to a Python f-string if it has params."""
    if ":" not in path:
        return f'"{path}"'
    converted = re.sub(r":(\w+)", r"{\1}", path)
    return f'f"{converted}"'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SDK code generation from handler route definitions"
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan handlers and report coverage",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate SDK code",
    )
    parser.add_argument(
        "--lang",
        choices=["typescript", "ts", "python", "py", "both"],
        default="typescript",
        help="Target language (default: typescript)",
    )
    parser.add_argument(
        "--handler-dir",
        type=Path,
        default=None,
        help="Handler directory (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated files",
    )
    args = parser.parse_args(argv)

    # Resolve paths
    repo_root = Path(__file__).resolve().parent.parent
    handler_dir = (
        args.handler_dir or repo_root / "aragora" / "server" / "handlers"
    )
    output_dir = args.output_dir or (
        repo_root / "sdk" / "generated" if args.generate else None
    )

    # Normalize language
    lang_map = {"ts": "typescript", "py": "python"}
    language = lang_map.get(args.lang, args.lang)

    gen = SDKCodeGenerator()

    if args.generate:
        report = gen.run(handler_dir, output_dir=output_dir, language=language)
        print(f"Generated SDK files in {output_dir}")
    elif args.scan:
        report = gen.run(handler_dir)
    else:
        parser.print_help()
        return 1

    # Print report
    print(f"\nEndpoint Coverage Report")
    print(f"=======================")
    print(f"Total endpoints scanned: {report.total_endpoints}")
    print(f"Covered by SDK:          {report.covered}")
    print(f"Missing from SDK:        {len(report.missing)}")
    print(f"Coverage:                {report.coverage_percent}%")

    if report.missing:
        print(f"\nMissing endpoints (first 20):")
        for ep in report.missing[:20]:
            print(f"  {ep.method:7s} {ep.path}")
            if ep.description:
                print(f"          {ep.description}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
