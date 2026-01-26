"""
Documentation Generator Agent.

AI agent specialized in generating documentation for code:
- Docstring generation for functions and classes
- API reference documentation
- README section generation
- Architecture Decision Records (ADRs)
- Inline comment suggestions
- Usage examples

Works with multi-agent workflows for comprehensive documentation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from aragora.agents.base import BaseDebateAgent

logger = logging.getLogger(__name__)


class DocType(str, Enum):
    """Types of documentation that can be generated."""

    DOCSTRING = "docstring"
    API_REFERENCE = "api_reference"
    README = "readme"
    ADR = "adr"
    TUTORIAL = "tutorial"
    CHANGELOG = "changelog"
    COMMENT = "comment"
    TYPE_HINTS = "type_hints"


class DocStyle(str, Enum):
    """Documentation style conventions."""

    GOOGLE = "google"
    NUMPY = "numpy"
    SPHINX = "sphinx"
    EPYTEXT = "epytext"
    PLAIN = "plain"


@dataclass
class CodeElement:
    """Represents a code element to document."""

    name: str
    element_type: str  # function, class, method, module, variable
    code: str
    file_path: str
    line_number: int
    existing_docstring: str | None = None
    parameters: list[dict[str, str]] = field(default_factory=list)
    return_type: str | None = None
    decorators: list[str] = field(default_factory=list)
    parent_class: str | None = None
    complexity: int = 1


@dataclass
class DocstringResult:
    """Generated docstring result."""

    element_name: str
    docstring: str
    style: DocStyle
    includes_examples: bool = False
    includes_raises: bool = False
    includes_returns: bool = True
    quality_score: float = 0.0


@dataclass
class APIDocResult:
    """Generated API documentation result."""

    module_name: str
    sections: list[dict[str, str]]
    format: str = "markdown"
    includes_examples: bool = True


@dataclass
class ADRResult:
    """Generated Architecture Decision Record."""

    title: str
    status: str  # proposed, accepted, deprecated, superseded
    context: str
    decision: str
    consequences: list[str]
    alternatives_considered: list[dict[str, Any]]
    references: list[str] = field(default_factory=list)


@dataclass
class DocumentationGap:
    """Identified gap in documentation."""

    file_path: str
    element_name: str
    gap_type: str  # "missing_docstring", "incomplete_params", "no_examples"
    description: str
    priority: int = 1


# Common parameter type descriptions
PARAM_TYPE_DESCRIPTIONS: dict[str, str] = {
    "str": "string value",
    "int": "integer value",
    "float": "floating-point number",
    "bool": "boolean flag",
    "list": "list of items",
    "dict": "dictionary mapping",
    "Optional": "optional value, can be None",
    "Any": "value of any type",
    "Callable": "callable function or method",
    "Path": "file system path",
}


class DocGeneratorAgent(BaseDebateAgent):
    """
    AI agent for generating documentation.

    Capabilities:
    - Generate docstrings in multiple styles (Google, NumPy, Sphinx)
    - Create API reference documentation
    - Write README sections
    - Generate Architecture Decision Records (ADRs)
    - Add inline comments for complex logic
    - Create usage examples
    """

    def __init__(self, style: DocStyle = DocStyle.GOOGLE, **kwargs):
        system_prompt = """You are a Documentation Specialist, an expert in technical writing.

Your expertise includes:
1. Writing clear, concise docstrings that explain what code does
2. Documenting parameters, return values, and exceptions
3. Creating usage examples that demonstrate typical use cases
4. Writing API reference documentation
5. Creating Architecture Decision Records (ADRs)
6. Explaining complex algorithms in simple terms

Guidelines:
- Focus on the "why" not just the "what"
- Use consistent terminology throughout
- Include examples for non-trivial functions
- Document edge cases and limitations
- Keep descriptions concise but complete
- Follow the specified documentation style consistently
"""
        super().__init__(
            name="doc_generator",
            model=kwargs.get("model", "claude-3.5-sonnet"),
            persona=system_prompt,
            **kwargs,
        )
        self.style = style

    def generate_docstring(
        self,
        element: CodeElement,
        include_examples: bool = True,
    ) -> DocstringResult:
        """
        Generate a docstring for a code element.

        Args:
            element: Code element to document
            include_examples: Whether to include usage examples

        Returns:
            Generated docstring with metadata
        """
        # Extract function signature info
        param_docs = self._document_parameters(element.parameters)
        return_doc = self._document_return(element.return_type)

        # Generate summary from code analysis
        summary = self._generate_summary(element)

        # Build docstring based on style
        if self.style == DocStyle.GOOGLE:
            docstring = self._format_google_docstring(
                summary=summary,
                params=param_docs,
                returns=return_doc,
                include_examples=include_examples,
                element=element,
            )
        elif self.style == DocStyle.NUMPY:
            docstring = self._format_numpy_docstring(
                summary=summary,
                params=param_docs,
                returns=return_doc,
                include_examples=include_examples,
                element=element,
            )
        else:
            docstring = self._format_sphinx_docstring(
                summary=summary,
                params=param_docs,
                returns=return_doc,
            )

        return DocstringResult(
            element_name=element.name,
            docstring=docstring,
            style=self.style,
            includes_examples=include_examples and bool(element.parameters),
            includes_returns=element.return_type is not None,
            quality_score=self._assess_docstring_quality(docstring, element),
        )

    def generate_api_reference(
        self,
        module_code: str,
        module_name: str,
        include_private: bool = False,
    ) -> APIDocResult:
        """
        Generate API reference documentation for a module.

        Args:
            module_code: Source code of the module
            module_name: Name of the module
            include_private: Include private functions/methods

        Returns:
            API documentation result
        """
        sections: list[dict[str, str]] = []

        # Parse module to extract elements
        elements = self._parse_module_elements(module_code, module_name)

        # Filter private if needed
        if not include_private:
            elements = [e for e in elements if not e.name.startswith("_")]

        # Group by type
        classes = [e for e in elements if e.element_type == "class"]
        functions = [e for e in elements if e.element_type == "function"]
        constants = [e for e in elements if e.element_type == "variable"]

        # Module overview section
        sections.append(
            {
                "title": f"# {module_name}",
                "content": self._generate_module_overview(module_code),
            }
        )

        # Classes section
        if classes:
            class_docs = []
            for cls in classes:
                class_doc = self._document_class(cls)
                class_docs.append(class_doc)

            sections.append({"title": "## Classes", "content": "\n\n".join(class_docs)})

        # Functions section
        if functions:
            func_docs = []
            for func in functions:
                docstring = self.generate_docstring(func)
                func_doc = f"### `{func.name}`\n\n```python\n{docstring.docstring}\n```"
                func_docs.append(func_doc)

            sections.append({"title": "## Functions", "content": "\n\n".join(func_docs)})

        # Constants section
        if constants:
            const_docs = []
            for const in constants:
                const_docs.append(f"- `{const.name}`: {self._infer_constant_description(const)}")

            sections.append({"title": "## Constants", "content": "\n".join(const_docs)})

        return APIDocResult(
            module_name=module_name,
            sections=sections,
            format="markdown",
            includes_examples=True,
        )

    def generate_readme_section(
        self,
        section_type: str,
        project_info: dict[str, Any],
    ) -> str:
        """
        Generate a README section.

        Args:
            section_type: Type of section (installation, usage, api, etc.)
            project_info: Project metadata

        Returns:
            Markdown content for the section
        """
        generators = {
            "installation": self._generate_installation_section,
            "usage": self._generate_usage_section,
            "api": self._generate_api_section,
            "contributing": self._generate_contributing_section,
            "license": self._generate_license_section,
            "features": self._generate_features_section,
            "quickstart": self._generate_quickstart_section,
        }

        generator = generators.get(section_type)
        if not generator:
            return f"## {section_type.title()}\n\n[Section content here]"

        return generator(project_info)

    def generate_adr(
        self,
        decision_context: dict[str, Any],
    ) -> ADRResult:
        """
        Generate an Architecture Decision Record.

        Args:
            decision_context: Context about the decision including:
                - title: Decision title
                - context: Background and problem statement
                - options: List of considered alternatives
                - chosen: The selected option
                - rationale: Why this option was chosen

        Returns:
            ADR result with all sections
        """
        title = decision_context.get("title", "Untitled Decision")
        context = decision_context.get("context", "")
        options = decision_context.get("options", [])
        chosen = decision_context.get("chosen", "")
        rationale = decision_context.get("rationale", "")

        # Build consequences
        consequences = self._analyze_consequences(chosen, options)

        # Build alternatives
        alternatives = []
        for opt in options:
            if opt != chosen:
                alternatives.append(
                    {
                        "option": opt.get("name", str(opt)) if isinstance(opt, dict) else str(opt),
                        "pros": opt.get("pros", []) if isinstance(opt, dict) else [],
                        "cons": opt.get("cons", []) if isinstance(opt, dict) else [],
                        "rejected_reason": (
                            opt.get("rejected_reason", "Not selected")
                            if isinstance(opt, dict)
                            else "Not selected"
                        ),
                    }
                )

        return ADRResult(
            title=title,
            status="proposed",
            context=context,
            decision=f"We will {chosen}. {rationale}",
            consequences=consequences,
            alternatives_considered=alternatives,
        )

    def analyze_documentation_gaps(
        self,
        code: str,
        file_path: str,
    ) -> list[DocumentationGap]:
        """
        Analyze code for documentation gaps.

        Args:
            code: Source code to analyze
            file_path: Path to the file

        Returns:
            List of identified documentation gaps
        """
        gaps: list[DocumentationGap] = []

        # Parse code elements
        elements = self._parse_module_elements(code, file_path)

        for element in elements:
            # Check for missing docstring
            if not element.existing_docstring:
                priority = 1 if element.element_type in ("class", "function") else 3
                gaps.append(
                    DocumentationGap(
                        file_path=file_path,
                        element_name=element.name,
                        gap_type="missing_docstring",
                        description=f"{element.element_type.title()} '{element.name}' has no docstring",
                        priority=priority,
                    )
                )

            # Check for incomplete parameter docs
            elif element.parameters and element.existing_docstring:
                documented_params = self._extract_documented_params(element.existing_docstring)
                for param in element.parameters:
                    param_name = param.get("name", "")
                    if param_name and param_name not in documented_params:
                        gaps.append(
                            DocumentationGap(
                                file_path=file_path,
                                element_name=element.name,
                                gap_type="incomplete_params",
                                description=f"Parameter '{param_name}' is not documented in '{element.name}'",
                                priority=2,
                            )
                        )

            # Check for missing return documentation
            if (
                element.return_type
                and element.return_type != "None"
                and element.existing_docstring
                and "Returns:" not in element.existing_docstring
                and ":returns:" not in element.existing_docstring
            ):
                gaps.append(
                    DocumentationGap(
                        file_path=file_path,
                        element_name=element.name,
                        gap_type="missing_return",
                        description=f"Return value not documented for '{element.name}'",
                        priority=2,
                    )
                )

            # Check for missing examples in complex functions
            if (
                element.complexity > 5
                and element.existing_docstring
                and "Example" not in element.existing_docstring
                and ">>>" not in element.existing_docstring
            ):
                gaps.append(
                    DocumentationGap(
                        file_path=file_path,
                        element_name=element.name,
                        gap_type="no_examples",
                        description=f"Complex function '{element.name}' lacks usage examples",
                        priority=3,
                    )
                )

        return gaps

    def suggest_inline_comments(
        self,
        code: str,
    ) -> list[dict[str, Any]]:
        """
        Suggest inline comments for complex code sections.

        Args:
            code: Source code to analyze

        Returns:
            List of suggested comments with line numbers
        """
        suggestions: list[dict[str, Any]] = []

        lines = code.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Complex conditionals
            if re.match(r"^if\s+.+and.+or.+:", stripped) or re.match(
                r"^if\s+.+or.+and.+:", stripped
            ):
                suggestions.append(
                    {
                        "line_number": i + 1,
                        "type": "complex_conditional",
                        "suggestion": "# Explain what this condition checks for",
                        "code_snippet": stripped[:50],
                    }
                )

            # Magic numbers
            if re.search(r"(?<![a-zA-Z_])\d{3,}(?![a-zA-Z_\d])", stripped) and "=" in stripped:
                suggestions.append(
                    {
                        "line_number": i + 1,
                        "type": "magic_number",
                        "suggestion": "# Consider extracting to a named constant and documenting its meaning",
                        "code_snippet": stripped[:50],
                    }
                )

            # Regex patterns
            if re.search(r'r["\'][^"\']{15,}["\']', stripped):
                suggestions.append(
                    {
                        "line_number": i + 1,
                        "type": "complex_regex",
                        "suggestion": "# Explain what this regex pattern matches",
                        "code_snippet": stripped[:50],
                    }
                )

            # Lambda with complex logic
            if "lambda" in stripped and len(stripped) > 60:
                suggestions.append(
                    {
                        "line_number": i + 1,
                        "type": "complex_lambda",
                        "suggestion": "# Explain what this lambda function does",
                        "code_snippet": stripped[:50],
                    }
                )

        return suggestions

    # Private helper methods

    def _document_parameters(self, params: list[dict[str, str]]) -> list[dict[str, str]]:
        """Generate documentation for parameters."""
        docs = []
        for param in params:
            name = param.get("name", "")
            type_hint = param.get("type", "Any")
            default = param.get("default")

            description = PARAM_TYPE_DESCRIPTIONS.get(type_hint.split("[")[0], f"{type_hint} value")

            if default:
                description += f" (default: {default})"

            docs.append({"name": name, "type": type_hint, "description": description})

        return docs

    def _document_return(self, return_type: str | None) -> dict[str, str]:
        """Generate return documentation."""
        if not return_type or return_type == "None":
            return {"type": "None", "description": "This function does not return a value"}

        base_type = return_type.split("[")[0]
        description = PARAM_TYPE_DESCRIPTIONS.get(base_type, f"{return_type} result")

        return {"type": return_type, "description": description}

    def _generate_summary(self, element: CodeElement) -> str:
        """Generate a summary description from code analysis."""
        name = element.name
        words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", name)
        readable = " ".join(w.lower() for w in words)

        if element.element_type == "class":
            return f"A class that represents {readable}."
        elif element.element_type == "function":
            if name.startswith("get"):
                return f"Retrieve the {readable.replace('get ', '')}."
            elif name.startswith("set"):
                return f"Set the {readable.replace('set ', '')}."
            elif name.startswith("is") or name.startswith("has"):
                return f"Check if {readable.replace('is ', '').replace('has ', '')}."
            elif name.startswith("create"):
                return f"Create a new {readable.replace('create ', '')}."
            elif name.startswith("delete") or name.startswith("remove"):
                return f"Remove the {readable.replace('delete ', '').replace('remove ', '')}."
            elif name.startswith("update"):
                return f"Update the {readable.replace('update ', '')}."
            else:
                return f"Perform {readable} operation."
        else:
            return f"The {readable}."

    def _format_google_docstring(
        self,
        summary: str,
        params: list[dict[str, str]],
        returns: dict[str, str],
        include_examples: bool,
        element: CodeElement,
    ) -> str:
        """Format docstring in Google style."""
        lines = [summary, ""]

        if params:
            lines.append("Args:")
            for p in params:
                lines.append(f"    {p['name']}: {p['description']}")
            lines.append("")

        if returns["type"] != "None":
            lines.append("Returns:")
            lines.append(f"    {returns['description']}")
            lines.append("")

        if include_examples and element.parameters:
            lines.append("Example:")
            lines.append(f"    >>> {element.name}(...)")
            lines.append("")

        return '"""' + "\n".join(lines).strip() + '\n"""'

    def _format_numpy_docstring(
        self,
        summary: str,
        params: list[dict[str, str]],
        returns: dict[str, str],
        include_examples: bool,
        element: CodeElement,
    ) -> str:
        """Format docstring in NumPy style."""
        lines = [summary, ""]

        if params:
            lines.append("Parameters")
            lines.append("-" * 10)
            for p in params:
                lines.append(f"{p['name']} : {p['type']}")
                lines.append(f"    {p['description']}")
            lines.append("")

        if returns["type"] != "None":
            lines.append("Returns")
            lines.append("-" * 7)
            lines.append(f"{returns['type']}")
            lines.append(f"    {returns['description']}")
            lines.append("")

        if include_examples and element.parameters:
            lines.append("Examples")
            lines.append("-" * 8)
            lines.append(f">>> {element.name}(...)")
            lines.append("")

        return '"""' + "\n".join(lines).strip() + '\n"""'

    def _format_sphinx_docstring(
        self,
        summary: str,
        params: list[dict[str, str]],
        returns: dict[str, str],
    ) -> str:
        """Format docstring in Sphinx style."""
        lines = [summary, ""]

        for p in params:
            lines.append(f":param {p['name']}: {p['description']}")
            lines.append(f":type {p['name']}: {p['type']}")

        if returns["type"] != "None":
            lines.append(f":returns: {returns['description']}")
            lines.append(f":rtype: {returns['type']}")

        return '"""' + "\n".join(lines).strip() + '\n"""'

    def _assess_docstring_quality(self, docstring: str, element: CodeElement) -> float:
        """Assess the quality of a docstring on a 0-1 scale."""
        score = 0.0
        max_score = 5.0

        # Has summary
        if len(docstring) > 20:
            score += 1.0

        # Documents all parameters
        for param in element.parameters:
            if param.get("name", "") in docstring:
                score += 0.5

        # Documents return value
        if element.return_type and element.return_type != "None":
            if "Returns:" in docstring or ":returns:" in docstring:
                score += 1.0

        # Has examples
        if "Example" in docstring or ">>>" in docstring:
            score += 1.0

        # Reasonable length
        if 50 < len(docstring) < 1000:
            score += 0.5

        return min(score / max_score, 1.0)

    def _parse_module_elements(
        self,
        code: str,
        module_name: str,
    ) -> list[CodeElement]:
        """Parse module to extract documentable elements."""
        elements: list[CodeElement] = []

        # Simple regex-based parsing (would use AST in production)
        # Find classes
        class_pattern = r"^class\s+(\w+).*?:"
        for match in re.finditer(class_pattern, code, re.MULTILINE):
            elements.append(
                CodeElement(
                    name=match.group(1),
                    element_type="class",
                    code=match.group(0),
                    file_path=module_name,
                    line_number=code[: match.start()].count("\n") + 1,
                )
            )

        # Find functions
        func_pattern = r"^(async\s+)?def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?:"
        for match in re.finditer(func_pattern, code, re.MULTILINE):
            _is_async = bool(match.group(1))  # noqa: F841
            name = match.group(2)
            params_str = match.group(3)
            return_type = match.group(4).strip() if match.group(4) else None

            # Parse parameters
            params = []
            if params_str:
                for param in params_str.split(","):
                    param = param.strip()
                    if param and param != "self" and param != "cls":
                        if ":" in param:
                            p_name, p_type = param.split(":", 1)
                            if "=" in p_type:
                                p_type, default = p_type.split("=", 1)
                                params.append(
                                    {
                                        "name": p_name.strip(),
                                        "type": p_type.strip(),
                                        "default": default.strip(),
                                    }
                                )
                            else:
                                params.append({"name": p_name.strip(), "type": p_type.strip()})
                        else:
                            params.append({"name": param.split("=")[0].strip(), "type": "Any"})

            elements.append(
                CodeElement(
                    name=name,
                    element_type="function",
                    code=match.group(0),
                    file_path=module_name,
                    line_number=code[: match.start()].count("\n") + 1,
                    parameters=params,
                    return_type=return_type,
                )
            )

        return elements

    def _generate_module_overview(self, code: str) -> str:
        """Generate module overview from code."""
        # Extract module docstring if present
        match = re.match(r'^["\'][\'"]{2}(.*?)["\'][\'"]{2}', code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "Module documentation."

    def _document_class(self, cls: CodeElement) -> str:
        """Generate documentation for a class."""
        return f"### `{cls.name}`\n\n{self._generate_summary(cls)}"

    def _infer_constant_description(self, const: CodeElement) -> str:
        """Infer description for a constant."""
        name = const.name.lower().replace("_", " ")
        return f"Defines the {name}"

    def _extract_documented_params(self, docstring: str) -> set[str]:
        """Extract parameter names that are documented in a docstring."""
        params = set()

        # Google style
        for match in re.finditer(r"^\s+(\w+):", docstring, re.MULTILINE):
            params.add(match.group(1))

        # Sphinx style
        for match in re.finditer(r":param\s+(\w+):", docstring):
            params.add(match.group(1))

        # NumPy style
        for match in re.finditer(r"^(\w+)\s*:", docstring, re.MULTILINE):
            params.add(match.group(1))

        return params

    def _analyze_consequences(
        self,
        chosen: Any,
        options: list[Any],
    ) -> list[str]:
        """Analyze consequences of a decision."""
        consequences = [
            "Team will need to learn and adopt this approach",
            "Future changes may be influenced by this decision",
        ]

        if isinstance(chosen, dict):
            if chosen.get("complexity") == "high":
                consequences.append("Higher initial implementation effort required")
            if chosen.get("scalability") == "high":
                consequences.append("System will be better prepared for growth")

        return consequences

    def _generate_installation_section(self, info: dict[str, Any]) -> str:
        """Generate installation section."""
        name = info.get("name", "package")
        return f"""## Installation

```bash
pip install {name}
```

Or with optional dependencies:

```bash
pip install {name}[all]
```
"""

    def _generate_usage_section(self, info: dict[str, Any]) -> str:
        """Generate usage section."""
        name = info.get("name", "package")
        return f"""## Usage

```python
from {name} import main_class

# Initialize
instance = main_class()

# Basic usage
result = instance.do_something()
```
"""

    def _generate_api_section(self, info: dict[str, Any]) -> str:
        """Generate API section."""
        return """## API Reference

See the [API documentation](./docs/api.md) for detailed information.
"""

    def _generate_contributing_section(self, info: dict[str, Any]) -> str:
        """Generate contributing section."""
        return """## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
"""

    def _generate_license_section(self, info: dict[str, Any]) -> str:
        """Generate license section."""
        license_type = info.get("license", "MIT")
        return f"""## License

This project is licensed under the {license_type} License - see the [LICENSE](LICENSE) file for details.
"""

    def _generate_features_section(self, info: dict[str, Any]) -> str:
        """Generate features section."""
        features = info.get("features", [])
        if features:
            feature_list = "\n".join(f"- {f}" for f in features)
            return f"""## Features

{feature_list}
"""
        return """## Features

- Feature 1
- Feature 2
- Feature 3
"""

    def _generate_quickstart_section(self, info: dict[str, Any]) -> str:
        """Generate quickstart section."""
        name = info.get("name", "package")
        return f"""## Quick Start

```python
# Import the library
import {name}

# Create an instance
app = {name}.create()

# Run
app.run()
```
"""


# Documentation workflow templates

DOCUMENTATION_WORKFLOW_TEMPLATE: dict[str, Any] = {
    "name": "Documentation Generation Workflow",
    "description": "Generate comprehensive documentation for a codebase",
    "category": "documentation",
    "version": "1.0",
    "tags": ["documentation", "docstrings", "api-docs", "readme"],
    "steps": [
        {
            "id": "scan_codebase",
            "type": "task",
            "name": "Scan Codebase",
            "description": "Scan codebase for documentable elements",
            "config": {
                "task_type": "function",
                "function_name": "scan_for_documentation_targets",
                "file_patterns": ["*.py"],
            },
        },
        {
            "id": "identify_gaps",
            "type": "task",
            "name": "Identify Documentation Gaps",
            "description": "Find elements missing documentation",
            "config": {
                "task_type": "function",
                "function_name": "analyze_documentation_gaps",
            },
        },
        {
            "id": "prioritize_gaps",
            "type": "debate",
            "name": "Prioritize Documentation",
            "description": "Multi-agent debate on documentation priorities",
            "config": {
                "agents": ["doc_generator", "code_quality_reviewer"],
                "rounds": 2,
                "topic_template": "Which documentation gaps should we address first? Gaps: {gaps}",
            },
        },
        {
            "id": "generate_docstrings",
            "type": "task",
            "name": "Generate Docstrings",
            "description": "Generate docstrings for prioritized elements",
            "config": {
                "task_type": "function",
                "function_name": "batch_generate_docstrings",
                "style": "google",
            },
        },
        {
            "id": "generate_api_docs",
            "type": "task",
            "name": "Generate API Documentation",
            "description": "Generate API reference documentation",
            "config": {
                "task_type": "function",
                "function_name": "generate_api_reference",
                "format": "markdown",
            },
        },
        {
            "id": "review_quality",
            "type": "debate",
            "name": "Review Documentation Quality",
            "description": "Multi-agent review of generated documentation",
            "config": {
                "agents": ["doc_generator", "code_quality_reviewer"],
                "rounds": 2,
                "topic_template": "Review documentation quality: {documentation}",
            },
        },
        {
            "id": "human_review",
            "type": "human_checkpoint",
            "name": "Developer Review",
            "description": "Developer reviews generated documentation",
            "config": {
                "approval_type": "review",
                "checklist": [
                    "Documentation is accurate",
                    "Examples are correct",
                    "No sensitive information exposed",
                    "Consistent style throughout",
                ],
            },
        },
    ],
    "transitions": [
        {"from": "scan_codebase", "to": "identify_gaps"},
        {"from": "identify_gaps", "to": "prioritize_gaps"},
        {"from": "prioritize_gaps", "to": "generate_docstrings"},
        {"from": "generate_docstrings", "to": "generate_api_docs"},
        {"from": "generate_api_docs", "to": "review_quality"},
        {"from": "review_quality", "to": "human_review"},
    ],
}


DOCUMENTATION_TEMPLATES = {
    "documentation_workflow": DOCUMENTATION_WORKFLOW_TEMPLATE,
}


def get_documentation_template(name: str) -> dict[str, Any]:
    """
    Get a documentation workflow template by name.

    Args:
        name: Template name

    Returns:
        Template dictionary

    Raises:
        KeyError: If template not found
    """
    if name not in DOCUMENTATION_TEMPLATES:
        raise KeyError(
            f"Unknown documentation template: {name}. "
            f"Available: {list(DOCUMENTATION_TEMPLATES.keys())}"
        )
    return DOCUMENTATION_TEMPLATES[name]
