"""
HTTP API Handlers for Code Intelligence Analysis.

Provides REST APIs for semantic code analysis:
- AST-based code parsing and symbol extraction
- Call graph construction and analysis
- Dead code detection
- Impact analysis
- Codebase understanding queries

Endpoints:
- POST /api/v1/codebase/{repo}/analyze - Analyze codebase structure
- GET /api/v1/codebase/{repo}/symbols - List symbols (classes, functions)
- GET /api/v1/codebase/{repo}/callgraph - Get call graph
- GET /api/v1/codebase/{repo}/deadcode - Find dead/unreachable code
- POST /api/v1/codebase/{repo}/impact - Analyze impact of changes
- POST /api/v1/codebase/{repo}/understand - Answer questions about code
- POST /api/v1/codebase/{repo}/audit - Run comprehensive audit
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    success_response,
)
from aragora.server.handlers.secure import (
    ForbiddenError,
    SecureHandler,
    UnauthorizedError,
)
from aragora.services import ServiceRegistry

# Permission constants for codebase intelligence operations
CODEBASE_READ_PERMISSION = "codebase:read"
CODEBASE_ANALYZE_PERMISSION = "codebase:analyze"
CODEBASE_AUDIT_PERMISSION = "codebase:audit"

logger = logging.getLogger(__name__)


# =============================================================================
# Service Registry Integration (Lazy Loading)
# =============================================================================


def _get_code_intelligence():
    """Get or create CodeIntelligence from service registry."""
    registry = ServiceRegistry.get()
    try:
        from aragora.analysis.code_intelligence import CodeIntelligence

        if not registry.has(CodeIntelligence):
            intel = CodeIntelligence()
            registry.register(CodeIntelligence, intel)
            logger.info("Registered CodeIntelligence with service registry")
        return registry.resolve(CodeIntelligence)
    except ImportError:
        logger.warning("CodeIntelligence not available")
        return None


def _get_call_graph_builder():
    """Get or create CallGraphBuilder from service registry."""
    registry = ServiceRegistry.get()
    try:
        from aragora.analysis.call_graph import CallGraphBuilder

        if not registry.has(CallGraphBuilder):
            intel = _get_code_intelligence()
            builder = CallGraphBuilder(intel)
            registry.register(CallGraphBuilder, builder)
            logger.info("Registered CallGraphBuilder with service registry")
        return registry.resolve(CallGraphBuilder)
    except ImportError:
        logger.warning("CallGraphBuilder not available")
        return None


def _get_security_scanner():
    """Get or create SecurityScanner from service registry."""
    registry = ServiceRegistry.get()
    try:
        from aragora.audit.security_scanner import SecurityScanner

        if not registry.has(SecurityScanner):
            scanner = SecurityScanner()
            registry.register(SecurityScanner, scanner)
            logger.info("Registered SecurityScanner with service registry")
        return registry.resolve(SecurityScanner)
    except ImportError:
        logger.warning("SecurityScanner not available")
        return None


def _get_bug_detector():
    """Get or create BugDetector from service registry."""
    registry = ServiceRegistry.get()
    try:
        from aragora.audit.bug_detector import BugDetector

        if not registry.has(BugDetector):
            detector = BugDetector()
            registry.register(BugDetector, detector)
            logger.info("Registered BugDetector with service registry")
        return registry.resolve(BugDetector)
    except ImportError:
        logger.warning("BugDetector not available")
        return None


# =============================================================================
# In-Memory Storage (replace with database in production)
# =============================================================================

_analysis_results: Dict[str, Dict[str, Any]] = {}  # repo_id -> {analysis_id -> result}
_analysis_lock = threading.Lock()
_running_analyses: Dict[str, asyncio.Task] = {}

_callgraph_cache: Dict[str, Any] = {}  # repo_id -> callgraph
_callgraph_lock = threading.Lock()

_audit_results: Dict[str, Dict[str, Any]] = {}  # repo_id -> {audit_id -> result}
_audit_lock = threading.Lock()
_running_audits: Dict[str, asyncio.Task] = {}


def _get_or_create_repo_analyses(repo_id: str) -> Dict[str, Any]:
    """Get or create analysis storage for a repository."""
    with _analysis_lock:
        if repo_id not in _analysis_results:
            _analysis_results[repo_id] = {}
        return _analysis_results[repo_id]


def _get_or_create_repo_audits(repo_id: str) -> Dict[str, Any]:
    """Get or create audit storage for a repository."""
    with _audit_lock:
        if repo_id not in _audit_results:
            _audit_results[repo_id] = {}
        return _audit_results[repo_id]


# =============================================================================
# Code Intelligence Handlers
# =============================================================================


class IntelligenceHandler(SecureHandler):
    """Handler class for code intelligence endpoints."""

    ROUTES = [
        "/api/codebase/analyze",
        "/api/codebase/symbols",
        "/api/codebase/callgraph",
        "/api/codebase/deadcode",
        "/api/codebase/impact",
        "/api/codebase/understand",
        "/api/codebase/audit",
        "/api/v1/codebase/analyze",
        "/api/v1/codebase/symbols",
        "/api/v1/codebase/callgraph",
        "/api/v1/codebase/deadcode",
        "/api/v1/codebase/impact",
        "/api/v1/codebase/understand",
        "/api/v1/codebase/audit",
    ]

    @classmethod
    def can_handle(cls, path: str) -> bool:
        """Check if this handler can process the given path."""
        # Match exact routes or routes with repo_id
        if path in cls.ROUTES:
            return True
        # Match /api/codebase/{repo_id}/... or /api/v1/codebase/{repo_id}/...
        if path.startswith("/api/codebase/") or path.startswith("/api/v1/codebase/"):
            return True
        return False

    async def _check_permission(self, handler: Any, permission: str) -> None:
        """Check if user has required permission."""
        auth_context = await self.get_auth_context(handler, require_auth=True)
        self.check_permission(auth_context, permission)

    async def analyze(
        self, repo_id: str, body: Dict[str, Any], handler: Any = None
    ) -> HandlerResult:
        """Analyze codebase structure and extract symbols."""
        # RBAC: Require codebase:analyze permission
        if handler:
            try:
                await self._check_permission(handler, CODEBASE_ANALYZE_PERMISSION)
            except UnauthorizedError:
                return error_response("Authentication required", 401)
            except ForbiddenError as e:
                return error_response(str(e), 403)
        return await handle_analyze_codebase(repo_id, body)

    async def get_symbols(
        self, repo_id: str, params: Dict[str, Any], handler: Any = None
    ) -> HandlerResult:
        """Get symbols from the codebase."""
        # RBAC: Require codebase:read permission
        if handler:
            try:
                await self._check_permission(handler, CODEBASE_READ_PERMISSION)
            except UnauthorizedError:
                return error_response("Authentication required", 401)
            except ForbiddenError as e:
                return error_response(str(e), 403)
        return await handle_get_symbols(repo_id, params)

    async def get_callgraph(
        self, repo_id: str, params: Dict[str, Any], handler: Any = None
    ) -> HandlerResult:
        """Get call graph for the codebase."""
        # RBAC: Require codebase:analyze permission
        if handler:
            try:
                await self._check_permission(handler, CODEBASE_ANALYZE_PERMISSION)
            except UnauthorizedError:
                return error_response("Authentication required", 401)
            except ForbiddenError as e:
                return error_response(str(e), 403)
        return await handle_get_callgraph(repo_id, params)

    async def find_deadcode(
        self, repo_id: str, params: Dict[str, Any], handler: Any = None
    ) -> HandlerResult:
        """Find dead/unreachable code."""
        # RBAC: Require codebase:analyze permission
        if handler:
            try:
                await self._check_permission(handler, CODEBASE_ANALYZE_PERMISSION)
            except UnauthorizedError:
                return error_response("Authentication required", 401)
            except ForbiddenError as e:
                return error_response(str(e), 403)
        return await handle_find_deadcode(repo_id, params)

    async def analyze_impact(
        self, repo_id: str, body: Dict[str, Any], handler: Any = None
    ) -> HandlerResult:
        """Analyze impact of changes to a symbol."""
        # RBAC: Require codebase:analyze permission
        if handler:
            try:
                await self._check_permission(handler, CODEBASE_ANALYZE_PERMISSION)
            except UnauthorizedError:
                return error_response("Authentication required", 401)
            except ForbiddenError as e:
                return error_response(str(e), 403)
        return await handle_analyze_impact(repo_id, body)

    async def understand(
        self, repo_id: str, body: Dict[str, Any], handler: Any = None
    ) -> HandlerResult:
        """Answer questions about the codebase."""
        # RBAC: Require codebase:read permission
        if handler:
            try:
                await self._check_permission(handler, CODEBASE_READ_PERMISSION)
            except UnauthorizedError:
                return error_response("Authentication required", 401)
            except ForbiddenError as e:
                return error_response(str(e), 403)
        return await handle_understand(repo_id, body)

    async def audit(self, repo_id: str, body: Dict[str, Any], handler: Any = None) -> HandlerResult:
        """Run comprehensive code audit."""
        # RBAC: Require codebase:audit permission
        if handler:
            try:
                await self._check_permission(handler, CODEBASE_AUDIT_PERMISSION)
            except UnauthorizedError:
                return error_response("Authentication required", 401)
            except ForbiddenError as e:
                return error_response(str(e), 403)
        return await handle_audit(repo_id, body)

    async def get_audit_status(
        self, repo_id: str, audit_id: str, params: Dict[str, Any], handler: Any = None
    ) -> HandlerResult:
        """Get status of an audit."""
        # RBAC: Require codebase:read permission
        if handler:
            try:
                await self._check_permission(handler, CODEBASE_READ_PERMISSION)
            except UnauthorizedError:
                return error_response("Authentication required", 401)
            except ForbiddenError as e:
                return error_response(str(e), 403)
        return await handle_get_audit_status(repo_id, audit_id, params)


# =============================================================================
# Analyze Codebase Handler
# =============================================================================


async def handle_analyze_codebase(repo_id: str, body: Dict[str, Any]) -> HandlerResult:
    """
    Analyze codebase structure and extract symbols.

    Request body:
    {
        "path": "/path/to/codebase",
        "exclude_patterns": ["__pycache__", "node_modules"],
        "languages": ["python", "javascript"],  # optional filter
        "include_imports": true,
        "include_complexity": true
    }

    Response:
    {
        "analysis_id": "uuid",
        "status": "completed",
        "summary": {
            "total_files": 100,
            "total_lines": 5000,
            "languages": {"python": 80, "javascript": 20},
            "classes": 50,
            "functions": 200,
            "imports": 300
        },
        "files": [...]
    }
    """
    path = body.get("path")
    if not path:
        return error_response("Missing required field: path", status=400)

    if not Path(path).exists():
        return error_response(f"Path does not exist: {path}", status=404)

    code_intel = _get_code_intelligence()
    if not code_intel:
        return error_response("Code intelligence not available", status=503)

    exclude_patterns = body.get(
        "exclude_patterns", ["__pycache__", ".git", "node_modules", ".venv", "venv"]
    )
    include_imports = body.get("include_imports", True)
    include_complexity = body.get("include_complexity", True)

    analysis_id = str(uuid.uuid4())[:8]
    start_time = datetime.now(timezone.utc)

    try:
        # Run analysis
        analyses_dict = code_intel.analyze_directory(path, exclude_patterns=exclude_patterns)

        # Aggregate results
        total_files = 0
        total_lines = 0
        languages: Dict[str, int] = {}
        all_classes: List[Dict[str, Any]] = []
        all_functions: List[Dict[str, Any]] = []
        all_imports: List[Dict[str, Any]] = []
        file_summaries: List[Dict[str, Any]] = []

        for file_path, analysis in analyses_dict.items():
            total_files += 1
            total_lines += analysis.lines_of_code + analysis.comment_lines + analysis.blank_lines

            lang = analysis.language.value if analysis.language else "unknown"
            languages[lang] = languages.get(lang, 0) + 1

            # Extract classes
            for cls in analysis.classes:
                all_classes.append(
                    {
                        "name": cls.name,
                        "file": file_path,
                        "line": cls.location.start_line if cls.location else None,
                        "bases": cls.bases,
                        "methods": len(cls.methods),
                        "docstring": cls.docstring[:200] if cls.docstring else None,
                    }
                )

            # Extract functions
            for func in analysis.functions:
                func_info = {
                    "name": func.name,
                    "file": file_path,
                    "line": func.location.start_line if func.location else None,
                    "is_async": func.is_async,
                    "parameters": len(func.parameters),
                    "docstring": func.docstring[:200] if func.docstring else None,
                }
                if include_complexity:
                    func_info["complexity"] = func.complexity
                all_functions.append(func_info)

            # Extract imports
            if include_imports:
                for imp in analysis.imports:
                    all_imports.append(
                        {
                            "module": imp.module,
                            "names": imp.names,
                            "alias": imp.alias,
                            "file": file_path,
                        }
                    )

            # File summary
            file_summaries.append(
                {
                    "path": file_path,
                    "language": lang,
                    "lines": analysis.lines_of_code,
                    "classes": len(analysis.classes),
                    "functions": len(analysis.functions),
                }
            )

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        result = {
            "analysis_id": analysis_id,
            "status": "completed",
            "path": path,
            "elapsed_seconds": elapsed,
            "summary": {
                "total_files": total_files,
                "total_lines": total_lines,
                "languages": languages,
                "classes": len(all_classes),
                "functions": len(all_functions),
                "imports": len(all_imports),
            },
            "classes": all_classes[:100],  # Limit response size
            "functions": all_functions[:200],
            "imports": all_imports[:100] if include_imports else [],
            "files": file_summaries,
        }

        # Cache result
        repo_analyses = _get_or_create_repo_analyses(repo_id)
        repo_analyses[analysis_id] = result

        logger.info(f"[{analysis_id}] Analyzed {total_files} files in {elapsed:.2f}s")

        return success_response(result)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return error_response(f"Analysis failed: {str(e)}", status=500)


# =============================================================================
# Get Symbols Handler
# =============================================================================


async def handle_get_symbols(repo_id: str, params: Dict[str, Any]) -> HandlerResult:
    """
    Get symbols (classes, functions) from the codebase.

    Query params:
    - path: Path to analyze
    - type: Symbol type filter (class, function, all)
    - name: Name filter (partial match)
    - file: File filter

    Response:
    {
        "symbols": [
            {"name": "MyClass", "kind": "class", "file": "...", "line": 10},
            ...
        ],
        "total": 100
    }
    """
    path = params.get("path")
    if not path:
        return error_response("Missing required parameter: path", status=400)

    if not Path(path).exists():
        return error_response(f"Path does not exist: {path}", status=404)

    code_intel = _get_code_intelligence()
    if not code_intel:
        return error_response("Code intelligence not available", status=503)

    symbol_type = params.get("type", "all")
    name_filter = params.get("name", "").lower()
    file_filter = params.get("file", "")

    try:
        analyses_dict = code_intel.analyze_directory(path)
        symbols: List[Dict[str, Any]] = []

        for file_path, analysis in analyses_dict.items():
            # Apply file filter
            if file_filter and file_filter not in file_path:
                continue

            # Extract classes
            if symbol_type in ("all", "class"):
                for cls in analysis.classes:
                    if name_filter and name_filter not in cls.name.lower():
                        continue
                    symbols.append(
                        {
                            "name": cls.name,
                            "kind": "class",
                            "file": file_path,
                            "line": cls.location.start_line if cls.location else None,
                            "bases": cls.bases,
                            "visibility": cls.visibility.value if cls.visibility else "public",
                        }
                    )

            # Extract functions
            if symbol_type in ("all", "function"):
                for func in analysis.functions:
                    if name_filter and name_filter not in func.name.lower():
                        continue
                    symbols.append(
                        {
                            "name": func.name,
                            "kind": "function",
                            "file": file_path,
                            "line": func.location.start_line if func.location else None,
                            "is_async": func.is_async,
                            "visibility": func.visibility.value if func.visibility else "public",
                        }
                    )

        return success_response(
            {
                "symbols": symbols[:500],  # Limit response size
                "total": len(symbols),
            }
        )

    except Exception as e:
        logger.error(f"Symbol extraction failed: {e}")
        return error_response(f"Symbol extraction failed: {str(e)}", status=500)


# =============================================================================
# Call Graph Handler
# =============================================================================


async def handle_get_callgraph(repo_id: str, params: Dict[str, Any]) -> HandlerResult:
    """
    Get call graph for the codebase.

    Query params:
    - path: Path to analyze
    - format: Output format (json, dot)
    - depth: Maximum depth for graph traversal

    Response:
    {
        "metrics": {
            "nodes": 100,
            "edges": 200,
            "density": 0.04
        },
        "nodes": [...],
        "edges": [...],
        "hotspots": [...]
    }
    """
    path = params.get("path")
    if not path:
        return error_response("Missing required parameter: path", status=400)

    if not Path(path).exists():
        return error_response(f"Path does not exist: {path}", status=404)

    builder = _get_call_graph_builder()
    if not builder:
        return error_response("Call graph builder not available", status=503)

    try:
        # Check cache
        cache_key = f"{repo_id}:{path}"
        with _callgraph_lock:
            if cache_key in _callgraph_cache:
                cached = _callgraph_cache[cache_key]
                # Return cached if less than 5 minutes old
                if (datetime.now(timezone.utc) - cached["timestamp"]).seconds < 300:
                    return success_response(cached["data"])

        # Build call graph
        graph = builder.build_from_directory(path)

        # Get metrics
        metrics = graph.get_complexity_metrics()

        # Get hotspots (most called functions)
        hotspots = graph.get_hotspots(top_n=20)
        hotspot_data = [
            {
                "name": node.qualified_name,
                "callers": count,
                "file": node.location.file_path if node.location else None,
                "line": node.location.start_line if node.location else None,
            }
            for node, count in hotspots
        ]

        # Convert to serializable format
        graph_data = graph.to_dict()

        result = {
            "metrics": metrics,
            "nodes": graph_data.get("nodes", [])[:200],
            "edges": graph_data.get("edges", [])[:500],
            "hotspots": hotspot_data,
            "entry_points": graph_data.get("entry_points", []),
        }

        # Cache result
        with _callgraph_lock:
            _callgraph_cache[cache_key] = {
                "data": result,
                "timestamp": datetime.now(timezone.utc),
            }

        return success_response(result)

    except Exception as e:
        logger.error(f"Call graph construction failed: {e}")
        return error_response(f"Call graph construction failed: {str(e)}", status=500)


# =============================================================================
# Dead Code Handler
# =============================================================================


async def handle_find_deadcode(repo_id: str, params: Dict[str, Any]) -> HandlerResult:
    """
    Find dead/unreachable code.

    Query params:
    - path: Path to analyze
    - entry_points: Comma-separated list of entry point functions

    Response:
    {
        "unreachable_functions": [...],
        "unreachable_classes": [...],
        "total_dead_lines": 500
    }
    """
    path = params.get("path")
    if not path:
        return error_response("Missing required parameter: path", status=400)

    if not Path(path).exists():
        return error_response(f"Path does not exist: {path}", status=404)

    builder = _get_call_graph_builder()
    if not builder:
        return error_response("Call graph builder not available", status=503)

    try:
        # Build call graph
        graph = builder.build_from_directory(path)

        # Mark entry points
        entry_points = params.get("entry_points", "").split(",")
        for ep in entry_points:
            ep = ep.strip()
            if ep:
                graph.mark_entry_point(ep)

        # Find dead code
        dead_code = graph.find_dead_code()

        result = {
            "unreachable_functions": [
                {
                    "name": n.qualified_name,
                    "file": n.location.file_path if n.location else None,
                    "line": n.location.start_line if n.location else None,
                }
                for n in dead_code.unreachable_functions[:100]
            ],
            "unreachable_classes": [
                {
                    "name": n.qualified_name,
                    "file": n.location.file_path if n.location else None,
                    "line": n.location.start_line if n.location else None,
                }
                for n in dead_code.unreachable_classes[:50]
            ],
            "total_dead_lines": dead_code.total_dead_lines,
            "summary": {
                "unreachable_functions_count": len(dead_code.unreachable_functions),
                "unreachable_classes_count": len(dead_code.unreachable_classes),
            },
        }

        return success_response(result)

    except Exception as e:
        logger.error(f"Dead code analysis failed: {e}")
        return error_response(f"Dead code analysis failed: {str(e)}", status=500)


# =============================================================================
# Impact Analysis Handler
# =============================================================================


async def handle_analyze_impact(repo_id: str, body: Dict[str, Any]) -> HandlerResult:
    """
    Analyze impact of changes to a symbol.

    Request body:
    {
        "path": "/path/to/codebase",
        "symbol": "module.ClassName.method_name",
        "change_type": "modify"  # modify, remove, signature_change
    }

    Response:
    {
        "changed_node": "module.ClassName.method_name",
        "directly_affected": [...],
        "transitively_affected": [...],
        "risk_level": "medium"
    }
    """
    path = body.get("path")
    if not path:
        return error_response("Missing required field: path", status=400)

    symbol = body.get("symbol")
    if not symbol:
        return error_response("Missing required field: symbol", status=400)

    if not Path(path).exists():
        return error_response(f"Path does not exist: {path}", status=404)

    builder = _get_call_graph_builder()
    if not builder:
        return error_response("Call graph builder not available", status=503)

    try:
        # Build call graph
        graph = builder.build_from_directory(path)

        # Analyze impact
        impact = graph.analyze_impact(symbol)

        # Calculate risk level based on affected count
        total_affected = len(impact.directly_affected) + len(impact.transitively_affected)
        if total_affected > 20:
            risk_level = "high"
        elif total_affected > 5:
            risk_level = "medium"
        else:
            risk_level = "low"

        result = {
            "changed_node": impact.changed_node,
            "directly_affected": list(impact.directly_affected)[:50],
            "transitively_affected": list(impact.transitively_affected)[:100],
            "risk_level": risk_level,
            "summary": {
                "directly_affected_count": len(impact.directly_affected),
                "transitively_affected_count": len(impact.transitively_affected),
            },
        }

        return success_response(result)

    except Exception as e:
        logger.error(f"Impact analysis failed: {e}")
        return error_response(f"Impact analysis failed: {str(e)}", status=500)


# =============================================================================
# Understand Handler
# =============================================================================


async def handle_understand(repo_id: str, body: Dict[str, Any]) -> HandlerResult:
    """
    Answer questions about the codebase.

    Request body:
    {
        "path": "/path/to/codebase",
        "question": "How does authentication work?",
        "max_files": 10
    }

    Response:
    {
        "question": "...",
        "answer": "...",
        "confidence": 0.85,
        "relevant_files": [...],
        "code_citations": [...]
    }
    """
    path = body.get("path")
    if not path:
        return error_response("Missing required field: path", status=400)

    question = body.get("question")
    if not question:
        return error_response("Missing required field: question", status=400)

    if not Path(path).exists():
        return error_response(f"Path does not exist: {path}", status=404)

    try:
        from aragora.agents.codebase_agent import CodebaseUnderstandingAgent
        from unittest.mock import Mock, patch

        # Patch abstract agent classes for now
        with (
            patch("aragora.agents.codebase_agent.CodeAnalystAgent", return_value=Mock()),
            patch("aragora.agents.codebase_agent.SecurityReviewerAgent", return_value=Mock()),
            patch("aragora.agents.codebase_agent.BugHunterAgent", return_value=Mock()),
        ):
            agent = CodebaseUnderstandingAgent(
                root_path=path,
                enable_debate=False,  # Faster response
            )

        max_files = body.get("max_files", 10)
        understanding = await agent.understand(question, max_files=max_files)

        return success_response(understanding.to_dict())

    except ImportError as e:
        logger.error(f"CodebaseUnderstandingAgent not available: {e}")
        return error_response("Codebase understanding not available", status=503)
    except Exception as e:
        logger.error(f"Understanding query failed: {e}")
        return error_response(f"Understanding query failed: {str(e)}", status=500)


# =============================================================================
# Audit Handler
# =============================================================================


async def handle_audit(repo_id: str, body: Dict[str, Any]) -> HandlerResult:
    """
    Run comprehensive code audit.

    Request body:
    {
        "path": "/path/to/codebase",
        "include_security": true,
        "include_bugs": true,
        "include_dead_code": true,
        "include_quality": true,
        "async": false  # Run in background
    }

    Response (sync):
    {
        "audit_id": "uuid",
        "status": "completed",
        "risk_score": 7.5,
        "security_findings": [...],
        "bug_findings": [...],
        ...
    }

    Response (async):
    {
        "audit_id": "uuid",
        "status": "running"
    }
    """
    path = body.get("path")
    if not path:
        return error_response("Missing required field: path", status=400)

    if not Path(path).exists():
        return error_response(f"Path does not exist: {path}", status=404)

    audit_id = str(uuid.uuid4())[:8]
    run_async = body.get("async", False)

    async def run_audit():
        """Run the audit."""
        start_time = datetime.now(timezone.utc)

        result = {
            "audit_id": audit_id,
            "status": "running",
            "started_at": start_time.isoformat(),
            "path": path,
            "security_findings": [],
            "bug_findings": [],
            "dead_code": [],
            "quality_issues": [],
            "files_analyzed": 0,
            "lines_analyzed": 0,
            "risk_score": 0.0,
            "error": None,
        }

        # Store initial state
        repo_audits = _get_or_create_repo_audits(repo_id)
        repo_audits[audit_id] = result

        try:
            include_security = body.get("include_security", True)
            include_bugs = body.get("include_bugs", True)
            include_dead_code = body.get("include_dead_code", True)

            # Security scan
            if include_security:
                scanner = _get_security_scanner()
                if scanner:
                    try:
                        report = scanner.scan_directory(path)
                        result["security_findings"] = [f.to_dict() for f in report.findings[:50]]
                        result["files_analyzed"] = report.files_scanned
                        result["lines_analyzed"] = report.lines_scanned
                    except Exception as e:
                        logger.warning(f"Security scan failed: {e}")

            # Bug detection
            if include_bugs:
                detector = _get_bug_detector()
                if detector:
                    try:
                        report = detector.detect_in_directory(path, exclude_patterns=[])
                        result["bug_findings"] = [b.to_dict() for b in report.bugs[:50]]
                        if not result["files_analyzed"]:
                            result["files_analyzed"] = report.files_scanned
                            result["lines_analyzed"] = report.lines_scanned
                    except Exception as e:
                        logger.warning(f"Bug detection failed: {e}")

            # Dead code analysis
            if include_dead_code:
                builder = _get_call_graph_builder()
                if builder:
                    try:
                        graph = builder.build_from_directory(path)
                        dead_code = graph.find_dead_code()
                        result["dead_code"] = [
                            {
                                "name": n.qualified_name,
                                "kind": n.kind.value,
                                "file": n.location.file_path if n.location else None,
                            }
                            for n in dead_code.unreachable_functions[:30]
                        ]
                    except Exception as e:
                        logger.warning(f"Dead code analysis failed: {e}")

            # Calculate risk score
            security_weight = len(result["security_findings"]) * 2
            bug_weight = len(result["bug_findings"]) * 1.5
            dead_code_weight = len(result["dead_code"]) * 0.5
            result["risk_score"] = min(10.0, (security_weight + bug_weight + dead_code_weight) / 10)

            result["status"] = "completed"
            result["completed_at"] = datetime.now(timezone.utc).isoformat()

            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(
                f"[{audit_id}] Audit completed in {elapsed:.2f}s: "
                f"{len(result['security_findings'])} security, "
                f"{len(result['bug_findings'])} bugs"
            )

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"[{audit_id}] Audit failed: {e}")

        # Update stored result
        repo_audits[audit_id] = result
        return result

    if run_async:
        # Run in background
        task = asyncio.create_task(run_audit())
        _running_audits[audit_id] = task

        return success_response(
            {
                "audit_id": audit_id,
                "status": "running",
                "path": path,
            }
        )
    else:
        # Run synchronously
        result = await run_audit()
        return success_response(result)


async def handle_get_audit_status(
    repo_id: str, audit_id: str, params: Dict[str, Any]
) -> HandlerResult:
    """Get status of an audit."""
    repo_audits = _get_or_create_repo_audits(repo_id)

    if audit_id not in repo_audits:
        return error_response(f"Audit not found: {audit_id}", status=404)

    return success_response(repo_audits[audit_id])


# =============================================================================
# Convenience Functions
# =============================================================================


async def quick_analyze(path: str) -> Dict[str, Any]:
    """Quick helper to analyze a path."""
    import json

    result = await handle_analyze_codebase("default", {"path": path})
    body = json.loads(result.body.decode("utf-8"))
    return body.get("data", {})


async def quick_audit(path: str) -> Dict[str, Any]:
    """Quick helper to audit a path."""
    import json

    result = await handle_audit("default", {"path": path})
    body = json.loads(result.body.decode("utf-8"))
    return body.get("data", {})
