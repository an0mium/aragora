"""
Training data export handler.

Provides API endpoints for exporting debate data for model training:
- SFT (Supervised Fine-Tuning) exports
- DPO (Direct Preference Optimization) exports
- Gauntlet adversarial exports
- Export statistics and job management
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .base import BaseHandler, HandlerResult, json_response, error_response

logger = logging.getLogger(__name__)


class TrainingHandler(BaseHandler):
    """Handler for training data export endpoints."""

    ROUTES = {
        "/api/training/export/sft": "handle_export_sft",
        "/api/training/export/dpo": "handle_export_dpo",
        "/api/training/export/gauntlet": "handle_export_gauntlet",
        "/api/training/stats": "handle_stats",
        "/api/training/formats": "handle_formats",
    }

    def __init__(self, ctx: dict[str, Any]):
        """Initialize with server context."""
        super().__init__(ctx)
        self._exporters: dict[str, Any] = {}
        self._export_dir = Path(os.environ.get(
            "ARAGORA_TRAINING_EXPORT_DIR",
            ".nomic/training_exports"
        ))
        self._export_dir.mkdir(parents=True, exist_ok=True)

    def can_handle(self, path: str, method: str = "GET") -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> Optional[HandlerResult]:
        """Route training requests to appropriate methods."""
        method_name = self.ROUTES.get(path)
        if method_name and hasattr(self, method_name):
            return getattr(self, method_name)(path, query_params, handler)
        return None

    def _get_sft_exporter(self):
        """Get or create SFT exporter."""
        if "sft" not in self._exporters:
            try:
                from aragora.training import SFTExporter
                self._exporters["sft"] = SFTExporter()
            except ImportError:
                return None
        return self._exporters["sft"]

    def _get_dpo_exporter(self):
        """Get or create DPO exporter."""
        if "dpo" not in self._exporters:
            try:
                from aragora.training import DPOExporter
                self._exporters["dpo"] = DPOExporter()
            except ImportError:
                return None
        return self._exporters["dpo"]

    def _get_gauntlet_exporter(self):
        """Get or create Gauntlet exporter."""
        if "gauntlet" not in self._exporters:
            try:
                from aragora.training import GauntletExporter
                self._exporters["gauntlet"] = GauntletExporter()
            except ImportError:
                return None
        return self._exporters["gauntlet"]

    def handle_export_sft(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult:
        """
        Export SFT training data.

        Query params:
            min_confidence: Minimum debate confidence (default 0.7)
            min_success_rate: Minimum pattern success rate (default 0.6)
            limit: Maximum records (default 1000)
            offset: Starting offset (default 0)
            include_critiques: Include critique data (default true)
            include_patterns: Include pattern data (default true)
            include_debates: Include debate data (default true)
            format: Output format (json, jsonl) (default json)
        """
        exporter = self._get_sft_exporter()
        if exporter is None:
            return error_response(
                "SFT exporter not available",
                500,
                details={"hint": "Training module may not be installed"},
            )

        try:
            # Parse parameters
            min_confidence = float(query_params.get("min_confidence", 0.7))
            min_success_rate = float(query_params.get("min_success_rate", 0.6))
            limit = int(query_params.get("limit", 1000))
            offset = int(query_params.get("offset", 0))
            include_critiques = query_params.get("include_critiques", "true").lower() == "true"
            include_patterns = query_params.get("include_patterns", "true").lower() == "true"
            include_debates = query_params.get("include_debates", "true").lower() == "true"
            output_format = query_params.get("format", "json")

            # Validate parameters
            min_confidence = max(0.0, min(1.0, min_confidence))
            min_success_rate = max(0.0, min(1.0, min_success_rate))
            limit = max(1, min(10000, limit))
            offset = max(0, offset)

            # Export data
            records = exporter.export(
                min_confidence=min_confidence,
                min_success_rate=min_success_rate,
                limit=limit,
                offset=offset,
                include_critiques=include_critiques,
                include_patterns=include_patterns,
                include_debates=include_debates,
            )

            response_data = {
                "export_type": "sft",
                "total_records": len(records),
                "parameters": {
                    "min_confidence": min_confidence,
                    "min_success_rate": min_success_rate,
                    "limit": limit,
                    "offset": offset,
                    "include_critiques": include_critiques,
                    "include_patterns": include_patterns,
                    "include_debates": include_debates,
                },
                "exported_at": datetime.now().isoformat(),
            }

            if output_format == "jsonl":
                # Return JSONL format inline
                jsonl_data = "\n".join(json.dumps(r) for r in records)
                response_data["data"] = jsonl_data
                response_data["format"] = "jsonl"
            else:
                # Return JSON array
                response_data["records"] = records
                response_data["format"] = "json"

            logger.info(
                "training_sft_export records=%d confidence=%.2f",
                len(records),
                min_confidence,
            )

            return json_response(response_data)

        except Exception as e:
            logger.error("training_sft_export_failed error=%s", e)
            return error_response(f"Export failed: {e}", 500)

    def handle_export_dpo(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult:
        """
        Export DPO (preference) training data.

        Query params:
            min_confidence_diff: Minimum confidence difference for pairs (default 0.1)
            limit: Maximum records (default 500)
            format: Output format (json, jsonl) (default json)
        """
        exporter = self._get_dpo_exporter()
        if exporter is None:
            return error_response(
                "DPO exporter not available",
                500,
                details={"hint": "Training module may not be installed"},
            )

        try:
            # Parse parameters
            min_confidence_diff = float(query_params.get("min_confidence_diff", 0.1))
            limit = int(query_params.get("limit", 500))
            output_format = query_params.get("format", "json")

            # Validate
            min_confidence_diff = max(0.0, min(1.0, min_confidence_diff))
            limit = max(1, min(5000, limit))

            # Export data
            records = exporter.export(
                min_confidence_diff=min_confidence_diff,
                limit=limit,
            )

            response_data = {
                "export_type": "dpo",
                "total_records": len(records),
                "parameters": {
                    "min_confidence_diff": min_confidence_diff,
                    "limit": limit,
                },
                "exported_at": datetime.now().isoformat(),
            }

            if output_format == "jsonl":
                jsonl_data = "\n".join(json.dumps(r) for r in records)
                response_data["data"] = jsonl_data
                response_data["format"] = "jsonl"
            else:
                response_data["records"] = records
                response_data["format"] = "json"

            logger.info(
                "training_dpo_export records=%d min_diff=%.2f",
                len(records),
                min_confidence_diff,
            )

            return json_response(response_data)

        except Exception as e:
            logger.error("training_dpo_export_failed error=%s", e)
            return error_response(f"Export failed: {e}", 500)

    def handle_export_gauntlet(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult:
        """
        Export Gauntlet adversarial training data.

        Query params:
            persona: Filter by persona (gdpr, hipaa, ai_act, all) (default all)
            min_severity: Minimum severity level (default 0.5)
            limit: Maximum records (default 500)
            format: Output format (json, jsonl) (default json)
        """
        exporter = self._get_gauntlet_exporter()
        if exporter is None:
            return error_response(
                "Gauntlet exporter not available",
                500,
                details={"hint": "Training module may not be installed"},
            )

        try:
            # Parse parameters
            persona = query_params.get("persona", "all")
            min_severity = float(query_params.get("min_severity", 0.5))
            limit = int(query_params.get("limit", 500))
            output_format = query_params.get("format", "json")

            # Validate
            min_severity = max(0.0, min(1.0, min_severity))
            limit = max(1, min(5000, limit))

            # Build export kwargs
            export_kwargs = {
                "min_severity": min_severity,
                "limit": limit,
            }
            if persona != "all":
                export_kwargs["persona"] = persona

            # Export data
            records = exporter.export(**export_kwargs)

            response_data = {
                "export_type": "gauntlet",
                "total_records": len(records),
                "parameters": {
                    "persona": persona,
                    "min_severity": min_severity,
                    "limit": limit,
                },
                "exported_at": datetime.now().isoformat(),
            }

            if output_format == "jsonl":
                jsonl_data = "\n".join(json.dumps(r) for r in records)
                response_data["data"] = jsonl_data
                response_data["format"] = "jsonl"
            else:
                response_data["records"] = records
                response_data["format"] = "json"

            logger.info(
                "training_gauntlet_export records=%d persona=%s",
                len(records),
                persona,
            )

            return json_response(response_data)

        except Exception as e:
            logger.error("training_gauntlet_export_failed error=%s", e)
            return error_response(f"Export failed: {e}", 500)

    def handle_stats(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult:
        """
        Get training data statistics.

        Returns counts of available training data by type.
        """
        try:
            stats = {
                "available_exporters": [],
                "export_directory": str(self._export_dir),
                "exported_files": [],
            }

            # Check available exporters
            if self._get_sft_exporter():
                stats["available_exporters"].append("sft")
            if self._get_dpo_exporter():
                stats["available_exporters"].append("dpo")
            if self._get_gauntlet_exporter():
                stats["available_exporters"].append("gauntlet")

            # List exported files
            if self._export_dir.exists():
                for f in self._export_dir.glob("*.jsonl"):
                    file_stat = f.stat()
                    stats["exported_files"].append({
                        "name": f.name,
                        "size_bytes": file_stat.st_size,
                        "created_at": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                        "modified_at": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                    })

            # Get data counts from each exporter
            sft_exporter = self._get_sft_exporter()
            if sft_exporter:
                try:
                    sft_sample = sft_exporter.export(limit=1)
                    stats["sft_available"] = len(sft_sample) > 0
                except Exception:
                    stats["sft_available"] = False

            return json_response(stats)

        except Exception as e:
            logger.error("training_stats_failed error=%s", e)
            return error_response(f"Failed to get stats: {e}", 500)

    def handle_formats(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult:
        """
        Get supported training data formats and schemas.

        Returns information about export formats and their structure.
        """
        formats = {
            "sft": {
                "description": "Supervised Fine-Tuning data",
                "schema": {
                    "instruction": "string - The task or question",
                    "response": "string - The model response",
                    "metadata": {
                        "source": "string - Origin (debate, pattern, critique)",
                        "confidence": "float - Debate confidence score",
                        "debate_id": "string - Source debate ID (optional)",
                    },
                },
                "use_case": "Training models on successful debate patterns and winning responses",
            },
            "dpo": {
                "description": "Direct Preference Optimization data",
                "schema": {
                    "prompt": "string - The input prompt",
                    "chosen": "string - The preferred response",
                    "rejected": "string - The less preferred response",
                    "metadata": {
                        "chosen_confidence": "float - Confidence of chosen response",
                        "rejected_confidence": "float - Confidence of rejected response",
                        "confidence_diff": "float - Difference in confidence",
                    },
                },
                "use_case": "Training models to prefer higher-quality debate responses",
            },
            "gauntlet": {
                "description": "Adversarial vulnerability training data",
                "schema": {
                    "instruction": "string - The adversarial prompt",
                    "response": "string - The appropriate response",
                    "metadata": {
                        "persona": "string - Gauntlet persona (gdpr, hipaa, ai_act)",
                        "vulnerability_type": "string - Type of vulnerability tested",
                        "severity": "float - Severity score",
                    },
                },
                "use_case": "Training models to handle adversarial compliance scenarios",
            },
        }

        return json_response({
            "formats": formats,
            "output_formats": ["json", "jsonl"],
            "endpoints": {
                "sft": "/api/training/export/sft",
                "dpo": "/api/training/export/dpo",
                "gauntlet": "/api/training/export/gauntlet",
            },
        })
