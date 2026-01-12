"""Base class for training data exporters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExportMetadata:
    """Metadata about an export operation."""

    exporter_type: str
    exported_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_records: int = 0
    filters_applied: dict[str, Any] = field(default_factory=dict)
    source_db: str = ""


@dataclass
class TrainingRecord:
    """A single training record."""

    instruction: str
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "instruction": self.instruction,
            "response": self.response,
            "metadata": self.metadata,
        }

    def to_jsonl(self) -> str:
        """Convert to JSONL line."""
        return json.dumps(self.to_dict())


@dataclass
class PreferenceRecord:
    """A preference pair for DPO training."""

    prompt: str
    chosen: str
    rejected: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": self.metadata,
        }

    def to_jsonl(self) -> str:
        """Convert to JSONL line."""
        return json.dumps(self.to_dict())


class BaseExporter(ABC):
    """Base class for training data exporters."""

    exporter_type: str = "base"

    @abstractmethod
    def export(self, **kwargs) -> list[dict[str, Any]]:
        """Export training data.

        Returns:
            List of training records as dictionaries.
        """
        pass

    def export_to_file(
        self,
        output_path: str | Path,
        **kwargs,
    ) -> ExportMetadata:
        """Export training data to a JSONL file.

        Args:
            output_path: Path to output file
            **kwargs: Arguments passed to export()

        Returns:
            Metadata about the export operation
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = self.export(**kwargs)

        with open(output_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        metadata = ExportMetadata(
            exporter_type=self.exporter_type,
            total_records=len(records),
            filters_applied=kwargs,
            source_db=getattr(self, "db_path", ""),
        )

        logger.info(
            "Exported %d records to %s (type=%s)",
            len(records),
            output_path,
            self.exporter_type,
        )

        return metadata

    def validate_record(self, record: dict[str, Any]) -> bool:
        """Validate a training record.

        Override in subclasses for specific validation.
        """
        return True
