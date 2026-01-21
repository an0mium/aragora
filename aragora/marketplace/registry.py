"""
Template Registry for the Aragora Marketplace.

Provides local storage and management of templates with
search, versioning, and validation capabilities.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional, Union

from .models import (
    AgentTemplate,
    DebateTemplate,
    WorkflowTemplate,
    TemplateCategory,
    TemplateMetadata,
    TemplateRating,
    BUILTIN_AGENT_TEMPLATES,
    BUILTIN_DEBATE_TEMPLATES,
)


class TemplateRegistry:
    """Local registry for marketplace templates."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the template registry."""
        if db_path is None:
            db_path = Path.home() / ".aragora" / "marketplace.db"
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._load_builtins()

    @contextmanager
    def _get_conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS templates (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    version TEXT NOT NULL,
                    author TEXT NOT NULL,
                    category TEXT NOT NULL,
                    tags TEXT,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    downloads INTEGER DEFAULT 0,
                    stars INTEGER DEFAULT 0,
                    is_builtin INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    template_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    review TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (template_id) REFERENCES templates(id),
                    UNIQUE(template_id, user_id)
                );

                CREATE INDEX IF NOT EXISTS idx_templates_category ON templates(category);
                CREATE INDEX IF NOT EXISTS idx_templates_author ON templates(author);
                CREATE INDEX IF NOT EXISTS idx_templates_type ON templates(type);
                CREATE INDEX IF NOT EXISTS idx_ratings_template ON ratings(template_id);
            """)
            conn.commit()

    def _load_builtins(self) -> None:
        """Load built-in templates into the registry."""
        for template in BUILTIN_AGENT_TEMPLATES:
            self._upsert_template(template, is_builtin=True)
        for template in BUILTIN_DEBATE_TEMPLATES:
            self._upsert_template(template, is_builtin=True)

    def _upsert_template(
        self,
        template: Union[AgentTemplate, DebateTemplate, WorkflowTemplate],
        is_builtin: bool = False,
    ) -> None:
        """Insert or update a template."""
        template_type = type(template).__name__
        data = template.to_dict()

        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO templates
                (id, type, name, description, version, author, category, tags,
                 content, content_hash, created_at, updated_at, downloads, stars, is_builtin)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    template.metadata.id,
                    template_type,
                    template.metadata.name,
                    template.metadata.description,
                    template.metadata.version,
                    template.metadata.author,
                    template.metadata.category.value,
                    json.dumps(template.metadata.tags),
                    json.dumps(data),
                    template.content_hash(),
                    template.metadata.created_at.isoformat(),
                    template.metadata.updated_at.isoformat(),
                    template.metadata.downloads,
                    template.metadata.stars,
                    1 if is_builtin else 0,
                ),
            )
            conn.commit()

    def register(
        self,
        template: Union[AgentTemplate, DebateTemplate, WorkflowTemplate],
    ) -> str:
        """Register a new template."""
        self._upsert_template(template)
        return template.metadata.id

    def get(
        self,
        template_id: str,
    ) -> Optional[Union[AgentTemplate, DebateTemplate, WorkflowTemplate]]:
        """Get a template by ID."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM templates WHERE id = ?", (template_id,)).fetchone()

        if row is None:
            return None

        return self._row_to_template(row)

    def _row_to_template(
        self, row: sqlite3.Row
    ) -> Union[AgentTemplate, DebateTemplate, WorkflowTemplate]:
        """Convert a database row to a template object."""
        data = json.loads(row["content"])
        template_type = row["type"]

        metadata = TemplateMetadata(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            version=row["version"],
            author=row["author"],
            category=TemplateCategory(row["category"]),
            tags=json.loads(row["tags"]) if row["tags"] else [],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            downloads=row["downloads"],
            stars=row["stars"],
        )

        if template_type == "AgentTemplate":
            return AgentTemplate(
                metadata=metadata,
                agent_type=data["agent_type"],
                system_prompt=data["system_prompt"],
                model_config=data.get("model_config", {}),
                capabilities=data.get("capabilities", []),
                constraints=data.get("constraints", []),
                examples=data.get("examples", []),
            )
        elif template_type == "DebateTemplate":
            return DebateTemplate(
                metadata=metadata,
                task_template=data["task_template"],
                agent_roles=data["agent_roles"],
                protocol=data["protocol"],
                evaluation_criteria=data.get("evaluation_criteria", []),
                success_metrics=data.get("success_metrics", {}),
            )
        elif template_type == "WorkflowTemplate":
            return WorkflowTemplate(
                metadata=metadata,
                nodes=data["nodes"],
                edges=data["edges"],
                inputs=data.get("inputs", {}),
                outputs=data.get("outputs", {}),
                variables=data.get("variables", {}),
            )
        else:
            raise ValueError(f"Unknown template type: {template_type}")

    def search(
        self,
        query: Optional[str] = None,
        category: Optional[TemplateCategory] = None,
        template_type: Optional[str] = None,
        author: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Union[AgentTemplate, DebateTemplate, WorkflowTemplate]]:
        """Search for templates."""
        conditions = []
        params: list[Any] = []

        if query:
            conditions.append("(name LIKE ? OR description LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])

        if category:
            conditions.append("category = ?")
            params.append(category.value)

        if template_type:
            conditions.append("type = ?")
            params.append(template_type)

        if author:
            conditions.append("author = ?")
            params.append(author)

        if tags:
            for tag in tags:
                conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._get_conn() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM templates
                WHERE {where_clause}
                ORDER BY stars DESC, downloads DESC
                LIMIT ? OFFSET ?
            """,
                params + [limit, offset],
            ).fetchall()

        return [self._row_to_template(row) for row in rows]

    def list_categories(self) -> list[dict[str, Any]]:
        """List all categories with template counts."""
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT category, COUNT(*) as count
                FROM templates
                GROUP BY category
                ORDER BY count DESC
            """).fetchall()

        return [{"category": row["category"], "count": row["count"]} for row in rows]

    def rate(self, rating: TemplateRating) -> None:
        """Add or update a rating for a template."""
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ratings
                (template_id, user_id, score, review, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    rating.template_id,
                    rating.user_id,
                    rating.score,
                    rating.review,
                    rating.created_at.isoformat(),
                ),
            )
            conn.commit()

    def get_ratings(self, template_id: str) -> list[TemplateRating]:
        """Get all ratings for a template."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM ratings WHERE template_id = ?
                ORDER BY created_at DESC
            """,
                (template_id,),
            ).fetchall()

        return [
            TemplateRating(
                user_id=row["user_id"],
                template_id=row["template_id"],
                score=row["score"],
                review=row["review"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def get_average_rating(self, template_id: str) -> Optional[float]:
        """Get average rating for a template."""
        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT AVG(score) as avg_score FROM ratings
                WHERE template_id = ?
            """,
                (template_id,),
            ).fetchone()

        return row["avg_score"] if row and row["avg_score"] else None

    def increment_downloads(self, template_id: str) -> None:
        """Increment download count for a template."""
        with self._get_conn() as conn:
            conn.execute(
                """
                UPDATE templates SET downloads = downloads + 1
                WHERE id = ?
            """,
                (template_id,),
            )
            conn.commit()

    def star(self, template_id: str) -> None:
        """Add a star to a template."""
        with self._get_conn() as conn:
            conn.execute(
                """
                UPDATE templates SET stars = stars + 1
                WHERE id = ?
            """,
                (template_id,),
            )
            conn.commit()

    def delete(self, template_id: str) -> bool:
        """Delete a template (non-builtins only)."""
        with self._get_conn() as conn:
            result = conn.execute(
                """
                DELETE FROM templates
                WHERE id = ? AND is_builtin = 0
            """,
                (template_id,),
            )
            conn.commit()
            return result.rowcount > 0

    def export_template(self, template_id: str) -> Optional[str]:
        """Export a template as JSON."""
        template = self.get(template_id)
        if template is None:
            return None
        return json.dumps(template.to_dict(), indent=2)

    def import_template(self, json_str: str) -> str:
        """Import a template from JSON."""
        data = json.loads(json_str)

        # Determine template type from data
        if "agent_type" in data:
            metadata = self._parse_metadata(data["metadata"])
            template = AgentTemplate(
                metadata=metadata,
                agent_type=data["agent_type"],
                system_prompt=data["system_prompt"],
                model_config=data.get("model_config", {}),
                capabilities=data.get("capabilities", []),
                constraints=data.get("constraints", []),
                examples=data.get("examples", []),
            )
        elif "task_template" in data:
            metadata = self._parse_metadata(data["metadata"])
            template = DebateTemplate(
                metadata=metadata,
                task_template=data["task_template"],
                agent_roles=data["agent_roles"],
                protocol=data["protocol"],
                evaluation_criteria=data.get("evaluation_criteria", []),
                success_metrics=data.get("success_metrics", {}),
            )
        elif "nodes" in data:
            metadata = self._parse_metadata(data["metadata"])
            template = WorkflowTemplate(
                metadata=metadata,
                nodes=data["nodes"],
                edges=data["edges"],
                inputs=data.get("inputs", {}),
                outputs=data.get("outputs", {}),
                variables=data.get("variables", {}),
            )
        else:
            raise ValueError("Unknown template format")

        return self.register(template)

    def _parse_metadata(self, data: dict[str, Any]) -> TemplateMetadata:
        """Parse metadata from dictionary."""
        return TemplateMetadata(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            version=data["version"],
            author=data["author"],
            category=TemplateCategory(data["category"]),
            tags=data.get("tags", []),
            downloads=data.get("downloads", 0),
            stars=data.get("stars", 0),
        )
