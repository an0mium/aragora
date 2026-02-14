"""
Tests for MarketplaceStore - SQLite and PostgreSQL marketplace storage.

Tests cover:
1. Store initialization and configuration
2. Template CRUD operations
3. Author/vendor management
4. Category and taxonomy handling
5. Ratings and download tracking
6. Search and filtering functionality
7. Featured and trending templates
8. Reviews and ratings
9. Data validation and constraints
10. Concurrent access handling
11. PostgreSQL async operations
12. Global store management
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.storage.marketplace_store import (
    MarketplaceStore,
    PostgresMarketplaceStore,
    StoredReview,
    StoredTemplate,
    get_marketplace_store,
    reset_marketplace_store,
    set_marketplace_store,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def tmp_db_path(tmp_path):
    """Provide a temporary database path."""
    return tmp_path / "marketplace_test.db"


@pytest.fixture
def store(tmp_db_path):
    """Create a fresh marketplace store instance."""
    return MarketplaceStore(db_path=tmp_db_path)


@pytest.fixture
def store_with_templates(store):
    """Create a store with pre-populated templates."""
    for i in range(5):
        store.create_template(
            name=f"Template {i}",
            description=f"Description for template {i}",
            author_id=f"author-{i % 2}",
            author_name=f"Author {i % 2}",
            category="security" if i % 2 == 0 else "code-review",
            pattern="adversarial" if i % 2 == 0 else "collaborative",
            workflow_definition={"nodes": [{"id": f"node-{i}"}]},
            tags=[f"tag-{i}", "common"],
        )
    return store


@pytest.fixture(autouse=True)
def reset_global_store():
    """Reset global store before and after each test."""
    reset_marketplace_store()
    yield
    reset_marketplace_store()


# ============================================================================
# Test StoredTemplate Dataclass
# ============================================================================


class TestStoredTemplate:
    """Tests for StoredTemplate dataclass."""

    def test_create_template_with_defaults(self):
        """Should create template with default values."""
        template = StoredTemplate(
            id="tpl-test123",
            name="Test Template",
            description="A test template",
            author_id="user-123",
            author_name="Test Author",
            category="security",
            pattern="adversarial",
        )

        assert template.id == "tpl-test123"
        assert template.name == "Test Template"
        assert template.tags == []
        assert template.workflow_definition == {}
        assert template.download_count == 0
        assert template.rating_sum == 0.0
        assert template.rating_count == 0
        assert template.is_featured is False
        assert template.is_trending is False
        assert template.created_at > 0

    def test_rating_property_zero_count(self):
        """Rating should return 0.0 when no ratings."""
        template = StoredTemplate(
            id="tpl-test",
            name="Test",
            description="Desc",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
        )

        assert template.rating == 0.0

    def test_rating_property_with_ratings(self):
        """Rating should calculate average correctly."""
        template = StoredTemplate(
            id="tpl-test",
            name="Test",
            description="Desc",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            rating_sum=15.0,
            rating_count=3,
        )

        assert template.rating == 5.0

    def test_to_dict(self):
        """Should convert to API response format."""
        template = StoredTemplate(
            id="tpl-test",
            name="Test",
            description="Desc",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            tags=["tag1", "tag2"],
            download_count=100,
            rating_sum=10.0,
            rating_count=2,
            created_at=1000.0,
            updated_at=2000.0,
        )

        data = template.to_dict()

        assert data["id"] == "tpl-test"
        assert data["name"] == "Test"
        assert data["tags"] == ["tag1", "tag2"]
        assert data["download_count"] == 100
        assert data["rating"] == 5.0
        assert data["rating_count"] == 2
        assert "workflow_definition" not in data

    def test_to_full_dict(self):
        """Should convert to full format including workflow."""
        workflow = {"nodes": [{"id": "1"}], "edges": []}
        template = StoredTemplate(
            id="tpl-test",
            name="Test",
            description="Desc",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition=workflow,
        )

        data = template.to_full_dict()

        assert data["workflow_definition"] == workflow


# ============================================================================
# Test StoredReview Dataclass
# ============================================================================


class TestStoredReview:
    """Tests for StoredReview dataclass."""

    def test_create_review_with_defaults(self):
        """Should create review with default values."""
        review = StoredReview(
            id="rev-test123",
            template_id="tpl-123",
            user_id="user-456",
            user_name="Test User",
            rating=5,
            title="Great Template",
            content="This template is excellent!",
        )

        assert review.id == "rev-test123"
        assert review.helpful_count == 0
        assert review.created_at > 0

    def test_to_dict(self):
        """Should convert to API response format."""
        review = StoredReview(
            id="rev-test",
            template_id="tpl-123",
            user_id="user-456",
            user_name="Test User",
            rating=4,
            title="Good Template",
            content="Works well",
            helpful_count=10,
            created_at=1000.0,
        )

        data = review.to_dict()

        assert data["id"] == "rev-test"
        assert data["template_id"] == "tpl-123"
        assert data["rating"] == 4
        assert data["helpful_count"] == 10


# ============================================================================
# Test Store Initialization
# ============================================================================


class TestMarketplaceStoreInit:
    """Tests for MarketplaceStore initialization."""

    def test_creates_database_file(self, tmp_db_path):
        """Store creates database file if it doesn't exist."""
        assert not tmp_db_path.exists()
        MarketplaceStore(db_path=tmp_db_path)
        assert tmp_db_path.exists()

    def test_creates_parent_directories(self, tmp_path):
        """Store creates parent directories if needed."""
        nested_path = tmp_path / "a" / "b" / "c" / "marketplace.db"
        assert not nested_path.parent.exists()
        MarketplaceStore(db_path=nested_path)
        assert nested_path.exists()

    def test_initializes_schema(self, tmp_db_path):
        """Store initializes schema with all tables."""
        store = MarketplaceStore(db_path=tmp_db_path)

        with store.connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            table_names = {t[0] for t in tables}

            assert "templates" in table_names
            assert "reviews" in table_names
            assert "ratings" in table_names
            assert "categories" in table_names

    def test_initializes_default_categories(self, tmp_db_path):
        """Store initializes default categories."""
        store = MarketplaceStore(db_path=tmp_db_path)
        categories = store.list_categories()

        assert len(categories) >= 7
        category_ids = {c["id"] for c in categories}
        assert "security" in category_ids
        assert "code-review" in category_ids
        assert "compliance" in category_ids

    def test_auto_init_false_skips_schema(self, tmp_db_path):
        """auto_init=False skips schema initialization."""
        store = MarketplaceStore(db_path=tmp_db_path, auto_init=False)

        with store.connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='templates'"
            ).fetchall()
            assert len(tables) == 0

    def test_default_db_path(self, monkeypatch, tmp_path):
        """Store uses default path when not specified (ARAGORA_DATA_DIR)."""
        data_dir = tmp_path / "data"
        # resolve_db_path() reads ARAGORA_DATA_DIR dynamically via get_default_data_dir().
        monkeypatch.setenv("ARAGORA_DATA_DIR", str(data_dir))
        store = MarketplaceStore()
        expected_path = data_dir / "marketplace.db"
        assert store.db_path == expected_path

    def test_schema_version(self, tmp_db_path):
        """Store sets correct schema version."""
        store = MarketplaceStore(db_path=tmp_db_path)
        assert store.SCHEMA_VERSION == 1
        assert store.SCHEMA_NAME == "marketplace_store"


# ============================================================================
# Test Template CRUD Operations
# ============================================================================


class TestTemplateCRUD:
    """Tests for template CRUD operations."""

    def test_create_template(self, store):
        """Should create a template with all fields."""
        template = store.create_template(
            name="Security Workflow",
            description="A comprehensive security analysis workflow",
            author_id="user-123",
            author_name="John Doe",
            category="security",
            pattern="adversarial",
            workflow_definition={"nodes": [], "edges": []},
            tags=["security", "analysis"],
        )

        assert template.id.startswith("tpl-")
        assert template.name == "Security Workflow"
        assert template.author_id == "user-123"
        assert template.category == "security"
        assert template.tags == ["security", "analysis"]

    def test_create_template_with_minimal_fields(self, store):
        """Should create template with minimal required fields."""
        template = store.create_template(
            name="Minimal Template",
            description="Minimal",
            author_id="user-1",
            author_name="Author",
            category="testing",
            pattern="simple",
            workflow_definition={},
        )

        assert template.tags == []
        assert template.download_count == 0

    def test_create_template_duplicate_name_raises(self, store):
        """Should raise ValueError for duplicate template name."""
        store.create_template(
            name="Unique Template",
            description="First",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        with pytest.raises(ValueError, match="already exists"):
            store.create_template(
                name="Unique Template",
                description="Second",
                author_id="user-2",
                author_name="Other",
                category="security",
                pattern="adversarial",
                workflow_definition={},
            )

    def test_create_template_updates_category_count(self, store):
        """Creating template should update category template count."""
        initial_categories = store.list_categories()
        security_count = next(
            (c["template_count"] for c in initial_categories if c["id"] == "security"),
            0,
        )

        store.create_template(
            name="Test Template",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        updated_categories = store.list_categories()
        new_count = next(c["template_count"] for c in updated_categories if c["id"] == "security")
        assert new_count == security_count + 1

    def test_get_template(self, store):
        """Should retrieve template by ID."""
        created = store.create_template(
            name="Get Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={"key": "value"},
        )

        retrieved = store.get_template(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == "Get Test"
        assert retrieved.workflow_definition == {"key": "value"}

    def test_get_template_nonexistent(self, store):
        """Should return None for nonexistent template."""
        result = store.get_template("tpl-nonexistent")
        assert result is None

    def test_list_templates_empty(self, store):
        """Should return empty list when no templates."""
        templates, total = store.list_templates()
        assert templates == []
        assert total == 0

    def test_list_templates_with_data(self, store_with_templates):
        """Should list all templates with pagination info."""
        templates, total = store_with_templates.list_templates()
        assert len(templates) == 5
        assert total == 5

    def test_list_templates_pagination(self, store_with_templates):
        """Should support pagination."""
        templates, total = store_with_templates.list_templates(limit=2, offset=0)
        assert len(templates) == 2
        assert total == 5

        templates2, total2 = store_with_templates.list_templates(limit=2, offset=2)
        assert len(templates2) == 2
        assert total2 == 5

        # No overlap in IDs
        ids1 = {t.id for t in templates}
        ids2 = {t.id for t in templates2}
        assert ids1.isdisjoint(ids2)

    def test_list_templates_filter_by_category(self, store_with_templates):
        """Should filter templates by category."""
        templates, total = store_with_templates.list_templates(category="security")
        assert all(t.category == "security" for t in templates)
        assert total == 3  # Templates 0, 2, 4

    def test_list_templates_search(self, store_with_templates):
        """Should search in name, description, and tags."""
        templates, total = store_with_templates.list_templates(search="Template 1")
        assert len(templates) == 1
        assert templates[0].name == "Template 1"

    def test_list_templates_search_in_tags(self, store_with_templates):
        """Should search in tags."""
        templates, total = store_with_templates.list_templates(search="tag-0")
        assert len(templates) == 1
        assert "tag-0" in templates[0].tags

    def test_list_templates_sort_by_rating(self, store):
        """Should sort by rating."""
        t1 = store.create_template(
            name="Low Rated",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )
        store.rate_template(t1.id, "user-a", 2)

        t2 = store.create_template(
            name="High Rated",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )
        store.rate_template(t2.id, "user-a", 5)

        templates, _ = store.list_templates(sort_by="rating")
        assert templates[0].id == t2.id

    def test_list_templates_sort_by_downloads(self, store):
        """Should sort by downloads."""
        t1 = store.create_template(
            name="Few Downloads",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        t2 = store.create_template(
            name="Many Downloads",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        for _ in range(10):
            store.increment_download(t2.id)

        templates, _ = store.list_templates(sort_by="downloads")
        assert templates[0].id == t2.id

    def test_list_templates_sort_by_newest(self, store):
        """Should sort by creation time."""
        t1 = store.create_template(
            name="Older",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        time.sleep(0.01)

        t2 = store.create_template(
            name="Newer",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        templates, _ = store.list_templates(sort_by="newest")
        assert templates[0].id == t2.id

    def test_list_templates_combined_filters(self, store_with_templates):
        """Should combine category filter and search."""
        templates, total = store_with_templates.list_templates(category="security", search="common")
        assert all(t.category == "security" for t in templates)
        assert all("common" in t.tags for t in templates)


# ============================================================================
# Test Template Ranking with Window Functions
# ============================================================================


class TestTemplateRanking:
    """Tests for list_templates_with_rank using window functions."""

    def test_list_templates_with_rank(self, store):
        """Should return templates with rank information."""
        for i in range(5):
            t = store.create_template(
                name=f"Template {i}",
                description=f"Desc {i}",
                author_id="user-1",
                author_name="Author",
                category="security" if i < 3 else "code-review",
                pattern="adversarial",
                workflow_definition={},
            )
            store.rate_template(t.id, "user-x", 5 - i)  # Higher rank for lower i

        results, total = store.list_templates_with_rank(sort_by="rating")

        assert total == 5
        assert len(results) == 5
        assert "global_rank" in results[0]
        assert "category_rank" in results[0]
        assert results[0]["global_rank"] == 1

    def test_list_templates_with_rank_category_filter(self, store):
        """Should rank within filtered category."""
        for i in range(3):
            t = store.create_template(
                name=f"Security {i}",
                description=f"Desc {i}",
                author_id="user-1",
                author_name="Author",
                category="security",
                pattern="adversarial",
                workflow_definition={},
            )
            store.rate_template(t.id, "user-x", 5 - i)

        results, total = store.list_templates_with_rank(category="security", sort_by="rating")

        assert total == 3
        assert results[0]["category_rank"] == 1

    def test_list_templates_with_rank_empty(self, store):
        """Should handle empty result set."""
        results, total = store.list_templates_with_rank()
        assert results == []
        assert total == 0


# ============================================================================
# Test Featured and Trending Templates
# ============================================================================


class TestFeaturedAndTrending:
    """Tests for featured and trending template management."""

    def test_set_featured(self, store):
        """Should set template as featured."""
        template = store.create_template(
            name="Featured Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        store.set_featured(template.id, True)
        retrieved = store.get_template(template.id)
        assert retrieved.is_featured is True

        store.set_featured(template.id, False)
        retrieved = store.get_template(template.id)
        assert retrieved.is_featured is False

    def test_set_trending(self, store):
        """Should set template as trending."""
        template = store.create_template(
            name="Trending Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        store.set_trending(template.id, True)
        retrieved = store.get_template(template.id)
        assert retrieved.is_trending is True

    def test_get_featured(self, store):
        """Should return only featured templates."""
        t1 = store.create_template(
            name="Featured 1",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )
        store.set_featured(t1.id, True)

        t2 = store.create_template(
            name="Not Featured",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        featured = store.get_featured()
        assert len(featured) == 1
        assert featured[0].id == t1.id

    def test_get_featured_with_limit(self, store):
        """Should respect limit parameter."""
        for i in range(5):
            t = store.create_template(
                name=f"Featured {i}",
                description="Test",
                author_id="user-1",
                author_name="Author",
                category="security",
                pattern="adversarial",
                workflow_definition={},
            )
            store.set_featured(t.id, True)

        featured = store.get_featured(limit=3)
        assert len(featured) == 3

    def test_get_trending(self, store):
        """Should return trending templates sorted by downloads."""
        t1 = store.create_template(
            name="Less Popular",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )
        store.set_trending(t1.id, True)
        store.increment_download(t1.id)

        t2 = store.create_template(
            name="More Popular",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )
        store.set_trending(t2.id, True)
        for _ in range(10):
            store.increment_download(t2.id)

        trending = store.get_trending()
        assert len(trending) == 2
        assert trending[0].id == t2.id  # More downloads first

    def test_increment_download(self, store):
        """Should increment download count."""
        template = store.create_template(
            name="Download Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        assert store.get_template(template.id).download_count == 0

        store.increment_download(template.id)
        assert store.get_template(template.id).download_count == 1

        store.increment_download(template.id)
        store.increment_download(template.id)
        assert store.get_template(template.id).download_count == 3


# ============================================================================
# Test Rating Operations
# ============================================================================


class TestRatings:
    """Tests for rating functionality."""

    def test_rate_template_new_rating(self, store):
        """Should add new rating to template."""
        template = store.create_template(
            name="Rating Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        avg, count = store.rate_template(template.id, "user-a", 5)
        assert avg == 5.0
        assert count == 1

    def test_rate_template_multiple_users(self, store):
        """Should calculate average across multiple users."""
        template = store.create_template(
            name="Rating Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        store.rate_template(template.id, "user-a", 5)
        store.rate_template(template.id, "user-b", 3)
        avg, count = store.rate_template(template.id, "user-c", 4)

        assert count == 3
        assert avg == 4.0  # (5 + 3 + 4) / 3

    def test_rate_template_update_existing(self, store):
        """Should update existing rating from same user."""
        template = store.create_template(
            name="Rating Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        store.rate_template(template.id, "user-a", 3)
        avg, count = store.rate_template(template.id, "user-a", 5)

        assert count == 1  # Still one rating
        assert avg == 5.0  # Updated to new value

    def test_rate_template_invalid_rating(self, store):
        """Should reject invalid rating values."""
        template = store.create_template(
            name="Rating Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        with pytest.raises(ValueError, match="between 1 and 5"):
            store.rate_template(template.id, "user-a", 0)

        with pytest.raises(ValueError, match="between 1 and 5"):
            store.rate_template(template.id, "user-a", 6)

    def test_rate_template_boundary_values(self, store):
        """Should accept boundary rating values."""
        template = store.create_template(
            name="Rating Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        avg1, _ = store.rate_template(template.id, "user-a", 1)
        assert avg1 == 1.0

        avg5, _ = store.rate_template(template.id, "user-b", 5)
        assert avg5 == 3.0  # (1 + 5) / 2


# ============================================================================
# Test Review Operations
# ============================================================================


class TestReviews:
    """Tests for review functionality."""

    def test_create_review(self, store):
        """Should create a review for template."""
        template = store.create_template(
            name="Review Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        review = store.create_review(
            template_id=template.id,
            user_id="reviewer-1",
            user_name="Reviewer",
            rating=5,
            title="Excellent template!",
            content="This template is very well designed and easy to use.",
        )

        assert review.id.startswith("rev-")
        assert review.template_id == template.id
        assert review.rating == 5
        assert review.helpful_count == 0

    def test_create_review_also_rates_template(self, store):
        """Creating review should also add rating to template."""
        template = store.create_template(
            name="Review Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        store.create_review(
            template_id=template.id,
            user_id="reviewer-1",
            user_name="Reviewer",
            rating=4,
            title="Good template",
            content="Works well",
        )

        updated_template = store.get_template(template.id)
        assert updated_template.rating_count == 1
        assert updated_template.rating == 4.0

    def test_create_review_invalid_rating(self, store):
        """Should reject review with invalid rating."""
        template = store.create_template(
            name="Review Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        with pytest.raises(ValueError, match="between 1 and 5"):
            store.create_review(
                template_id=template.id,
                user_id="reviewer-1",
                user_name="Reviewer",
                rating=0,
                title="Bad",
                content="Invalid rating",
            )

    def test_list_reviews(self, store):
        """Should list reviews for template."""
        template = store.create_template(
            name="Review Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        for i in range(3):
            store.create_review(
                template_id=template.id,
                user_id=f"reviewer-{i}",
                user_name=f"Reviewer {i}",
                rating=5 - i,
                title=f"Review {i}",
                content=f"Content {i}",
            )

        reviews = store.list_reviews(template.id)
        assert len(reviews) == 3

    def test_list_reviews_pagination(self, store):
        """Should support review pagination."""
        template = store.create_template(
            name="Review Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        for i in range(5):
            store.create_review(
                template_id=template.id,
                user_id=f"reviewer-{i}",
                user_name=f"Reviewer {i}",
                rating=4,
                title=f"Review {i}",
                content=f"Content {i}",
            )

        reviews = store.list_reviews(template.id, limit=2, offset=0)
        assert len(reviews) == 2

        reviews2 = store.list_reviews(template.id, limit=2, offset=2)
        assert len(reviews2) == 2

    def test_list_reviews_empty(self, store):
        """Should return empty list for template with no reviews."""
        template = store.create_template(
            name="No Reviews",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        reviews = store.list_reviews(template.id)
        assert reviews == []

    def test_review_replaces_existing_by_same_user(self, store):
        """Same user's new review should replace existing."""
        template = store.create_template(
            name="Review Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        store.create_review(
            template_id=template.id,
            user_id="reviewer-1",
            user_name="Reviewer",
            rating=3,
            title="First Review",
            content="Initial thoughts",
        )

        store.create_review(
            template_id=template.id,
            user_id="reviewer-1",
            user_name="Reviewer",
            rating=5,
            title="Updated Review",
            content="Changed my mind!",
        )

        reviews = store.list_reviews(template.id)
        assert len(reviews) == 1
        assert reviews[0].title == "Updated Review"
        assert reviews[0].rating == 5


# ============================================================================
# Test Category Operations
# ============================================================================


class TestCategories:
    """Tests for category functionality."""

    def test_list_categories(self, store):
        """Should list all categories."""
        categories = store.list_categories()
        assert len(categories) >= 7
        assert all("id" in c for c in categories)
        assert all("name" in c for c in categories)
        assert all("description" in c for c in categories)
        assert all("template_count" in c for c in categories)

    def test_categories_ordered_by_count(self, store):
        """Should order categories by template count."""
        # Create templates in different categories
        for i in range(5):
            store.create_template(
                name=f"Security Template {i}",
                description="Test",
                author_id="user-1",
                author_name="Author",
                category="security",
                pattern="adversarial",
                workflow_definition={},
            )

        for i in range(2):
            store.create_template(
                name=f"Compliance Template {i}",
                description="Test",
                author_id="user-1",
                author_name="Author",
                category="compliance",
                pattern="adversarial",
                workflow_definition={},
            )

        categories = store.list_categories()

        # Security should be first (most templates)
        security_idx = next(i for i, c in enumerate(categories) if c["id"] == "security")
        compliance_idx = next(i for i, c in enumerate(categories) if c["id"] == "compliance")
        assert security_idx < compliance_idx


# ============================================================================
# Test Data Validation
# ============================================================================


class TestDataValidation:
    """Tests for data validation and constraints."""

    def test_template_requires_name(self, store):
        """Template name must be non-empty for meaningful display."""
        # While the database allows empty strings, templates should have names
        # This test documents the current behavior - empty names are allowed at DB level
        template = store.create_template(
            name="",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )
        # Document that empty names are technically allowed
        assert template.name == ""

        # However, duplicate empty names should still raise
        with pytest.raises(ValueError, match="already exists"):
            store.create_template(
                name="",
                description="Another",
                author_id="user-2",
                author_name="Other",
                category="security",
                pattern="adversarial",
                workflow_definition={},
            )

    def test_template_preserves_json_structure(self, store):
        """Should preserve complex JSON in workflow_definition."""
        complex_workflow = {
            "nodes": [
                {"id": "1", "type": "input", "data": {"nested": {"deep": [1, 2, 3]}}},
                {"id": "2", "type": "output", "config": None},
            ],
            "edges": [{"source": "1", "target": "2"}],
            "metadata": {"version": 1.5, "tags": ["a", "b"]},
        }

        template = store.create_template(
            name="Complex Workflow",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition=complex_workflow,
        )

        retrieved = store.get_template(template.id)
        assert retrieved.workflow_definition == complex_workflow

    def test_template_preserves_tags_list(self, store):
        """Should preserve tags as list."""
        tags = ["tag-1", "tag-2", "tag with spaces", "tag-123"]

        template = store.create_template(
            name="Tagged Template",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
            tags=tags,
        )

        retrieved = store.get_template(template.id)
        assert retrieved.tags == tags


# ============================================================================
# Test Concurrent Access
# ============================================================================


class TestConcurrentAccess:
    """Tests for concurrent database access."""

    def test_concurrent_reads(self, tmp_db_path):
        """Should handle concurrent reads safely."""
        store = MarketplaceStore(db_path=tmp_db_path)

        # Populate with data
        for i in range(10):
            store.create_template(
                name=f"Template {i}",
                description="Test",
                author_id="user-1",
                author_name="Author",
                category="security",
                pattern="adversarial",
                workflow_definition={},
            )

        results = []
        errors = []

        def reader(thread_id):
            try:
                for _ in range(10):
                    templates, total = store.list_templates()
                    results.append((thread_id, len(templates)))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 50
        assert all(count == 10 for _, count in results)

    def test_concurrent_writes(self, tmp_db_path):
        """Should handle concurrent writes safely."""
        store = MarketplaceStore(db_path=tmp_db_path)
        errors = []

        def writer(thread_id):
            try:
                for i in range(5):
                    store.create_template(
                        name=f"Thread {thread_id} Template {i}",
                        description="Test",
                        author_id=f"user-{thread_id}",
                        author_name=f"Author {thread_id}",
                        category="security",
                        pattern="adversarial",
                        workflow_definition={},
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        templates, total = store.list_templates(limit=100)
        assert total == 25

    def test_concurrent_ratings(self, tmp_db_path):
        """Should handle concurrent rating updates safely."""
        store = MarketplaceStore(db_path=tmp_db_path)

        template = store.create_template(
            name="Concurrent Rating Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        def rater(user_id):
            for _ in range(5):
                store.rate_template(template.id, f"user-{user_id}", (user_id % 5) + 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(rater, i) for i in range(20)]
            concurrent.futures.wait(futures)

        updated = store.get_template(template.id)
        assert updated.rating_count == 20


# ============================================================================
# Test PostgresMarketplaceStore
# ============================================================================


class TestPostgresMarketplaceStore:
    """Tests for PostgreSQL marketplace store."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock asyncpg pool."""
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=AsyncMock())
        return pool

    def test_init(self, mock_pool):
        """Should initialize with pool."""
        store = PostgresMarketplaceStore(mock_pool)
        assert store._pool is mock_pool
        assert store._initialized is False

    @pytest.mark.asyncio
    async def test_initialize(self, mock_pool):
        """Should initialize schema."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresMarketplaceStore(mock_pool)
        await store.initialize()

        mock_conn.execute.assert_called()
        assert store._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, mock_pool):
        """Should not reinitialize if already initialized."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresMarketplaceStore(mock_pool)
        store._initialized = True

        await store.initialize()

        mock_conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_template_async(self, mock_pool):
        """Should create template asynchronously."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresMarketplaceStore(mock_pool)

        template = await store.create_template_async(
            name="Async Template",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        assert template.name == "Async Template"
        assert template.id.startswith("tpl-")
        mock_conn.execute.assert_called()

    @pytest.mark.asyncio
    async def test_create_template_duplicate_name(self, mock_pool):
        """Should raise ValueError for duplicate name."""
        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = Exception("duplicate key value")
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresMarketplaceStore(mock_pool)

        with pytest.raises(ValueError, match="already exists"):
            await store.create_template_async(
                name="Duplicate",
                description="Test",
                author_id="user-1",
                author_name="Author",
                category="security",
                pattern="adversarial",
                workflow_definition={},
            )

    @pytest.mark.asyncio
    async def test_get_template_async(self, mock_pool):
        """Should get template asynchronously."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {
            "id": "tpl-123",
            "name": "Test",
            "description": "Desc",
            "author_id": "user-1",
            "author_name": "Author",
            "category": "security",
            "pattern": "adversarial",
            "tags": "[]",
            "workflow_definition": "{}",
            "download_count": 0,
            "rating_sum": 0.0,
            "rating_count": 0,
            "is_featured": False,
            "is_trending": False,
            "created_at": 1000.0,
            "updated_at": 1000.0,
        }
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresMarketplaceStore(mock_pool)
        template = await store.get_template_async("tpl-123")

        assert template is not None
        assert template.id == "tpl-123"

    @pytest.mark.asyncio
    async def test_get_template_not_found(self, mock_pool):
        """Should return None for nonexistent template."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresMarketplaceStore(mock_pool)
        result = await store.get_template_async("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_rate_template_invalid(self, mock_pool):
        """Should reject invalid rating."""
        store = PostgresMarketplaceStore(mock_pool)

        with pytest.raises(ValueError, match="between 1 and 5"):
            await store.rate_template_async("tpl-123", "user-1", 10)

    @pytest.mark.asyncio
    async def test_create_review_invalid_rating(self, mock_pool):
        """Should reject review with invalid rating."""
        store = PostgresMarketplaceStore(mock_pool)

        with pytest.raises(ValueError, match="between 1 and 5"):
            await store.create_review_async(
                template_id="tpl-123",
                user_id="user-1",
                user_name="User",
                rating=0,
                title="Bad",
                content="Invalid",
            )

    def test_close_is_noop(self, mock_pool):
        """Close should be a no-op for pool-based stores."""
        store = PostgresMarketplaceStore(mock_pool)
        store.close()  # Should not raise


# ============================================================================
# Test PostgresMarketplaceStore Sync Wrappers
# ============================================================================


class TestPostgresMarketplaceStoreSyncWrappers:
    """Tests for PostgreSQL store sync wrapper methods."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock asyncpg pool."""
        pool = MagicMock()
        return pool

    def test_sync_create_template(self, mock_pool, monkeypatch):
        """Sync create_template should call async version."""
        import aragora.storage.marketplace_store_postgres as pg_mod

        store = PostgresMarketplaceStore(mock_pool)

        expected = StoredTemplate(
            id="tpl-123",
            name="Test",
            description="Desc",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
        )

        with patch.object(store, "create_template_async") as mock_async:
            mock_async.return_value = expected

            mock_run = MagicMock(return_value=expected)
            monkeypatch.setattr(pg_mod._marketplace_mod, "run_async", mock_run)

            result = store.create_template(
                name="Test",
                description="Desc",
                author_id="user-1",
                author_name="Author",
                category="security",
                pattern="adversarial",
                workflow_definition={},
            )

            assert result.name == "Test"
            mock_run.assert_called_once()

    def test_sync_get_template(self, mock_pool, monkeypatch):
        """Sync get_template should call async version."""
        import aragora.storage.marketplace_store_postgres as pg_mod

        store = PostgresMarketplaceStore(mock_pool)

        mock_run = MagicMock(return_value=None)
        monkeypatch.setattr(pg_mod._marketplace_mod, "run_async", mock_run)

        result = store.get_template("tpl-123")
        assert result is None
        mock_run.assert_called_once()


# ============================================================================
# Test Global Store Management
# ============================================================================


class TestGlobalStoreManagement:
    """Tests for global store functions."""

    def test_set_marketplace_store(self, tmp_db_path):
        """Should set custom marketplace store."""
        custom_store = MarketplaceStore(db_path=tmp_db_path)
        set_marketplace_store(custom_store)

        retrieved = get_marketplace_store()
        assert retrieved is custom_store

    def test_reset_marketplace_store(self, tmp_db_path):
        """Should reset global store instance."""
        store1 = MarketplaceStore(db_path=tmp_db_path)
        set_marketplace_store(store1)

        reset_marketplace_store()

        # After reset, get_marketplace_store should create a new instance
        # (but we can't easily test this without mocking the factory)

    def test_multi_instance_sqlite_raises(self, monkeypatch):
        """Should raise for multi-instance mode with SQLite."""
        monkeypatch.setenv("ARAGORA_MULTI_INSTANCE", "true")
        monkeypatch.setenv("ARAGORA_DB_BACKEND", "sqlite")

        with pytest.raises(RuntimeError, match="ARAGORA_MULTI_INSTANCE"):
            get_marketplace_store()


# ============================================================================
# Test Row Conversion Helpers
# ============================================================================


class TestRowConversion:
    """Tests for row to dataclass conversion methods."""

    def test_sqlite_row_to_template(self, store):
        """Should convert SQLite row to StoredTemplate."""
        template = store.create_template(
            name="Conversion Test",
            description="Test description",
            author_id="user-1",
            author_name="Author Name",
            category="security",
            pattern="adversarial",
            workflow_definition={"key": "value"},
            tags=["tag1", "tag2"],
        )

        retrieved = store.get_template(template.id)

        assert isinstance(retrieved, StoredTemplate)
        assert retrieved.tags == ["tag1", "tag2"]
        assert retrieved.workflow_definition == {"key": "value"}

    def test_sqlite_row_to_review(self, store):
        """Should convert SQLite row to StoredReview."""
        template = store.create_template(
            name="Review Conversion Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        store.create_review(
            template_id=template.id,
            user_id="reviewer-1",
            user_name="Reviewer Name",
            rating=5,
            title="Great!",
            content="Love it",
        )

        reviews = store.list_reviews(template.id)
        assert len(reviews) == 1
        assert isinstance(reviews[0], StoredReview)
        assert reviews[0].user_name == "Reviewer Name"


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_tags(self, store):
        """Should handle empty tags list."""
        template = store.create_template(
            name="No Tags",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
            tags=[],
        )

        retrieved = store.get_template(template.id)
        assert retrieved.tags == []

    def test_empty_workflow_definition(self, store):
        """Should handle empty workflow definition."""
        template = store.create_template(
            name="Empty Workflow",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        retrieved = store.get_template(template.id)
        assert retrieved.workflow_definition == {}

    def test_special_characters_in_name(self, store):
        """Should handle special characters in template name."""
        template = store.create_template(
            name="Template with 'quotes' and \"double quotes\"",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        retrieved = store.get_template(template.id)
        assert retrieved.name == "Template with 'quotes' and \"double quotes\""

    def test_unicode_content(self, store):
        """Should handle unicode content."""
        template = store.create_template(
            name="Unicode Template",
            description="Description with emojis: 🚀 and special chars: é, ñ, 中文",
            author_id="user-1",
            author_name="Author with émoji 👤",
            category="security",
            pattern="adversarial",
            workflow_definition={"message": "Hello 世界"},
            tags=["тег", "標籤"],
        )

        retrieved = store.get_template(template.id)
        assert "🚀" in retrieved.description
        assert "中文" in retrieved.description
        assert retrieved.workflow_definition["message"] == "Hello 世界"

    def test_very_long_description(self, store):
        """Should handle very long descriptions."""
        long_desc = "A" * 10000

        template = store.create_template(
            name="Long Description",
            description=long_desc,
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        retrieved = store.get_template(template.id)
        assert retrieved.description == long_desc

    def test_null_workflow_definition_handling(self, store):
        """Should handle None-like workflow definitions."""
        with store.connection() as conn:
            # Manually insert a row with NULL-like workflow_definition
            conn.execute(
                """
                INSERT INTO templates (
                    id, name, description, author_id, author_name,
                    category, pattern, tags, workflow_definition,
                    download_count, rating_sum, rating_count,
                    is_featured, is_trending, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "tpl-null-test",
                    "Null Test",
                    "Test",
                    "user-1",
                    "Author",
                    "security",
                    "adversarial",
                    "[]",
                    "",  # Empty string for workflow
                    0,
                    0.0,
                    0,
                    0,
                    0,
                    time.time(),
                    time.time(),
                ),
            )

        retrieved = store.get_template("tpl-null-test")
        assert retrieved is not None
        assert retrieved.workflow_definition == {}


# ============================================================================
# Test Persistence
# ============================================================================


class TestPersistence:
    """Tests for data persistence across store instances."""

    def test_persistence_across_instances(self, tmp_db_path):
        """Should persist data across store instances."""
        # First instance
        store1 = MarketplaceStore(db_path=tmp_db_path)
        template = store1.create_template(
            name="Persistent Template",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={"key": "value"},
        )

        # Second instance (simulates restart)
        store2 = MarketplaceStore(db_path=tmp_db_path)
        retrieved = store2.get_template(template.id)

        assert retrieved is not None
        assert retrieved.name == "Persistent Template"
        assert retrieved.workflow_definition == {"key": "value"}

    def test_ratings_persist(self, tmp_db_path):
        """Should persist ratings across instances."""
        store1 = MarketplaceStore(db_path=tmp_db_path)
        template = store1.create_template(
            name="Rating Persistence",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        store1.rate_template(template.id, "user-a", 5)
        store1.rate_template(template.id, "user-b", 3)

        store2 = MarketplaceStore(db_path=tmp_db_path)
        retrieved = store2.get_template(template.id)

        assert retrieved.rating_count == 2
        assert retrieved.rating == 4.0

    def test_reviews_persist(self, tmp_db_path):
        """Should persist reviews across instances."""
        store1 = MarketplaceStore(db_path=tmp_db_path)
        template = store1.create_template(
            name="Review Persistence",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        store1.create_review(
            template_id=template.id,
            user_id="reviewer-1",
            user_name="Reviewer",
            rating=5,
            title="Great!",
            content="Excellent template",
        )

        store2 = MarketplaceStore(db_path=tmp_db_path)
        reviews = store2.list_reviews(template.id)

        assert len(reviews) == 1
        assert reviews[0].title == "Great!"


# ============================================================================
# Test Schema Compliance
# ============================================================================


class TestSchemaCompliance:
    """Tests for schema compliance and constraints."""

    def test_schema_name_and_version(self, store):
        """Should have correct schema name and version."""
        assert store.SCHEMA_NAME == "marketplace_store"
        assert store.SCHEMA_VERSION == 1

    def test_template_unique_name_constraint(self, store):
        """Templates table should enforce unique name constraint."""
        store.create_template(
            name="Unique Name",
            description="First",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        with pytest.raises(ValueError, match="already exists"):
            store.create_template(
                name="Unique Name",
                description="Second",
                author_id="user-2",
                author_name="Other",
                category="code-review",
                pattern="collaborative",
                workflow_definition={},
            )

    def test_review_unique_per_user_per_template(self, store):
        """Should enforce one review per user per template."""
        template = store.create_template(
            name="Review Constraint",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        # First review
        store.create_review(
            template_id=template.id,
            user_id="reviewer-1",
            user_name="Reviewer",
            rating=3,
            title="First",
            content="Initial",
        )

        # Second review from same user should replace (INSERT OR REPLACE)
        store.create_review(
            template_id=template.id,
            user_id="reviewer-1",
            user_name="Reviewer",
            rating=5,
            title="Second",
            content="Updated",
        )

        reviews = store.list_reviews(template.id)
        assert len(reviews) == 1
        assert reviews[0].title == "Second"

    def test_indexes_created(self, store):
        """Should create necessary indexes."""
        with store.connection() as conn:
            indexes = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
            ).fetchall()
            index_names = {i[0] for i in indexes}

            assert "idx_templates_category" in index_names
            assert "idx_templates_author" in index_names
            assert "idx_reviews_template" in index_names


# ============================================================================
# Test Author/Vendor Management
# ============================================================================


class TestAuthorManagement:
    """Tests for author/vendor management functionality."""

    def test_filter_templates_by_author(self, store_with_templates):
        """Should be able to find templates by author."""
        # Use list_templates with custom query through direct SQL
        with store_with_templates.connection() as conn:
            rows = conn.execute(
                "SELECT COUNT(*) FROM templates WHERE author_id = ?",
                ("author-0",),
            ).fetchone()
            count = rows[0]
            # Authors 0, 2, 4 have author-0
            assert count >= 2

    def test_multiple_templates_same_author(self, store):
        """Same author can create multiple templates."""
        for i in range(5):
            store.create_template(
                name=f"Author Template {i}",
                description="Test",
                author_id="same-author",
                author_name="Same Author",
                category="security",
                pattern="adversarial",
                workflow_definition={},
            )

        with store.connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM templates WHERE author_id = ?",
                ("same-author",),
            ).fetchone()[0]
            assert count == 5

    def test_author_name_preserved(self, store):
        """Author name should be preserved in template."""
        template = store.create_template(
            name="Author Name Test",
            description="Test",
            author_id="user-123",
            author_name="Dr. Jane Smith",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        retrieved = store.get_template(template.id)
        assert retrieved.author_name == "Dr. Jane Smith"


# ============================================================================
# Test Timestamps
# ============================================================================


class TestTimestamps:
    """Tests for timestamp handling."""

    def test_created_at_set_on_create(self, store):
        """created_at should be set when template is created."""
        before = time.time()
        template = store.create_template(
            name="Timestamp Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )
        after = time.time()

        assert before <= template.created_at <= after

    def test_updated_at_matches_created_at_initially(self, store):
        """updated_at should match created_at for new templates."""
        template = store.create_template(
            name="Timestamp Test 2",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        assert template.updated_at == template.created_at

    def test_rating_updates_updated_at(self, store):
        """Rating a template should update updated_at."""
        template = store.create_template(
            name="Rating Timestamp",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        original_updated = template.updated_at
        time.sleep(0.01)  # Small delay

        store.rate_template(template.id, "user-x", 5)

        updated = store.get_template(template.id)
        assert updated.updated_at > original_updated


# ============================================================================
# Test Search Functionality
# ============================================================================


class TestSearchFunctionality:
    """Tests for search and filtering."""

    def test_search_case_insensitive(self, store):
        """Search should be case-insensitive."""
        store.create_template(
            name="UPPERCASE Template",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        templates, _ = store.list_templates(search="uppercase")
        assert len(templates) == 1

        templates, _ = store.list_templates(search="UPPERCASE")
        assert len(templates) == 1

    def test_search_partial_match(self, store):
        """Search should find partial matches."""
        store.create_template(
            name="Complete Workflow Template",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        templates, _ = store.list_templates(search="Workflow")
        assert len(templates) == 1

    def test_search_in_description(self, store):
        """Search should find matches in description."""
        store.create_template(
            name="Generic Name",
            description="This template uses advanced machine learning algorithms",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        templates, _ = store.list_templates(search="machine learning")
        assert len(templates) == 1

    def test_search_returns_empty_for_no_matches(self, store):
        """Search should return empty for no matches."""
        store.create_template(
            name="Some Template",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        templates, total = store.list_templates(search="nonexistent12345")
        assert templates == []
        assert total == 0


# ============================================================================
# Test Sort Functionality
# ============================================================================


class TestSortFunctionality:
    """Tests for sorting templates."""

    def test_sort_by_updated(self, store):
        """Should sort by updated_at."""
        t1 = store.create_template(
            name="Old Updated",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        time.sleep(0.01)

        t2 = store.create_template(
            name="New Updated",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        # Update t1 to make it more recently updated
        time.sleep(0.01)
        store.rate_template(t1.id, "user-x", 5)

        templates, _ = store.list_templates(sort_by="updated")
        assert templates[0].id == t1.id  # t1 was updated most recently

    def test_invalid_sort_uses_default(self, store):
        """Invalid sort field should use default (rating)."""
        for i in range(3):
            t = store.create_template(
                name=f"Sort Test {i}",
                description="Test",
                author_id="user-1",
                author_name="Author",
                category="security",
                pattern="adversarial",
                workflow_definition={},
            )
            store.rate_template(t.id, "user-x", 5 - i)

        # Invalid sort should default to rating
        templates, _ = store.list_templates(sort_by="invalid_field")
        assert templates[0].rating == 5.0


# ============================================================================
# Test PostgresMarketplaceStore Additional Methods
# ============================================================================


class TestPostgresMarketplaceStoreAdditional:
    """Additional tests for PostgreSQL marketplace store."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock asyncpg pool."""
        pool = MagicMock()
        pool.acquire = MagicMock(return_value=AsyncMock())
        return pool

    @pytest.mark.asyncio
    async def test_list_templates_async(self, mock_pool):
        """Should list templates asynchronously."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = (0,)  # count
        mock_conn.fetch.return_value = []
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresMarketplaceStore(mock_pool)
        templates, total = await store.list_templates_async()

        assert templates == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_get_featured_async(self, mock_pool):
        """Should get featured templates asynchronously."""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresMarketplaceStore(mock_pool)
        featured = await store.get_featured_async()

        assert featured == []

    @pytest.mark.asyncio
    async def test_get_trending_async(self, mock_pool):
        """Should get trending templates asynchronously."""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresMarketplaceStore(mock_pool)
        trending = await store.get_trending_async()

        assert trending == []

    @pytest.mark.asyncio
    async def test_list_categories_async(self, mock_pool):
        """Should list categories asynchronously."""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [("security", "Security", "Security workflows", 5)]
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresMarketplaceStore(mock_pool)
        categories = await store.list_categories_async()

        assert len(categories) == 1
        assert categories[0]["id"] == "security"

    @pytest.mark.asyncio
    async def test_increment_download_async(self, mock_pool):
        """Should increment download count asynchronously."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresMarketplaceStore(mock_pool)
        await store.increment_download_async("tpl-123")

        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_featured_async(self, mock_pool):
        """Should set featured status asynchronously."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresMarketplaceStore(mock_pool)
        await store.set_featured_async("tpl-123", True)

        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_trending_async(self, mock_pool):
        """Should set trending status asynchronously."""
        mock_conn = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresMarketplaceStore(mock_pool)
        await store.set_trending_async("tpl-123", True)

        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_template_new_rating_async(self, mock_pool):
        """Should add new rating asynchronously."""

        # Create mock rows that support both index and key access like asyncpg Record
        class MockRecord:
            def __init__(self, data):
                self._data = data
                self._keys = list(data.keys())

            def __getitem__(self, key):
                if isinstance(key, int):
                    return self._data[self._keys[key]]
                return self._data[key]

        mock_conn = AsyncMock()
        mock_conn.fetchrow.side_effect = [
            None,  # No existing rating
            MockRecord({"rating_sum": 5.0, "rating_count": 1}),  # Updated stats
        ]
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresMarketplaceStore(mock_pool)
        avg, count = await store.rate_template_async("tpl-123", "user-1", 5)

        assert avg == 5.0
        assert count == 1

    @pytest.mark.asyncio
    async def test_rate_template_update_existing_async(self, mock_pool):
        """Should update existing rating asynchronously."""

        # Create mock rows that support both index and key access like asyncpg Record
        class MockRecord:
            def __init__(self, data):
                self._data = data
                self._keys = list(data.keys())

            def __getitem__(self, key):
                if isinstance(key, int):
                    return self._data[self._keys[key]]
                return self._data[key]

        mock_conn = AsyncMock()
        mock_conn.fetchrow.side_effect = [
            MockRecord({"rating": 3}),  # Existing rating
            MockRecord({"rating_sum": 5.0, "rating_count": 1}),  # Updated stats
        ]
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresMarketplaceStore(mock_pool)
        await store.rate_template_async("tpl-123", "user-1", 5)

        # Should have called execute for update
        assert mock_conn.execute.call_count >= 2

    @pytest.mark.asyncio
    async def test_list_reviews_async(self, mock_pool):
        """Should list reviews asynchronously."""
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn

        store = PostgresMarketplaceStore(mock_pool)
        reviews = await store.list_reviews_async("tpl-123")

        assert reviews == []


# ============================================================================
# Test Postgres Row Conversion with Dict-like Rows
# ============================================================================


class TestPostgresRowConversion:
    """Tests for PostgreSQL row conversion with dict-like rows."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock asyncpg pool."""
        pool = MagicMock()
        return pool

    def test_row_to_template_with_list_tags(self, mock_pool):
        """Should handle tags as list (not string)."""
        store = PostgresMarketplaceStore(mock_pool)

        row = {
            "id": "tpl-123",
            "name": "Test",
            "description": "Desc",
            "author_id": "user-1",
            "author_name": "Author",
            "category": "security",
            "pattern": "adversarial",
            "tags": ["tag1", "tag2"],  # Already a list
            "workflow_definition": {"key": "value"},  # Already a dict
            "download_count": 0,
            "rating_sum": 0.0,
            "rating_count": 0,
            "is_featured": False,
            "is_trending": False,
            "created_at": 1000.0,
            "updated_at": 1000.0,
        }

        template = store._row_to_template(row)

        assert template.tags == ["tag1", "tag2"]
        assert template.workflow_definition == {"key": "value"}

    def test_row_to_template_with_empty_tags(self, mock_pool):
        """Should handle None/empty tags."""
        store = PostgresMarketplaceStore(mock_pool)

        row = {
            "id": "tpl-123",
            "name": "Test",
            "description": "Desc",
            "author_id": "user-1",
            "author_name": "Author",
            "category": "security",
            "pattern": "adversarial",
            "tags": None,
            "workflow_definition": None,
            "download_count": 0,
            "rating_sum": 0.0,
            "rating_count": 0,
            "is_featured": False,
            "is_trending": False,
            "created_at": 1000.0,
            "updated_at": 1000.0,
        }

        template = store._row_to_template(row)

        assert template.tags == []
        assert template.workflow_definition == {}


# ============================================================================
# Test Transaction Behavior
# ============================================================================


class TestTransactionBehavior:
    """Tests for transaction behavior in store operations."""

    def test_create_template_atomic(self, store):
        """Template creation should be atomic."""
        # Create one template successfully
        store.create_template(
            name="Atomic Test",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        # Verify it was committed
        templates, total = store.list_templates()
        assert total == 1

    def test_rating_atomic(self, store):
        """Rating should be atomic."""
        template = store.create_template(
            name="Rating Atomic",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        store.rate_template(template.id, "user-a", 5)
        store.rate_template(template.id, "user-b", 3)

        retrieved = store.get_template(template.id)
        assert retrieved.rating_count == 2
        assert retrieved.rating_sum == 8.0


# ============================================================================
# Test Rating Edge Cases
# ============================================================================


class TestRatingEdgeCases:
    """Tests for rating edge cases."""

    def test_rating_nonexistent_template(self, store):
        """Rating nonexistent template should not fail immediately."""
        # This documents current behavior - rating a nonexistent template
        # creates a ratings record but template stats are 0
        avg, count = store.rate_template("nonexistent-template", "user-1", 5)
        assert count == 0  # Template doesn't exist, so stats are 0

    def test_many_ratings_same_template(self, store):
        """Should handle many ratings on same template."""
        template = store.create_template(
            name="Many Ratings",
            description="Test",
            author_id="user-1",
            author_name="Author",
            category="security",
            pattern="adversarial",
            workflow_definition={},
        )

        total_rating = 0
        for i in range(100):
            rating = (i % 5) + 1
            total_rating += rating
            store.rate_template(template.id, f"user-{i}", rating)

        retrieved = store.get_template(template.id)
        assert retrieved.rating_count == 100
        assert abs(retrieved.rating_sum - total_rating) < 0.01
