"""
Tests for MarketplaceStore - SQLite-backed template persistence.
"""

import os
import tempfile
import pytest
from pathlib import Path

from aragora.storage.marketplace_store import MarketplaceStore, StoredTemplate, StoredReview


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def store(temp_db):
    """Create a fresh store instance."""
    return MarketplaceStore(db_path=temp_db)


class TestMarketplaceStore:
    """Test MarketplaceStore functionality."""

    def test_create_template(self, store):
        template = store.create_template(
            name="Test Template",
            description="A test template",
            author_id="user-123",
            author_name="Test User",
            category="security",
            pattern="review_cycle",
            workflow_definition={"nodes": [], "edges": []},
            tags=["test", "unit-test"],
        )

        assert template.id.startswith("tpl-")
        assert template.name == "Test Template"
        assert template.author_name == "Test User"
        assert template.category == "security"
        assert template.tags == ["test", "unit-test"]

    def test_get_template(self, store):
        created = store.create_template(
            name="Get Test",
            description="Test get",
            author_id="user-1",
            author_name="User 1",
            category="testing",
            pattern="pipeline",
            workflow_definition={},
        )

        retrieved = store.get_template(created.id)
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == "Get Test"

    def test_get_nonexistent_template(self, store):
        result = store.get_template("nonexistent-id")
        assert result is None

    def test_list_templates(self, store):
        # Create multiple templates
        store.create_template(
            name="Template 1",
            description="First",
            author_id="u1",
            author_name="User 1",
            category="security",
            pattern="p1",
            workflow_definition={},
        )
        store.create_template(
            name="Template 2",
            description="Second",
            author_id="u2",
            author_name="User 2",
            category="testing",
            pattern="p2",
            workflow_definition={},
        )

        templates, total = store.list_templates()
        assert total == 2
        assert len(templates) == 2

    def test_list_templates_with_category_filter(self, store):
        store.create_template(
            name="Security Template",
            description="Security",
            author_id="u1",
            author_name="U1",
            category="security",
            pattern="p1",
            workflow_definition={},
        )
        store.create_template(
            name="Testing Template",
            description="Testing",
            author_id="u2",
            author_name="U2",
            category="testing",
            pattern="p2",
            workflow_definition={},
        )

        templates, total = store.list_templates(category="security")
        assert total == 1
        assert templates[0].category == "security"

    def test_list_templates_with_search(self, store):
        store.create_template(
            name="Code Analyzer",
            description="Analyzes code",
            author_id="u1",
            author_name="U1",
            category="code-review",
            pattern="p1",
            workflow_definition={},
        )
        store.create_template(
            name="Contract Review",
            description="Reviews contracts",
            author_id="u2",
            author_name="U2",
            category="legal",
            pattern="p2",
            workflow_definition={},
        )

        templates, total = store.list_templates(search="code")
        assert total == 1
        assert "Code" in templates[0].name

    def test_rate_template(self, store):
        template = store.create_template(
            name="Rate Test",
            description="Test rating",
            author_id="u1",
            author_name="U1",
            category="testing",
            pattern="p1",
            workflow_definition={},
        )

        # Rate the template
        avg, count = store.rate_template(template.id, "voter-1", 5)
        assert avg == 5.0
        assert count == 1

        # Second rating
        avg, count = store.rate_template(template.id, "voter-2", 3)
        assert avg == 4.0  # (5 + 3) / 2
        assert count == 2

    def test_rate_template_update_existing(self, store):
        template = store.create_template(
            name="Update Rate Test",
            description="Test",
            author_id="u1",
            author_name="U1",
            category="testing",
            pattern="p1",
            workflow_definition={},
        )

        # Initial rating
        avg, count = store.rate_template(template.id, "voter-1", 3)
        assert avg == 3.0
        assert count == 1

        # Update rating from same user
        avg, count = store.rate_template(template.id, "voter-1", 5)
        assert avg == 5.0
        assert count == 1  # Count should not increase

    def test_rate_template_invalid_rating(self, store):
        template = store.create_template(
            name="Invalid Rate Test",
            description="Test",
            author_id="u1",
            author_name="U1",
            category="testing",
            pattern="p1",
            workflow_definition={},
        )

        with pytest.raises(ValueError, match="between 1 and 5"):
            store.rate_template(template.id, "voter-1", 6)

        with pytest.raises(ValueError, match="between 1 and 5"):
            store.rate_template(template.id, "voter-1", 0)

    def test_create_review(self, store):
        template = store.create_template(
            name="Review Test",
            description="Test",
            author_id="u1",
            author_name="U1",
            category="testing",
            pattern="p1",
            workflow_definition={},
        )

        review = store.create_review(
            template_id=template.id,
            user_id="reviewer-1",
            user_name="Reviewer One",
            rating=5,
            title="Great template!",
            content="This template saved me hours.",
        )

        assert review.id.startswith("rev-")
        assert review.template_id == template.id
        assert review.rating == 5
        assert review.title == "Great template!"

    def test_list_reviews(self, store):
        template = store.create_template(
            name="Review List Test",
            description="Test",
            author_id="u1",
            author_name="U1",
            category="testing",
            pattern="p1",
            workflow_definition={},
        )

        store.create_review(
            template_id=template.id,
            user_id="r1",
            user_name="R1",
            rating=4,
            title="Good",
            content="Nice",
        )
        store.create_review(
            template_id=template.id,
            user_id="r2",
            user_name="R2",
            rating=5,
            title="Great",
            content="Excellent",
        )

        reviews = store.list_reviews(template.id)
        assert len(reviews) == 2

    def test_increment_download(self, store):
        template = store.create_template(
            name="Download Test",
            description="Test",
            author_id="u1",
            author_name="U1",
            category="testing",
            pattern="p1",
            workflow_definition={},
        )

        # Initially 0 downloads
        retrieved = store.get_template(template.id)
        assert retrieved.download_count == 0

        # Increment downloads
        store.increment_download(template.id)
        store.increment_download(template.id)

        retrieved = store.get_template(template.id)
        assert retrieved.download_count == 2

    def test_set_featured(self, store):
        template = store.create_template(
            name="Featured Test",
            description="Test",
            author_id="u1",
            author_name="U1",
            category="testing",
            pattern="p1",
            workflow_definition={},
        )

        # Initially not featured
        assert not store.get_template(template.id).is_featured

        # Set as featured
        store.set_featured(template.id, True)
        assert store.get_template(template.id).is_featured

        # Get featured templates
        featured = store.get_featured()
        assert len(featured) == 1
        assert featured[0].id == template.id

    def test_list_categories(self, store):
        categories = store.list_categories()
        assert len(categories) >= 7  # Default categories

        # Check structure
        for cat in categories:
            assert "id" in cat
            assert "name" in cat
            assert "template_count" in cat

    def test_duplicate_template_name(self, store):
        store.create_template(
            name="Unique Name",
            description="First",
            author_id="u1",
            author_name="U1",
            category="testing",
            pattern="p1",
            workflow_definition={},
        )

        with pytest.raises(ValueError, match="already exists"):
            store.create_template(
                name="Unique Name",
                description="Duplicate",
                author_id="u2",
                author_name="U2",
                category="testing",
                pattern="p2",
                workflow_definition={},
            )


class TestStoredTemplate:
    """Test StoredTemplate dataclass."""

    def test_rating_calculation(self):
        template = StoredTemplate(
            id="t1",
            name="Test",
            description="Test",
            author_id="a1",
            author_name="Author",
            category="test",
            pattern="p1",
            rating_sum=15.0,
            rating_count=3,
        )
        assert template.rating == 5.0

    def test_rating_zero_count(self):
        template = StoredTemplate(
            id="t1",
            name="Test",
            description="Test",
            author_id="a1",
            author_name="Author",
            category="test",
            pattern="p1",
            rating_sum=0.0,
            rating_count=0,
        )
        assert template.rating == 0.0

    def test_to_dict(self):
        template = StoredTemplate(
            id="t1",
            name="Test",
            description="Description",
            author_id="a1",
            author_name="Author",
            category="test",
            pattern="p1",
            tags=["tag1", "tag2"],
        )
        data = template.to_dict()

        assert data["id"] == "t1"
        assert data["name"] == "Test"
        assert data["tags"] == ["tag1", "tag2"]
        assert "workflow_definition" not in data  # Not in summary

    def test_to_full_dict(self):
        template = StoredTemplate(
            id="t1",
            name="Test",
            description="Description",
            author_id="a1",
            author_name="Author",
            category="test",
            pattern="p1",
            workflow_definition={"nodes": [{"id": "n1"}]},
        )
        data = template.to_full_dict()

        assert "workflow_definition" in data
        assert data["workflow_definition"]["nodes"][0]["id"] == "n1"


class TestListTemplatesWithRank:
    """Test the list_templates_with_rank method using window functions."""

    def test_empty_store_returns_empty(self, store):
        """Should return empty list and zero total for empty store."""
        templates, total = store.list_templates_with_rank()
        assert templates == []
        assert total == 0

    def test_includes_global_rank(self, store):
        """Should include global_rank in results."""
        # Create templates with different ratings
        store.create_template(
            name="Low Rated",
            description="Low",
            author_id="u1",
            author_name="U1",
            category="test",
            pattern="p1",
            workflow_definition={},
        )
        store.create_template(
            name="High Rated",
            description="High",
            author_id="u2",
            author_name="U2",
            category="test",
            pattern="p2",
            workflow_definition={},
        )

        templates, total = store.list_templates_with_rank(sort_by="newest")
        assert total == 2
        assert len(templates) == 2
        # Each template should have global_rank
        for t in templates:
            assert "global_rank" in t
            assert isinstance(t["global_rank"], int)

    def test_includes_category_rank(self, store):
        """Should include category_rank in results."""
        # Create templates in different categories
        store.create_template(
            name="Security 1",
            description="Sec1",
            author_id="u1",
            author_name="U1",
            category="security",
            pattern="p1",
            workflow_definition={},
        )
        store.create_template(
            name="Security 2",
            description="Sec2",
            author_id="u2",
            author_name="U2",
            category="security",
            pattern="p2",
            workflow_definition={},
        )
        store.create_template(
            name="Testing 1",
            description="Test1",
            author_id="u3",
            author_name="U3",
            category="testing",
            pattern="p3",
            workflow_definition={},
        )

        templates, total = store.list_templates_with_rank(category="security")
        assert total == 2
        # All should be in security category with category ranks 1 and 2
        for t in templates:
            assert "category_rank" in t
            assert t["category_rank"] in [1, 2]

    def test_respects_pagination(self, store):
        """Should respect limit and offset for pagination."""
        # Create 5 templates
        for i in range(5):
            store.create_template(
                name=f"Template {i}",
                description=f"Desc {i}",
                author_id=f"u{i}",
                author_name=f"U{i}",
                category="test",
                pattern="p1",
                workflow_definition={},
            )

        templates, total = store.list_templates_with_rank(limit=2, offset=0)
        assert total == 5  # Total count from window function
        assert len(templates) == 2  # Only 2 returned due to limit

    def test_search_filter(self, store):
        """Should filter by search term."""
        store.create_template(
            name="Python Security",
            description="Python security template",
            author_id="u1",
            author_name="U1",
            category="security",
            pattern="p1",
            workflow_definition={},
        )
        store.create_template(
            name="Java Testing",
            description="Java testing template",
            author_id="u2",
            author_name="U2",
            category="testing",
            pattern="p2",
            workflow_definition={},
        )

        templates, total = store.list_templates_with_rank(search="Python")
        assert total == 1
        assert templates[0]["name"] == "Python Security"

    def test_sort_by_downloads(self, store):
        """Should sort by downloads when requested."""
        t1 = store.create_template(
            name="Popular",
            description="Many downloads",
            author_id="u1",
            author_name="U1",
            category="test",
            pattern="p1",
            workflow_definition={},
        )
        t2 = store.create_template(
            name="Unpopular",
            description="Few downloads",
            author_id="u2",
            author_name="U2",
            category="test",
            pattern="p2",
            workflow_definition={},
        )

        # Increment downloads for first template
        for _ in range(10):
            store.increment_download(t1.id)

        templates, _ = store.list_templates_with_rank(sort_by="downloads")
        assert templates[0]["name"] == "Popular"
        assert templates[0]["global_rank"] == 1
