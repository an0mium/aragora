"""
Tests for Google Analytics GA4 Connector.

Tests for Google Analytics Data API integration.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


class TestGoogleAnalyticsCredentials:
    """Tests for Google Analytics credentials."""

    def test_credentials_creation(self):
        """Test GoogleAnalyticsCredentials dataclass."""
        from aragora.connectors.analytics.google_analytics import GoogleAnalyticsCredentials

        creds = GoogleAnalyticsCredentials(
            access_token="ya29.abc123",
            property_id="properties/123456789",
        )

        assert creds.access_token == "ya29.abc123"
        assert creds.property_id == "properties/123456789"
        assert creds.base_url == "https://analyticsdata.googleapis.com/v1beta"

    def test_credentials_default_base_url(self):
        """Test default base URL."""
        from aragora.connectors.analytics.google_analytics import GoogleAnalyticsCredentials

        creds = GoogleAnalyticsCredentials(property_id="properties/123")
        assert "analyticsdata.googleapis.com" in creds.base_url


class TestGoogleAnalyticsDataclasses:
    """Tests for Google Analytics dataclass parsing."""

    def test_dimension_creation(self):
        """Test Dimension dataclass."""
        from aragora.connectors.analytics.google_analytics import Dimension

        dim = Dimension(name="country")
        assert dim.name == "country"
        assert dim.expression is None

    def test_dimension_to_api_simple(self):
        """Test Dimension.to_api for simple dimension."""
        from aragora.connectors.analytics.google_analytics import Dimension

        dim = Dimension(name="city")
        api_data = dim.to_api()

        assert api_data == {"name": "city"}

    def test_dimension_to_api_with_expression(self):
        """Test Dimension.to_api with expression."""
        from aragora.connectors.analytics.google_analytics import Dimension

        dim = Dimension(name="combined", expression="country")
        api_data = dim.to_api()

        assert api_data["name"] == "combined"
        assert "dimensionExpression" in api_data

    def test_metric_creation(self):
        """Test Metric dataclass."""
        from aragora.connectors.analytics.google_analytics import Metric

        metric = Metric(name="activeUsers")
        assert metric.name == "activeUsers"
        assert metric.expression is None

    def test_metric_to_api(self):
        """Test Metric.to_api conversion."""
        from aragora.connectors.analytics.google_analytics import Metric

        metric = Metric(name="sessions")
        api_data = metric.to_api()

        assert api_data == {"name": "sessions"}

    def test_metric_to_api_with_expression(self):
        """Test Metric.to_api with expression."""
        from aragora.connectors.analytics.google_analytics import Metric

        metric = Metric(name="custom", expression="activeUsers/sessions")
        api_data = metric.to_api()

        assert api_data["name"] == "custom"
        assert api_data["expression"] == "activeUsers/sessions"

    def test_date_range_creation(self):
        """Test DateRange dataclass."""
        from aragora.connectors.analytics.google_analytics import DateRange

        date_range = DateRange(start_date="7daysAgo", end_date="today")
        assert date_range.start_date == "7daysAgo"
        assert date_range.end_date == "today"
        assert date_range.name is None

    def test_date_range_to_api(self):
        """Test DateRange.to_api conversion."""
        from aragora.connectors.analytics.google_analytics import DateRange

        date_range = DateRange(start_date="2024-01-01", end_date="2024-01-31", name="January")
        api_data = date_range.to_api()

        assert api_data["startDate"] == "2024-01-01"
        assert api_data["endDate"] == "2024-01-31"
        assert api_data["name"] == "January"

    def test_filter_expression_creation(self):
        """Test FilterExpression dataclass."""
        from aragora.connectors.analytics.google_analytics import FilterExpression

        filter_expr = FilterExpression(
            field_name="country",
            string_filter={"matchType": "EXACT", "value": "United States"},
        )
        assert filter_expr.field_name == "country"
        assert filter_expr.string_filter["value"] == "United States"

    def test_filter_expression_to_api(self):
        """Test FilterExpression.to_api conversion."""
        from aragora.connectors.analytics.google_analytics import FilterExpression

        filter_expr = FilterExpression(
            field_name="city",
            string_filter={"matchType": "CONTAINS", "value": "York"},
        )
        api_data = filter_expr.to_api()

        assert "filter" in api_data
        assert api_data["filter"]["fieldName"] == "city"
        assert api_data["filter"]["stringFilter"]["value"] == "York"


class TestGoogleAnalyticsResponseParsing:
    """Tests for response parsing."""

    def test_dimension_header_from_api(self):
        """Test DimensionHeader.from_api parsing."""
        from aragora.connectors.analytics.google_analytics import DimensionHeader

        data = {"name": "country"}
        header = DimensionHeader.from_api(data)

        assert header.name == "country"

    def test_metric_header_from_api(self):
        """Test MetricHeader.from_api parsing."""
        from aragora.connectors.analytics.google_analytics import MetricHeader

        data = {"name": "activeUsers", "type": "TYPE_INTEGER"}
        header = MetricHeader.from_api(data)

        assert header.name == "activeUsers"
        assert header.type == "TYPE_INTEGER"

    def test_row_from_api(self):
        """Test Row.from_api parsing."""
        from aragora.connectors.analytics.google_analytics import Row

        data = {
            "dimensionValues": [{"value": "United States"}, {"value": "New York"}],
            "metricValues": [{"value": "1000"}, {"value": "500"}],
        }
        row = Row.from_api(data)

        assert row.dimension_values == ["United States", "New York"]
        assert row.metric_values == ["1000", "500"]

    def test_report_response_from_api(self):
        """Test ReportResponse.from_api parsing."""
        from aragora.connectors.analytics.google_analytics import ReportResponse

        data = {
            "dimensionHeaders": [{"name": "country"}],
            "metricHeaders": [{"name": "activeUsers", "type": "TYPE_INTEGER"}],
            "rows": [
                {"dimensionValues": [{"value": "US"}], "metricValues": [{"value": "1000"}]},
                {"dimensionValues": [{"value": "UK"}], "metricValues": [{"value": "500"}]},
            ],
            "rowCount": 2,
            "metadata": {"currencyCode": "USD"},
            "propertyQuota": {"tokensPerDay": {"consumed": 100}},
        }
        response = ReportResponse.from_api(data)

        assert len(response.dimension_headers) == 1
        assert len(response.metric_headers) == 1
        assert len(response.rows) == 2
        assert response.row_count == 2
        assert response.metadata["currencyCode"] == "USD"

    def test_report_response_to_dataframe_dict(self):
        """Test ReportResponse.to_dataframe_dict conversion."""
        from aragora.connectors.analytics.google_analytics import (
            ReportResponse,
            DimensionHeader,
            MetricHeader,
            Row,
        )

        response = ReportResponse(
            dimension_headers=[DimensionHeader(name="country")],
            metric_headers=[MetricHeader(name="activeUsers")],
            rows=[
                Row(dimension_values=["US"], metric_values=["1000"]),
                Row(dimension_values=["UK"], metric_values=["500"]),
            ],
        )

        df_dict = response.to_dataframe_dict()

        assert "country" in df_dict
        assert "activeUsers" in df_dict
        assert df_dict["country"] == ["US", "UK"]
        assert df_dict["activeUsers"] == ["1000", "500"]

    def test_realtime_report_from_api(self):
        """Test RealTimeReport.from_api parsing."""
        from aragora.connectors.analytics.google_analytics import RealTimeReport

        data = {
            "dimensionHeaders": [{"name": "country"}],
            "metricHeaders": [{"name": "activeUsers"}],
            "rows": [{"dimensionValues": [{"value": "US"}], "metricValues": [{"value": "50"}]}],
            "rowCount": 1,
        }
        report = RealTimeReport.from_api(data)

        assert len(report.rows) == 1
        assert report.row_count == 1


class TestGoogleAnalyticsEnums:
    """Tests for Google Analytics enum values."""

    def test_metric_aggregation_enum(self):
        """Test MetricAggregation enum values."""
        from aragora.connectors.analytics.google_analytics import MetricAggregation

        assert MetricAggregation.TOTAL.value == "TOTAL"
        assert MetricAggregation.MINIMUM.value == "MINIMUM"
        assert MetricAggregation.MAXIMUM.value == "MAXIMUM"
        assert MetricAggregation.COUNT.value == "COUNT"

    def test_order_type_enum(self):
        """Test OrderType enum values."""
        from aragora.connectors.analytics.google_analytics import OrderType

        assert OrderType.ALPHANUMERIC.value == "ALPHANUMERIC"
        assert OrderType.NUMERIC.value == "NUMERIC"


class TestGoogleAnalyticsError:
    """Tests for Google Analytics error handling."""

    def test_error_creation(self):
        """Test GoogleAnalyticsError creation."""
        from aragora.connectors.analytics.google_analytics import GoogleAnalyticsError

        error = GoogleAnalyticsError(
            message="Invalid property ID",
            status_code=400,
            error_details={"code": "INVALID_ARGUMENT"},
        )

        assert str(error) == "Invalid property ID"
        assert error.status_code == 400
        assert error.error_details["code"] == "INVALID_ARGUMENT"

    def test_error_default_details(self):
        """Test error with default details."""
        from aragora.connectors.analytics.google_analytics import GoogleAnalyticsError

        error = GoogleAnalyticsError("Something went wrong")
        assert error.error_details == {}


class TestGoogleAnalyticsMocks:
    """Tests for mock data generation."""

    def test_mock_report(self):
        """Test mock report generation."""
        from aragora.connectors.analytics.google_analytics import get_mock_report

        report = get_mock_report()

        assert len(report.dimension_headers) == 2
        assert len(report.metric_headers) == 1
        assert len(report.rows) == 2
        assert report.row_count == 2


class TestGoogleAnalyticsConnector:
    """Tests for GoogleAnalyticsConnector."""

    def test_connector_initialization(self):
        """Test connector initialization."""
        from aragora.connectors.analytics.google_analytics import (
            GoogleAnalyticsConnector,
            GoogleAnalyticsCredentials,
        )

        creds = GoogleAnalyticsCredentials(
            access_token="test_token",
            property_id="properties/123",
        )
        connector = GoogleAnalyticsConnector(creds)

        assert connector.credentials == creds
        assert connector._client is None

    @pytest.mark.asyncio
    async def test_connector_context_manager(self):
        """Test connector as context manager."""
        from aragora.connectors.analytics.google_analytics import (
            GoogleAnalyticsConnector,
            GoogleAnalyticsCredentials,
        )

        creds = GoogleAnalyticsCredentials(
            access_token="test_token",
            property_id="properties/123",
        )

        async with GoogleAnalyticsConnector(creds) as connector:
            assert connector is not None

    @pytest.mark.asyncio
    async def test_run_report_builds_request(self):
        """Test run_report builds correct request."""
        from aragora.connectors.analytics.google_analytics import (
            GoogleAnalyticsConnector,
            GoogleAnalyticsCredentials,
            DateRange,
            ReportResponse,
        )

        creds = GoogleAnalyticsCredentials(
            access_token="test_token",
            property_id="properties/123",
        )
        connector = GoogleAnalyticsConnector(creds)

        # Mock the _request method
        mock_response = {
            "dimensionHeaders": [{"name": "country"}],
            "metricHeaders": [{"name": "activeUsers"}],
            "rows": [{"dimensionValues": [{"value": "US"}], "metricValues": [{"value": "100"}]}],
            "rowCount": 1,
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.run_report(
                dimensions=["country"],
                metrics=["activeUsers"],
                date_ranges=[DateRange(start_date="7daysAgo", end_date="today")],
            )

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert "runReport" in call_args[0][1]

            assert isinstance(result, ReportResponse)
            assert result.row_count == 1

        await connector.close()

    @pytest.mark.asyncio
    async def test_run_realtime_report(self):
        """Test run_realtime_report method."""
        from aragora.connectors.analytics.google_analytics import (
            GoogleAnalyticsConnector,
            GoogleAnalyticsCredentials,
            RealTimeReport,
        )

        creds = GoogleAnalyticsCredentials(
            access_token="test_token",
            property_id="properties/123",
        )
        connector = GoogleAnalyticsConnector(creds)

        mock_response = {
            "dimensionHeaders": [{"name": "country"}],
            "metricHeaders": [{"name": "activeUsers"}],
            "rows": [{"dimensionValues": [{"value": "US"}], "metricValues": [{"value": "25"}]}],
            "rowCount": 1,
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.run_realtime_report()

            mock_request.assert_called_once()
            assert isinstance(result, RealTimeReport)

        await connector.close()

    @pytest.mark.asyncio
    async def test_get_metadata(self):
        """Test get_metadata method."""
        from aragora.connectors.analytics.google_analytics import (
            GoogleAnalyticsConnector,
            GoogleAnalyticsCredentials,
        )

        creds = GoogleAnalyticsCredentials(
            access_token="test_token",
            property_id="properties/123",
        )
        connector = GoogleAnalyticsConnector(creds)

        mock_response = {
            "dimensions": [{"apiName": "country", "uiName": "Country"}],
            "metrics": [{"apiName": "activeUsers", "uiName": "Active Users"}],
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.get_metadata()

            assert "dimensions" in result
            assert "metrics" in result

        await connector.close()

    @pytest.mark.asyncio
    async def test_get_active_users_now(self):
        """Test get_active_users_now convenience method."""
        from aragora.connectors.analytics.google_analytics import (
            GoogleAnalyticsConnector,
            GoogleAnalyticsCredentials,
            RealTimeReport,
            Row,
        )

        creds = GoogleAnalyticsCredentials(
            access_token="test_token",
            property_id="properties/123",
        )
        connector = GoogleAnalyticsConnector(creds)

        mock_response = {
            "dimensionHeaders": [],
            "metricHeaders": [{"name": "activeUsers"}],
            "rows": [{"dimensionValues": [], "metricValues": [{"value": "42"}]}],
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.get_active_users_now()
            assert result == 42

        await connector.close()

    @pytest.mark.asyncio
    async def test_batch_run_reports(self):
        """Test batch_run_reports method."""
        from aragora.connectors.analytics.google_analytics import (
            GoogleAnalyticsConnector,
            GoogleAnalyticsCredentials,
        )

        creds = GoogleAnalyticsCredentials(
            access_token="test_token",
            property_id="properties/123",
        )
        connector = GoogleAnalyticsConnector(creds)

        mock_response = {
            "reports": [
                {
                    "dimensionHeaders": [{"name": "country"}],
                    "metricHeaders": [{"name": "activeUsers"}],
                    "rows": [],
                },
                {
                    "dimensionHeaders": [{"name": "city"}],
                    "metricHeaders": [{"name": "sessions"}],
                    "rows": [],
                },
            ]
        }

        with patch.object(connector, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await connector.batch_run_reports([{}, {}])
            assert len(result) == 2

        await connector.close()


class TestGoogleAnalyticsPackageImports:
    """Test that Google Analytics imports work correctly."""

    def test_google_analytics_imports(self):
        """Test Google Analytics can be imported from analytics package."""
        from aragora.connectors.analytics.google_analytics import (
            GoogleAnalyticsConnector,
            GoogleAnalyticsCredentials,
            Dimension,
            Metric,
            DateRange,
            FilterExpression,
            ReportResponse,
            RealTimeReport,
            GoogleAnalyticsError,
        )

        assert GoogleAnalyticsConnector is not None
        assert GoogleAnalyticsCredentials is not None
        assert Dimension is not None
        assert Metric is not None
        assert DateRange is not None
