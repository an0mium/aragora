"""Tests for aragora.server.openapi.endpoints.pulse."""

from aragora.server.openapi.endpoints.pulse import PULSE_ENDPOINTS


class TestPulseEndpointsStructure:
    def test_has_trending_endpoint(self):
        assert "/api/pulse/trending" in PULSE_ENDPOINTS

    def test_has_suggest_endpoint(self):
        assert "/api/pulse/suggest" in PULSE_ENDPOINTS

    def test_exactly_two_endpoints(self):
        assert len(PULSE_ENDPOINTS) == 2


class TestTrendingEndpoint:
    def setup_method(self):
        self.endpoint = PULSE_ENDPOINTS["/api/pulse/trending"]["get"]

    def test_tags(self):
        assert self.endpoint["tags"] == ["Pulse"]

    def test_operation_id(self):
        assert self.endpoint["operationId"] == "listPulseTrending"

    def test_summary(self):
        assert self.endpoint["summary"] == "Trending topics"

    def test_limit_parameter(self):
        params = self.endpoint["parameters"]
        assert len(params) == 1
        param = params[0]
        assert param["name"] == "limit"
        assert param["in"] == "query"
        assert param["schema"]["type"] == "integer"
        assert param["schema"]["default"] == 10
        assert param["schema"]["maximum"] == 50

    def test_response_200_exists(self):
        assert "200" in self.endpoint["responses"]

    def test_response_schema_has_topics_array(self):
        resp = self.endpoint["responses"]["200"]
        schema = resp["content"]["application/json"]["schema"]
        assert "topics" in schema["properties"]
        topics = schema["properties"]["topics"]
        assert topics["type"] == "array"
        item_props = topics["items"]["properties"]
        assert "title" in item_props
        assert "score" in item_props


class TestSuggestEndpoint:
    def setup_method(self):
        self.endpoint = PULSE_ENDPOINTS["/api/pulse/suggest"]["get"]

    def test_tags(self):
        assert self.endpoint["tags"] == ["Pulse"]

    def test_operation_id(self):
        assert self.endpoint["operationId"] == "listPulseSuggest"

    def test_category_parameter(self):
        params = self.endpoint["parameters"]
        assert len(params) == 1
        assert params[0]["name"] == "category"
        assert params[0]["schema"]["type"] == "string"

    def test_response_schema_has_expected_fields(self):
        resp = self.endpoint["responses"]["200"]
        schema = resp["content"]["application/json"]["schema"]
        props = schema["properties"]
        assert "topic" in props
        assert "category" in props
        assert "confidence" in props
        assert props["confidence"]["type"] == "number"

    def test_description(self):
        assert "AI-suggested" in self.endpoint["description"]
