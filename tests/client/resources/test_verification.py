"""Tests for VerificationAPI resource."""

import pytest

from aragora.client import AragoraClient
from aragora.client.resources.verification import VerificationAPI


class TestVerificationAPI:
    """Tests for VerificationAPI resource."""

    def test_verification_api_exists(self):
        """Test that VerificationAPI is accessible on client."""
        client = AragoraClient()
        assert isinstance(client.verification, VerificationAPI)

    def test_verification_api_has_basic_methods(self):
        """Test that VerificationAPI has basic methods."""
        client = AragoraClient()
        api = client.verification
        assert api is not None
        assert api._client is not None


class TestVerificationModels:
    """Tests for Verification model classes."""

    def test_verify_claim_request_import(self):
        """Test VerifyClaimRequest model can be imported."""
        from aragora.client.models import VerifyClaimRequest

        # Check that the model can be imported
        assert VerifyClaimRequest is not None

    def test_verify_claim_response_import(self):
        """Test VerifyClaimResponse model can be imported."""
        from aragora.client.models import VerifyClaimResponse

        # Check that the model can be imported
        assert VerifyClaimResponse is not None

    def test_verification_status_enum_import(self):
        """Test VerificationStatus enum can be imported."""
        from aragora.client.models import VerificationStatus

        # Check enum values exist
        assert hasattr(VerificationStatus, "VALID")
        assert hasattr(VerificationStatus, "INVALID")

    def test_verification_backend_enum_import(self):
        """Test VerificationBackend enum can be imported."""
        from aragora.client.models import VerificationBackend

        assert VerificationBackend.Z3 == "z3"
