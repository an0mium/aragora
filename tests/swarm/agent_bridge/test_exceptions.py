from __future__ import annotations

import pytest

from aragora.swarm.agent_bridge.exceptions import FooterValidationError
from aragora.swarm.agent_bridge.exceptions import MissingSessionIdentityError
from aragora.swarm.agent_bridge.exceptions import TransportLaunchError
from aragora.swarm.agent_bridge.exceptions import TransportNotAvailableError
from aragora.swarm.agent_bridge.exceptions import TransportOutputParseError
from aragora.swarm.agent_bridge.exceptions import TransportResumeError


@pytest.mark.parametrize(
    "error_type",
    [
        TransportNotAvailableError,
        TransportLaunchError,
        TransportResumeError,
        TransportOutputParseError,
        FooterValidationError,
    ],
)
def test_each_specific_error_class_is_distinct(error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        raise error_type("boom")


def test_missing_session_identity_is_specific_parse_error() -> None:
    with pytest.raises(MissingSessionIdentityError):
        raise MissingSessionIdentityError("missing session")

    with pytest.raises(TransportOutputParseError):
        raise MissingSessionIdentityError("missing session")
