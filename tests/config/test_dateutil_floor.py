"""Regression guard for the python-dateutil floor.

python-dateutil < 2.8.1 calls the removed ``collections.Callable`` and crashes
on Python 3.10+ when botocore parses AWS response timestamps with ``tzinfos=``.
That abort cascaded through ``hydrate_env_from_secrets`` and broke the entire
settlement/secrets CLI path (observed with 2.6.1 in the field). pyproject pins
``python-dateutil>=2.8.2``; these tests fail loudly if a broken version is ever
resolved into the environment again.
"""

from __future__ import annotations

from importlib import metadata

from dateutil import parser as dateutil_parser
from dateutil.tz import tzutc


def test_dateutil_version_is_at_or_above_floor():
    version = metadata.version("python-dateutil")
    parts = tuple(int(p) for p in version.split(".")[:3] if p.isdigit())
    assert parts >= (2, 8, 2), f"python-dateutil {version} is below the >=2.8.2 floor"


def test_parse_timestamp_with_tzinfos_does_not_crash():
    # Mirrors botocore's AWS timestamp parsing: dateutil.parser.parse(value,
    # tzinfos={'GMT': tzutc()}). On <2.8.1 this raised
    # AttributeError: module 'collections' has no attribute 'Callable'.
    parsed = dateutil_parser.parse("2026-06-04T23:44:00 GMT", tzinfos={"GMT": tzutc()})
    assert parsed.year == 2026
    assert parsed.tzinfo is not None
