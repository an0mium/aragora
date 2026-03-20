from __future__ import annotations

from subprocess import CompletedProcess

import scripts.generate_python_sdk_types as generate_python_sdk_types


def test_get_datamodel_codegen_version_parses_cli_output(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        return CompletedProcess(
            args=["datamodel-codegen", "--version"],
            returncode=0,
            stdout="datamodel-codegen 0.54.0\n",
        )

    monkeypatch.setattr(generate_python_sdk_types.subprocess, "run", fake_run)

    assert generate_python_sdk_types.get_datamodel_codegen_version() == "0.54.0"


def test_check_datamodel_codegen_requires_pinned_version(monkeypatch) -> None:
    monkeypatch.setattr(
        generate_python_sdk_types,
        "get_datamodel_codegen_version",
        lambda: generate_python_sdk_types.REQUIRED_DATAMODEL_CODEGEN_VERSION,
    )
    assert generate_python_sdk_types.check_datamodel_codegen() == (
        True,
        generate_python_sdk_types.REQUIRED_DATAMODEL_CODEGEN_VERSION,
    )

    monkeypatch.setattr(
        generate_python_sdk_types, "get_datamodel_codegen_version", lambda: "0.55.0"
    )
    assert generate_python_sdk_types.check_datamodel_codegen() == (False, "0.55.0")
