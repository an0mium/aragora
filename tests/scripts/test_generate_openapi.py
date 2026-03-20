from __future__ import annotations

from pathlib import Path

import scripts.generate_openapi as generate_openapi
from aragora.server.handlers.openapi_decorator import (
    api_endpoint,
    clear_registry,
    get_registered_endpoints,
)


def test_ast_extract_endpoints_cleans_docstrings(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(generate_openapi, "PROJECT_ROOT", tmp_path)
    handler_path = tmp_path / "example_handler.py"
    handler_path.write_text(
        """
from aragora.server.handlers.openapi_decorator import api_endpoint


class ExampleHandler:
    @api_endpoint(path="/api/example", method="GET", summary="Example")
    def handle(self):
        \"\"\"Example summary.

        Details line.
            Preserved nested indentation.
        \"\"\"
        return None
""".strip()
        + "\n",
        encoding="utf-8",
    )

    endpoints = generate_openapi._ast_extract_endpoints_from_file(handler_path)

    assert len(endpoints) == 1
    assert endpoints[0]["description"] == (
        "Example summary.\n\nDetails line.\n    Preserved nested indentation."
    )


def test_api_endpoint_cleans_docstrings() -> None:
    clear_registry()

    @api_endpoint(path="/api/example", method="GET", summary="Example")
    def handle() -> None:
        """Example summary.

        Details line.
            Preserved nested indentation.
        """

    try:
        endpoints = get_registered_endpoints()
        assert endpoints[-1].description == (
            "Example summary.\n\nDetails line.\n    Preserved nested indentation."
        )
    finally:
        clear_registry()
