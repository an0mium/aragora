from __future__ import annotations

from aragora.server.openapi import generate_openapi_schema


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}

EXPECTED_DECLARED_METHODS = {
    "/api/v1/accounting/expenses": {"get", "post"},
    "/api/v1/accounting/expenses/upload": {"post"},
    "/api/v1/accounting/expenses/stats": {"get"},
    "/api/v1/accounting/expenses/{expense_id}": {"get", "put", "delete"},
    "/api/v1/accounting/expenses/{expense_id}/approve": {"post"},
    "/api/v1/accounting/invoices/{invoice_id}": {"get"},
    "/api/v1/accounting/invoices/{invoice_id}/approve": {"post"},
    "/api/v1/rlm/stats": {"get"},
    "/api/v1/rlm/contexts": {"get"},
    "/api/v1/rlm/query": {"post"},
    "/api/v1/rlm/context/{context_id}": {"delete", "get"},
}


def _operation_methods(path_spec: dict[str, object]) -> set[str]:
    return {method for method in path_spec if method in HTTP_METHODS}


def test_dict_style_routes_export_declared_methods() -> None:
    paths = generate_openapi_schema()["paths"]

    for path, expected_methods in EXPECTED_DECLARED_METHODS.items():
        assert _operation_methods(paths[path]) == expected_methods
