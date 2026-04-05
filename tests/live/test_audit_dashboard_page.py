from pathlib import Path


AUDIT_DASHBOARD_PAGE = Path("aragora/live/src/app/(app)/audit/page.tsx")


def test_audit_dashboard_uses_versioned_audit_session_endpoints() -> None:
    source = AUDIT_DASHBOARD_PAGE.read_text(encoding="utf-8")

    assert "`${backendConfig.api}/api/v1/audit/sessions`" in source
    assert "`${backendConfig.api}/api/v1/audit/sessions/${session.id}/report?format=html`" in source
    assert "`${backendConfig.api}/api/audit/sessions`" not in source
    assert (
        "`${backendConfig.api}/api/audit/sessions/${session.id}/report?format=html`" not in source
    )
