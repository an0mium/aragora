from __future__ import annotations

import os
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import TCPServer


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_frontend_routes.sh"


def _run_verify(
    base_url: str, *routes: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    return subprocess.run(
        ["bash", str(SCRIPT_PATH), base_url, *routes],
        cwd=REPO_ROOT,
        env=merged_env,
        capture_output=True,
        text=True,
        check=False,
    )


def _handler(expected_header: tuple[str, str] | None = None) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if expected_header is not None:
                name, value = expected_header
                if self.headers.get(name) != value:
                    self.send_response(401)
                    self.end_headers()
                    self.wfile.write(b"unauthorized")
                    return

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<html><body>ok</body></html>")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    return _Handler


@contextmanager
def _serve(handler_cls: type[BaseHTTPRequestHandler]):
    with TCPServer(("127.0.0.1", 0), handler_cls) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            yield f"http://{host}:{port}"
        finally:
            server.shutdown()
            thread.join(timeout=2)


def test_verify_frontend_routes_fails_on_401_by_default() -> None:
    with _serve(_handler(expected_header=("x-vercel-protection-bypass", "token"))) as base_url:
        result = _run_verify(base_url, "/")

    assert result.returncode != 0
    assert "status 401" in result.stdout


def test_verify_frontend_routes_uses_auth_header_when_configured() -> None:
    with _serve(_handler(expected_header=("x-vercel-protection-bypass", "token"))) as base_url:
        result = _run_verify(
            base_url,
            "/",
            env={"VERIFY_FRONTEND_AUTH_HEADER": "x-vercel-protection-bypass: token"},
        )

    assert result.returncode == 0
    assert "OK" in result.stdout


def test_verify_frontend_routes_soft_fail_is_non_blocking() -> None:
    with _serve(_handler(expected_header=("x-vercel-protection-bypass", "token"))) as base_url:
        result = _run_verify(
            base_url,
            "/",
            env={
                "VERIFY_FRONTEND_SOFT_FAIL": "1",
                "VERIFY_FRONTEND_ANNOTATION_LEVEL": "warning",
            },
        )

    assert result.returncode == 0
    assert "::warning::Route check failed" in result.stdout
    assert "soft-fail mode" in result.stdout
