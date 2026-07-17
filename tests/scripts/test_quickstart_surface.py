from __future__ import annotations

import textwrap
from pathlib import Path

from scripts.check_quickstart_surface import check_documents, extract_python_blocks


def _write(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return path


def _manifest() -> dict[str, object]:
    return {
        "version": "2.8.0",
        "clients": {
            "AragoraAsyncClient": {
                "context_managers": {"async": True, "sync": False},
                "methods": ["close"],
                "namespaces": {"debates": ["create", "list"], "agents": ["list"]},
            },
            "AragoraClient": {
                "context_managers": {"async": False, "sync": True},
                "methods": ["close"],
                "namespaces": {"debates": ["create", "list"], "agents": ["list"]},
            },
        },
    }


def test_extract_python_blocks_preserves_source_line(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "guide.md",
        """
        # Guide

        ```bash
        pip install aragora-sdk
        ```

        ```python
        print("hello")
        ```
        """,
    )

    blocks = extract_python_blocks(path)

    assert len(blocks) == 1
    assert blocks[0].start_line == 8
    assert blocks[0].source == 'print("hello")'


def test_checker_accepts_released_async_quickstart(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "guide.md",
        """
        ```python
        import asyncio
        from aragora_sdk import AragoraAsyncClient

        async def main():
            async with AragoraAsyncClient(demo=True) as client:
                created = await client.debates.create(task="Test")
                debates = await client.debates.list(limit=5)
                agents = await client.agents.list()
                print(created, debates, agents)

        asyncio.run(main())
        ```
        """,
    )

    assert check_documents([path], _manifest()) == []


def test_checker_rejects_missing_released_call(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "guide.md",
        """
        ```python
        from aragora_sdk import AragoraAsyncClient

        async def main():
            async with AragoraAsyncClient() as client:
                await client.debates.run(task="Test")
        ```
        """,
    )

    findings = check_documents([path], _manifest())

    assert len(findings) == 1
    assert findings[0].line == 6
    assert "AragoraAsyncClient.debates.run()" in findings[0].message


def test_checker_rejects_sync_client_in_async_context(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "guide.md",
        """
        ```python
        from aragora_sdk import AragoraClient

        async def main():
            async with AragoraClient() as client:
                await client.debates.create(task="Test")
        ```
        """,
    )

    findings = check_documents([path], _manifest())

    assert len(findings) == 1
    assert findings[0].line == 5
    assert findings[0].message == "AragoraClient does not support async with"


def test_checker_ignores_other_client_packages(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "guide.md",
        """
        ```python
        from aragora.client import AragoraClient

        client = AragoraClient()
        client.debates.run(task="Test")
        ```
        """,
    )

    assert check_documents([path], _manifest()) == []
