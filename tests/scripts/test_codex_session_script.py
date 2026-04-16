from pathlib import Path


def test_codex_session_forwards_full_auto_to_codex() -> None:
    script = Path("scripts/codex_session.sh").read_text(encoding="utf-8")

    assert "--full-auto)" in script
    assert "CODEX_ARGS+=(--full-auto)" in script
    assert 'codex "${CODEX_ARGS[@]}"' in script
    assert 'SESSION_ARGS_JSON="$(python3 - "${CODEX_ARGS[@]}"' in script
