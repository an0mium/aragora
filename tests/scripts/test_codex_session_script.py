from pathlib import Path


def test_codex_session_translates_autonomous_aliases_to_supported_codex_flags() -> None:
    script = Path("scripts/codex_session.sh").read_text(encoding="utf-8")

    assert "--full-auto|--yolo)" in script
    assert "AUTONOMOUS_CODEX=true" in script
    assert "CODEX_ARGS+=(--dangerously-bypass-approvals-and-sandbox)" in script
    assert "--ask-for-approval" not in script
    assert "CODEX_ARGS+=(--full-auto)" not in script
    assert 'codex "${CODEX_ARGS[@]}"' in script
    assert 'SESSION_ARGS_JSON="$(python3 - "${CODEX_ARGS[@]}"' in script
