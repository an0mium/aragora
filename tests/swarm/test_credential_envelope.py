from __future__ import annotations

from aragora.swarm.credential_envelope import CredentialEnvelope


def test_from_environment_defaults_to_unknown_provider():
    env = {}
    envelope = CredentialEnvelope.from_environment(env)
    assert envelope.provider.provider_name == "unknown"
    assert envelope.provider.api_key_present is False


def test_runner_profile_sets_auth_mode_profile():
    env = {"ARAGORA_CLAUDE_PROFILE": "default"}
    envelope = CredentialEnvelope.from_environment(env)
    assert envelope.runner.profile == "default"
    assert envelope.runner.auth_mode == "profile"


def test_runner_command_sets_auth_mode_command():
    env = {"CLAUDE_COMMAND": "/usr/local/bin/claude"}
    envelope = CredentialEnvelope.from_environment(env)
    assert envelope.runner.command_path.endswith("claude")
    assert envelope.runner.auth_mode == "command"


def test_git_detects_ssh_key():
    env = {"SSH_AUTH_SOCK": "/tmp/agent.sock"}
    envelope = CredentialEnvelope.from_environment(env)
    assert envelope.git.ssh_key_available is True


def test_git_detects_https_token():
    env = {"GH_TOKEN": "token"}
    envelope = CredentialEnvelope.from_environment(env)
    assert envelope.git.https_token_available is True


def test_git_safe_env_strips_github_tokens():
    env = {"GH_TOKEN": "token", "GITHUB_TOKEN": "token2", "PATH": "/bin"}
    envelope = CredentialEnvelope.from_environment(env)
    assert "GH_TOKEN" not in envelope.git.safe_env
    assert "GITHUB_TOKEN" not in envelope.git.safe_env
    assert envelope.git.safe_env["PATH"] == "/bin"


def test_github_api_token_source_user():
    env = {"GITHUB_TOKEN": "token"}
    envelope = CredentialEnvelope.from_environment(env)
    assert envelope.github_api.token_source == "user"


def test_github_api_token_source_app():
    env = {"GITHUB_APP_ID": "1", "GITHUB_TOKEN": "token"}
    envelope = CredentialEnvelope.from_environment(env)
    assert envelope.github_api.token_source == "app"


def test_github_api_rate_limit_parsing():
    env = {"GITHUB_RATE_LIMIT_REMAINING": "42"}
    envelope = CredentialEnvelope.from_environment(env)
    assert envelope.github_api.rate_limit_remaining == 42


def test_github_api_rate_limit_invalid():
    env = {"GITHUB_RATE_LIMIT_REMAINING": "nope"}
    envelope = CredentialEnvelope.from_environment(env)
    assert envelope.github_api.rate_limit_remaining is None


def test_provider_api_key_detected():
    env = {"OPENAI_API_KEY": "key", "ARAGORA_PROVIDER": "openai"}
    envelope = CredentialEnvelope.from_environment(env)
    assert envelope.provider.api_key_present is True
    assert envelope.provider.provider_name == "openai"


def test_provider_name_fallback():
    env = {"PROVIDER_NAME": "custom"}
    envelope = CredentialEnvelope.from_environment(env)
    assert envelope.provider.provider_name == "custom"


def test_verification_detects_pytest_and_ruff():
    env = {"PYTEST_AVAILABLE": "true", "RUFF_AVAILABLE": "true"}
    envelope = CredentialEnvelope.from_environment(env)
    assert envelope.verification.can_run_pytest is True
    assert envelope.verification.can_run_ruff is True


def test_is_complete_false_when_missing():
    env = {"GITHUB_TOKEN": "token"}
    envelope = CredentialEnvelope.from_environment(env)
    assert envelope.is_complete() is False
    assert "provider" in envelope.missing_slices()


def test_is_complete_true_when_all_present():
    env = {
        "ARAGORA_CLAUDE_PROFILE": "default",
        "GITHUB_TOKEN": "token",
        "OPENAI_API_KEY": "key",
        "ARAGORA_PROVIDER": "openai",
        "PYTEST_AVAILABLE": "true",
        "RUFF_AVAILABLE": "true",
        "SSH_AUTH_SOCK": "/tmp/agent.sock",
    }
    envelope = CredentialEnvelope.from_environment(env)
    assert envelope.is_complete() is True
    assert envelope.missing_slices() == []


def test_missing_slices_order_is_stable():
    env = {}
    envelope = CredentialEnvelope.from_environment(env)
    assert envelope.missing_slices() == [
        "runner",
        "git",
        "github_api",
        "provider",
        "verification",
    ]


def test_to_dict_round_trip():
    env = {
        "ARAGORA_CLAUDE_PROFILE": "default",
        "GITHUB_TOKEN": "token",
        "OPENAI_API_KEY": "key",
        "ARAGORA_PROVIDER": "openai",
        "PYTEST_AVAILABLE": "true",
        "RUFF_AVAILABLE": "true",
        "SSH_AUTH_SOCK": "/tmp/agent.sock",
    }
    envelope = CredentialEnvelope.from_environment(env)
    restored = CredentialEnvelope.from_dict(envelope.to_dict())
    assert restored.to_dict() == envelope.to_dict()


def test_seal_is_stable_for_same_envelope():
    env = {"GITHUB_TOKEN": "token"}
    envelope = CredentialEnvelope.from_environment(env)
    assert envelope.seal() == envelope.seal()


def test_seal_changes_when_envelope_changes():
    env = {"GITHUB_TOKEN": "token"}
    envelope = CredentialEnvelope.from_environment(env)
    altered = CredentialEnvelope.from_environment(
        {"GITHUB_TOKEN": "token", "ARAGORA_PROVIDER": "openai", "OPENAI_API_KEY": "key"}
    )
    assert envelope.seal() != altered.seal()
