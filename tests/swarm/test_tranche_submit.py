from types import SimpleNamespace

import pytest

from aragora.swarm.tranche_submit import (
    classify_source_ref,
    determine_decomposition_action,
    enrich_github_refs,
    normalize_lanes,
    submit_intake_bundle,
)


def test_classify_github_issue_ref() -> None:
    result = classify_source_ref("https://github.com/synaptent/aragora/issues/1064")
    assert result["kind"] == "github"
    assert result["github_kind"] == "issue"
    assert result["number"] == 1064


def test_classify_github_pr_ref() -> None:
    result = classify_source_ref("https://github.com/synaptent/aragora/pull/1065")
    assert result["kind"] == "github"
    assert result["github_kind"] == "pull_request"


def test_classify_local_file_ref() -> None:
    result = classify_source_ref("/path/to/local/file.md")
    assert result["kind"] == "context"
    assert result["gated"] is False


def test_classify_doc_url_ref() -> None:
    result = classify_source_ref("https://docs.example.com/guide")
    assert result["kind"] == "context"
    assert result["gated"] is False


def test_enrich_github_refs_resolves_issue_and_preserves_context() -> None:
    client = SimpleNamespace(
        get_issue=lambda repo, number: {
            "number": number,
            "state": "OPEN",
            "title": "Fix PMF path",
            "url": f"https://github.com/{repo}/issues/{number}",
            "labels": [{"name": "pmf"}],
            "closedAt": None,
        }
    )
    refs = [
        classify_source_ref("https://github.com/synaptent/aragora/issues/1064"),
        classify_source_ref("https://docs.example.com/guide"),
    ]

    enriched = enrich_github_refs(refs, client)

    assert enriched[0]["observed_state"] == "open"
    assert enriched[0]["status"] == "actionable"
    assert enriched[0]["stale"] is False
    assert enriched[0]["title"] == "Fix PMF path"
    assert enriched[1]["kind"] == "context"
    assert "observed_state" not in enriched[1]


def test_enrich_github_refs_marks_merged_pr_stale() -> None:
    client = SimpleNamespace(
        get_pr=lambda repo, number: {
            "number": number,
            "state": "CLOSED",
            "mergedAt": "2026-03-19T00:00:00Z",
            "title": "Land PMF path",
            "url": f"https://github.com/{repo}/pull/{number}",
            "labels": [],
        }
    )
    refs = [classify_source_ref("https://github.com/synaptent/aragora/pull/1065")]

    enriched = enrich_github_refs(refs, client)

    assert enriched[0]["observed_state"] == "merged"
    assert enriched[0]["status"] == "stale"
    assert enriched[0]["stale"] is True


def test_no_lanes_triggers_full_decomposition() -> None:
    bundle = {"objective": "Fix the user journey", "candidate_lanes": []}
    result = determine_decomposition_action(bundle)
    assert result == "full_decomposition"


def test_lane_missing_prompt_triggers_augment() -> None:
    bundle = {
        "objective": "Fix things",
        "candidate_lanes": [{"lane_id": "a", "title": "Fix auth"}],
    }
    result = determine_decomposition_action(bundle)
    assert result == "augment"


def test_lane_missing_scope_triggers_inference_only() -> None:
    bundle = {
        "objective": "Fix things",
        "candidate_lanes": [
            {
                "lane_id": "a",
                "title": "Fix auth",
                "prompt": "Fix the auth flow",
                "owner_role": "engineer",
            },
        ],
    }
    result = determine_decomposition_action(bundle)
    assert result == "inference_only"


def test_complete_lanes_skip_decomposition() -> None:
    bundle = {
        "objective": "Fix things",
        "candidate_lanes": [
            {
                "lane_id": "a",
                "title": "Fix auth",
                "prompt": "Fix auth",
                "owner_role": "engineer",
                "allowed_write_scope": ["aragora/auth/**"],
                "verification_commands": ["pytest tests/auth/"],
            }
        ],
    }
    result = determine_decomposition_action(bundle)
    assert result == "none"


def test_normalize_lanes_generates_lane_id_and_infers_scope_from_hints() -> None:
    bundle = {
        "objective": "Fix auth",
        "candidate_lanes": [
            {
                "title": "Fix auth flow",
                "prompt": "Fix the auth flow.",
                "owner_role": "engineer",
                "file_scope_hints": ["aragora/auth"],
            }
        ],
    }

    lanes = normalize_lanes(bundle, planner=None)

    assert lanes[0]["lane_id"] == "fix-auth-flow"
    assert lanes[0]["allowed_write_scope"] == ["aragora/auth/**"]
    assert lanes[0]["verification_commands"] == []
    assert lanes[0]["dependencies"] == []


def test_normalize_lanes_infers_scope_from_prompt_text() -> None:
    bundle = {
        "objective": "Fix UI",
        "candidate_lanes": [
            {
                "title": "Refine dashboard",
                "prompt": "Refine the aragora/live dashboard flow and tests/live coverage.",
                "owner_role": "ui_engineer",
                "verification_commands": ["cd aragora/live && npm run lint"],
            }
        ],
    }

    lanes = normalize_lanes(bundle, planner=None)

    assert lanes[0]["allowed_write_scope"] == ["aragora/live/**", "tests/live/**"]


def test_normalize_lanes_requires_planner_for_augment() -> None:
    bundle = {
        "objective": "Fix things",
        "candidate_lanes": [{"title": "Fix auth"}],
    }

    with pytest.raises(ValueError, match="planner"):
        normalize_lanes(bundle, planner=None)


def test_submit_returns_dual_status(tmp_path) -> None:
    bundle = {
        "objective": "Bump supabase in aragora/live",
        "candidate_lanes": [
            {
                "lane_id": "bump",
                "title": "Bump supabase",
                "prompt": "Bump @supabase/supabase-js to latest",
                "owner_role": "engineer",
                "allowed_write_scope": ["aragora/live/**"],
                "verification_commands": ["cd aragora/live && npm run lint"],
            }
        ],
        "autonomy_mode": "adaptive",
    }

    result = submit_intake_bundle(
        bundle,
        repo_root=tmp_path,
        skip_github_resolution=True,
    )

    assert "inspection_status" in result
    assert "submission_status" in result
    assert "recommended_action" in result
    assert result["inspection_status"] in ("ok", "blocked")
    assert result["submission_status"] in ("ready_to_prepare", "awaiting_confirmation", "blocked")
    assert "manifest_id" in result


def test_submit_persists_three_layers(tmp_path) -> None:
    bundle = {"objective": "Test persistence", "candidate_lanes": []}

    result = submit_intake_bundle(
        bundle,
        repo_root=tmp_path,
        skip_github_resolution=True,
    )

    manifest_id = result["manifest_id"]
    tranche_dir = tmp_path / ".aragora" / "tranches" / manifest_id
    assert (tranche_dir / "intake_bundle.yaml").exists()
    assert (tranche_dir / "normalized_bundle.yaml").exists()
    assert (tranche_dir / "tranche.yaml").exists()
    assert (tranche_dir / "run_state.yaml").exists()


def test_submit_recommends_design_review_for_adaptive_writable_tranche(tmp_path) -> None:
    result = submit_intake_bundle(
        {
            "objective": "Ship feature",
            "candidate_lanes": [
                {
                    "lane_id": "lane_a",
                    "title": "Build it",
                    "owner_role": "engineer",
                    "prompt": "Implement the feature",
                    "allowed_write_scope": ["aragora/server/**"],
                }
            ],
            "autonomy_mode": "adaptive",
        },
        repo_root=tmp_path,
        skip_github_resolution=True,
    )

    assert result["recommended_action"] == "design-review"
