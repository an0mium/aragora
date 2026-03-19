from types import SimpleNamespace

from aragora.swarm.tranche_submit import classify_source_ref, enrich_github_refs


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
