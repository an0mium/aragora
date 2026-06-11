from __future__ import annotations

import pytest

from scripts import dogfood_score


def test_load_timeout_payload_reads_valid_report_file(tmp_path) -> None:
    report = tmp_path / "timeout.json"
    report.write_text(
        '{"status": "timeout", "timeout_seconds": 30, "elapsed_seconds": 30.1}',
        encoding="utf-8",
    )

    payload = dogfood_score._load_timeout_payload(report, "")

    assert payload == {
        "status": "timeout",
        "timeout_seconds": 30,
        "elapsed_seconds": 30.1,
    }


def test_load_timeout_payload_rejects_malformed_report_file(tmp_path) -> None:
    report = tmp_path / "timeout.json"
    report.write_text('{"status": ', encoding="utf-8")

    with pytest.raises(dogfood_score.DogfoodScoreInputError, match="invalid timeout JSON"):
        dogfood_score._load_timeout_payload(report, "")


def test_load_timeout_payload_rejects_non_object_report_file(tmp_path) -> None:
    report = tmp_path / "timeout.json"
    report.write_text('["timeout"]', encoding="utf-8")

    with pytest.raises(dogfood_score.DogfoodScoreInputError, match="must be an object"):
        dogfood_score._load_timeout_payload(report, "")


def test_load_timeout_payload_rejects_malformed_stdout_sentinel() -> None:
    stdout = "setup\nARAGORA_TIMEOUT_JSON={bad\n"

    with pytest.raises(
        dogfood_score.DogfoodScoreInputError,
        match="invalid timeout JSON in ARAGORA_TIMEOUT_JSON= line",
    ):
        dogfood_score._load_timeout_payload(None, stdout)


def test_main_rejects_malformed_timeout_report_before_writing_outputs(tmp_path, capsys) -> None:
    baseline_stdout = tmp_path / "baseline.out"
    enhanced_stdout = tmp_path / "enhanced.out"
    timeout_report = tmp_path / "baseline-timeout.json"
    output_json = tmp_path / "score.json"

    baseline_stdout.write_text("FINAL ANSWER:\n", encoding="utf-8")
    enhanced_stdout.write_text("FINAL ANSWER:\n", encoding="utf-8")
    timeout_report.write_text("{bad", encoding="utf-8")

    rc = dogfood_score.main(
        [
            "--baseline-stdout",
            str(baseline_stdout),
            "--enhanced-stdout",
            str(enhanced_stdout),
            "--baseline-timeout-report",
            str(timeout_report),
            "--repo-root",
            str(tmp_path),
            "--output-json",
            str(output_json),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "invalid timeout JSON" in captured.err
    assert not output_json.exists()
