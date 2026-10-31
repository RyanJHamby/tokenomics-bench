"""Spec bench-claim-guard: README/docs never quote a measured number without a results/ capture."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
XFAIL = pytest.mark.xfail(
    strict=True, reason="spec:bench-claim-guard (remove this marker when implemented)"
)


def _c():
    return importlib.import_module("tokbench.claimcheck")


def _capture(root: Path, name: str = "20261101-b1", files=("cmd.txt", "env.txt", "output.log")):
    d = root / "results" / name
    d.mkdir(parents=True)
    for f in files:
        (d / f).write_text("x")


@XFAIL
@pytest.mark.parametrize(
    "line",
    [
        "Goodput reached 1,234 tok/s on the L40S.",
        "We saw 45 tokens/s per stream.",
        "Capacity was 12.5 req/s at the SLO.",
        "Energy was 3.1 J/token at saturation.",
        "Cost came to $0.42/Mtok.",
        "Cost came to 0.42 $/Mtok.",
        "Throughput: 900 tok/s",
    ],
)
def test_measured_style_quantities_are_flagged(line):
    found = _c().scan_text("intro\n" + line + "\n")
    assert len(found) == 1 and found[0].line == 2


@XFAIL
@pytest.mark.parametrize(
    "line",
    [
        "Operators buy goodput: tokens per second from requests that meet the SLO.",
        "Primary SLO: p99 TTFT <= 1.0 s and p99 TPOT <= 50 ms.",
        "A clock lock at ~70% of max SM clock cuts energy per token (superior, 10%).",
        "| P4 | FP8 weights raise SLO capacity | capacity vs FP16 | superior, 20% |",
        "J/token and tokens/s describe identical work.",
        "Planned total is about 18 GPU-hours.",
    ],
)
def test_design_prose_and_pre_registered_margins_are_not_flagged(line):
    assert _c().scan_text(line + "\n") == []


@XFAIL
def test_a_claim_is_backed_only_by_a_complete_capture_linked_on_the_same_line(tmp_path):
    _capture(tmp_path)
    _capture(tmp_path, "partial", files=("cmd.txt", "env.txt"))
    claim = "Goodput was 1,234 tok/s"
    ok = f"{claim} ([run](results/20261101-b1/)).\n"
    assert _c().scan_text(ok, root=tmp_path) == []
    assert _c().scan_text(ok.replace("20261101-b1", "missing"), root=tmp_path) != []
    assert _c().scan_text(ok.replace("20261101-b1", "partial"), root=tmp_path) != []
    # the link on a different line does not back the number
    split = f"{claim}.\nSee [run](results/20261101-b1/).\n"
    assert len(_c().scan_text(split, root=tmp_path)) == 1
    # without a root nothing can be backed
    assert _c().scan_text(ok) != []


@XFAIL
def test_check_repo_scans_readme_and_docs_but_not_the_dated_log_predictions_or_prereg(tmp_path):
    bad = "Goodput was 1,234 tok/s.\n"
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# T\n" + bad)
    (tmp_path / "docs" / "NOTES.md").write_text(bad)
    (tmp_path / "docs" / "journal.md").write_text(bad)  # dated log, quotes labelled predictions
    (tmp_path / "docs" / "PREDICTIONS.md").write_text(bad)  # model output, labelled predicted
    (tmp_path / "docs" / "PREREG-v2.md").write_text(bad)
    found = _c().check_repo(tmp_path)
    assert {(f.path, f.line) for f in found} == {
        ("README.md", 2),
        ("docs/NOTES.md", 1),
    }


@XFAIL
def test_missing_readme_is_an_error_not_a_clean_pass(tmp_path):
    with pytest.raises(FileNotFoundError):
        _c().check_repo(tmp_path)


@XFAIL
def test_this_repository_is_clean_today():
    assert _c().check_repo(REPO) == []
    assert _c().main(["--root", str(REPO)]) == 0


@XFAIL
def test_cli_exit_code_and_report_line(tmp_path, capsys):
    (tmp_path / "README.md").write_text("fine\nGoodput was 9 tok/s\n")
    assert _c().main(["--root", str(tmp_path)]) == 1
    assert "README.md:2" in capsys.readouterr().out
    (tmp_path / "README.md").write_text("fine\n")
    assert _c().main(["--root", str(tmp_path)]) == 0
