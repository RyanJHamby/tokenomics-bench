import pytest
import yaml

from tokbench import budget


@pytest.fixture
def d(tmp_path):
    (tmp_path / "budget.yaml").write_text(yaml.safe_dump({"cap_usd": 60.0, "safety_margin": 0.2}))
    return tmp_path


def _add(d, kind, usd, session=None):
    budget.append({"kind": kind, "usd": usd, **({"session": session} if session else {})}, d)


def test_empty_ledger_has_full_budget(d):
    assert budget.remaining(d) == 60.0


def test_invoice_supersedes_runner_lower_bound_for_same_session(d):
    _add(d, "runner_wall", 5.0, "pod1")  # runner saw 5, but the pod really billed 8
    _add(d, "invoice", 8.0, "pod1")
    _add(d, "runner_wall", 3.0, "pod2")  # no invoice yet: counted as a lower bound
    assert budget.spent(budget.read_ledger(d)) == pytest.approx(11.0)
    assert budget.remaining(d) == pytest.approx(49.0)


def test_check_pads_estimate_and_refuses_overrun(d):
    _add(d, "invoice", 40.0, "pod1")
    assert budget.check(15.0, d)["remaining"] == 20.0  # 15*1.2 = 18 <= 20
    with pytest.raises(budget.BudgetExceeded, match="exceeds remaining"):
        budget.check(17.0, d)  # 17*1.2 = 20.4 > 20


def test_exact_fit_is_allowed_and_negative_spend_rejected(d):
    _add(d, "invoice", 24.0, "p")  # remaining 36; 30*1.2 = 36 exactly
    budget.check(30.0, d)
    with pytest.raises(ValueError):
        budget.append({"kind": "invoice", "usd": -1}, d)


def test_cli_status_add_check(d, monkeypatch, capsys):
    monkeypatch.setattr(budget, "DIR", d)
    assert (
        budget.main(
            [
                "add",
                "--provider",
                "p",
                "--gpu",
                "L40S",
                "--usd-per-hr",
                "2",
                "--hours",
                "10",
                "--session",
                "s1",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert budget.main(["status"]) == 0
    assert "spent $20.00  remaining $40.00" in capsys.readouterr().out
    assert budget.main(["check", "--estimate", "10"]) == 0
    assert budget.main(["check", "--estimate", "40"]) == 1  # 48 > 40 remaining
