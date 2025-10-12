from torchwindow import diagnostics


def test_diagnostics_structure():
    report = diagnostics.run_diagnostics()
    assert "status" in report
    assert "checks" in report
    assert isinstance(report["checks"], list)
