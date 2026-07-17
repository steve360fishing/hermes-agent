"""Regression checks for profile reconciliation permission diagnostics."""

from pathlib import Path


def test_reconciler_does_not_swallow_permission_denial() -> None:
    root = Path(__file__).parents[2]
    script = (root / "docker" / "cont-init.d" / "02-reconcile-profiles").read_text(
        encoding="utf-8"
    )

    assert "record_profile_reconciliation_failure" in script
    assert "chown hermes:hermes /run/service 2>/dev/null || true" not in script
