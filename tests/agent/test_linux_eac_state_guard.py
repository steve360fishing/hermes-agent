import os
import tempfile
from pathlib import Path

import pytest

from agent.linux_eac_state_guard import LinuxEACStateGuard, LinuxEACStateGuardError

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Linux host adapter")


def _root() -> Path:
    path = Path(tempfile.mkdtemp())
    os.chmod(path, 0o700)
    if os.name != "nt" and os.geteuid() != 0:
        pytest.skip("host ownership invariant needs root")
    return path


def test_compare_and_set_is_monotonic_and_durable():
    guard = LinuxEACStateGuard(_root())
    first = {"checkpoint_mac": "a" * 64, "sequence": 1}
    second = {"checkpoint_mac": "b" * 64, "sequence": 2}
    assert guard.compare_and_set("contract-1", None, first)
    assert guard.read("contract-1") == first
    assert not guard.compare_and_set("contract-1", None, second)
    assert guard.compare_and_set("contract-1", "a" * 64, second)
    assert LinuxEACStateGuard(guard.root).read("contract-1") == second


def test_rejects_symlinked_checkpoint():
    root = _root()
    guard = LinuxEACStateGuard(root)
    target = root / "target"
    target.write_text("{}")
    os.chmod(target, 0o600)
    (root / "contract-1.checkpoint.json").symlink_to(target)
    with pytest.raises(LinuxEACStateGuardError):
        guard.read("contract-1")


def test_rejects_public_state_root():
    root = Path(tempfile.mkdtemp())
    os.chmod(root, 0o755)
    with pytest.raises(LinuxEACStateGuardError):
        LinuxEACStateGuard(root)
