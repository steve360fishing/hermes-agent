"""Real s6/non-root continuity harness for rescue telemetry.

The regular Linux tests exercise the state machine deterministically. This
test uses the image built by docker.yml to prove the same boundary through
the actual s6 service, Unix peer credentials, and volatile /run filesystem.
"""
from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Iterator

import pytest

from tests.docker.conftest import docker_exec, docker_exec_sh, poll_container


def _docker(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture
def rescue_container(
    request: pytest.FixtureRequest,
    built_image: str,
) -> Iterator[str]:
    safe = request.node.name.replace("[", "_").replace("]", "_")
    name = f"hermes-rescue-{safe}"
    secrets_volume = f"{name}-secrets"
    safe_mode_volume = f"{name}-safe-mode"
    for volume in (secrets_volume, safe_mode_volume):
        _docker("volume", "rm", "-f", volume, timeout=10)
        _docker("volume", "create", volume, timeout=10).check_returncode()

    provision = (
        "set -eu; "
        "chown 10001:10000 /secrets; chmod 0750 /secrets; "
        f"printf '{'q' * 32}' > /secrets/hmac-current; "
        "printf 'current-v1' > /secrets/key-id-current; "
        f"printf '{'a' * 40}' > /secrets/source-sha; "
        f"printf 'sha256:{'b' * 64}' > /secrets/image-id; "
        "chown 10001:10000 /secrets/*; chmod 0400 /secrets/*"
    )
    _docker(
        "run",
        "--rm",
        "--entrypoint",
        "sh",
        "-v",
        f"{secrets_volume}:/secrets",
        built_image,
        "-c",
        provision,
    ).check_returncode()

    _docker("rm", "-f", name, timeout=10)
    _docker(
        "run",
        "-d",
        "--name",
        name,
        "-v",
        f"{secrets_volume}:/run/hermes-rescue-secrets:ro",
        "-v",
        f"{safe_mode_volume}:/run/hermes-rescue:ro",
        built_image,
        "sleep",
        "infinity",
    ).check_returncode()
    ready, output = poll_container(
        name,
        "test -S /run/hermes-rescue-reporter/events.sock && "
        "test -f /run/hermes-rescue-reporter/quiescence-v1.json",
        user="root",
        deadline_s=45,
    )
    assert ready, f"rescue reporter did not become ready: {output}"
    yield name
    _docker("rm", "-f", name, timeout=10)
    for volume in (secrets_volume, safe_mode_volume):
        _docker("volume", "rm", "-f", volume, timeout=10)


def _snapshot(container: str) -> dict[str, object]:
    # The signed rescue snapshot is deliberately 0600 and reporter-owned.
    # This test inspects supervisor evidence, so read it through the root-only
    # test boundary instead of weakening the production file permissions.
    result = docker_exec(
        container,
        "cat",
        "/run/hermes-rescue-reporter/quiescence-v1.json",
        user="root",
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _wait_snapshot(
    container: str,
    predicate,
    *,
    deadline_s: float = 20,
) -> dict[str, object]:
    deadline = time.monotonic() + deadline_s
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        try:
            last = _snapshot(container)
        except (AssertionError, json.JSONDecodeError):
            time.sleep(0.2)
            continue
        if predicate(last):
            return last
        time.sleep(0.2)
    raise AssertionError(f"snapshot predicate not reached; last={last!r}")


def _reporter_pid(container: str) -> int:
    result = docker_exec_sh(
        container,
        "/command/s6-svstat -o pid "
        "/run/service/rescue-quiescence-reporter",
        user="root",
    )
    assert result.returncode == 0, result.stderr
    return int(result.stdout.strip())


def test_s6_reporter_retains_background_across_death_restart_and_runtime_loss(
    rescue_container: str,
) -> None:
    container = rescue_container
    reporter_pid = _reporter_pid(container)
    uid = docker_exec_sh(
        container,
        f"awk '/^Uid:/ {{print $2}}' /proc/{reporter_pid}/status",
        user="root",
    )
    assert uid.stdout.strip() == "10001"

    worker_code = (
        "import os,time;"
        "from agent.rescue_plane_core import RescueTelemetryClient;"
        "RescueTelemetryClient().emit({"
        "'event':'background_start','event_id':'docker-background-start',"
        "'turn_id':'docker-turn','work_id':'terminal:docker-worker'});"
        "print(os.getpid(),flush=True);time.sleep(120)"
    )
    worker = subprocess.Popen(
        [
            "docker",
            "exec",
            "-u",
            "hermes",
            container,
            "python3",
            "-c",
            worker_code,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert worker.stdout is not None
    worker_pid = int(worker.stdout.readline().strip())
    active = _wait_snapshot(
        container,
        lambda value: value.get("active_tool_count") == 1,
    )

    docker_exec(
        container, "kill", "-9", str(worker_pid), user="root"
    ).check_returncode()
    worker.wait(timeout=10)
    degraded = _wait_snapshot(
        container,
        lambda value: (
            value.get("active_tool_count") == 1
            and value.get("telemetry_health") == "degraded"
        ),
    )
    assert degraded["producer_epoch"] == active["producer_epoch"]

    before_restart = _snapshot(container)
    docker_exec(
        container,
        "/command/s6-svc",
        "-r",
        "/run/service/rescue-quiescence-reporter",
        user="root",
    ).check_returncode()
    changed_pid, _ = poll_container(
        container,
        f"test \"$(/command/s6-svstat -o pid "
        "/run/service/rescue-quiescence-reporter)\" != "
        f"\"{reporter_pid}\"",
        user="root",
    )
    assert changed_pid
    restarted = _wait_snapshot(
        container,
        lambda value: (
            value.get("producer_epoch") == active["producer_epoch"]
            and int(value.get("sequence", 0)) > int(before_restart["sequence"])
            and value.get("active_tool_count") == 1
        ),
    )
    assert _reporter_pid(container) != reporter_pid

    docker_exec(
        container,
        "/command/s6-svc",
        "-d",
        "/run/service/rescue-quiescence-reporter",
        user="root",
    ).check_returncode()
    down, _ = poll_container(
        container,
        "/command/s6-svstat /run/service/rescue-quiescence-reporter "
        "| grep -q '^down '",
        user="root",
    )
    assert down
    reset = docker_exec_sh(
        container,
        "rm -rf /run/hermes-rescue-reporter && "
        "install -d -o hermes-rescue -g hermes -m 0750 "
        "/run/hermes-rescue-reporter",
        user="root",
    )
    assert reset.returncode == 0, reset.stderr
    docker_exec(
        container,
        "/command/s6-svc",
        "-u",
        "/run/service/rescue-quiescence-reporter",
        user="root",
    ).check_returncode()
    recovered = _wait_snapshot(
        container,
        lambda value: (
            value.get("producer_epoch") == active["producer_epoch"]
            and int(value.get("sequence", 0)) > int(restarted["sequence"])
            and value.get("active_tool_count") == 1
            and value.get("telemetry_health") == "degraded"
        ),
    )
    assert recovered["key_id"] == "current-v1"
