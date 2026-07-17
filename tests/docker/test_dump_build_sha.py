"""Regression test: ``hermes dump`` reports a real git SHA inside the container.

Background: ``.dockerignore`` excludes ``.git``, so ``git rev-parse HEAD``
fails inside the published image and ``hermes dump`` used to report
``version: ... [(unknown)]``.  The Dockerfile now writes the build-time
``$HERMES_GIT_SHA`` build-arg to ``/opt/hermes/.hermes_build_sha`` and
``hermes_cli/build_info.py`` reads it as a fallback.

CI (``.github/workflows/docker.yml``) and the local ``built_image`` fixture
both pass a valid ``HERMES_GIT_SHA``. The Dockerfile rejects an image build
without one and binds the same value to the baked file and OCI revision label.
"""
from __future__ import annotations

import re
import subprocess


_VERSION_LINE = re.compile(r"^version:\s+(?P<rest>.+)$", re.MULTILINE)
_SHA_BRACKET = re.compile(r"\[(?P<sha>[^\]]+)\]\s*$")


def _run_dump(image: str) -> str:
    """Return the stdout of ``docker run <image> dump``.

    Relies on Docker's anonymous VOLUME for ``/opt/data`` (declared by the
    Dockerfile) so the container's hermes user (UID 10000) can bootstrap
    its config.  Anonymous volumes are auto-cleaned by ``--rm``, so unlike
    a host bind-mount we don't have to chown anything to UID 10000 (which
    would break cleanup on non-root hosts).
    """
    r = subprocess.run(
        ["docker", "run", "--rm", image, "dump"],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, (
        f"hermes dump exited {r.returncode}: "
        f"stderr={r.stderr[-1000:]!r}\nstdout={r.stdout[-1000:]!r}"
    )
    return r.stdout


def _read_baked_sha_from_image(image: str) -> str | None:
    """Return the ``/opt/hermes/.hermes_build_sha`` content, or None if absent."""
    r = subprocess.run(
        [
            "docker", "run", "--rm", "--entrypoint", "cat", image,
            "/opt/hermes/.hermes_build_sha",
        ],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def _read_revision_label(image: str) -> str:
    """Return the OCI source-revision label from an image."""
    r = subprocess.run(
        [
            "docker", "image", "inspect",
            "--format", "{{ index .Config.Labels \"org.opencontainers.image.revision\" }}",
            image,
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, f"docker image inspect failed: {r.stderr[-1000:]!r}"
    return r.stdout.strip()


def test_dump_reports_baked_sha_when_present(built_image: str) -> None:
    """When the image was built with ``HERMES_GIT_SHA``, dump must surface it.

    Together with the OCI-label assertion and smoke-test action (which
    exercises ``--help``), this closes the regression loop for the missing-SHA
    bug: any future change that breaks the provenance -> baked-file -> dump
    pipeline will fail CI here.
    """
    baked = _read_baked_sha_from_image(built_image)
    stdout = _run_dump(built_image)

    match = _VERSION_LINE.search(stdout)
    assert match, f"no `version:` line in dump output:\n{stdout[:2000]}"
    sha_match = _SHA_BRACKET.search(match.group("rest"))
    assert sha_match, (
        f"`version:` line missing [<sha>] bracket: {match.group('rest')!r}"
    )
    reported = sha_match.group("sha")

    assert baked is not None, "production image must contain a baked source revision"
    assert _read_revision_label(built_image) == baked, (
        "OCI revision label must exactly match the baked source revision"
    )

    # ``hermes dump``
    # truncates to 8 chars via ``git rev-parse --short=8`` semantics.
    assert reported != "(unknown)", (
        "baked SHA file present in image but dump still reported "
        f"'(unknown)' — the build-info fallback is broken.  "
        f"Baked file content: {baked!r}"
    )
    assert reported == baked[:8], (
        f"dump reported {reported!r} but baked file contained {baked!r} "
        f"(expected first 8 chars: {baked[:8]!r})"
    )
