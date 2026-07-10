from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_all_docker_runtime_files_use_lf_line_endings() -> None:
    docker_root = ROOT / "docker"
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in docker_root.rglob("*")
        if path.is_file() and b"\r\n" in path.read_bytes()
    ]

    assert offenders == []


def test_git_attributes_enforce_lf_for_extensionless_docker_runtime_files() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert "docker/** text eol=lf" in attributes
