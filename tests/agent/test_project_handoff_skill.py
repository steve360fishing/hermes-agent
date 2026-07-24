from pathlib import Path


SKILL = Path(__file__).resolve().parents[2] / "skills" / "project-handoff" / "SKILL.md"


def test_project_handoff_skill_has_required_context_and_safety_rules():
    text = SKILL.read_text(encoding="utf-8")
    for required in (
        "Goal and finish condition",
        "Current truth",
        "Workspace/source map",
        "Open items in order",
        "Approval claims",
        "Receiver first action",
        "Redaction record",
        "safe boundary",
        "PREPARED_NOT_DELIVERED",
        "not authority",
    ):
        assert required in text


def test_project_handoff_skill_excludes_sensitive_packet_content():
    text = SKILL.read_text(encoding="utf-8")
    for forbidden in ("API_KEY=", "Bearer ", "bot token"):
        assert forbidden not in text


def test_project_handoff_skill_is_discoverable_from_the_hermes_skills_root(monkeypatch):
    from tools import skills_tool

    monkeypatch.setattr(skills_tool, "SKILLS_DIR", SKILL.parent.parent)
    names = [entry["name"] for entry in skills_tool._find_all_skills()]
    assert "project-handoff" in names
