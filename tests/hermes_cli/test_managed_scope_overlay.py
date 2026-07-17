"""apply_managed_overlay() — the shared helper used by every standalone loader."""
import textwrap

import pytest


@pytest.fixture
def managed(tmp_path, monkeypatch):
    md = tmp_path / "managed"
    md.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(md))
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    return md


def _write(md, body):
    (md / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")
    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()


def test_overlay_noop_without_scope(tmp_path, monkeypatch):
    from hermes_cli import managed_scope

    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "nope"))
    managed_scope.invalidate_managed_cache()
    src = {"display": {"skin": "user"}}
    assert managed_scope.apply_managed_overlay(src) == {"display": {"skin": "user"}}


def test_overlay_managed_wins(managed):
    from hermes_cli import managed_scope

    _write(managed, "display:\n  skin: charizard\n")
    out = managed_scope.apply_managed_overlay({"display": {"skin": "user"}})
    assert out["display"]["skin"] == "charizard"


def test_overlay_preserves_user_siblings(managed):
    from hermes_cli import managed_scope

    _write(managed, "display:\n  skin: charizard\n")
    out = managed_scope.apply_managed_overlay(
        {"display": {"skin": "user", "show_reasoning": True}}
    )
    assert out["display"]["skin"] == "charizard"
    assert out["display"]["show_reasoning"] is True


def test_overlay_normalizes_root_model_string(managed):
    """A managed bare `model: x/y` must promote to model.default, not clobber the dict."""
    from hermes_cli import managed_scope

    _write(managed, "model: org/locked\n")
    out = managed_scope.apply_managed_overlay({"model": {"default": "user/m", "fallback": "u/fb"}})
    assert out["model"]["default"] == "org/locked"  # managed wins
    assert out["model"]["fallback"] == "u/fb"  # user sibling preserved (dict shape intact)


def test_overlay_user_envref_cannot_shadow_managed_literal(managed, monkeypatch):
    from hermes_cli import managed_scope

    monkeypatch.setenv("EVIL", "user/override")
    _write(managed, "model:\n  default: managed/locked\n")
    out = managed_scope.apply_managed_overlay({"model": {"default": "${EVIL}"}})
    assert out["model"]["default"] == "managed/locked"


def test_strict_overlay_raises_when_managed_config_is_malformed(managed):
    from hermes_cli import managed_scope

    _write(managed, "model: [unterminated\n")
    assert hasattr(managed_scope, "apply_managed_overlay_strict")
    with pytest.raises(managed_scope.ManagedScopeError):
        managed_scope.apply_managed_overlay_strict({"model": {"default": "user/model"}})


def test_strict_overlay_raises_when_overlay_application_fails(managed, monkeypatch):
    from hermes_cli import config as config_module
    from hermes_cli import managed_scope

    _write(managed, "model:\n  default: managed/locked\n")

    def fail_merge(_base, _overlay):
        raise TypeError("merge failed")

    monkeypatch.setattr(config_module, "_deep_merge", fail_merge)
    with pytest.raises(managed_scope.ManagedScopeError, match="failed to apply"):
        managed_scope.apply_managed_overlay_strict({"model": {"default": "user/model"}})
