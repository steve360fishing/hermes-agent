import json
import sys
from types import SimpleNamespace

import pytest

from tools import tournament_source_capture_tool as tool


def test_capture_schema_exposes_only_registered_source_id():
    assert set(tool.TOURNAMENT_SOURCE_CAPTURE_SCHEMA["parameters"]["properties"]) == {"source_id"}


def test_capture_uses_runtime_owned_factory_without_accepting_model_network_inputs(monkeypatch, tmp_path):
    roots = SimpleNamespace(journal_root=tmp_path, source_snapshot_root=tmp_path, receipt_root=tmp_path)
    monkeypatch.setattr(tool, "configured_runtime_roots", lambda: roots)
    runtime = object()
    monkeypatch.setattr(tool, "_runtime", lambda: runtime)
    captured = {}

    def capture_registered_source(**kwargs):
        captured.update(kwargs)
        return {"source_id": "event-1", "source_snapshot_path": "private.json"}

    class TrustedCaptureError(ValueError):
        pass

    module = SimpleNamespace(
        capture_registered_source=capture_registered_source,
        TrustedCaptureError=TrustedCaptureError,
    )
    monkeypatch.setitem(sys.modules, "audit_agent.tournament_trusted_capture", module)
    result = json.loads(tool.run_tournament_source_capture({"source_id": "event-1", "url": "https://evil.invalid"}))
    assert result["code"] == "captured"
    assert captured["source_id"] == "event-1"
    assert captured["runtime"] is runtime
    assert "url" not in captured


def test_capture_preserves_specific_safe_audit_rejection_code(monkeypatch, tmp_path):
    roots = SimpleNamespace(journal_root=tmp_path, source_snapshot_root=tmp_path, receipt_root=tmp_path)
    monkeypatch.setattr(tool, "configured_runtime_roots", lambda: roots)
    monkeypatch.setattr(tool, "_runtime", lambda: object())

    class TrustedCaptureError(ValueError):
        pass

    def rejected(**_kwargs):
        raise TrustedCaptureError("TCAP_SSRF_PRIVATE_ADDRESS: rejected")

    module = SimpleNamespace(
        capture_registered_source=rejected,
        TrustedCaptureError=TrustedCaptureError,
    )
    monkeypatch.setitem(sys.modules, "audit_agent.tournament_trusted_capture", module)
    result = json.loads(tool.run_tournament_source_capture({"source_id": "event-1"}))
    assert result["code"] == "TCAP_SSRF_PRIVATE_ADDRESS"


def test_runtime_resolver_uses_specific_safe_private_address_code(monkeypatch):
    monkeypatch.setattr(
        tool.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("169.254.169.254", 443))],
    )
    with pytest.raises(ValueError, match="TCAP_SSRF_PRIVATE_ADDRESS"):
        tool._resolve_public("results.example.test")


@pytest.mark.parametrize(
    ("detail", "expected", "forbidden"),
    (
        ("source host resolved to private address 10.9.8.7", "TCAP_SSRF_PRIVATE_ADDRESS", "10.9.8.7"),
        ("source host resolved to metadata address 169.254.169.254", "TCAP_SSRF_PRIVATE_ADDRESS", "169.254.169.254"),
        ("source host resolved to link-local address 169.254.22.9", "TCAP_SSRF_PRIVATE_ADDRESS", "169.254.22.9"),
        ("source host resolved to RFC1918 address 192.168.1.5", "TCAP_SSRF_PRIVATE_ADDRESS", "192.168.1.5"),
        ("source host resolver returned invalid address", "TCAP_DNS_INVALID", "invalid address"),
        ("response exceeds configured bound", "TCAP_SIZE_LIMIT", "configured bound"),
        ("trusted capture dedup snapshot must not be a symlink", "TCAP_SYMLINK_ESCAPE", "dedup snapshot"),
        ("trusted snapshot escapes approved root", "TCAP_PATH_ESCAPE", "approved root"),
        ("URL credentials are forbidden: https://user:secret@example.test", "TCAP_URL_CREDENTIALS_FORBIDDEN", "secret"),
        ("only HTTPS source routes are allowed: file:///etc/passwd", "TCAP_HTTPS_REQUIRED", "file:///etc/passwd"),
        ("redirect has no location", "TCAP_REDIRECT_INVALID", "no location"),
        ("response content type is not evidence-safe", "TCAP_CONTENT_TYPE", "evidence-safe"),
    ),
)
def test_model_facing_adapter_maps_local_rejection_boundaries_without_leaking_details(
    monkeypatch, tmp_path, detail, expected, forbidden
):
    roots = SimpleNamespace(journal_root=tmp_path, source_snapshot_root=tmp_path, receipt_root=tmp_path)
    monkeypatch.setattr(tool, "configured_runtime_roots", lambda: roots)
    monkeypatch.setattr(tool, "_runtime", lambda: object())

    class TrustedCaptureError(ValueError):
        pass

    def rejected(**_kwargs):
        raise ValueError(detail)

    monkeypatch.setitem(
        sys.modules,
        "audit_agent.tournament_trusted_capture",
        SimpleNamespace(
            capture_registered_source=rejected,
            TrustedCaptureError=TrustedCaptureError,
        ),
    )
    raw_result = tool.run_tournament_source_capture({"source_id": "event-1"})
    result = json.loads(raw_result)
    assert result["code"] == expected
    assert forbidden not in raw_result


def test_model_facing_adapter_rejects_unknown_tcap_prefix_without_echoing_it(monkeypatch, tmp_path):
    roots = SimpleNamespace(journal_root=tmp_path, source_snapshot_root=tmp_path, receipt_root=tmp_path)
    monkeypatch.setattr(tool, "configured_runtime_roots", lambda: roots)
    monkeypatch.setattr(tool, "_runtime", lambda: object())

    class TrustedCaptureError(ValueError):
        pass

    def rejected(**_kwargs):
        raise TrustedCaptureError("TCAP_UNTRUSTED_secret-token: https://private.invalid")

    monkeypatch.setitem(
        sys.modules,
        "audit_agent.tournament_trusted_capture",
        SimpleNamespace(capture_registered_source=rejected, TrustedCaptureError=TrustedCaptureError),
    )
    raw_result = tool.run_tournament_source_capture({"source_id": "event-1"})
    result = json.loads(raw_result)
    assert result["code"] == "TCAP_CAPTURE_INVALID"
    assert "secret-token" not in raw_result
    assert "private.invalid" not in raw_result
