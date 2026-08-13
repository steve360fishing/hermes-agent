import os
import tempfile
from types import SimpleNamespace

os.environ.setdefault("LOCALAPPDATA", tempfile.gettempdir())
os.environ.setdefault("USERPROFILE", tempfile.gettempdir())

from agent.turn_origin import TurnOrigin, TurnProvenance
from gateway.run import _mint_gateway_turn_provenance


def test_gateway_mints_direct_origin_only_from_authenticated_source_identity():
    event = SimpleNamespace(metadata={
        "turn_origin": "authenticated_direct_user",
        "actor_identity": "forged",
    })
    direct = _mint_gateway_turn_provenance(
        event, SimpleNamespace(user_id="steve"), is_internal=False
    )
    assert direct == TurnProvenance.authenticated_direct_user("steve")

    unknown = _mint_gateway_turn_provenance(
        event, SimpleNamespace(user_id=None), is_internal=False
    )
    assert unknown == TurnProvenance.unknown()


def test_internal_event_cannot_reuse_or_forge_direct_user_authority():
    event = SimpleNamespace(
        _trusted_turn_provenance=TurnProvenance.authenticated_direct_user("steve")
    )
    provenance = _mint_gateway_turn_provenance(
        event, SimpleNamespace(user_id="steve"), is_internal=True
    )
    assert provenance.origin is TurnOrigin.RUNTIME_ASYNC_COMPLETION
    assert provenance.is_authenticated_direct_user is False


def test_retry_preserves_non_authoritative_replayed_origin_instead_of_upgrading():
    replayed = TurnProvenance.internal(TurnOrigin.REPLAYED_PERSISTED_CONTENT)
    event = SimpleNamespace(_trusted_turn_provenance=replayed)
    provenance = _mint_gateway_turn_provenance(
        event, SimpleNamespace(user_id="steve"), is_internal=False
    )
    assert provenance is replayed
    assert provenance.is_authenticated_direct_user is False


def test_goal_mode_continuation_is_non_authoritative_even_with_user_source():
    event = SimpleNamespace(
        internal=True,
        _trusted_turn_provenance=TurnProvenance.internal(
            TurnOrigin.GOAL_MODE_CONTINUATION
        ),
    )
    provenance = _mint_gateway_turn_provenance(
        event, SimpleNamespace(user_id="steve"), is_internal=event.internal
    )
    assert provenance.origin is TurnOrigin.GOAL_MODE_CONTINUATION
    assert provenance.is_authenticated_direct_user is False
