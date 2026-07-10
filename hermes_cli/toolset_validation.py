"""Validation for the ``platform_toolsets`` config section.

The core validator is pure and accepts its registry predicate as an argument.
The startup wrapper first performs enabled-plugin discovery, then delegates to
that pure validator.

Motivated by #38798: a config migration silently rewrote the valid toolset name
``hermes-cli`` to the non-existent ``hermes``. ``resolve_toolset('hermes')``
returns an empty list, so every tool silently disappeared with no error, warning,
or log entry — the agent degraded to text-only replies and the cause took
significant debugging to find. Surfacing invalid toolset names (and the
zero-tools end state) loudly turns that silent failure into an actionable one.
"""

from typing import Callable, Dict, List


def validate_platform_toolsets(
    platform_toolsets: object,
    is_valid_toolset: Callable[[str], bool],
) -> List[str]:
    """Return human-readable warnings for a ``platform_toolsets`` mapping.

    Two failure modes are reported:

    1. A toolset name that ``is_valid_toolset`` rejects — usually a corrupted or
       renamed entry. When ``hermes-<platform>`` would have been valid (the exact
       #38798 shape, where ``cli`` held ``hermes`` instead of ``hermes-cli``),
       the warning includes that as a suggestion.
    2. The mapping is non-empty but resolves to *zero* valid toolsets, so the
       agent would start with no tools at all.

    ``is_valid_toolset`` is injected (normally :func:`toolsets.validate_toolset`)
    so this function performs no imports or I/O and is testable in isolation.

    Args:
        platform_toolsets: The raw ``platform_toolsets`` value from config. Only
            ``dict`` values carry toolset entries; anything else yields no
            warnings (nothing to validate).
        is_valid_toolset: Predicate returning ``True`` for a known toolset name.

    Returns:
        A list of warning strings (empty when everything is valid).
    """
    warnings: List[str] = []
    if not isinstance(platform_toolsets, dict) or not platform_toolsets:
        return warnings

    valid_count = 0
    for platform, raw in platform_toolsets.items():
        names = raw if isinstance(raw, list) else [raw]
        for name in names:
            if not isinstance(name, str) or not name:
                continue
            if is_valid_toolset(name):
                valid_count += 1
                continue
            suggestion = f"hermes-{platform}"
            hint = (
                f" — did you mean '{suggestion}'?"
                if is_valid_toolset(suggestion)
                else ""
            )
            warnings.append(
                f"platform '{platform}' references unknown toolset "
                f"'{name}'{hint}"
            )

    if valid_count == 0:
        warnings.append(
            "platform_toolsets resolves to zero valid toolsets — the agent will "
            "have no tools. Run `hermes tools` to reconfigure."
        )
    return warnings


def validate_platform_toolsets_after_plugin_discovery(
    platform_toolsets: object,
    known_plugin_toolsets: object = None,
) -> List[str]:
    """Validate against built-in and enabled plugin toolsets.

    Plugin toolsets are registered dynamically, so validating before discovery
    misclassifies enabled plugins as unknown. Discovery is opt-in aware: a
    disabled plugin does not register its tools and therefore remains invalid.
    """
    from hermes_cli.plugins import discover_plugins, get_plugin_toolsets
    from toolsets import validate_toolset

    # Reconcile against current config rather than trusting an earlier
    # in-process discovery snapshot. ``known_plugin_toolsets`` preserves plugin
    # identity across enabled -> disabled transitions so stale registry entries
    # cannot make a disabled plugin appear valid.
    discover_plugins(force=True)
    active_plugin_toolsets = {entry[0] for entry in get_plugin_toolsets()}
    known_plugin_names: set[str] = set()
    if isinstance(known_plugin_toolsets, dict):
        for raw_names in known_plugin_toolsets.values():
            names = raw_names if isinstance(raw_names, list) else [raw_names]
            known_plugin_names.update(
                name for name in names if isinstance(name, str) and name
            )

    def is_currently_valid(name: str) -> bool:
        if name in known_plugin_names:
            return name in active_plugin_toolsets
        return validate_toolset(name)

    return validate_platform_toolsets(platform_toolsets, is_currently_valid)
