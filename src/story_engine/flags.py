"""World flags, FlagSet container, and flag-overlap texture helpers.

This module is intentionally minimal — it defines `WorldFlag` and the helpers
that derive narrative texture from flag combinations. Full `FlagSet` logic and
overlap conditions will be fleshed out when the engine is wired together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class WorldFlag(Enum):
    """Campus-wide conditions active at a given simulation moment.

    Each flag shapes which engine rules fire, which characters are most
    exposed, and what narrative texture the scene carries. Flags are not
    mutually exclusive — overlapping flags produce compounded conditions
    (see `overlap_texture`).
    """

    SEMESTER_OPENING = auto()
    """New academic year. Hierarchy re-establishing. Meera arrives."""

    ELECTION_SEASON = auto()
    """Student union elections. Neel's strongest flag.
    Everyone must align or be read as aligned."""

    CULTURAL_FEST = auto()
    """Rules loosen. Savar thrives. Meera most exposed."""

    EXAM_SEASON = auto()
    """Institutional weapons sharpest. Dhruv calculates hardest."""

    POLITICAL_AGITATION = auto()
    """Campus erupts. Ranveer's institutional power most stressed."""

    INCIDENT_AFTERMATH = auto()
    """Always follows a significant event. Who people really are becomes visible."""

    SEMESTER_END = auto()
    """Third-years leaving. Natural resolution container.
    Students with nothing to lose act accordingly."""


def overlap_texture(flags: frozenset[WorldFlag]) -> str | None:
    """Return a narrative texture note for a known flag combination, or None.

    These notes are injected into `SceneBrief.world_state.flag_texture_note`
    to guide the LLM renderer without hardcoding story beats.

    Args:
        flags: The active flags at the current simulation moment.

    Returns:
        A single descriptive string capturing the compound condition, or
        ``None`` if no significant overlap is present.
    """
    F = WorldFlag

    if F.ELECTION_SEASON in flags and F.POLITICAL_AGITATION in flags:
        return (
            "Most volatile combination. Neel's control is most tested. "
            "Every public move is read as factional positioning."
        )
    if F.EXAM_SEASON in flags and F.INCIDENT_AFTERMATH in flags:
        return (
            "Kavya most exposed. Institutional weapons are active while "
            "the campus is still processing what just happened."
        )
    if F.CULTURAL_FEST in flags and F.INCIDENT_AFTERMATH in flags:
        return (
            "Campus performing while carrying aftermath. "
            "The gap between surface and reality is at its widest."
        )
    if F.SEMESTER_OPENING in flags and F.ELECTION_SEASON in flags:
        return (
            "Rare combination. Meera most exposed — she arrives into an "
            "already-mobilised campus. Neel most efficient."
        )
    if F.SEMESTER_END in flags and F.POLITICAL_AGITATION in flags:
        return (
            "Leaving students with nothing to lose. "
            "Agitation has no natural brake; costs feel abstract."
        )
    return None


@dataclass
class FlagSet:
    """Immutable set of currently active WorldFlags.

    All mutation methods return a new ``FlagSet``; the original is unchanged.
    This makes it safe to store in ``WorldState`` and diff across steps.

    Args:
        flags: The active flags at this simulation moment.
    """

    flags: frozenset[WorldFlag] = field(default_factory=frozenset)

    def is_active(self, flag: WorldFlag) -> bool:
        """Return True if *flag* is currently active.

        Args:
            flag: The flag to test.
        """
        return flag in self.flags

    def with_flag(self, flag: WorldFlag) -> FlagSet:
        """Return a new FlagSet with *flag* added.

        Args:
            flag: The flag to add.
        """
        return FlagSet(flags=self.flags | {flag})

    def without_flag(self, flag: WorldFlag) -> FlagSet:
        """Return a new FlagSet with *flag* removed (no-op if not present).

        Args:
            flag: The flag to remove.
        """
        return FlagSet(flags=self.flags - {flag})

    def texture_note(self) -> str | None:
        """Return the narrative texture note for the current flag combination.

        Returns:
            A compound-condition description string, or ``None`` if no
            significant overlap is present.
        """
        return overlap_texture(self.flags)
