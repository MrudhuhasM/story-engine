"""Trigger taxonomy, Trigger dataclass, and factory functions.

A ``Trigger`` is a discrete event that breaks equilibrium and starts an engine
chain. It is immutable — the engine reads it and produces state mutations; the
trigger itself is never altered.

Taxonomy
~~~~~~~~
TYPE 01 DIRECT_CHALLENGE      — one side moves directly, no deniability
TYPE 02 INSTITUTIONAL_MOVE    — campus structure used as weapon, full deniability
TYPE 03 POLITICAL_MOVE        — student union machinery, Neel's territory
TYPE 04 AMBIENT_TRIGGER       — nobody planned this; world is now different

Each type has named variants (see ``TriggerVariant``). Factory functions create
fully-populated ``Trigger`` instances for the most common patterns.

Flag sensitivity is handled by the engine's ``fire_trigger()`` method, not here.

Import contract
~~~~~~~~~~~~~~~
triggers.py → locations.py (for ``LocationName``)
triggers.py has no imports from characters.py or world_state.py.
engine.py imports from triggers.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from story_engine.locations import LocationName


# ---------------------------------------------------------------------------
# TriggerType
# ---------------------------------------------------------------------------


class TriggerType(Enum):
    """The four top-level trigger categories in the world model.

    Each type has a distinct speed of consequence, deniability profile, and
    set of character response rules it activates.
    """

    DIRECT_CHALLENGE = auto()
    """One side moves directly against the other — no intermediary, no plausible
    deniability. Fastest escalation path. Immediately engages both pride cores."""

    INSTITUTIONAL_MOVE = auto()
    """Campus structure used as a weapon. No direct confrontation; maximum
    deniability. Engages Vikram's blind spot — there is no one to confront."""

    POLITICAL_MOVE = auto()
    """Student union machinery activated. Slowest consequence delivery.
    Effects are not immediately visible; by the time they land, untraceable.
    Neel's territory."""

    AMBIENT_TRIGGER = auto()
    """Nobody planned this. Something happened. Introduces variables neither
    gang controls; breaks the bilateral character of the conflict temporarily."""


# ---------------------------------------------------------------------------
# TriggerVariant
# ---------------------------------------------------------------------------


class TriggerVariant(Enum):
    """Named sub-variants within each TriggerType.

    Naming convention: ``{TYPE}_{DESCRIPTION}``.
    The engine uses the variant to select specific chain-rule branches.
    """

    # TYPE 01 — DIRECT CHALLENGE
    DIRECT_CHALLENGE_PUBLIC_HUMILIATION = auto()
    """01-A: Public humiliation attempt. Initiator: Ranveer. Target: Vikram.
    Location: high-visibility. Fastest pride-ratchet advancement."""

    DIRECT_CHALLENGE_PHYSICAL_CONFRONTATION = auto()
    """01-B: Physical confrontation initiated. Initiator: Karan (typically).
    Location: variable visibility. Karan activated immediately."""

    DIRECT_CHALLENGE_REFUSAL = auto()
    """01-C: Vikram refuses a direct order or expectation from Ranveer's gang.
    Initiator: Vikram. Target: Ranveer. High visibility. Classic non-submission."""

    DIRECT_CHALLENGE_PUBLIC_CALLOUT = auto()
    """01-D: Vikram publicly calls out something Ranveer's gang did.
    Initiator: Vikram. Target: Ranveer. Maximum visibility. Highest escalation."""

    # TYPE 02 — INSTITUTIONAL MOVE
    INSTITUTIONAL_ACADEMIC_THREAT = auto()
    """02-A: Attendance manipulation, internal assessment interference, or
    exam irregularity targeting Vikram or his gang. Sharpest during EXAM_SEASON."""

    INSTITUTIONAL_ADMINISTRATIVE_ACTION = auto()
    """02-B: Disciplinary summons, manufactured misconduct record, or hostel
    room reassignment. Conflict moves from social to structural."""

    INSTITUTIONAL_NOTICE_BOARD = auto()
    """02-C: Official-looking declaration or selective information posted that
    damages without technically lying. Neel's most surgical weapon."""

    INSTITUTIONAL_OPPORTUNITY_DENIAL = auto()
    """02-D: Blocking access to events, selections, or positions Vikram or his
    gang wanted. Invisible until the denial is discovered."""

    # TYPE 03 — POLITICAL MOVE
    POLITICAL_ELECTION_POSITIONING = auto()
    """03-A: Election candidate positioned against someone in Vikram's orbit.
    Isolates Vikram politically without direct confrontation."""

    POLITICAL_FACTION_APPROACH = auto()
    """03-B: Opposing faction approached and offered something.
    Vikram's name used as currency without his consent."""

    POLITICAL_AGITATION_SHAPING = auto()
    """03-C: Student agitation shaped or contained to serve Ranveer's interests.
    Genuine anger redirected. Dangerous — can produce the opposite of control."""

    POLITICAL_DHRUV_CONTACT = auto()
    """03-D: Dhruv contacted directly by Neel. The most reachable member of
    Vikram's gang. Neel and Dhruv speak the same language."""

    # TYPE 04 — AMBIENT TRIGGER
    AMBIENT_MEERA_INTERSECTION = auto()
    """04-A: Meera intersects with the conflict through her own actions —
    not through Vikram's. Vikram's response is undefined even to himself."""

    AMBIENT_KAVYA_EXPOSED = auto()
    """04-B: Kavya's position becomes visible in a way she cannot control —
    a classroom moment, an administrative decision she is seen to make or not."""

    AMBIENT_OUTSIDE_ACTOR = auto()
    """04-C: Someone outside both gangs does something that forces both sides
    to respond — a junior, a professor, an external political event."""

    AMBIENT_INFORMATION_SURFACE = auto()
    """04-D: Information surfaces that one side had and the other didn't know
    about. Surya's trigger — he knows something, and its surfacing changes everything."""

    AMBIENT_GANG_MEMBER_ACTS_ALONE = auto()
    """04-E: A member of Vikram's gang acts without Vikram — Rajan escalating
    independently, Dhruv making a contact, Savar saying something he cannot unsay."""


# ---------------------------------------------------------------------------
# Trigger dataclass
# ---------------------------------------------------------------------------

#: The sentinel value for ``Trigger.target`` when a trigger has no single target.
NO_TARGET: str = "__diffuse__"


@dataclass(frozen=True)
class Trigger:
    """An immutable event that breaks equilibrium and starts an engine chain.

    The engine reads ``Trigger`` fields to select chain rules, apply visibility
    multipliers, and log incidents. Triggers are never mutated after creation.

    Args:
        trigger_type: Top-level category from ``TriggerType``.
        variant: Specific sub-variant from ``TriggerVariant``.
        location: Where this event occurs (``LocationName``).
        initiator: Lowercase name of the character who initiates the trigger
            (e.g. ``"ranveer"``, ``"neel"``, ``"vikram"``).
        target: Lowercase name of the character the trigger lands on, or
            ``NO_TARGET`` (``"__diffuse__"``) for ambient / diffuse triggers.
        is_public: Whether the general student body witnesses this event.
            Used together with ``location.consequence_multiplier()`` in
            ``apply_visibility_multiplier``.
        description: Human-readable description for logging and brief generation.
    """

    trigger_type: TriggerType
    variant: TriggerVariant
    location: LocationName
    initiator: str
    target: str
    is_public: bool
    description: str

    def is_direct(self) -> bool:
        """Return True if this trigger has no intermediary (TYPE 01 only)."""
        return self.trigger_type is TriggerType.DIRECT_CHALLENGE

    def has_target(self) -> bool:
        """Return True if this trigger has a specific named target."""
        return self.target != NO_TARGET

    def involves_character(self, name: str) -> bool:
        """Return True if *name* appears as initiator or target.

        Args:
            name: Lowercase character name to check.
        """
        return self.initiator == name or self.target == name


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------
#
# Each factory pre-fills trigger_type, variant, and sensible is_public defaults.
# Callers supply description and any variant-specific overrides.
# ---------------------------------------------------------------------------


def make_public_humiliation(
    location: LocationName,
    initiator: str,
    target: str,
    description: str,
) -> Trigger:
    """Create a TYPE 01-A: public humiliation attempt trigger.

    Used when Ranveer's gang attempts to humiliate Vikram (or another) in front
    of an audience. Forces both pride cores to engage.

    Args:
        location: Where the humiliation attempt occurs. Should be a
            high-visibility location for full pride-ratchet effect.
        initiator: Character who initiates (typically ``"ranveer"``).
        target: Character targeted (typically ``"vikram"``).
        description: What specifically happened.
    """
    return Trigger(
        trigger_type=TriggerType.DIRECT_CHALLENGE,
        variant=TriggerVariant.DIRECT_CHALLENGE_PUBLIC_HUMILIATION,
        location=location,
        initiator=initiator,
        target=target,
        is_public=True,
        description=description,
    )


def make_physical_confrontation(
    location: LocationName,
    initiator: str,
    target: str,
    description: str,
    *,
    is_public: bool = False,
) -> Trigger:
    """Create a TYPE 01-B: physical confrontation trigger.

    Karan is the typical initiator. Visibility is variable — confrontations on
    dead paths carry no multiplier; confrontations in the main canteen double
    every consequence.

    Args:
        location: Where the confrontation occurs.
        initiator: Character who starts it (typically ``"karan"``).
        target: Character targeted.
        description: What specifically happened.
        is_public: Whether bystanders are present. Defaults to ``False`` —
            most physical confrontations are engineered for deniability.
    """
    return Trigger(
        trigger_type=TriggerType.DIRECT_CHALLENGE,
        variant=TriggerVariant.DIRECT_CHALLENGE_PHYSICAL_CONFRONTATION,
        location=location,
        initiator=initiator,
        target=target,
        is_public=is_public,
        description=description,
    )


def make_vikram_refusal(
    location: LocationName,
    description: str,
    *,
    target: str = "ranveer",
) -> Trigger:
    """Create a TYPE 01-C: Vikram refuses a direct order or expectation.

    Classic non-submission trigger. Advances Ranveer's phase by one step via
    the pride ratchet. Initiator is always ``"vikram"``.

    Args:
        location: Where the refusal occurs. High-visibility locations amplify
            the consequence — refusing in the main canteen is not the same
            as refusing on a dead path.
        description: What expectation was refused and how.
        target: Who the refusal is directed at. Defaults to ``"ranveer"``.
    """
    return Trigger(
        trigger_type=TriggerType.DIRECT_CHALLENGE,
        variant=TriggerVariant.DIRECT_CHALLENGE_REFUSAL,
        location=location,
        initiator="vikram",
        target=target,
        is_public=True,
        description=description,
    )


def make_public_callout(
    location: LocationName,
    description: str,
    *,
    target: str = "ranveer",
) -> Trigger:
    """Create a TYPE 01-D: Vikram publicly calls out something Ranveer's gang did.

    Highest-escalation variant of DIRECT_CHALLENGE. Initiator is always
    ``"vikram"``. The callout must happen in a high-visibility location to carry
    its full weight.

    Args:
        location: Where the callout happens. MAIN_GROUND or MAIN_CANTEEN
            are the canonical locations for maximum effect.
        description: What Vikram called out, and how.
        target: Who is called out. Defaults to ``"ranveer"``.
    """
    return Trigger(
        trigger_type=TriggerType.DIRECT_CHALLENGE,
        variant=TriggerVariant.DIRECT_CHALLENGE_PUBLIC_CALLOUT,
        location=location,
        initiator="vikram",
        target=target,
        is_public=True,
        description=description,
    )


def make_academic_threat(
    description: str,
    *,
    initiator: str = "neel",
    target: str = "vikram",
    location: LocationName = LocationName.FACULTY_BUILDINGS,
) -> Trigger:
    """Create a TYPE 02-A: academic threat trigger.

    Attendance manipulation, internal assessment interference, or exam
    irregularity. Sharpest during EXAM_SEASON. No direct confrontation —
    Vikram cannot respond with presence.

    Args:
        description: The specific academic weapon used.
        initiator: Who orchestrated this (typically ``"neel"``).
        target: Who is targeted (typically ``"vikram"``).
        location: Where this is administered. Defaults to FACULTY_BUILDINGS.
    """
    return Trigger(
        trigger_type=TriggerType.INSTITUTIONAL_MOVE,
        variant=TriggerVariant.INSTITUTIONAL_ACADEMIC_THREAT,
        location=location,
        initiator=initiator,
        target=target,
        is_public=False,
        description=description,
    )


def make_administrative_action(
    description: str,
    *,
    initiator: str = "neel",
    target: str = "vikram",
) -> Trigger:
    """Create a TYPE 02-B: administrative action trigger.

    Disciplinary summons, manufactured misconduct record, or hostel room
    reassignment. Being summoned to the administration building means the
    conflict has moved from social to structural.

    Args:
        description: The specific administrative action taken.
        initiator: Who drove this (typically ``"neel"``).
        target: Who receives the action (typically ``"vikram"``).
    """
    return Trigger(
        trigger_type=TriggerType.INSTITUTIONAL_MOVE,
        variant=TriggerVariant.INSTITUTIONAL_ADMINISTRATIVE_ACTION,
        location=LocationName.ADMINISTRATION_BUILDING,
        initiator=initiator,
        target=target,
        is_public=False,
        description=description,
    )


def make_notice_board_move(
    description: str,
    *,
    initiator: str = "neel",
    target: str = "vikram",
) -> Trigger:
    """Create a TYPE 02-C: notice board move trigger.

    Official-looking declaration or selective information that damages without
    lying. Posted publicly but the authorship is deniable. Neel's most surgical
    weapon — what is posted here is official.

    Args:
        description: What was posted and what damage it does.
        initiator: Who arranged this (typically ``"neel"``).
        target: Whose reputation is damaged.
    """
    return Trigger(
        trigger_type=TriggerType.INSTITUTIONAL_MOVE,
        variant=TriggerVariant.INSTITUTIONAL_NOTICE_BOARD,
        location=LocationName.NOTICE_BOARD_CLUSTER,
        initiator=initiator,
        target=target,
        is_public=True,
        description=description,
    )


def make_opportunity_denial(
    description: str,
    *,
    initiator: str = "neel",
    target: str = "vikram",
    location: LocationName = LocationName.STUDENT_UNION_BUILDING,
) -> Trigger:
    """Create a TYPE 02-D: opportunity denial trigger.

    Blocking access to events, selections, or positions Vikram or his gang
    wanted. Invisible until the denial is discovered — forces Dhruv's
    cost-benefit recalculation.

    Args:
        description: What opportunity was denied and how.
        initiator: Who blocked it (typically ``"neel"``).
        target: Who was denied.
        location: Where this was administered.
    """
    return Trigger(
        trigger_type=TriggerType.INSTITUTIONAL_MOVE,
        variant=TriggerVariant.INSTITUTIONAL_OPPORTUNITY_DENIAL,
        location=location,
        initiator=initiator,
        target=target,
        is_public=False,
        description=description,
    )


def make_election_positioning(
    description: str,
    *,
    initiator: str = "neel",
    target: str = "vikram",
) -> Trigger:
    """Create a TYPE 03-A: election candidate positioning trigger.

    A candidate positioned against someone in Vikram's orbit, isolating Vikram
    politically without direct confrontation. Maximum power during ELECTION_SEASON.

    Args:
        description: What positioning move was made.
        initiator: Who arranged this (typically ``"neel"``).
        target: Who is isolated.
    """
    return Trigger(
        trigger_type=TriggerType.POLITICAL_MOVE,
        variant=TriggerVariant.POLITICAL_ELECTION_POSITIONING,
        location=LocationName.STUDENT_UNION_BUILDING,
        initiator=initiator,
        target=target,
        is_public=False,
        description=description,
    )


def make_dhruv_contact(
    description: str,
    *,
    initiator: str = "neel",
) -> Trigger:
    """Create a TYPE 03-D: Dhruv contacted directly by Neel trigger.

    The most direct test of Dhruv's drift state. Neel and Dhruv speak the same
    language — this trigger forces Dhruv's cost-benefit recalculation to the
    surface. Occurs in the secondary canteen by convention.

    Args:
        description: What Neel offered or communicated.
        initiator: Who made contact (typically ``"neel"``).
    """
    return Trigger(
        trigger_type=TriggerType.POLITICAL_MOVE,
        variant=TriggerVariant.POLITICAL_DHRUV_CONTACT,
        location=LocationName.SECONDARY_CANTEEN,
        initiator=initiator,
        target="dhruv",
        is_public=False,
        description=description,
    )


def make_meera_intersection(
    location: LocationName,
    description: str,
    *,
    initiator: str = "meera",
) -> Trigger:
    """Create a TYPE 04-A: Meera intersects with the conflict trigger.

    Through her own actions, not Vikram's. The gap between her awareness of the
    danger and their awareness of her is where the risk lives. Vikram's response
    is undefined — even to himself.

    Args:
        location: Where the intersection occurs.
        description: What Meera did and how it touched the conflict.
        initiator: Defaults to ``"meera"`` — she acts, not Vikram.
    """
    return Trigger(
        trigger_type=TriggerType.AMBIENT_TRIGGER,
        variant=TriggerVariant.AMBIENT_MEERA_INTERSECTION,
        location=location,
        initiator=initiator,
        target=NO_TARGET,
        is_public=False,
        description=description,
    )


def make_information_surface(
    location: LocationName,
    description: str,
    *,
    initiator: str = "surya",
) -> Trigger:
    """Create a TYPE 04-D: information surfaces trigger.

    Information one side had that the other didn't know about. Surya's trigger
    — he knows something, and its surfacing changes everything. The most
    consequential ambient variant for shifting Surya's reveal conditions.

    Args:
        location: Where the information emerges.
        description: What information surfaced and which side is changed by it.
        initiator: Who surfaced it. Defaults to ``"surya"``.
    """
    return Trigger(
        trigger_type=TriggerType.AMBIENT_TRIGGER,
        variant=TriggerVariant.AMBIENT_INFORMATION_SURFACE,
        location=location,
        initiator=initiator,
        target=NO_TARGET,
        is_public=False,
        description=description,
    )


def make_gang_member_acts_alone(
    initiator: str,
    location: LocationName,
    description: str,
    *,
    is_public: bool = False,
) -> Trigger:
    """Create a TYPE 04-E: gang member acts without Vikram trigger.

    Rajan escalating independently, Dhruv making a contact, Savar saying
    something he cannot unsay. Forces Vikram to decide whether to claim or
    distance himself — a character-defining decision.

    Args:
        initiator: Which gang member acted (``"rajan"``, ``"dhruv"``, ``"savar"``).
        location: Where the action occurred.
        description: What the member did and what it has set in motion.
        is_public: Whether bystanders were present. Rajan's escalations
            often are; Savar's statements usually are.
    """
    return Trigger(
        trigger_type=TriggerType.AMBIENT_TRIGGER,
        variant=TriggerVariant.AMBIENT_GANG_MEMBER_ACTS_ALONE,
        location=location,
        initiator=initiator,
        target=NO_TARGET,
        is_public=is_public,
        description=description,
    )
