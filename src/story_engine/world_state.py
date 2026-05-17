"""WorldState dataclass — single mutable source of truth for the story engine.

All engine methods read from and write to a ``WorldState`` instance. Nothing
persists outside it during a simulation step. Character states are stored as
named fields for direct attribute access; the relationship graph uses
``(source_name, target_name)`` string pairs as keys.

Serialization contract
~~~~~~~~~~~~~~~~~~~~~~
``WorldState.to_dict()``  — produces a plain-Python dict with JSON-safe values
``WorldState.from_dict()`` — reconstructs a full ``WorldState`` from such a dict
``WorldState.to_json()``   — thin wrapper: ``json.dumps(self.to_dict(), indent=2)``
``WorldState.from_json()`` — thin wrapper: ``cls.from_dict(json.loads(s))``

Enum values are serialised as their ``.name`` strings.
``frozenset`` fields are serialised as sorted lists.
Relationship graph keys ``(str, str)`` are serialised as ``"source|target"`` strings.

Import contract
~~~~~~~~~~~~~~~
world_state.py → characters.py → flags.py   (no reverse imports)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from story_engine.characters import (
    ArjunState,
    CoreTrait,
    DhruvDriftState,
    DhruvState,
    KaranState,
    KavyaState,
    MeeraState,
    NeelState,
    RajanState,
    RanveerPhase,
    RanveerState,
    SavarState,
    SuryaState,
    SuryaAllegiance,
    VikramState,
    Year,
)
from story_engine.flags import FlagSet, WorldFlag


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConflictPhase(Enum):
    """Escalation phase of the Vikram / Ranveer conflict.

    Advances through the enum in response to triggers and chain rules. The
    engine reads this to determine which rules become available and which
    resolution conditions can be checked.
    """

    COLD_EQUILIBRIUM = auto()
    """Tension exists; no active moves. Default opening state."""

    FRICTION = auto()
    """Trigger fired; both sides responded; no irreversible move yet."""

    OPEN_CONFLICT = auto()
    """Publicly known; faculty aware; institutional weapons may be active."""

    RESOLUTION_ONE_SIDE_UP = auto()
    """Winner visible to campus; neither side acknowledges it explicitly."""

    PYRRHIC = auto()
    """Both sides paid; nobody won. Often accompanies DhruvDriftState.GONE."""

    CRISIS = auto()
    """Irreversible move made; cannot be absorbed into normal campus life."""


class ResolutionType(Enum):
    """The resolution type targeted at story initialisation.

    The engine's ``check_resolution_condition`` tests whether the current
    ``WorldState`` satisfies the conditions of the target type.
    """

    R1_VISIBLE_DEFEAT = auto()
    """Vikram publicly damaged, still present, hasn't acknowledged it."""

    R2_VISIBLE_WIN = auto()
    """Ranveer made to look like a non-owner; campus reads it."""

    R3_PYRRHIC = auto()
    """Both paid; Dhruv gone; nobody won."""

    R4_SUSPENDED = auto()
    """Ends mid-conflict; consequences still arriving; no clean resolution."""

    R5_STRUCTURAL = auto()
    """Institutional conflict resolved; personal tension between Vikram and
    Ranveer persists unresolved."""


class TimeOfDay(Enum):
    """Coarse time-of-day for scene context.

    Injected into ``SceneBrief.world_state`` to constrain which locations
    are plausible and what ambient crowd density looks like.
    """

    EARLY_MORNING = auto()
    MORNING = auto()
    AFTERNOON = auto()
    EVENING = auto()
    NIGHT = auto()


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RelationshipState:
    """Directed relationship state between two named characters.

    Stored in ``WorldState.relationship_graph`` keyed by
    ``(source_name, target_name)`` — lowercase character first names.

    Args:
        tension: Active tension level. 0 = none; 10 = maximum.
        trust: Trust level. 0 = none; 10 = unconditional.
        history_notes: Ordered significant past events between these two.
        is_public: Whether this relationship is campus-common-knowledge.
    """

    tension: int = 0
    trust: int = 5
    history_notes: tuple[str, ...] = field(default_factory=tuple)
    is_public: bool = False

    def __post_init__(self) -> None:
        if not (0 <= self.tension <= 10):
            raise ValueError(
                f"RelationshipState.tension must be in [0, 10], got {self.tension}"
            )
        if not (0 <= self.trust <= 10):
            raise ValueError(
                f"RelationshipState.trust must be in [0, 10], got {self.trust}"
            )


@dataclass
class IncidentEntry:
    """A single recorded event in the simulation incident log.

    Appended to ``WorldState.incident_log`` each time a trigger fires.
    The ``SceneBriefGenerator`` pulls from this list to populate
    ``SceneBrief.prior_context``.

    Args:
        step: Simulation step at which the incident occurred.
        trigger_type: The ``TriggerType`` enum value name (string).
        variant: The ``TriggerVariant`` enum value name (string).
        location_name: Name of the location where the incident occurred.
        initiator: Lowercase name of the character who fired the trigger.
        target: Lowercase name of the primary target, or ``"__diffuse__"``
            for untargeted triggers.
        description: Human-readable summary of what happened.
        consequence_notes: Downstream effects identified at firing time.
        is_public: Whether the general student body witnessed this.
    """

    step: int
    trigger_type: str
    variant: str
    location_name: str
    initiator: str
    target: str
    description: str
    consequence_notes: tuple[str, ...] = field(default_factory=tuple)
    is_public: bool = False


# ---------------------------------------------------------------------------
# WorldState
# ---------------------------------------------------------------------------


@dataclass
class WorldState:
    """Single mutable source of truth for all engine state.

    Constructed by ``StoryEngine.initialize_story()`` from validated
    ``StoryInitParams`` and mutated in place by every engine chain-rule method.
    No engine state lives outside this object during a simulation step.

    Character states are named fields — access is ``state.vikram.phase``, not
    ``state.characters["vikram"].phase``. The relationship graph uses
    ``(source, target)`` lowercase-name pairs as keys.

    Args:
        active_flags: Set of currently active world flags.
        conflict_phase: Current escalation phase of the conflict.
        resolution_type: Target resolution set at initialisation.
        time_of_day: Current time of day in the simulation.
        step: Step counter; incremented by ``advance_state()``.
        vikram: Mutable state for Vikram.
        ranveer: Mutable state for Ranveer.
        karan: Mutable state for Karan.
        neel: Mutable state for Neel.
        arjun: Mutable state for Arjun.
        savar: Mutable state for Savar.
        dhruv: Mutable state for Dhruv.
        rajan: Mutable state for Rajan.
        surya: Mutable state for Surya (includes hidden true_state).
        kavya: Mutable state for Kavya.
        meera: Mutable state for Meera.
        relationship_graph: Directed edges keyed by (source, target) name pairs.
        incident_log: Ordered record of all incidents fired this simulation.
    """

    # Context
    active_flags: FlagSet
    conflict_phase: ConflictPhase
    resolution_type: ResolutionType
    time_of_day: TimeOfDay
    step: int = 0

    # Character states
    vikram: VikramState = field(default_factory=VikramState)
    ranveer: RanveerState = field(default_factory=RanveerState)
    karan: KaranState = field(default_factory=KaranState)
    neel: NeelState = field(default_factory=NeelState)
    arjun: ArjunState = field(default_factory=ArjunState)
    savar: SavarState = field(default_factory=SavarState)
    dhruv: DhruvState = field(default_factory=DhruvState)
    rajan: RajanState = field(default_factory=RajanState)
    surya: SuryaState = field(default_factory=SuryaState)
    kavya: KavyaState = field(default_factory=KavyaState)
    meera: MeeraState = field(default_factory=MeeraState)

    # Relationship graph: (source_name, target_name) → state
    relationship_graph: dict[tuple[str, str], RelationshipState] = field(
        default_factory=dict
    )

    # Incident log
    incident_log: list[IncidentEntry] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain-Python dict with JSON-safe values only.

        Enums → their ``.name`` string.
        ``frozenset`` → sorted list of member names (or values).
        ``tuple[str, ...]`` → list.
        Nested dataclasses → recursively serialised dicts.
        Relationship graph keys ``(str, str)`` → ``"source|target"`` strings.

        Returns:
            JSON-safe dict representing the complete world state.
        """
        return {
            "active_flags": sorted(f.name for f in self.active_flags.flags),
            "conflict_phase": self.conflict_phase.name,
            "resolution_type": self.resolution_type.name,
            "time_of_day": self.time_of_day.name,
            "step": self.step,
            "vikram": _vikram_to_dict(self.vikram),
            "ranveer": _ranveer_to_dict(self.ranveer),
            "karan": _karan_to_dict(self.karan),
            "neel": _neel_to_dict(self.neel),
            "arjun": _arjun_to_dict(self.arjun),
            "savar": _savar_to_dict(self.savar),
            "dhruv": _dhruv_to_dict(self.dhruv),
            "rajan": _rajan_to_dict(self.rajan),
            "surya": _surya_to_dict(self.surya),
            "kavya": _kavya_to_dict(self.kavya),
            "meera": _meera_to_dict(self.meera),
            "relationship_graph": {
                f"{src}|{tgt}": _rel_to_dict(v)
                for (src, tgt), v in self.relationship_graph.items()
            },
            "incident_log": [_incident_to_dict(e) for e in self.incident_log],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorldState:
        """Reconstruct a WorldState from a dict produced by ``to_dict()``.

        Args:
            data: A plain-Python dict as returned by ``to_dict()``.

        Returns:
            A fully reconstructed ``WorldState`` instance.
        """
        return cls(
            active_flags=FlagSet(
                flags=frozenset(WorldFlag[n] for n in data["active_flags"])
            ),
            conflict_phase=ConflictPhase[data["conflict_phase"]],
            resolution_type=ResolutionType[data["resolution_type"]],
            time_of_day=TimeOfDay[data["time_of_day"]],
            step=data["step"],
            vikram=_vikram_from_dict(data["vikram"]),
            ranveer=_ranveer_from_dict(data["ranveer"]),
            karan=_karan_from_dict(data["karan"]),
            neel=_neel_from_dict(data["neel"]),
            arjun=_arjun_from_dict(data["arjun"]),
            savar=_savar_from_dict(data["savar"]),
            dhruv=_dhruv_from_dict(data["dhruv"]),
            rajan=_rajan_from_dict(data["rajan"]),
            surya=_surya_from_dict(data["surya"]),
            kavya=_kavya_from_dict(data["kavya"]),
            meera=_meera_from_dict(data["meera"]),
            relationship_graph={
                (src, tgt): _rel_from_dict(v)
                for k, v in data["relationship_graph"].items()
                for src, tgt in [k.split("|", 1)]
            },
            incident_log=[_incident_from_dict(e) for e in data["incident_log"]],
        )

    def to_json(self) -> str:
        """Serialise to an indented JSON string.

        Returns:
            Indented JSON string representing the complete world state.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> WorldState:
        """Reconstruct a WorldState from a JSON string produced by ``to_json()``.

        Args:
            s: JSON string as returned by ``to_json()``.

        Returns:
            A fully reconstructed ``WorldState`` instance.
        """
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Private serialisation helpers — one pair per character/type
# ---------------------------------------------------------------------------


def _vikram_to_dict(v: VikramState) -> dict[str, Any]:
    return {
        "year": v.year.name,
        "core_trait": v.core_trait.name,
        "last_trigger_type": v.last_trigger_type,
    }


def _vikram_from_dict(d: dict[str, Any]) -> VikramState:
    return VikramState(
        year=Year[d["year"]],
        core_trait=CoreTrait[d["core_trait"]],
        last_trigger_type=d["last_trigger_type"],
    )


def _ranveer_to_dict(r: RanveerState) -> dict[str, Any]:
    return {
        "year": r.year.name,
        "core_trait": r.core_trait.name,
        "phase": r.phase.name,
        "consecutive_unacknowledged_non_submissions": r.consecutive_unacknowledged_non_submissions,
        "last_weakness_was_strategy": r.last_weakness_was_strategy,
    }


def _ranveer_from_dict(d: dict[str, Any]) -> RanveerState:
    return RanveerState(
        year=Year[d["year"]],
        core_trait=CoreTrait[d["core_trait"]],
        phase=RanveerPhase[d["phase"]],
        consecutive_unacknowledged_non_submissions=d[
            "consecutive_unacknowledged_non_submissions"
        ],
        last_weakness_was_strategy=d["last_weakness_was_strategy"],
    )


def _karan_to_dict(k: KaranState) -> dict[str, Any]:
    return {
        "year": k.year.name,
        "core_trait": k.core_trait.name,
        "unfinished_feeling": k.unfinished_feeling,
        "is_activated": k.is_activated,
    }


def _karan_from_dict(d: dict[str, Any]) -> KaranState:
    return KaranState(
        year=Year[d["year"]],
        core_trait=CoreTrait[d["core_trait"]],
        unfinished_feeling=d["unfinished_feeling"],
        is_activated=d["is_activated"],
    )


def _neel_to_dict(n: NeelState) -> dict[str, Any]:
    return {
        "year": n.year.name,
        "core_trait": n.core_trait.name,
        "effective_capacity": n.effective_capacity,
    }


def _neel_from_dict(d: dict[str, Any]) -> NeelState:
    return NeelState(
        year=Year[d["year"]],
        core_trait=CoreTrait[d["core_trait"]],
        effective_capacity=d["effective_capacity"],
    )


def _arjun_to_dict(a: ArjunState) -> dict[str, Any]:
    return {
        "year": a.year.name,
        "core_trait": a.core_trait.name,
        "arjun_acts_in_window": a.arjun_acts_in_window,
        "window_is_open": a.window_is_open,
    }


def _arjun_from_dict(d: dict[str, Any]) -> ArjunState:
    return ArjunState(
        year=Year[d["year"]],
        core_trait=CoreTrait[d["core_trait"]],
        arjun_acts_in_window=d["arjun_acts_in_window"],
        window_is_open=d["window_is_open"],
    )


def _savar_to_dict(s: SavarState) -> dict[str, Any]:
    return {
        "year": s.year.name,
        "core_trait": s.core_trait.name,
        "visibility_level": s.visibility_level,
    }


def _savar_from_dict(d: dict[str, Any]) -> SavarState:
    return SavarState(
        year=Year[d["year"]],
        core_trait=CoreTrait[d["core_trait"]],
        visibility_level=d["visibility_level"],
    )


def _dhruv_to_dict(dh: DhruvState) -> dict[str, Any]:
    return {
        "year": dh.year.name,
        "core_trait": dh.core_trait.name,
        "cost_benefit_total": dh.cost_benefit_total,
        "consecutive_negative_events": dh.consecutive_negative_events,
        "drift_state": dh.drift_state.name,
    }


def _dhruv_from_dict(d: dict[str, Any]) -> DhruvState:
    return DhruvState(
        year=Year[d["year"]],
        core_trait=CoreTrait[d["core_trait"]],
        cost_benefit_total=d["cost_benefit_total"],
        consecutive_negative_events=d["consecutive_negative_events"],
        drift_state=DhruvDriftState[d["drift_state"]],
    )


def _rajan_to_dict(r: RajanState) -> dict[str, Any]:
    return {
        "year": r.year.name,
        "core_trait": r.core_trait.name,
        "has_direction_from_vikram": r.has_direction_from_vikram,
    }


def _rajan_from_dict(d: dict[str, Any]) -> RajanState:
    return RajanState(
        year=Year[d["year"]],
        core_trait=CoreTrait[d["core_trait"]],
        has_direction_from_vikram=d["has_direction_from_vikram"],
    )


def _surya_to_dict(s: SuryaState) -> dict[str, Any]:
    return {
        "year": s.year.name,
        "core_trait": s.core_trait.name,
        "true_state": s.true_state.name,
        "is_revealed": s.is_revealed,
    }


def _surya_from_dict(d: dict[str, Any]) -> SuryaState:
    return SuryaState(
        year=Year[d["year"]],
        core_trait=CoreTrait[d["core_trait"]],
        true_state=SuryaAllegiance[d["true_state"]],
        is_revealed=d["is_revealed"],
    )


def _kavya_to_dict(k: KavyaState) -> dict[str, Any]:
    return {
        "core_trait": k.core_trait.name,
        "is_active": k.is_active,
        "condition_a_met": k.condition_a_met,
        "condition_b_met": k.condition_b_met,
    }


def _kavya_from_dict(d: dict[str, Any]) -> KavyaState:
    return KavyaState(
        core_trait=CoreTrait[d["core_trait"]],
        is_active=d["is_active"],
        condition_a_met=d["condition_a_met"],
        condition_b_met=d["condition_b_met"],
    )


def _meera_to_dict(m: MeeraState) -> dict[str, Any]:
    return {
        "year": m.year.name,
        "core_trait": m.core_trait.name,
        "flags_lived_through": sorted(f.name for f in m.flags_lived_through),
        "response_set": sorted(m.response_set),
    }


def _meera_from_dict(d: dict[str, Any]) -> MeeraState:
    return MeeraState(
        year=Year[d["year"]],
        core_trait=CoreTrait[d["core_trait"]],
        flags_lived_through=frozenset(WorldFlag[n] for n in d["flags_lived_through"]),
        response_set=frozenset(d["response_set"]),
    )


def _rel_to_dict(r: RelationshipState) -> dict[str, Any]:
    return {
        "tension": r.tension,
        "trust": r.trust,
        "history_notes": list(r.history_notes),
        "is_public": r.is_public,
    }


def _rel_from_dict(d: dict[str, Any]) -> RelationshipState:
    return RelationshipState(
        tension=d["tension"],
        trust=d["trust"],
        history_notes=tuple(d["history_notes"]),
        is_public=d["is_public"],
    )


def _incident_to_dict(e: IncidentEntry) -> dict[str, Any]:
    return {
        "step": e.step,
        "trigger_type": e.trigger_type,
        "variant": e.variant,
        "location_name": e.location_name,
        "initiator": e.initiator,
        "target": e.target,
        "description": e.description,
        "consequence_notes": list(e.consequence_notes),
        "is_public": e.is_public,
    }


def _incident_from_dict(d: dict[str, Any]) -> IncidentEntry:
    return IncidentEntry(
        step=d["step"],
        trigger_type=d["trigger_type"],
        variant=d.get("variant", ""),
        location_name=d["location_name"],
        initiator=d.get("initiator", ""),
        target=d.get("target", ""),
        description=d["description"],
        consequence_notes=tuple(d["consequence_notes"]),
        is_public=d["is_public"],
    )
