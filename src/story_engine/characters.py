"""Character dataclasses and enums for the story engine.

Each character in the world is represented by:
- A set of enums capturing their phase / hidden state / drift trajectory.
- A ``@dataclass`` holding all mutable state the engine reads and writes.

No LLM calls live here. This module is pure Python state.

Import contract:
    characters.py → flags.py (one-directional; no reverse import).
    world_state.py → characters.py (aggregates all character states).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from story_engine.flags import WorldFlag


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------


class Year(Enum):
    """Academic year of a student character."""

    FIRST = 1
    SECOND = 2
    THIRD = 3


class CoreTrait(Enum):
    """Dominant character trait that governs engine response rules.

    Each value maps to the immutable behavioural constraint described in the
    world model. The engine uses these as keys when selecting which chain
    rules apply to a given trigger.
    """

    PRIDE = auto()
    """Vikram — cannot submit, cannot perform fear, cannot ignore public challenge."""

    CRUEL_AND_CALCULATIVE = auto()
    """Ranveer — sadistic precision; wants the authored humiliation, not just the outcome."""

    ENJOYS_IT = auto()
    """Karan — loyalty like a weapon; takes pleasure in the act itself."""

    PATIENCE = auto()
    """Neel — institutional systems thinking; debt collection without announcement."""

    COLD_ASSESSMENT = auto()
    """Arjun — sees people as problems or non-problems; no emotional dimension."""

    PERFORMANCE = auto()
    """Savar — courage, loyalty, anger are all performed; audience-dependent."""

    SELF_INTEREST = auto()
    """Dhruv — worn as pragmatism; always calculating the cost/return ratio."""

    CONTROLLED_RECKLESSNESS = auto()
    """Rajan — genuine indifference to consequence; does not stop."""

    UNKNOWABLE = auto()
    """Surya — cultivated opacity; the unknown is his trait, not a gap."""

    PRAGMATIC_SURVIVAL = auto()
    """Kavya — everything evaluated against what holds her life together."""

    BECOMING = auto()
    """Meera — still forming; the campus conflict is the test."""


# ---------------------------------------------------------------------------
# Ranveer phase
# ---------------------------------------------------------------------------


class RanveerPhase(Enum):
    """Escalation phase of Ranveer's obsession with Vikram.

    Integer values allow direct arithmetic in ``apply_pride_ratchet``.
    Phase 1 → 4 is a one-way ratchet under normal conditions; see engine
    rules for regression and +2 jump conditions.
    """

    COLD = 1
    """Vikram is an irregularity. Will be corrected."""

    IRRITATED = 2
    """Vikram is an irritant. Becoming interesting."""

    OBSESSED = 3
    """Vikram is personal. This is no longer about campus order."""

    PERSONAL = 4
    """Vikram is the only thing on this campus that matters to Ranveer.
    He would never say this. Everyone around him can see it."""


# ---------------------------------------------------------------------------
# Dhruv drift state
# ---------------------------------------------------------------------------


class DhruvDriftState(Enum):
    """Dhruv's departure trajectory, triggered by consecutive net-negative events.

    States advance in sequence; ``GONE`` cannot be reversed and cannot be
    called betrayal — it simply functions as it.
    """

    PRESENT = auto()
    """Fully engaged. Cost/benefit still positive."""

    LESS_AVAILABLE = auto()
    """First signal. Harder to reach; reasons feel circumstantial."""

    PRESENT_BUT_UNINVESTED = auto()
    """Body here, attention elsewhere. Reads are still accurate but volunteered less."""

    MAKING_EXIT_ARRANGEMENTS = auto()
    """Actively reducing exposure. Will not be seen to leave."""

    GONE = auto()
    """Not betrayal. Functions as it."""


# ---------------------------------------------------------------------------
# Surya's true allegiance (hidden from all in-world characters)
# ---------------------------------------------------------------------------


class SuryaAllegiance(Enum):
    """Surya's actual allegiance — hidden from every character in the world.

    Set at story initialisation via ``surya_true_state`` parameter.
    The engine only reveals this when ``check_surya_reveal`` conditions are met:
    (A) CRISIS phase, (B) direct private confrontation, or (C) operationally
    necessary.
    """

    WITH_VIKRAM = auto()
    """Genuinely loyal to Vikram. The silence is protection, not distance."""

    RANVEER_PLANT = auto()
    """Placed inside Vikram's gang by Ranveer's network. Long-term intelligence."""

    OWN_AGENDA = auto()
    """Operating toward personal goals unknown to both sides."""

    DRIFTER = auto()
    """Allegiance undefined. Present for reasons even Surya has not examined."""


# ---------------------------------------------------------------------------
# Character state dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VikramState:
    """Mutable engine state for Vikram, the MC.

    CoreTrait.PRIDE governs all response rules:
    - DIRECT_CHALLENGE → immediate, instinctive, pride before strategy.
    - INSTITUTIONAL_MOVE → delayed, frustrated, tries to make it personal.
    - POLITICAL_MOVE → slow recognition, last to see it.
    - AMBIENT_TRIGGER → undefined when Meera is involved.
    """

    year: Year = Year.SECOND
    core_trait: CoreTrait = CoreTrait.PRIDE
    last_trigger_type: str | None = None
    """Most recent trigger type that fired against Vikram. Engine-internal."""


@dataclass
class RanveerState:
    """Mutable engine state for Ranveer, the antagonist leader.

    ``phase`` is the primary output of ``apply_pride_ratchet``. It advances
    on every unacknowledged non-submission by Vikram, regresses if Vikram
    appears genuinely weakened (−1), and jumps +2 if apparent weakness is
    later revealed as strategy.
    """

    year: Year = Year.THIRD
    core_trait: CoreTrait = CoreTrait.CRUEL_AND_CALCULATIVE
    phase: RanveerPhase = RanveerPhase.COLD
    consecutive_unacknowledged_non_submissions: int = 0
    """Running count driving phase advancement. Resets on phase change."""
    last_weakness_was_strategy: bool = False
    """If True, the next phase recalculation adds +2 instead of −1."""


@dataclass
class KaranState:
    """Mutable engine state for Karan, Ranveer's enforcer.

    Activated immediately on physical triggers; peripheral on political ones.
    ``unfinished_feeling`` is the single unresolved variable in his psychology
    — Vikram never broke in year one, and Karan has never encountered that before.
    """

    year: Year = Year.THIRD
    core_trait: CoreTrait = CoreTrait.ENJOYS_IT
    unfinished_feeling: bool = True
    """True whenever Vikram remains unbroken. The only thing that confuses Karan."""
    is_activated: bool = False
    """True when a physical trigger has drawn Karan into active involvement."""


@dataclass
class NeelState:
    """Mutable engine state for Neel, Ranveer's shadow strategist.

    ``effective_capacity`` is computed by ``apply_neel_management_threshold``:
    - RanveerPhase.OBSESSED → 0.70 (30 % consumed managing Ranveer)
    - RanveerPhase.PERSONAL → 0.40 (60 % consumed managing Ranveer)
    - All other phases → 1.0
    """

    year: Year = Year.THIRD
    core_trait: CoreTrait = CoreTrait.PATIENCE
    effective_capacity: float = 1.0
    """Fraction of Neel's resources available against Vikram. Range [0.0, 1.0]."""


@dataclass
class ArjunState:
    """Mutable engine state for Arjun, the cold second-year assessor.

    The Arjun window (RULE_10) opens when ``RanveerPhase == OBSESSED`` and
    closes when ``RanveerPhase == PERSONAL``. Whether Arjun actually acts
    within the window is the per-story ``arjun_acts_in_window`` variable.
    """

    year: Year = Year.SECOND
    core_trait: CoreTrait = CoreTrait.COLD_ASSESSMENT
    arjun_acts_in_window: bool = False
    """Per-story variable set at initialisation. Engine reads but does not mutate."""
    window_is_open: bool = False
    """Computed by ``check_arjun_window``; True iff RanveerPhase == OBSESSED."""


@dataclass
class SavarState:
    """Mutable engine state for Savar, the performance-driven gang face.

    ``visibility_level`` is an **inverse** health signal per RULE_04:
    high Savar volume → something real is fracturing quietly inside the gang.
    Valid range: 1 (quiet, gang healthy) to 5 (very loud, gang in trouble).
    """

    year: Year = Year.SECOND
    core_trait: CoreTrait = CoreTrait.PERFORMANCE
    visibility_level: int = 1
    """Inverse gang-health indicator. Range [1, 5]."""

    def __post_init__(self) -> None:
        if not (1 <= self.visibility_level <= 5):
            raise ValueError(
                f"SavarState.visibility_level must be in [1, 5], got {self.visibility_level}"
            )


@dataclass
class DhruvState:
    """Mutable engine state for Dhruv, the calculating one.

    ``cost_benefit_total`` is updated on every trigger.
    Three consecutive net-negative events advance ``drift_state`` by one step
    (RULE_03_DHRUV_DRIFT). The progression is one-directional once started.
    """

    year: Year = Year.SECOND
    core_trait: CoreTrait = CoreTrait.SELF_INTEREST
    cost_benefit_total: float = 0.0
    """Running cost/benefit sum. Negative drift triggers state advance."""
    consecutive_negative_events: int = 0
    """Resets to 0 on any net-positive event; at 3 advances drift_state."""
    drift_state: DhruvDriftState = DhruvDriftState.PRESENT


@dataclass
class RajanState:
    """Mutable engine state for Rajan, the recklessly present one.

    RULE_08_RAJAN_CONSTANT: any trigger → Rajan shows up. No calculation.
    Without direction from Vikram he escalates to whatever the situation allows.
    There is no ``is_present`` toggle — presence is his constant.
    """

    year: Year = Year.SECOND
    core_trait: CoreTrait = CoreTrait.CONTROLLED_RECKLESSNESS
    has_direction_from_vikram: bool = False
    """If False, Rajan escalates unchecked when the engine fires his constant."""


@dataclass
class SuryaState:
    """Mutable engine state for Surya, the silent unknown.

    ``true_state`` is set at story initialisation and hidden from all
    in-world characters until ``check_surya_reveal`` returns True.
    Once ``is_revealed`` is set, the engine may surface the allegiance
    in scene briefs.
    """

    year: Year = Year.SECOND
    core_trait: CoreTrait = CoreTrait.UNKNOWABLE
    true_state: SuryaAllegiance = SuryaAllegiance.WITH_VIKRAM
    """Hidden allegiance. Engine-only until reveal conditions are met."""
    is_revealed: bool = False
    """Flipped by ``check_surya_reveal`` when conditions are satisfied."""


@dataclass
class KavyaState:
    """Mutable engine state for Kavya, professor and Vikram's mother.

    RULE_06_KAVYA_THRESHOLD: moves from passive to active **only when both**:
    - condition_a: conflict reached her professionally or domestically in
      an unavoidable way.
    - condition_b: she has calculated that acting costs less than not acting.

    ``is_active`` is set True by ``apply_kavya_threshold`` when both conditions
    are simultaneously met.
    """

    core_trait: CoreTrait = CoreTrait.PRAGMATIC_SURVIVAL
    is_active: bool = False
    """False until both threshold conditions are met."""
    condition_a_met: bool = False
    """Conflict has reached Kavya professionally or domestically."""
    condition_b_met: bool = False
    """Acting costs less than not acting — Kavya has made this calculation."""


@dataclass
class MeeraState:
    """Mutable engine state for Meera, Vikram's first-year sister.

    RULE_07_MEERA_TRANSFORMATION: each ``WorldFlag`` she lives through expands
    her available response set. The engine filters her scene response through
    ``flags_lived_through`` to determine what she is now capable of.

    ``response_set`` is a frozenset of string response-type identifiers.
    It starts empty (she arrived open) and is populated by
    ``apply_meera_transformation`` as each flag accumulates.
    """

    year: Year = Year.FIRST
    core_trait: CoreTrait = CoreTrait.BECOMING
    flags_lived_through: frozenset[WorldFlag] = field(default_factory=frozenset)
    """Flags Meera has personally experienced. Each one unlocks new responses."""
    response_set: frozenset[str] = field(default_factory=frozenset)
    """Available response-type identifiers. Grows with flags_lived_through."""
