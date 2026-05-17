"""StoryEngine — deterministic rule-based state machine.

The engine holds a single mutable ``WorldState`` and exposes a public API of
six top-level methods plus ten chain-rule methods. No LLM calls live here.
All logic is deterministic Python.

Chain rules are implemented as public methods so callers can invoke them
independently (e.g. after a narrative decision that does not originate from
a ``Trigger``).

Import contract
~~~~~~~~~~~~~~~
engine.py → brief_generator.py → world_state.py → characters.py → flags.py
engine.py → locations.py
engine.py → triggers.py → locations.py
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator

from story_engine.brief_generator import SceneBrief, SceneBriefGenerator
from story_engine.characters import (
    ArjunState,
    DhruvDriftState,
    DhruvState,
    RanveerPhase,
    RanveerState,
    SuryaCharacter,
    SuryaState,
)
from story_engine.flags import FlagSet, WorldFlag
from story_engine.locations import Location, LocationName, get_location
from story_engine.triggers import Trigger, TriggerType, TriggerVariant
from story_engine.world_state import (
    ConflictPhase,
    IncidentEntry,
    RelationshipState,
    ResolutionType,
    TimeOfDay,
    WorldState,
)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class EngineStateError(RuntimeError):
    """Raised when a chain-rule or API method is called before initialize_story()."""


# ---------------------------------------------------------------------------
# Story init params (Pydantic — external boundary)
# ---------------------------------------------------------------------------


class RelationshipStateParam(BaseModel):
    """A single relationship edge for story initialisation.

    Keys in ``StoryInitParams.active_relationship_states`` are formatted as
    ``"source_name|target_name"`` (e.g. ``"vikram|ranveer"``).
    """

    tension: int = 0
    trust: int = 5
    history_notes: list[str] = []
    is_public: bool = False


class StoryInitParams(BaseModel):
    """Validated input for ``StoryEngine.initialize_story()``.

    All enum values are passed as their ``.name`` string (e.g. ``"COLD"``
    for ``RanveerPhase.COLD``). Pydantic validators reject unknown names at
    the boundary so the engine never receives bad state.

    Args:
        active_flags: WorldFlag names active at story start.
        ranveer_phase_start: Initial RanveerPhase name.
        surya_true_state: Initial SuryaState name (hidden from in-world chars).
        dhruv_cost_start: Initial cost/benefit total for Dhruv.
        resolution_type: Target ResolutionType name for this story.
        active_relationship_states: Graph edges as ``"source|target"`` → params.
        arjun_acts_in_window: Whether Arjun will act during the OBSESSED window.
        time_of_day: Initial TimeOfDay name. Defaults to ``"MORNING"``.
        initial_conflict_phase: Initial ConflictPhase name. Defaults to
            ``"COLD_EQUILIBRIUM"``.
    """

    active_flags: list[str]
    ranveer_phase_start: str
    surya_true_state: str
    dhruv_cost_start: float
    resolution_type: str
    active_relationship_states: dict[str, RelationshipStateParam] = {}
    arjun_acts_in_window: bool = False
    time_of_day: str = "MORNING"
    initial_conflict_phase: str = "COLD_EQUILIBRIUM"

    @field_validator("active_flags")
    @classmethod
    def validate_active_flags(cls, v: list[str]) -> list[str]:
        for name in v:
            try:
                WorldFlag[name]
            except KeyError:
                raise ValueError(f"Unknown WorldFlag name: {name!r}")
        return v

    @field_validator("ranveer_phase_start")
    @classmethod
    def validate_ranveer_phase(cls, v: str) -> str:
        try:
            RanveerPhase[v]
        except KeyError:
            raise ValueError(f"Unknown RanveerPhase name: {v!r}")
        return v

    @field_validator("surya_true_state")
    @classmethod
    def validate_surya_state(cls, v: str) -> str:
        try:
            SuryaState[v]
        except KeyError:
            raise ValueError(f"Unknown SuryaState name: {v!r}")
        return v

    @field_validator("resolution_type")
    @classmethod
    def validate_resolution_type(cls, v: str) -> str:
        try:
            ResolutionType[v]
        except KeyError:
            raise ValueError(f"Unknown ResolutionType name: {v!r}")
        return v

    @field_validator("time_of_day")
    @classmethod
    def validate_time_of_day(cls, v: str) -> str:
        try:
            TimeOfDay[v]
        except KeyError:
            raise ValueError(f"Unknown TimeOfDay name: {v!r}")
        return v

    @field_validator("initial_conflict_phase")
    @classmethod
    def validate_conflict_phase(cls, v: str) -> str:
        try:
            ConflictPhase[v]
        except KeyError:
            raise ValueError(f"Unknown ConflictPhase name: {v!r}")
        return v


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Meera response types unlocked by each WorldFlag (RULE_07_MEERA_TRANSFORMATION).
# Responses accumulate — living through a flag expands the set permanently.
_MEERA_FLAG_RESPONSES: dict[WorldFlag, frozenset[str]] = {
    WorldFlag.SEMESTER_OPENING: frozenset(
        {
            "watchful",  # she has learned to watch before speaking
            "performing_openness",  # openness is now chosen, not instinctive
        }
    ),
    WorldFlag.CULTURAL_FEST: frozenset(
        {
            "exposed",  # she knows what unregulated attention costs
            "selective",  # she now chooses who gets access to her
        }
    ),
    WorldFlag.ELECTION_SEASON: frozenset(
        {
            "politically_legible",  # she can read moves for what they are
            "aligned_by_default",  # she knows she is read as aligned whether she intends to be or not
        }
    ),
    WorldFlag.EXAM_SEASON: frozenset(
        {
            "institutionally_aware",  # understands how the institution can be weaponised
            "calculating_cost",  # calculates before acting
        }
    ),
    WorldFlag.POLITICAL_AGITATION: frozenset(
        {
            "reads_movements",  # distinguishes genuine anger from shaped anger
            "positioned",  # understands she is positioned by others without consent
        }
    ),
    WorldFlag.INCIDENT_AFTERMATH: frozenset(
        {
            "carrying_weight",  # knows what it means to carry what has happened
            "pragmatic",  # has learned to be practical about what cannot be changed
            "withdrawn",  # manages openness more carefully
        }
    ),
    WorldFlag.SEMESTER_END: frozenset(
        {
            "calculating_what_remains",  # sees who is still there after things cost something
            "transformed",  # no longer who she was when she arrived
        }
    ),
}

# Default Dhruv event cost by trigger type.
# Positive = benefit to Dhruv's calculus; negative = cost.
# Callers may override via the ``dhruv_event_cost`` parameter of ``fire_trigger``.
_DHRUV_DEFAULT_COST: dict[TriggerType, float] = {
    TriggerType.DIRECT_CHALLENGE: -1.0,  # dangerous and visible
    TriggerType.INSTITUTIONAL_MOVE: -0.5,  # threatens stability and future prospects
    TriggerType.POLITICAL_MOVE: -0.25,  # isolating; slow-burn cost
    TriggerType.AMBIENT_TRIGGER: -0.25,  # unpredictable; disrupts Dhruv's calculation
}


# ---------------------------------------------------------------------------
# StoryEngine
# ---------------------------------------------------------------------------


class StoryEngine:
    """Deterministic rule-based state machine for the story world.

    Call ``initialize_story()`` first. All other methods require an
    initialised state and will raise ``EngineStateError`` otherwise.

    Chain rule methods may be called independently of ``fire_trigger()``
    when the caller needs to apply a rule in response to a narrative decision
    that does not originate from a ``Trigger`` (e.g. Dhruv being directly
    contacted off-screen, or Meera living through a flag between steps).
    """

    def __init__(self) -> None:
        self._state: WorldState | None = None

    # ------------------------------------------------------------------
    # State guard property
    # ------------------------------------------------------------------

    @property
    def state(self) -> WorldState:
        """Return current WorldState, raising EngineStateError if uninitialised."""
        if self._state is None:
            raise EngineStateError(
                "StoryEngine.initialize_story() must be called before "
                "accessing engine state."
            )
        return self._state

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def initialize_story(self, params: StoryInitParams) -> WorldState:
        """Seed a new WorldState from validated initialisation parameters.

        Resets all engine state. Safe to call multiple times (starts fresh).

        Args:
            params: Validated ``StoryInitParams`` instance.

        Returns:
            The freshly constructed WorldState.
        """
        graph: dict[tuple[str, str], RelationshipState] = {}
        for edge_key, rel_param in params.active_relationship_states.items():
            src, tgt = edge_key.split("|", 1)
            graph[(src, tgt)] = RelationshipState(
                tension=rel_param.tension,
                trust=rel_param.trust,
                history_notes=tuple(rel_param.history_notes),
                is_public=rel_param.is_public,
            )

        ranveer = RanveerState(phase=RanveerPhase[params.ranveer_phase_start])
        dhruv = DhruvState(cost_benefit_total=params.dhruv_cost_start)
        surya = SuryaCharacter(true_state=SuryaState[params.surya_true_state])
        arjun = ArjunState(arjun_acts_in_window=params.arjun_acts_in_window)

        self._state = WorldState(
            active_flags=FlagSet(
                flags=frozenset(WorldFlag[n] for n in params.active_flags)
            ),
            conflict_phase=ConflictPhase[params.initial_conflict_phase],
            resolution_type=ResolutionType[params.resolution_type],
            time_of_day=TimeOfDay[params.time_of_day],
            step=0,
            ranveer=ranveer,
            dhruv=dhruv,
            surya=surya,
            arjun=arjun,
            relationship_graph=graph,
        )

        # Apply initial derived state
        self.apply_neel_management_threshold()
        self.check_arjun_window()

        return self._state

    def fire_trigger(
        self,
        trigger: Trigger,
        *,
        dhruv_event_cost: float | None = None,
    ) -> WorldState:
        """Apply a trigger, cascade all applicable chain rules, and log the incident.

        Chain rules are applied in this order:
        1. Compute visibility multiplier (RULE_02).
        2. Type-specific handler (direct challenge, institutional, political, ambient).
        3. Dhruv drift (RULE_03) — default cost or caller override.
        4. Neel management threshold (RULE_05) — always recomputed.
        5. Rajan constant (RULE_08) — always fires.
        6. Arjun window check (RULE_10) — always recomputed.
        7. Update Vikram's ``last_trigger_type``.
        8. Append ``IncidentEntry`` to incident log.

        Args:
            trigger: The trigger to fire.
            dhruv_event_cost: Override the default Dhruv cost for this trigger.
                Positive = net benefit to Dhruv; negative = net cost.

        Returns:
            Updated WorldState (same object, mutated in place).
        """
        state = self.state

        location = get_location(trigger.location)
        multiplier = self.apply_visibility_multiplier(trigger, location)

        if trigger.trigger_type is TriggerType.DIRECT_CHALLENGE:
            self._handle_direct_challenge(trigger, multiplier)
        elif trigger.trigger_type is TriggerType.INSTITUTIONAL_MOVE:
            self._handle_institutional_move(trigger, multiplier)
        elif trigger.trigger_type is TriggerType.POLITICAL_MOVE:
            self._handle_political_move(trigger, multiplier)
        elif trigger.trigger_type is TriggerType.AMBIENT_TRIGGER:
            self._handle_ambient_trigger(trigger, multiplier)

        # Dhruv drift — DHRUV_CONTACT is a positive signal for Dhruv's calculus
        cost: float
        if dhruv_event_cost is not None:
            cost = dhruv_event_cost
        elif trigger.variant is TriggerVariant.POLITICAL_DHRUV_CONTACT:
            cost = 0.5
        else:
            cost = _DHRUV_DEFAULT_COST[trigger.trigger_type]
        self.apply_dhruv_drift(cost)

        # Always-on rules
        self.apply_neel_management_threshold()
        rajan_escalated = self.apply_rajan_constant(trigger)
        arjun_window = self.check_arjun_window()

        # Update Vikram's last trigger type
        state.vikram.last_trigger_type = trigger.trigger_type.name

        # Log incident
        state.incident_log.append(
            IncidentEntry(
                step=state.step,
                trigger_type=trigger.trigger_type.name,
                location_name=trigger.location.name,
                description=trigger.description,
                consequence_notes=self._derive_consequence_notes(
                    trigger, multiplier, rajan_escalated, arjun_window
                ),
                is_public=trigger.is_public,
            )
        )

        return state

    def get_current_state(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the current WorldState.

        Returns:
            JSON-safe dict produced by ``WorldState.to_dict()``.
        """
        return self.state.to_dict()

    def generate_scene_brief(self, location_name: LocationName) -> SceneBrief:
        """Generate a rich SceneBrief for the current state at the given location.

        Delegates to ``SceneBriefGenerator``. All fields are derived from
        WorldState — no placeholder strings, no empty lists.

        Args:
            location_name: The location where the scene will be set.

        Returns:
            A fully populated SceneBrief ready for LLM prose rendering.
        """
        return SceneBriefGenerator().generate(self.state, location_name)

    def advance_state(self, *, time_of_day: TimeOfDay | None = None) -> WorldState:
        """Step the simulation forward by one unit.

        Increments the step counter, optionally advances time of day, and
        recomputes periodic derived state (Neel threshold, Arjun window).
        Does NOT advance the conflict phase — that is driven by triggers.

        Args:
            time_of_day: If provided, set the time of day to this value.
                If None, the time of day is unchanged.

        Returns:
            Updated WorldState (same object, mutated in place).
        """
        state = self.state
        state.step += 1

        if time_of_day is not None:
            state.time_of_day = time_of_day

        self.apply_neel_management_threshold()
        self.check_arjun_window()

        return state

    def check_resolution_condition(self) -> ResolutionType | None:
        """Test whether the current WorldState satisfies the target resolution.

        Checks conditions for the ``ResolutionType`` set at story initialisation.

        Returns:
            The target ``ResolutionType`` if conditions are met; None otherwise.
        """
        state = self.state
        target = state.resolution_type

        if target is ResolutionType.R1_VISIBLE_DEFEAT:
            public_hits_on_vikram = sum(
                1
                for e in state.incident_log
                if "vikram" in e.description.lower() and e.is_public
            )
            if (
                state.conflict_phase
                in {ConflictPhase.OPEN_CONFLICT, ConflictPhase.CRISIS}
                and state.ranveer.phase.value >= RanveerPhase.IRRITATED.value
                and public_hits_on_vikram >= 2
            ):
                return ResolutionType.R1_VISIBLE_DEFEAT

        elif target is ResolutionType.R2_VISIBLE_WIN:
            if (
                state.ranveer.phase is RanveerPhase.PERSONAL
                and state.conflict_phase
                in {
                    ConflictPhase.OPEN_CONFLICT,
                    ConflictPhase.CRISIS,
                    ConflictPhase.RESOLUTION_ONE_SIDE_UP,
                }
                and state.neel.effective_capacity <= 0.4
            ):
                return ResolutionType.R2_VISIBLE_WIN

        elif target is ResolutionType.R3_PYRRHIC:
            if state.dhruv.drift_state is DhruvDriftState.GONE:
                return ResolutionType.R3_PYRRHIC

        elif target is ResolutionType.R4_SUSPENDED:
            if state.step >= 6 and state.conflict_phase in {
                ConflictPhase.FRICTION,
                ConflictPhase.OPEN_CONFLICT,
            }:
                return ResolutionType.R4_SUSPENDED

        elif target is ResolutionType.R5_STRUCTURAL:
            if (
                state.conflict_phase is ConflictPhase.RESOLUTION_ONE_SIDE_UP
                and state.karan.unfinished_feeling is True
            ):
                return ResolutionType.R5_STRUCTURAL

        return None

    # ------------------------------------------------------------------
    # Chain rules
    # ------------------------------------------------------------------

    def apply_pride_ratchet(
        self,
        trigger: Trigger,
        *,
        is_apparent_weakness: bool = False,
        weakness_revealed_as_strategy: bool = False,
    ) -> RanveerPhase:
        """RULE_01_PRIDE_RATCHET: update Ranveer's phase in response to Vikram.

        Every unacknowledged non-submission (default) → ranveer_phase += 1.
        Apparent weakness → ranveer_phase -= 1; sets ``last_weakness_was_strategy``.
        Weakness later revealed as strategy → ranveer_phase += 2.

        Phase is clamped to [COLD(1), PERSONAL(4)].

        Args:
            trigger: The trigger that prompted this evaluation (used for logging).
            is_apparent_weakness: True if Vikram appears genuinely weakened.
            weakness_revealed_as_strategy: True if a prior apparent weakness is
                now revealed to have been deliberate — triggers the +2 jump.

        Returns:
            The new RanveerPhase after applying the rule.
        """
        state = self.state
        current_val = state.ranveer.phase.value

        if weakness_revealed_as_strategy:
            new_val = min(current_val + 2, RanveerPhase.PERSONAL.value)
            state.ranveer.last_weakness_was_strategy = False
        elif is_apparent_weakness:
            new_val = max(current_val - 1, RanveerPhase.COLD.value)
            state.ranveer.last_weakness_was_strategy = True
        else:
            # Non-submission: Vikram never breaks (CoreTrait.PRIDE)
            new_val = min(current_val + 1, RanveerPhase.PERSONAL.value)
            state.ranveer.consecutive_unacknowledged_non_submissions += 1
            state.ranveer.last_weakness_was_strategy = False

        state.ranveer.phase = RanveerPhase(new_val)
        return state.ranveer.phase

    def apply_visibility_multiplier(
        self, trigger: Trigger, location: Location
    ) -> float:
        """RULE_02_VISIBILITY_MULTIPLIER: return consequence weight for a trigger.

        A non-public trigger is not witnessed by the student body regardless of
        location — its multiplier is always 1.0.
        A public trigger inherits the location's consequence multiplier.

        Args:
            trigger: The trigger being evaluated.
            location: The ``Location`` instance for ``trigger.location``.

        Returns:
            1.0, 1.5, or 2.0 depending on publicity and location visibility.
        """
        if not trigger.is_public:
            return 1.0
        return location.consequence_multiplier()

    def apply_dhruv_drift(self, event_cost: float) -> DhruvDriftState:
        """RULE_03_DHRUV_DRIFT: update Dhruv's cost/benefit total and drift state.

        Adds ``event_cost`` to ``cost_benefit_total``.
        A positive ``event_cost`` resets the consecutive negative event counter.
        A negative ``event_cost`` increments it.
        When the counter reaches 3, ``drift_state`` advances one step and the
        counter resets. ``GONE`` is the terminal state.

        Args:
            event_cost: Net benefit (positive) or cost (negative) for Dhruv.

        Returns:
            Current ``DhruvDriftState`` after applying the rule.
        """
        state = self.state
        dhruv = state.dhruv

        dhruv.cost_benefit_total += event_cost

        if event_cost < 0.0:
            dhruv.consecutive_negative_events += 1
        else:
            dhruv.consecutive_negative_events = 0

        if dhruv.consecutive_negative_events >= 3:
            drift_states = list(DhruvDriftState)
            current_idx = drift_states.index(dhruv.drift_state)
            if current_idx < len(drift_states) - 1:
                dhruv.drift_state = drift_states[current_idx + 1]
            dhruv.consecutive_negative_events = 0

        return dhruv.drift_state

    def apply_savar_inversion(self) -> int:
        """RULE_04_SAVAR_INVERSION: return inverted gang health signal.

        Savar's ``visibility_level`` [1–5] is an inverse health indicator:
        high volume → something real is fracturing quietly inside the gang.

        Returns:
            Gang health signal as int in [1, 5].
            5 = gang is healthy (Savar quiet).
            1 = gang is in serious trouble (Savar very loud).
        """
        return 6 - self.state.savar.visibility_level

    def apply_neel_management_threshold(self) -> float:
        """RULE_05_NEEL_MANAGEMENT_THRESHOLD: compute Neel's effective capacity.

        As Ranveer's phase advances, Neel spends increasing resources managing
        Ranveer instead of applying pressure on Vikram:

        - COLD or IRRITATED → 1.0 (full capacity against Vikram)
        - OBSESSED → 0.70 (30% consumed managing Ranveer)
        - PERSONAL → 0.40 (60% consumed managing Ranveer)

        Updates ``neel.effective_capacity`` in place.

        Returns:
            The new ``effective_capacity`` value.
        """
        state = self.state
        phase = state.ranveer.phase

        if phase is RanveerPhase.PERSONAL:
            state.neel.effective_capacity = 0.40
        elif phase is RanveerPhase.OBSESSED:
            state.neel.effective_capacity = 0.70
        else:
            state.neel.effective_capacity = 1.0

        return state.neel.effective_capacity

    def apply_kavya_threshold(self, condition_a: bool, condition_b: bool) -> bool:
        """RULE_06_KAVYA_THRESHOLD: update Kavya's activation conditions.

        Kavya moves from passive to active only when BOTH:
        - Condition A: conflict has reached her professionally or domestically
          in an unavoidable way.
        - Condition B: she has calculated that acting costs less than not acting.

        Updates ``kavya.condition_a_met``, ``kavya.condition_b_met``, and
        ``kavya.is_active`` in place.

        Args:
            condition_a: Whether condition A is now met.
            condition_b: Whether condition B is now met.

        Returns:
            True if Kavya is now active; False otherwise.
        """
        state = self.state
        state.kavya.condition_a_met = condition_a
        state.kavya.condition_b_met = condition_b
        state.kavya.is_active = condition_a and condition_b
        return state.kavya.is_active

    def apply_meera_transformation(self, flag: WorldFlag) -> frozenset[str]:
        """RULE_07_MEERA_TRANSFORMATION: expand Meera's response set by flag lived.

        Adds ``flag`` to ``meera.flags_lived_through`` and unlocks the response
        types associated with it. Responses accumulate — nothing is removed once
        unlocked.

        Args:
            flag: The WorldFlag Meera has lived through.

        Returns:
            The updated ``meera.response_set`` (full accumulated set after this flag).
        """
        state = self.state
        meera = state.meera

        meera.flags_lived_through = meera.flags_lived_through | {flag}
        new_responses = _MEERA_FLAG_RESPONSES.get(flag, frozenset())
        meera.response_set = meera.response_set | new_responses

        return meera.response_set

    def apply_rajan_constant(self, trigger: Trigger) -> bool:
        """RULE_08_RAJAN_CONSTANT: Rajan always shows up; escalates without direction.

        Any trigger → Rajan is present. Without direction from Vikram, he
        escalates to whatever the situation allows. With direction, he executes.

        Args:
            trigger: The trigger that Rajan is responding to.

        Returns:
            True if Rajan escalated (no direction from Vikram); False if directed.
        """
        return not self.state.rajan.has_direction_from_vikram

    def check_surya_reveal(
        self,
        phase: ConflictPhase,
        confronted: bool,
        operationally_necessary: bool,
    ) -> SuryaState | None:
        """RULE_09_SURYA_REVEAL_CONDITION: reveal Surya's true state if warranted.

        Reveal when ANY of:
        - A: ``phase`` is ``ConflictPhase.CRISIS``
        - B: ``confronted`` is True (direct private confrontation)
        - C: ``operationally_necessary`` is True

        Sets ``surya.is_revealed = True`` if revealed.

        Args:
            phase: Current ConflictPhase (typically ``self.state.conflict_phase``).
            confronted: True if Surya has been directly and privately confronted.
            operationally_necessary: True if the story requires Surya to act in a
                way that exposes his allegiance.

        Returns:
            Surya's ``SuryaState`` if revealed; None if conditions are not met.
        """
        state = self.state
        should_reveal = (
            phase is ConflictPhase.CRISIS or confronted or operationally_necessary
        )

        if should_reveal:
            state.surya.is_revealed = True
            return state.surya.true_state

        return None

    def check_arjun_window(self) -> bool:
        """RULE_10_ARJUN_WINDOW: determine if the Arjun window is open and active.

        Window opens when Ranveer reaches OBSESSED.
        Window closes when Ranveer reaches PERSONAL.
        Whether Arjun acts within the window is the per-story
        ``arjun_acts_in_window`` variable.

        Updates ``arjun.window_is_open`` in place.

        Returns:
            True if window is open AND ``arjun_acts_in_window`` is True.
        """
        state = self.state
        state.arjun.window_is_open = state.ranveer.phase is RanveerPhase.OBSESSED
        return state.arjun.window_is_open and state.arjun.arjun_acts_in_window

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _derive_consequence_notes(
        self,
        trigger: Trigger,
        multiplier: float,
        rajan_escalated: bool,
        arjun_window: bool,
    ) -> tuple[str, ...]:
        """Derive consequence notes for the incident log entry."""
        notes: list[str] = []

        if multiplier >= 2.0:
            notes.append(
                "Event witnessed by general student body — consequences doubled."
            )
        elif multiplier >= 1.5:
            notes.append(
                "Event witnessed by partial audience — consequences amplified."
            )

        if rajan_escalated:
            notes.append("Rajan escalated without direction from Vikram.")

        if arjun_window:
            notes.append("Arjun window is open — Arjun may act this step.")

        state = self.state
        if state.ranveer.phase is RanveerPhase.PERSONAL:
            notes.append("Ranveer is operating from obsession, not strategy.")

        if state.dhruv.drift_state is not DhruvDriftState.PRESENT:
            notes.append(f"Dhruv drift state: {state.dhruv.drift_state.name}.")

        return tuple(notes)

    def _handle_direct_challenge(self, trigger: Trigger, multiplier: float) -> None:
        """Apply chain rules for TYPE 01 DIRECT_CHALLENGE triggers.

        Fires the pride ratchet (Vikram's PRIDE means he never breaks).
        Activates Karan on physical confrontations.
        Advances conflict phase toward OPEN_CONFLICT.
        """
        state = self.state

        # Pride ratchet — Vikram never submits (CoreTrait.PRIDE)
        self.apply_pride_ratchet(trigger)

        # Physical confrontation always activates Karan
        if trigger.variant is TriggerVariant.DIRECT_CHALLENGE_PHYSICAL_CONFRONTATION:
            state.karan.is_activated = True

        # Conflict phase advancement: fastest escalation path
        if state.conflict_phase is ConflictPhase.COLD_EQUILIBRIUM:
            state.conflict_phase = ConflictPhase.FRICTION
        elif state.conflict_phase is ConflictPhase.FRICTION and trigger.is_public:
            state.conflict_phase = ConflictPhase.OPEN_CONFLICT

    def _handle_institutional_move(self, trigger: Trigger, multiplier: float) -> None:
        """Apply chain rules for TYPE 02 INSTITUTIONAL_MOVE triggers.

        Institutional moves do not automatically advance the conflict phase —
        their effect is slower and delivered through Dhruv's drift and Neel's
        capacity (both handled in ``fire_trigger``).

        Kavya threshold updates are the caller's responsibility via
        ``apply_kavya_threshold()``, since the engine cannot infer whether
        a specific institutional move has reached her professionally.
        """

    def _handle_political_move(self, trigger: Trigger, multiplier: float) -> None:
        """Apply chain rules for TYPE 03 POLITICAL_MOVE triggers.

        Political moves work through Neel's machinery. State effects arrive
        via Dhruv drift and Neel threshold (both handled in ``fire_trigger``).
        No immediate conflict phase change.
        """

    def _handle_ambient_trigger(self, trigger: Trigger, multiplier: float) -> None:
        """Apply chain rules for TYPE 04 AMBIENT_TRIGGER triggers.

        Meera intersection: no automatic state change — Vikram's response to
        Meera is undefined by the engine (RULE: undefined if Meera is involved).

        Kavya exposed: threshold update is the caller's responsibility.

        Surya information surface: reveal check is the caller's responsibility
        via ``check_surya_reveal()``.

        Gang member acts alone with Savar: bumps ``savar.visibility_level`` by 1,
        signalling something fracturing inside the gang (RULE_04_SAVAR_INVERSION).
        """
        state = self.state

        if trigger.variant is TriggerVariant.AMBIENT_GANG_MEMBER_ACTS_ALONE:
            if trigger.initiator == "savar":
                state.savar.visibility_level = min(state.savar.visibility_level + 1, 5)
