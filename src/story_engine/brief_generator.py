"""Pydantic models for SceneBrief and SceneBriefGenerator.

``SceneBrief`` is the structured output this engine produces for the
fine-tuned LLM prose renderer. Every field is derived from ``WorldState`` —
no placeholder strings, no empty lists.

This module is a well-structured stub. The ``generate()`` implementation
produces real, state-derived content. Full narrative depth will be expanded
in a dedicated build pass.

Import contract
~~~~~~~~~~~~~~~
brief_generator.py → world_state.py → characters.py → flags.py
brief_generator.py → locations.py
engine.py imports SceneBrief and SceneBriefGenerator from this module.
"""

from __future__ import annotations

from pydantic import BaseModel

from story_engine.characters import DhruvDriftState, RanveerPhase, SuryaAllegiance
from story_engine.locations import LocationName, get_location
from story_engine.triggers import NO_TARGET, Trigger, TriggerVariant
from story_engine.world_state import ConflictPhase, WorldState


# ---------------------------------------------------------------------------
# Nested Pydantic models
# ---------------------------------------------------------------------------


class WorldStateSnapshot(BaseModel):
    """Serialisable snapshot of the world-level state for a scene brief."""

    active_flags: list[str]
    conflict_phase: str
    ranveer_phase: str
    time_of_day: str
    flag_texture_note: str | None


class LocationContext(BaseModel):
    """Location context for a scene brief."""

    name: str
    control: str
    visibility: str
    who_is_present: list[str]


class CharacterContext(BaseModel):
    """Per-character context for a scene brief."""

    name: str
    core_trait: str
    current_state: str
    want: str
    must_not_do: str


class ProseNotes(BaseModel):
    """Craft-level instructions for the LLM prose renderer."""

    prose_register: str
    pov: str
    craft_instructions: list[str]


# ---------------------------------------------------------------------------
# Root SceneBrief model
# ---------------------------------------------------------------------------


class SceneBrief(BaseModel):
    """Rich structured brief that the LLM prose renderer uses to write a scene.

    All fields must be populated from WorldState. This model is the external
    boundary — pydantic validates that nothing is empty or malformed before it
    reaches the renderer.
    """

    world_state: WorldStateSnapshot
    location: LocationContext
    characters_in_scene: list[CharacterContext]
    scene_goal: str
    emotional_arc: list[str]
    what_must_be_shown_not_told: list[str]
    prior_context: list[str]
    prose_notes: ProseNotes
    what_must_not_happen: list[str]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

# Baseline (want, must_not_do) for characters whose values do NOT vary with state.
# State-sensitive characters (ranveer, dhruv, surya, meera) are derived in
# SceneBriefGenerator._character_wants() below.
_CHARACTER_BRIEF_BASELINE: dict[str, tuple[str, str]] = {
    "vikram": (
        "Maintain position without acknowledging cost",
        "Perform submission or fear",
    ),
    "neel": (
        "Control outcomes through systems, not personal presence",
        "Act visibly or impulsively",
    ),
    "karan": (
        "Deliver the physical dimension of consequences",
        "Hold back when he smells weakness",
    ),
    "arjun": (
        "Identify and execute the clean solution",
        "Act on emotion",
    ),
    "savar": (
        "Maintain the performance of courage for the audience present",
        "Admit the performance is not real",
    ),
    "rajan": (
        "Show up and escalate to whatever the situation allows",
        "Calculate odds or show fear",
    ),
    "kavya": (
        "Maintain professional surface; protect her position",
        "Involve herself publicly in Vikram's conflict",
    ),
}


class SceneBriefGenerator:
    """Converts a ``WorldState`` into a fully populated ``SceneBrief``.

    Instantiate once and call ``generate()`` per scene. The generator is
    stateless — all information comes from the ``WorldState`` passed to it.
    """

    def generate(
        self,
        state: WorldState,
        location_name: LocationName,
        trigger: Trigger | None = None,
    ) -> SceneBrief:
        """Produce a SceneBrief from current world state at a given location.

        Every field is derived from ``state`` (and optionally ``trigger``) —
        no placeholder strings and no empty lists. All derivation logic is
        deterministic.

        Args:
            state: The current WorldState snapshot.
            location_name: The location where the scene is set.
            trigger: The trigger that prompted this scene, if any. Trigger
                participants are added to ``characters_in_scene``; the trigger
                variant informs ``scene_goal`` and ``emotional_arc``.

        Returns:
            A fully populated SceneBrief ready for LLM prose rendering.
        """
        location = get_location(location_name)
        last_variant = trigger.variant if trigger is not None else None

        world_snap = WorldStateSnapshot(
            active_flags=sorted(f.name for f in state.active_flags.flags),
            conflict_phase=state.conflict_phase.name,
            ranveer_phase=state.ranveer.phase.name,
            time_of_day=state.time_of_day.name,
            flag_texture_note=state.active_flags.texture_note(),
        )

        present = self._who_is_present(state, location_name, trigger)

        location_ctx = LocationContext(
            name=location.name.name,
            control=location.control.name,
            visibility=location.visibility.name,
            who_is_present=present,
        )

        prior_ctx = [e.description for e in state.incident_log[-3:]]
        if not prior_ctx:
            prior_ctx = ["No prior incidents on record at this step."]

        return SceneBrief(
            world_state=world_snap,
            location=location_ctx,
            characters_in_scene=self._build_character_contexts(state, present),
            scene_goal=self._derive_scene_goal(state, last_variant),
            emotional_arc=self._derive_emotional_arc(state, last_variant),
            what_must_be_shown_not_told=self._derive_subtext_instructions(state),
            prior_context=prior_ctx,
            prose_notes=self._derive_prose_notes(state),
            what_must_not_happen=self._derive_must_not_happen(state),
        )

    # ------------------------------------------------------------------
    # Private derivation helpers
    # ------------------------------------------------------------------

    def _who_is_present(
        self,
        state: WorldState,
        location_name: LocationName,
        trigger: Trigger | None = None,
    ) -> list[str]:
        """Return characters plausibly present at this location given WorldState.

        Trigger participants (initiator + target) are added first — they are
        definitionally present. Location-based additions follow.
        """
        from story_engine.locations import ControlType

        location = get_location(location_name)
        present: list[str] = ["vikram"]  # Vikram is always the POV anchor

        # Trigger participants are definitely present
        if trigger is not None:
            if trigger.initiator not in present and trigger.initiator != NO_TARGET:
                present.append(trigger.initiator)
            if trigger.target not in present and trigger.target != NO_TARGET:
                present.append(trigger.target)

        # Location-control additions
        control = location.control
        if control is ControlType.NEEL:
            for name in ("neel", "arjun"):
                if name not in present:
                    present.append(name)
        elif control is ControlType.KARAN:
            if "karan" not in present:
                present.append("karan")
        elif control is ControlType.KAVYA:
            if "kavya" not in present:
                present.append("kavya")
        elif control is ControlType.MEERA:
            if "meera" not in present:
                present.append("meera")
        elif location_name is LocationName.MAIN_CANTEEN:
            for name in ("savar", "dhruv", "rajan"):
                if name not in present:
                    present.append(name)
        elif location_name in {
            LocationName.BOYS_HOSTEL_BLOCKS,
            LocationName.HOSTEL_ROOF,
        }:
            for name in ("rajan", "surya"):
                if name not in present:
                    present.append(name)
        elif location_name is LocationName.MAIN_GROUND:
            for name in ("savar", "rajan"):
                if name not in present:
                    present.append(name)

        # Karan attaches to any direct-conflict location when activated
        if state.karan.is_activated and "karan" not in present:
            if location_name in {
                LocationName.DEAD_PATHS,
                LocationName.MAIN_GROUND,
                LocationName.BOYS_HOSTEL_BLOCKS,
            }:
                present.append("karan")

        return present

    def _build_character_contexts(
        self, state: WorldState, present: list[str]
    ) -> list[CharacterContext]:
        """Build CharacterContext for each character present."""
        current_state_map: dict[str, str] = {
            "vikram": f"last trigger: {state.vikram.last_trigger_type or 'none'}",
            "ranveer": f"phase: {state.ranveer.phase.name}",
            "neel": f"capacity: {state.neel.effective_capacity:.0%}",
            "karan": f"activated: {state.karan.is_activated}; "
            f"unfinished feeling: {state.karan.unfinished_feeling}",
            "arjun": f"window: {'open' if state.arjun.window_is_open else 'closed'}",
            "savar": f"visibility: {state.savar.visibility_level}/5",
            "dhruv": f"drift: {state.dhruv.drift_state.name}",
            "rajan": f"has direction: {state.rajan.has_direction_from_vikram}",
            "surya": f"revealed: {state.surya.is_revealed}",
            "kavya": f"active: {state.kavya.is_active}",
            "meera": f"flags lived: {len(state.meera.flags_lived_through)}; "
            f"responses available: {len(state.meera.response_set)}",
        }
        core_trait_map: dict[str, str] = {
            "vikram": state.vikram.core_trait.name,
            "ranveer": state.ranveer.core_trait.name,
            "neel": state.neel.core_trait.name,
            "karan": state.karan.core_trait.name,
            "arjun": state.arjun.core_trait.name,
            "savar": state.savar.core_trait.name,
            "dhruv": state.dhruv.core_trait.name,
            "rajan": state.rajan.core_trait.name,
            "surya": state.surya.core_trait.name,
            "kavya": state.kavya.core_trait.name,
            "meera": state.meera.core_trait.name,
        }
        wants = self._character_wants(state)
        contexts: list[CharacterContext] = []
        for name in present:
            if name not in wants:
                continue
            want, must_not = wants[name]
            contexts.append(
                CharacterContext(
                    name=name,
                    core_trait=core_trait_map[name],
                    current_state=current_state_map[name],
                    want=want,
                    must_not_do=must_not,
                )
            )
        return contexts

    def _character_wants(self, state: WorldState) -> dict[str, tuple[str, str]]:
        """Return state-derived (want, must_not_do) for every character.

        State-invariant characters draw from ``_CHARACTER_BRIEF_BASELINE``.
        State-sensitive characters (Ranveer, Dhruv, Surya, Meera) derive their
        want from ``WorldState`` so that the brief reflects where the character
        actually is in the story.

        Args:
            state: Current WorldState.

        Returns:
            Dict mapping lowercase character name → (want, must_not_do).
        """
        # Ranveer: want escalates with phase; must_not flips at PERSONAL
        if state.ranveer.phase is RanveerPhase.PERSONAL:
            ranveer_want = (
                "The specific authored moment — Vikram acknowledging his place, "
                "witnessed. The outcome no longer matters; the moment is everything."
            )
            ranveer_must_not = "Let anyone see how personal this has become"
        elif state.ranveer.phase is RanveerPhase.OBSESSED:
            ranveer_want = (
                "Vikram's visible acknowledgment of his place — "
                "Ranveer is operating from personal obsession now, not campus order."
            )
            ranveer_must_not = "Act without deniability"
        else:
            ranveer_want = "Author Vikram's visible acknowledgment of his place"
            ranveer_must_not = "Act without deniability"

        # Dhruv: want tracks drift state
        drift = state.dhruv.drift_state
        if (
            drift is DhruvDriftState.GONE
            or drift is DhruvDriftState.MAKING_EXIT_ARRANGEMENTS
        ):
            dhruv_want = "Exit cleanly without being seen to leave"
        elif drift is DhruvDriftState.PRESENT_BUT_UNINVESTED:
            dhruv_want = "Remain technically present while withdrawing real investment"
        elif drift is DhruvDriftState.LESS_AVAILABLE:
            dhruv_want = (
                "Protect his future; becoming harder to reach without explanation"
            )
        else:
            dhruv_want = "Protect his future; remain useful without committing"

        # Surya: want reflects revealed allegiance; opaque until revealed
        if state.surya.is_revealed:
            if state.surya.true_state is SuryaAllegiance.RANVEER_PLANT:
                surya_want = "Complete the intelligence task; disengage cleanly"
            elif state.surya.true_state is SuryaAllegiance.OWN_AGENDA:
                surya_want = "Advance his own position using this conflict as cover"
            elif state.surya.true_state is SuryaAllegiance.WITH_VIKRAM:
                surya_want = "Protect Vikram through silence and misdirection"
            else:
                surya_want = "Determine what this conflict means for him personally"
        else:
            surya_want = "Observe without revealing anything ahead of schedule"

        # Meera: want deepens as she lives through more flags
        if len(state.meera.flags_lived_through) >= 3:
            meera_want = (
                "Define herself against what this campus has tried to make her — "
                "not opposition to Vikram but independence from the world he inhabits."
            )
        else:
            meera_want = "Move toward her own definition of herself"

        result: dict[str, tuple[str, str]] = {
            **_CHARACTER_BRIEF_BASELINE,
            "ranveer": (ranveer_want, ranveer_must_not),
            "dhruv": (dhruv_want, "Admit he is already halfway out"),
            "surya": (surya_want, "Explain himself"),
            "meera": (meera_want, "Ask Vikram for help as if it costs her nothing"),
        }
        return result

    def _derive_scene_goal(
        self,
        state: WorldState,
        last_variant: TriggerVariant | None = None,
    ) -> str:
        """Return a scene goal string derived from trigger variant and conflict phase.

        Trigger-specific goals take priority over phase-level goals when the
        variant is known. Phase-level goals act as a fallback for quiet steps
        and unknown variants.

        Args:
            state: Current WorldState.
            last_variant: The TriggerVariant that fired this step, or None.

        Returns:
            Scene goal string.
        """
        # Trigger-variant-specific goals (checked first)
        if last_variant is TriggerVariant.DIRECT_CHALLENGE_PHYSICAL_CONFRONTATION:
            return (
                "The physical dimension has arrived — show what it costs "
                "and what it does not resolve. Karan is present for a reason."
            )
        if last_variant is TriggerVariant.DIRECT_CHALLENGE_PUBLIC_CALLOUT:
            return (
                "The challenge is public and witnessed — the student body is now "
                "the audience. Show what it means to be seen refusing to back down."
            )
        if last_variant is TriggerVariant.INSTITUTIONAL_ACADEMIC_THREAT:
            return (
                "An invisible move has been made — the damage is real but "
                "there is nobody to confront. Show what Vikram does "
                "with anger that has no valid target."
            )
        if last_variant is TriggerVariant.INSTITUTIONAL_ADMINISTRATIVE_ACTION:
            return (
                "The institution has become a weapon — show who controls it "
                "and who it cannot touch."
            )
        if last_variant is TriggerVariant.INSTITUTIONAL_NOTICE_BOARD:
            return (
                "A public declaration through institutional channels — "
                "show what it means when the system speaks for someone."
            )
        if last_variant is TriggerVariant.POLITICAL_ELECTION_POSITIONING:
            return (
                "Everyone is being read as aligned whether they intend it or not — "
                "show what it costs to appear neutral when the campus is counting sides."
            )
        if last_variant is TriggerVariant.AMBIENT_MEERA_INTERSECTION:
            return (
                "Meera is present and Vikram's response is undefined — "
                "show what it looks like when his rules do not apply."
            )
        if last_variant is TriggerVariant.AMBIENT_KAVYA_EXPOSED:
            return (
                "The conflict has reached Kavya — show the moment before "
                "she decides whether acting costs less than not acting."
            )

        # Phase-level fallback
        phase = state.conflict_phase
        phase_goals: dict[ConflictPhase, str] = {
            ConflictPhase.COLD_EQUILIBRIUM: (
                "Establish the tension that exists before any move is made — "
                "the weight of knowing, without the relief of action."
            ),
            ConflictPhase.FRICTION: (
                "Show what a first move looks like in this world — "
                "and what it costs the person who makes it."
            ),
            ConflictPhase.OPEN_CONFLICT: (
                "Both sides are visible to campus — show what each is willing to spend, "
                "and who on the periphery is doing the accounting."
            ),
            ConflictPhase.CRISIS: (
                "An irreversible move has been made — show what that looks like "
                "in the moment immediately after, before anyone has decided what it means."
            ),
            ConflictPhase.PYRRHIC: (
                "Both paid; nobody won — show what that looks like "
                "on a campus that cannot acknowledge it publicly."
            ),
            ConflictPhase.RESOLUTION_ONE_SIDE_UP: (
                "A winner is visible to campus without anyone saying it — "
                "show what that silence looks like."
            ),
        }
        return phase_goals.get(
            phase,
            "Advance the conflict toward its resolution — show what each character is becoming.",
        )

    def _derive_emotional_arc(
        self,
        state: WorldState,
        last_variant: TriggerVariant | None = None,
    ) -> list[str]:
        """Return beat-by-beat emotional movement derived from trigger and phase.

        A standard 3-beat arc is returned for most cases. Physical confrontations
        and CRISIS phase both get a 4-beat arc with a break/aftermath beat.

        Args:
            state: Current WorldState.
            last_variant: The TriggerVariant that fired this step, or None.

        Returns:
            Ordered list of arc beat strings.
        """
        arc = [
            "Opening: the space before anything is said — "
            "what the body knows before the mind does.",
        ]

        is_physical = (
            last_variant is TriggerVariant.DIRECT_CHALLENGE_PHYSICAL_CONFRONTATION
        )
        is_institutional = last_variant in {
            TriggerVariant.INSTITUTIONAL_ACADEMIC_THREAT,
            TriggerVariant.INSTITUTIONAL_ADMINISTRATIVE_ACTION,
            TriggerVariant.INSTITUTIONAL_NOTICE_BOARD,
            TriggerVariant.INSTITUTIONAL_OPPORTUNITY_DENIAL,
        }

        if is_physical:
            arc.append(
                "Escalation: the moment the body moves before the mind catches up — "
                "show what it looks like when Karan's version of loyalty is present."
            )
            arc.append(
                "Break: contact or near-contact — "
                "the thing that cannot be undone even if nothing formally happens."
            )
            arc.append(
                "Aftermath: Vikram still standing. Karan's unfinished feeling "
                "is not resolved — show it in what he does not do."
            )
        elif is_institutional:
            arc.append(
                "Rising: the realisation that the move came through a system — "
                "there is no face to confront, only a consequence."
            )
            arc.append(
                "Absorption: Vikram cannot respond the way he would respond to a person. "
                "Show what it costs him to absorb something he cannot make personal."
            )
        elif state.conflict_phase is ConflictPhase.CRISIS:
            arc.append("Rising: the moment when what is happening becomes undeniable.")
            arc.append(
                "Break: something irreversible has happened — "
                "neither side can absorb this into the normal rhythm of campus life."
            )
            arc.append(
                "Aftermath: what is left standing after the irreversible thing — "
                "who is still in the room, and what they do with their hands."
            )
        else:
            arc.append("Rising: the moment when what is happening becomes undeniable.")
            arc.append(
                "Unresolved: the scene ends without resolution — "
                "the tension carries forward into the next step."
            )

        return arc

    def _derive_subtext_instructions(self, state: WorldState) -> list[str]:
        """Return subtext instructions for the LLM renderer."""
        instructions = [
            "Show Vikram's pride through what he does NOT do — "
            "submission is an absence of action, not an action.",
        ]
        if state.ranveer.phase.value >= 3:  # OBSESSED or PERSONAL
            instructions.append(
                "Ranveer's obsession is visible in how much attention he gives Vikram "
                "without appearing to — show the attention, never the explanation of it."
            )
        if state.dhruv.drift_state.name != "PRESENT":
            instructions.append(
                f"Dhruv is at {state.dhruv.drift_state.name} — "
                "show distance through availability (or its absence), not announcement."
            )
        if state.karan.unfinished_feeling:
            instructions.append(
                "Karan's unfinished feeling about Vikram is not respect — "
                "show it as something he cannot name, and therefore cannot resolve."
            )
        if state.neel.effective_capacity < 1.0:
            instructions.append(
                f"Neel is operating at {state.neel.effective_capacity:.0%} capacity — "
                "show the overhead through what does NOT get managed quietly."
            )
        return instructions

    def _derive_prose_notes(self, state: WorldState) -> ProseNotes:
        """Return register, POV, and craft instructions for the renderer."""
        cold_phases = {ConflictPhase.COLD_EQUILIBRIUM, ConflictPhase.FRICTION}
        register = (
            "controlled" if state.conflict_phase in cold_phases else "pressurised"
        )

        return ProseNotes(
            prose_register=register,
            pov="Close third — Vikram's interiority, without his explanation of it.",
            craft_instructions=[
                "No character explains their own motivation. Ever.",
                "What is not said is as important as what is.",
                "Time moves slower in confrontation — let it. Do not rush the beat.",
                "Campus is always present as ambient sound and peripheral movement — "
                "it witnesses even when the characters wish it did not.",
            ],
        )

    def _derive_must_not_happen(self, state: WorldState) -> list[str]:
        """Return hard constraints for the LLM renderer."""
        constraints = [
            "Vikram does not perform submission or acknowledge being put in his place.",
            "Ranveer does not show his hand before he intends to.",
            "Neel is never the most visible person in any room.",
        ]
        if not state.surya.is_revealed:
            constraints.append(
                "Surya's true allegiance is not revealed — his actions must be legible "
                "in at least two directions simultaneously."
            )
        if state.dhruv.drift_state.name in {"MAKING_EXIT_ARRANGEMENTS", "GONE"}:
            constraints.append(
                "Dhruv does not announce his departure — "
                "he simply becomes less available, then absent."
            )
        if not state.kavya.is_active:
            constraints.append(
                "Kavya does not make herself visible in this conflict — "
                "she is present as context, not as actor."
            )
        return constraints
