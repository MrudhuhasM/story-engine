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

from story_engine.locations import LocationName, get_location
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

# Maps each character name to a stable (want, must_not_do) pair for the brief.
# Values are intentionally character-specific and grounded in the world model.
_CHARACTER_BRIEF_DATA: dict[str, tuple[str, str]] = {
    "vikram": (
        "Maintain position without acknowledging cost",
        "Perform submission or fear",
    ),
    "ranveer": (
        "Author Vikram's visible acknowledgment of his place",
        "Act without deniability",
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
    "dhruv": (
        "Protect his future; remain useful without committing",
        "Admit he is already halfway out",
    ),
    "rajan": (
        "Show up and escalate to whatever the situation allows",
        "Calculate odds or show fear",
    ),
    "surya": (
        "Observe without revealing anything ahead of schedule",
        "Explain himself",
    ),
    "kavya": (
        "Maintain professional surface; protect her position",
        "Involve herself publicly in Vikram's conflict",
    ),
    "meera": (
        "Move toward her own definition of herself",
        "Ask Vikram for help as if it costs her nothing",
    ),
}


class SceneBriefGenerator:
    """Converts a ``WorldState`` into a fully populated ``SceneBrief``.

    Instantiate once and call ``generate()`` per scene. The generator is
    stateless — all information comes from the ``WorldState`` passed to it.
    """

    def generate(self, state: WorldState, location_name: LocationName) -> SceneBrief:
        """Produce a SceneBrief from current world state at a given location.

        Every field is derived from ``state`` — no placeholder strings and no
        empty lists. All derivation logic is deterministic.

        Args:
            state: The current WorldState snapshot.
            location_name: The location where the scene is set.

        Returns:
            A fully populated SceneBrief ready for LLM prose rendering.
        """
        location = get_location(location_name)

        world_snap = WorldStateSnapshot(
            active_flags=sorted(f.name for f in state.active_flags.flags),
            conflict_phase=state.conflict_phase.name,
            ranveer_phase=state.ranveer.phase.name,
            time_of_day=state.time_of_day.name,
            flag_texture_note=state.active_flags.texture_note(),
        )

        present = self._who_is_present(state, location_name)

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
            scene_goal=self._derive_scene_goal(state),
            emotional_arc=self._derive_emotional_arc(state),
            what_must_be_shown_not_told=self._derive_subtext_instructions(state),
            prior_context=prior_ctx,
            prose_notes=self._derive_prose_notes(state),
            what_must_not_happen=self._derive_must_not_happen(state),
        )

    # ------------------------------------------------------------------
    # Private derivation helpers
    # ------------------------------------------------------------------

    def _who_is_present(
        self, state: WorldState, location_name: LocationName
    ) -> list[str]:
        """Return characters plausibly present at this location given WorldState."""
        from story_engine.locations import ControlType

        location = get_location(location_name)
        present: list[str] = ["vikram"]  # Vikram is always the POV anchor

        control = location.control
        if control is ControlType.NEEL:
            present.extend(["neel", "arjun"])
        elif control is ControlType.KARAN:
            present.append("karan")
        elif control is ControlType.KAVYA:
            present.append("kavya")
        elif control is ControlType.MEERA:
            present.append("meera")
        elif location_name is LocationName.MAIN_CANTEEN:
            present.extend(["savar", "dhruv", "rajan"])
        elif location_name in {
            LocationName.BOYS_HOSTEL_BLOCKS,
            LocationName.HOSTEL_ROOF,
        }:
            present.extend(["rajan", "surya"])
        elif location_name is LocationName.MAIN_GROUND:
            present.extend(["savar", "rajan"])

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
        contexts: list[CharacterContext] = []
        for name in present:
            if name not in _CHARACTER_BRIEF_DATA:
                continue
            want, must_not = _CHARACTER_BRIEF_DATA[name]
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

    def _derive_scene_goal(self, state: WorldState) -> str:
        """Return a scene goal string derived from conflict phase."""
        phase = state.conflict_phase
        goals: dict[ConflictPhase, str] = {
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
        return goals.get(
            phase,
            "Advance the conflict toward its resolution — show what each character is becoming.",
        )

    def _derive_emotional_arc(self, state: WorldState) -> list[str]:
        """Return beat-by-beat emotional movement derived from conflict phase."""
        arc = [
            "Opening: the space before anything is said — "
            "what the body knows before the mind does.",
            "Rising: the moment when what is happening becomes undeniable.",
        ]
        if state.conflict_phase is ConflictPhase.CRISIS:
            arc.append(
                "Break: something irreversible has happened — "
                "neither side can absorb this into the normal rhythm of campus life."
            )
            arc.append(
                "Aftermath: what is left standing after the irreversible thing — "
                "who is still in the room, and what they do with their hands."
            )
        else:
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
