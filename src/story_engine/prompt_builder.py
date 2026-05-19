"""PromptBuilder: converts an enriched SceneBrief dict into LLM messages.

The previous version rendered the brief as labeled sections (CHARACTERS,
CONSTRAINTS, EMOTIONAL ARC) — a structured form. This caused the model
to write like it was filling out a form: mechanical, repetitive, checked.

This version renders the brief as a director's note:
  - What is happening in this specific scene
  - What the scene is doing beneath the surface
  - Physical specifics that anchor the prose
  - Memory context woven in naturally

The system prompt is short — establishes world and voice, then gets out
of the way. Rules in a system prompt become ceilings, not floors.

Import contract
~~~~~~~~~~~~~~~
prompt_builder.py has no imports from other engine modules.
inference.py imports PromptBuilder from here.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Caps on memory context (keeps prompt within token budget)
# ---------------------------------------------------------------------------

_MAX_PRIOR_SCENES = 3
_MAX_VOICE_SAMPLES = 2
_MAX_THREADS = 3


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# Short. Establishes world, central conflict, and voice.
# No rule lists — rules become ceilings.

_SYSTEM_PROMPT = """\
You are writing scenes for a literary novel set at a Hindi-medium Indian \
college campus. The central conflict is between Vikram (second year, \
CoreTrait: PRIDE — cannot submit, cannot be seen to acknowledge his place) \
and Ranveer (third year, CoreTrait: CRUEL AND CALCULATIVE — wants the \
authored humiliation, not just the outcome).

The campus is always present as witness. Hierarchy is physical: \
who sits where, who moves first, who does not move at all.

Voice: Close third person, Vikram's interiority — without his explanation \
of it. Dialogue is spare. Silence is load-bearing. No character explains \
their motivation. Time moves slower in confrontation.

Write the scene only. No title. No scene header. Begin mid-scene.\
"""


# ---------------------------------------------------------------------------
# Conflict phase and Ranveer phase — written as plain language
# ---------------------------------------------------------------------------

_CONFLICT_PHASE_NOTE: dict[str, str] = {
    "COLD_EQUILIBRIUM": (
        "The conflict has not broken open yet. Both sides know what is happening. "
        "Neither has moved visibly."
    ),
    "FRICTION":         (
        "The first moves have been made. Both sides have responded. "
        "No irreversible act yet — but the campus is reading the situation."
    ),
    "OPEN_CONFLICT":    (
        "The conflict is publicly known. Faculty are aware. "
        "Both sides are spending something."
    ),
    "CRISIS":           (
        "An irreversible move has been made. This cannot be absorbed "
        "into normal campus life."
    ),
    "PYRRHIC":          (
        "Both sides paid. Nobody won. The campus knows this even if no one says it."
    ),
    "RESOLUTION_ONE_SIDE_UP": (
        "A winner is visible to campus without anyone naming it. "
        "The silence is the acknowledgment."
    ),
}

_RANVEER_PHASE_NOTE: dict[str, str] = {
    "COLD":     "Vikram is an irregularity to Ranveer. Will be corrected.",
    "IRRITATED":"Vikram has become an irritant. Ranveer is paying attention now.",
    "OBSESSED": (
        "Ranveer is operating from personal obsession, not campus order. "
        "He would never say this. Everyone can see it."
    ),
    "PERSONAL": (
        "Vikram is the only thing on this campus that matters to Ranveer right now. "
        "The outcome no longer matters — the moment is everything."
    ),
}


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Converts an enriched SceneBrief dict into (system_prompt, user_prompt).

    The user prompt is a director's note: what is happening, what it means
    beneath the surface, who is present and what they are carrying.
    """

    def build(self, enriched: dict[str, Any]) -> tuple[str, str]:
        return _SYSTEM_PROMPT.strip(), self._build_user(enriched)

    # ------------------------------------------------------------------
    # User prompt
    # ------------------------------------------------------------------

    def _build_user(self, d: dict[str, Any]) -> str:
        parts: list[str] = []

        ws = d.get("world_state", {})
        loc = d.get("location", {})
        chars = d.get("characters_in_scene", [])
        goal = d.get("scene_goal", "")
        subtext = d.get("what_must_be_shown_not_told", [])
        prior = d.get("prior_context", [])
        must_not = d.get("what_must_not_happen", [])
        prose_notes = d.get("prose_notes", {})
        memory = d.get("memory", {})

        # --- Line 1: step context ---
        step_line = self._step_context(ws, loc)
        if step_line:
            parts.append(step_line)

        # --- Scene situation: what is literally happening ---
        situation = self._scene_situation(goal, loc, chars, ws)
        if situation:
            parts.append(situation)

        # --- Characters: who is here and what they are carrying ---
        char_block = self._character_block(chars, ws)
        if char_block:
            parts.append(char_block)

        # --- Beneath the surface: subtext instructions ---
        subtext_block = self._subtext_block(subtext)
        if subtext_block:
            parts.append(subtext_block)

        # --- What must not happen — folded into a single sentence ---
        constraint_note = self._constraint_note(must_not)
        if constraint_note:
            parts.append(constraint_note)

        # --- Prior context ---
        prior_block = self._prior_block(prior)
        if prior_block:
            parts.append(prior_block)

        # --- Memory: prior scenes + voice samples + threads ---
        memory_block = self._memory_block(memory)
        if memory_block:
            parts.append(memory_block)

        # --- Register note (only if pressurised) ---
        register = prose_notes.get("prose_register", "")
        if register == "pressurised":
            parts.append(
                "The register is pressurised — every exchange carries weight. "
                "Do not let the scene relax."
            )

        parts.append("Write the scene.")

        return "\n\n".join(p for p in parts if p.strip())

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _step_context(self, ws: dict, loc: dict) -> str:
        """One-line orientation: location, conflict phase, time."""
        loc_name   = loc.get("name", "")
        time_name  = ws.get("time_of_day", "").lower().replace("_", " ")
        phase      = ws.get("conflict_phase", "")
        ranveer_ph = ws.get("ranveer_phase", "")
        flags      = ws.get("active_flags", [])

        phase_note   = _CONFLICT_PHASE_NOTE.get(phase, "")
        ranveer_note = _RANVEER_PHASE_NOTE.get(ranveer_ph, "")

        lines = [f"{loc_name} — {time_name}."]
        if phase_note:
            lines.append(phase_note)
        if ranveer_note:
            lines.append(f"Ranveer: {ranveer_note}")
        if ws.get("flag_texture_note"):
            lines.append(ws["flag_texture_note"])

        return "\n".join(lines)

    def _scene_situation(
        self, goal: str, loc: dict, chars: list, ws: dict
    ) -> str:
        """The scene goal rewritten as what is literally happening."""
        if not goal:
            return ""

        # The goal from brief_generator is already a directive sentence.
        # Prepend the physical anchor: who is present and where.
        present = loc.get("who_is_present", [])
        control = loc.get("control", "")

        lines = []

        # Physical setting
        if present:
            present_str = ", ".join(p.title() for p in present)
            lines.append(f"Present: {present_str}.")

        if control:
            lines.append(f"This space is {control.lower().replace('_', ' ')} territory.")

        lines.append("")
        lines.append(goal)

        return "\n".join(lines)

    def _character_block(self, chars: list, ws: dict) -> str:
        """What each character is carrying into this scene — written naturally."""
        if not chars:
            return ""

        lines = ["Who is here and what they are carrying:"]
        lines.append("")

        for c in chars:
            name = c.get("name", "?")
            want = c.get("want", "")
            current = c.get("current_state", "")
            must_not = c.get("must_not_do", "")

            # Write as one or two plain sentences, not as labeled fields
            char_line = f"{name.title()}:"

            if want:
                char_line += f" {want}."
            if must_not:
                char_line += f" Will not: {must_not.lower()}."

            lines.append(char_line)

        return "\n".join(lines)

    def _subtext_block(self, instructions: list[str]) -> str:
        """Show-not-tell instructions written as observations, not rules."""
        if not instructions:
            return ""

        lines = ["What must be visible without being stated:"]
        for item in instructions:
            lines.append(f"— {item}")
        return "\n".join(lines)

    def _constraint_note(self, constraints: list[str]) -> str:
        """Hard constraints folded into a brief note rather than a list."""
        if not constraints:
            return ""
        # Take the most important ones (first 3) and write them as a note
        key = constraints[:3]
        return "Hard constraints: " + " / ".join(k.rstrip(".") for k in key) + "."

    def _prior_block(self, prior: list[str]) -> str:
        """What happened before this scene."""
        real = [
            p for p in prior
            if p and "No prior incidents" not in p
        ]
        if not real:
            return ""

        lines = ["What came before:"]
        for item in real:
            lines.append(f"— {item}")
        return "\n".join(lines)

    def _memory_block(self, memory: dict) -> str:
        """Prior scenes and voice samples woven into natural prose direction."""
        if not memory:
            return ""

        parts: list[str] = []

        # Prior scenes
        prior = memory.get("relevant_prior_scenes", [])
        sentinels = {"No prior scenes on record.", "No relevant prior scenes found."}
        useful = [s for s in prior if s not in sentinels]
        if useful:
            parts.append("From memory — scenes that bear on this one:")
            for s in useful[:_MAX_PRIOR_SCENES]:
                parts.append(f"  {s}")

        # Voice samples — anchor the model to how these characters have been written
        voice = memory.get("character_voice_samples", {})
        if voice:
            parts.append("How these characters have been written before:")
            for char, samples in voice.items():
                for s in samples[:_MAX_VOICE_SAMPLES]:
                    if s.get("dialogue"):
                        parts.append(f'  {char.title()} said: "{s["dialogue"]}"')
                    if s.get("behavior"):
                        parts.append(f"  {char.title()} did: {s['behavior']}")

        # Open threads
        threads = memory.get("open_narrative_threads", [])
        if threads:
            parts.append("Narrative threads still open:")
            for t in threads[:_MAX_THREADS]:
                parts.append(f"  — {t}")

        return "\n".join(parts) if parts else ""