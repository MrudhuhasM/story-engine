"""PromptBuilder: converts an enriched SceneBrief dict into LLM messages.

The enriched dict comes from MemorySystem.enrich_brief(), which merges
a SceneBrief with memory context (prior scenes, voice samples, threads).

Produces two strings:
  system_prompt — role definition, register, hard craft constraints
  user_prompt   — the full scene brief rendered as structured text

Token budget target for an 8096-context server:
  System prompt:     ~400 tokens
  Scene brief:      ~1200 tokens
  Memory context:    ~700 tokens
  Output (prose):   ~1800 tokens
  Total:            ~4100 / 8096  (leaves headroom for longer scenes)

Import contract
~~~~~~~~~~~~~~~
prompt_builder.py has no imports from other engine modules.
inference.py imports PromptBuilder from here.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Retrieval caps (keeps memory section within token budget)
# ---------------------------------------------------------------------------

_MAX_PRIOR_SCENES = 3    # scene summaries to include from vector store
_MAX_VOICE_SAMPLES = 2   # samples per character
_MAX_THREADS = 4         # open narrative threads to surface


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

# This is sent as the system message on every generation call.
# It establishes the renderer role and hard craft constraints.
# Kept short to preserve context budget for the scene brief.

_SYSTEM_PROMPT = """\
You are a prose renderer for a literary fiction engine set at a Hindi-medium \
Indian college campus. Your task is to write one scene as polished literary prose.

You receive a structured scene brief: world state, location, characters, \
scene goal, emotional arc, subtext instructions, memory context. \
You render this as a scene. You do not summarise, plan, or explain. You write.

CRAFT CONSTRAINTS — NON-NEGOTIABLE

Point of view: Close third, Vikram's interiority, without his explanation of it.
Show not tell: Subtext instructions are hard constraints, not suggestions.
Silence: What is not said is as load-bearing as what is said. \
Dialogue is spare.
Motivation: No character explains their own motivation. Ever.
Time: Moves slower in confrontation. Do not rush the beat.
Campus: Always present as ambient sound and peripheral movement — \
it witnesses even when characters wish it did not.
Resolution: The scene ends without resolution unless the brief specifies \
otherwise. Carry the tension forward.

OUTPUT FORMAT

Prose only. No scene title. No chapter header. No authorial commentary. \
Begin mid-scene. End when the beat is complete.\
"""


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Converts an enriched SceneBrief dict into (system_prompt, user_prompt).

    The enriched dict is the output of ``MemorySystem.enrich_brief()``:
    a SceneBrief serialised to dict, with an additional ``"memory"`` key
    containing prior scenes, voice samples, and open threads.

    Usage::

        builder = PromptBuilder()
        system, user = builder.build(enriched_brief_dict)
        response = llm_client.generate(system, user)
    """

    def build(self, enriched: dict[str, Any]) -> tuple[str, str]:
        """Return ``(system_prompt, user_prompt)`` ready for ``LLMClient.generate()``.

        Args:
            enriched: Output of ``MemorySystem.enrich_brief()``.

        Returns:
            Two strings: system prompt and user prompt.
        """
        return _SYSTEM_PROMPT.strip(), self._build_user_prompt(enriched)

    # ------------------------------------------------------------------
    # User prompt assembly
    # ------------------------------------------------------------------

    def _build_user_prompt(self, d: dict[str, Any]) -> str:
        sections: list[str] = []

        sections.append(self._render_world_state(d.get("world_state", {})))
        sections.append(self._render_location(d.get("location", {})))
        sections.append(self._render_characters(d.get("characters_in_scene", [])))
        sections.append(self._render_scene_goal(d.get("scene_goal", "")))
        sections.append(self._render_emotional_arc(d.get("emotional_arc", [])))
        sections.append(self._render_subtext(d.get("what_must_be_shown_not_told", [])))
        sections.append(self._render_prior_context(d.get("prior_context", [])))
        sections.append(self._render_constraints(d.get("what_must_not_happen", [])))
        sections.append(self._render_prose_notes(d.get("prose_notes", {})))

        memory = d.get("memory", {})
        if memory:
            mem_section = self._render_memory(memory)
            if mem_section:
                sections.append(mem_section)

        # Filter empty sections, join with double newline
        return "\n\n".join(s for s in sections if s and s.strip())

    # ------------------------------------------------------------------
    # Section renderers — one per SceneBrief field
    # ------------------------------------------------------------------

    def _render_world_state(self, ws: dict[str, Any]) -> str:
        if not ws:
            return ""
        lines = ["WORLD STATE"]
        lines.append(
            f"Conflict: {ws.get('conflict_phase', '?')}  |  "
            f"Ranveer: {ws.get('ranveer_phase', '?')}  |  "
            f"Time: {ws.get('time_of_day', '?')}"
        )
        flags = ws.get("active_flags", [])
        if flags:
            lines.append(f"Flags: {', '.join(flags)}")
        note = ws.get("flag_texture_note")
        if note:
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def _render_location(self, loc: dict[str, Any]) -> str:
        if not loc:
            return ""
        lines = [f"LOCATION: {loc.get('name', '?')}"]
        lines.append(
            f"Control: {loc.get('control', '?')}  |  "
            f"Visibility: {loc.get('visibility', '?')}"
        )
        present = loc.get("who_is_present", [])
        if present:
            lines.append(f"Present: {', '.join(present)}")
        return "\n".join(lines)

    def _render_characters(self, chars: list[dict[str, Any]]) -> str:
        if not chars:
            return ""
        lines = ["CHARACTERS"]
        for c in chars:
            name = c.get("name", "?").upper()
            trait = c.get("core_trait", "?")
            lines.append(f"\n{name} [{trait}]")
            lines.append(f"  State: {c.get('current_state', '?')}")
            lines.append(f"  Want: {c.get('want', '?')}")
            lines.append(f"  Must not: {c.get('must_not_do', '?')}")
        return "\n".join(lines)

    def _render_scene_goal(self, goal: str) -> str:
        if not goal:
            return ""
        return f"SCENE GOAL\n{goal}"

    def _render_emotional_arc(self, arc: list[str]) -> str:
        if not arc:
            return ""
        lines = ["EMOTIONAL ARC"]
        for i, beat in enumerate(arc, 1):
            lines.append(f"{i}. {beat}")
        return "\n".join(lines)

    def _render_subtext(self, instructions: list[str]) -> str:
        if not instructions:
            return ""
        lines = ["SHOW NOT TELL"]
        for item in instructions:
            lines.append(f"- {item}")
        return "\n".join(lines)

    def _render_prior_context(self, context: list[str]) -> str:
        if not context:
            return ""
        lines = ["PRIOR CONTEXT"]
        for item in context:
            lines.append(f"- {item}")
        return "\n".join(lines)

    def _render_constraints(self, constraints: list[str]) -> str:
        if not constraints:
            return ""
        lines = ["CONSTRAINTS"]
        for item in constraints:
            lines.append(f"- {item}")
        return "\n".join(lines)

    def _render_prose_notes(self, notes: dict[str, Any]) -> str:
        if not notes:
            return ""
        lines = ["PROSE NOTES"]
        register = notes.get("prose_register")
        pov = notes.get("pov")
        craft = notes.get("craft_instructions", [])
        if register:
            lines.append(f"Register: {register}")
        if pov:
            lines.append(f"POV: {pov}")
        for instruction in craft:
            lines.append(f"- {instruction}")
        return "\n".join(lines)

    def _render_memory(self, memory: dict[str, Any]) -> str:
        """Render memory context section. Returns empty string if nothing useful."""
        sections: list[str] = []

        prior = memory.get("relevant_prior_scenes", [])
        sentinel = ["No prior scenes on record.", "No relevant prior scenes found."]
        useful_prior = [s for s in prior if s not in sentinel]
        if useful_prior:
            sections.append("Prior scenes:")
            for summary in useful_prior[:_MAX_PRIOR_SCENES]:
                # Indent each line of the summary for readability
                indented = "\n".join(f"  {line}" for line in summary.splitlines())
                sections.append(indented)

        voice = memory.get("character_voice_samples", {})
        if voice:
            sections.append("Voice samples (anchor your prose to these):")
            for char, samples in voice.items():
                if not samples:
                    continue
                sections.append(f"  {char.upper()}:")
                for s in samples[:_MAX_VOICE_SAMPLES]:
                    if s.get("dialogue"):
                        sections.append(f'    said: "{s["dialogue"]}"')
                    if s.get("behavior"):
                        sections.append(f"    did: {s['behavior']}")

        threads = memory.get("open_narrative_threads", [])
        if threads:
            sections.append("Open threads (honor or acknowledge these):")
            for t in threads[:_MAX_THREADS]:
                sections.append(f"  - {t}")

        if not sections:
            return ""

        return "MEMORY CONTEXT\n" + "\n".join(sections)