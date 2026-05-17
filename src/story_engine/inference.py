"""InferencePipeline — connects StoryEngine, MemorySystem, and LLMClient.

For each scene:
  1. Fires trigger through engine (chain rules apply)
  2. Generates SceneBrief from updated WorldState
  3. Retrieves memory context (prior scenes, voice samples, threads)
  4. Builds the LLM prompt via PromptBuilder
  5. Calls the LLM — returns prose + thinking
  6. Returns RenderedScene for caller to review before storing

store_scene() is a separate explicit call so you can read/edit the
summary before committing to memory. Step counter advances there.

auto_summarize() makes a focused low-temperature call to convert
rendered prose into a factual 2-4 sentence memory summary.

Import contract
~~~~~~~~~~~~~~~
inference.py → engine.py, memory.py, prompt_builder.py, llm_interface.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

from story_engine.engine import StoryEngine
from story_engine.brief_generator import SceneBrief
from story_engine.llm_interface import LLMClient, LLMResponse, StreamChunk
from story_engine.locations import LocationName
from story_engine.memory import MemoryInput, MemorySystem, NarrativeThread, VoiceSample
from story_engine.prompt_builder import PromptBuilder
from story_engine.triggers import Trigger
from story_engine.world_state import WorldState


# ---------------------------------------------------------------------------
# RenderedScene
# ---------------------------------------------------------------------------


@dataclass
class RenderedScene:
    """Complete output for one rendered scene.

    Args:
        step: Engine step counter at time of generation.
        trigger: The trigger fired at this step, or None.
        location_name: Location where the scene was set.
        brief: The SceneBrief used to generate the prose.
        prose: Rendered prose from the LLM.
        thinking: Content of the model's reasoning block, or None.
        tokens_used: Total tokens consumed, or None if not reported.
        world_state_snapshot: WorldState.to_dict() captured BEFORE
            advance_state() — reflects what the prose was generated from.
        enriched_brief: Full brief dict sent to LLM including memory context.
    """

    step: int
    trigger: Trigger | None
    location_name: LocationName
    brief: SceneBrief
    prose: str
    thinking: str | None
    tokens_used: int | None
    world_state_snapshot: dict[str, Any]
    enriched_brief: dict[str, Any]


# ---------------------------------------------------------------------------
# SceneInput
# ---------------------------------------------------------------------------


@dataclass
class SceneInput:
    """Specification for one scene to generate.

    Args:
        location: Where the scene is set.
        trigger: Trigger to fire before generating. None = quiet step.
        dhruv_event_cost: Override Dhruv's event cost. None = engine default.
        characters_for_memory: Override which characters are used for the
            memory retrieval query. Defaults to scene brief participants.
    """

    location: LocationName
    trigger: Trigger | None = None
    dhruv_event_cost: float | None = None
    characters_for_memory: list[str] | None = None


# ---------------------------------------------------------------------------
# InferencePipeline
# ---------------------------------------------------------------------------


class InferencePipeline:
    """Connects StoryEngine, MemorySystem, PromptBuilder, and LLMClient.

    Args:
        engine: Pre-initialised StoryEngine.
        memory: MemorySystem instance.
        llm: LLMClient. If None, a default client is constructed.
        max_tokens: Max tokens per scene (thinking + prose combined).
            3000 is recommended for a thinking model on 8096-context server.
        temperature: Sampling temperature for prose generation.
    """

    def __init__(
        self,
        engine: StoryEngine,
        memory: MemorySystem,
        llm: LLMClient | None = None,
        *,
        max_tokens: int = 3000,
        temperature: float = 0.88,
    ) -> None:
        self._engine = engine
        self._memory = memory
        self._llm = llm or LLMClient()
        self._builder = PromptBuilder()
        self._max_tokens = max_tokens
        self._temperature = temperature

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def run_scene(self, scene_input: SceneInput) -> RenderedScene:
        """Blocking generation. Does NOT store or advance step counter."""
        if scene_input.trigger is not None:
            self._engine.fire_trigger(
                scene_input.trigger,
                dhruv_event_cost=scene_input.dhruv_event_cost,
            )

        brief = self._engine.generate_scene_brief(
            scene_input.location, trigger=scene_input.trigger
        )
        enriched = self._build_enriched(brief, scene_input)
        system, user = self._builder.build(enriched)

        response = self._llm.generate(
            system, user,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        return RenderedScene(
            step=self._engine.state.step,
            trigger=scene_input.trigger,
            location_name=scene_input.location,
            brief=brief,
            prose=response.prose,
            thinking=response.thinking,
            tokens_used=response.tokens_used,
            world_state_snapshot=self._engine.get_current_state(),
            enriched_brief=enriched,
        )

    def run_scene_stream(
        self,
        scene_input: SceneInput,
    ) -> tuple[dict[str, Any], Iterator[StreamChunk]]:
        """Streaming generation. Returns (metadata, StreamChunk iterator).

        StreamChunk.is_thinking=True  → reasoning token (show dim)
        StreamChunk.is_thinking=False → prose token (show normal)

        The engine fires and the brief is generated synchronously before
        the stream starts. Only the LLM call is lazy.

        metadata keys: step, brief, enriched_brief, state_snapshot
        """
        if scene_input.trigger is not None:
            self._engine.fire_trigger(
                scene_input.trigger,
                dhruv_event_cost=scene_input.dhruv_event_cost,
            )

        brief = self._engine.generate_scene_brief(
            scene_input.location, trigger=scene_input.trigger
        )
        enriched = self._build_enriched(brief, scene_input)
        system, user = self._builder.build(enriched)

        metadata: dict[str, Any] = {
            "step": self._engine.state.step,
            "brief": brief,
            "enriched_brief": enriched,
            "state_snapshot": self._engine.get_current_state(),
        }

        stream = self._llm.generate_stream(
            system, user,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        return metadata, stream

    def store_scene(
        self,
        scene: RenderedScene,
        summary: str,
        *,
        voice_samples: list[VoiceSample] | None = None,
        threads_introduced: list[NarrativeThread] | None = None,
        threads_resolved: list[str] | None = None,
    ) -> None:
        """Store scene in memory and advance the engine step counter.

        The summary is embedded in the vector store and retrieved for future
        scenes. Write it factually: who did what, what was not said, what
        was established between characters.

        Always call this after run_scene() or run_scene_stream().
        """
        state = scene.world_state_snapshot

        self._memory.store_scene(
            MemoryInput(
                scene_id=f"step_{scene.step}_{scene.location_name.name}",
                step=scene.step,
                location_name=scene.location_name.name,
                characters_present=[c.name for c in scene.brief.characters_in_scene],
                active_flags=state.get("active_flags", []),
                conflict_phase=state.get("conflict_phase", ""),
                ranveer_phase=state.get("ranveer", {}).get("phase", ""),
                summary=summary,
                prose=scene.prose,
                voice_samples=voice_samples or [],
                threads_introduced=threads_introduced or [],
                threads_resolved=threads_resolved or [],
            )
        )
        self._engine.advance_state()

    # ------------------------------------------------------------------
    # Auto-summarize
    # ------------------------------------------------------------------

    def auto_summarize(self, prose: str) -> str:
        """Generate a factual memory summary from rendered prose.

        Makes a focused low-temperature LLM call. The summary is what
        gets embedded in the vector store — factual and specific, not
        a prose paraphrase.

        Returns the summary string. Returns a fallback if the call fails.

        Args:
            prose: The rendered scene prose to summarize.
        """
        system = (
            "You write factual scene summaries for a story memory system. "
            "2-4 sentences only. State what happened: who did what, "
            "what was not said, what was established between characters, "
            "what changed and what did not. "
            "No interpretation. No prose flourish. Pure facts of the scene. "
            "Do not start with 'In this scene' or 'The scene'."
        )
        user = f"Summarize this scene:\n\n{prose}"

        try:
            response = self._llm.generate(
                system, user,
                max_tokens=200,
                temperature=0.2,
                top_p=0.9,
            )
            return response.prose.strip()
        except Exception:
            # Fallback: use first 200 chars of prose
            return prose[:200].strip()

    # ------------------------------------------------------------------
    # Resolution / state helpers
    # ------------------------------------------------------------------

    def check_resolution(self) -> str | None:
        """Return resolution type name if conditions are met, else None."""
        result = self._engine.check_resolution_condition()
        return result.name if result is not None else None

    def get_state(self) -> dict[str, Any]:
        """Return current WorldState as a JSON-safe dict."""
        return self._engine.get_current_state()

    @property
    def engine(self) -> StoryEngine:
        return self._engine

    @property
    def memory(self) -> MemorySystem:
        return self._memory

    # ------------------------------------------------------------------
    # Optional: voice sample extraction
    # ------------------------------------------------------------------

    def extract_voice_samples(
        self,
        scene: RenderedScene,
        *,
        max_tokens: int = 600,
    ) -> list[VoiceSample]:
        """Extract voice samples via a focused LLM call (temp=0.2).

        Returns empty list on any parse failure.
        """
        characters = [c.name for c in scene.brief.characters_in_scene]
        system = (
            "Extract character voice samples from literary prose. "
            "Return ONLY a JSON array. No commentary, no markdown fences. "
            "Each object: character (string), dialogue (string or null), "
            "behavior (string or null). "
            "dialogue = one verbatim spoken line or null. "
            "behavior = one behavioral moment, one sentence, or null. "
            "Omit characters who neither speak nor act meaningfully."
        )
        user = (
            f"Characters: {', '.join(characters)}\n\n"
            f"Prose:\n{scene.prose}\n\nReturn JSON array now."
        )
        response = self._llm.generate(
            system, user, max_tokens=max_tokens, temperature=0.2
        )
        return self._parse_voice_samples(
            response.prose,
            scene_id=f"step_{scene.step}_{scene.location_name.name}",
            step=scene.step,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_enriched(
        self, brief: SceneBrief, scene_input: SceneInput
    ) -> dict[str, Any]:
        """Retrieve memory and merge with brief into enriched dict."""
        chars = scene_input.characters_for_memory or [
            c.name for c in brief.characters_in_scene
        ]
        enrichment = self._memory.retrieve(
            characters_present=chars,
            active_flags=brief.world_state.active_flags,
            location_name=scene_input.location.name,
        )
        return self._memory.enrich_brief(brief, enrichment)

    def _parse_voice_samples(
        self, raw_json: str, scene_id: str, step: int
    ) -> list[VoiceSample]:
        text = raw_json.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:]).rstrip("`").strip()
        try:
            items: list[dict[str, Any]] = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return []
        samples = []
        for item in items:
            if not isinstance(item, dict):
                continue
            character = item.get("character", "").strip().lower()
            if not character:
                continue
            samples.append(
                VoiceSample(
                    scene_id=scene_id,
                    character=character,
                    dialogue=item.get("dialogue") or None,
                    behavior=item.get("behavior") or None,
                    step=step,
                )
            )
        return samples