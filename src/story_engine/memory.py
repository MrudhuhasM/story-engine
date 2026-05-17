"""Memory system for the story engine.

Three-layer architecture:
  Layer 1 — World fact memory: WorldState (already handled by engine).
  Layer 2 — Prose continuity: vector store of scene summaries.
  Layer 3 — Character voice cache: per-character dialogue and behavior samples.
  Layer 4 — Open thread tracker: narrative debts the story owes the reader.

No LLM calls live here. This module is pure Python + ChromaDB.

The memory system sits between the engine and the LLM prose renderer.
It enriches scene briefs with prose history before they reach the LLM,
and updates its stores after each scene is rendered.

Import contract
~~~~~~~~~~~~~~~
memory.py → brief_generator.py (SceneBrief)
memory.py has no imports from engine.py or simulation.py.
The inference pipeline imports both engine.py and memory.py separately.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

import chromadb
from pydantic import BaseModel, Field

from story_engine.brief_generator import SceneBrief


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maximum voice samples stored per character.
# Older samples are evicted when this limit is reached.
_MAX_VOICE_SAMPLES_PER_CHARACTER = 5

# Number of prior scene summaries to retrieve per scene brief.
_RETRIEVAL_COUNT = 4

# Maximum open threads tracked simultaneously.
_MAX_OPEN_THREADS = 20


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class VoiceSample:
    """A single dialogue or behavioral moment from a rendered scene.

    Used to anchor the LLM to how this character has been written,
    not just how they are described.

    Args:
        scene_id: ID of the scene this sample came from.
        character: Lowercase character name.
        dialogue: A line or exchange this character actually said.
            None if the sample is behavioral only.
        behavior: A behavioral moment — what the character did,
            how they moved, what they didn't say.
            None if the sample is dialogue only.
        step: Engine step at which this scene occurred.
    """

    scene_id: str
    character: str
    dialogue: str | None
    behavior: str | None
    step: int

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "scene_id": self.scene_id,
            "character": self.character,
            "dialogue": self.dialogue,
            "behavior": self.behavior,
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VoiceSample:
        """Deserialise from a dict produced by ``to_dict()``."""
        return cls(
            scene_id=d["scene_id"],
            character=d["character"],
            dialogue=d.get("dialogue"),
            behavior=d.get("behavior"),
            step=d["step"],
        )


@dataclass
class NarrativeThread:
    """An unresolved narrative element the story owes the reader a return on.

    These are things introduced in prose that have not been paid off —
    a look that went unexplained, something said that meant something else,
    a character noticing something they didn't act on.

    Args:
        thread_id: Unique identifier.
        description: What the thread is — specific enough to reconstruct.
        introduced_in_scene: Scene ID where this thread was introduced.
        introduced_at_step: Engine step when introduced.
        characters_involved: Names of characters involved in this thread.
        resolved: Whether this thread has been paid off.
        resolved_in_scene: Scene ID where resolved, if resolved.
    """

    thread_id: str
    description: str
    introduced_in_scene: str
    introduced_at_step: int
    characters_involved: list[str]
    resolved: bool = False
    resolved_in_scene: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "thread_id": self.thread_id,
            "description": self.description,
            "introduced_in_scene": self.introduced_in_scene,
            "introduced_at_step": self.introduced_at_step,
            "characters_involved": self.characters_involved,
            "resolved": self.resolved,
            "resolved_in_scene": self.resolved_in_scene,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NarrativeThread:
        """Deserialise from a dict produced by ``to_dict()``."""
        return cls(
            thread_id=d["thread_id"],
            description=d["description"],
            introduced_in_scene=d["introduced_in_scene"],
            introduced_at_step=d["introduced_at_step"],
            characters_involved=d["characters_involved"],
            resolved=d["resolved"],
            resolved_in_scene=d.get("resolved_in_scene"),
        )


class SceneRecord(BaseModel):
    """Complete record of a rendered scene stored after generation.

    This is the primary unit the memory system works with.
    The LLM produces prose; the caller extracts voice samples
    and threads from it and stores everything here.

    Args:
        scene_id: Unique identifier (auto-generated if not provided).
        step: Engine step at which this scene was generated.
        location_name: Where the scene was set.
        characters_present: Names of characters in the scene.
        active_flags: World flags active during this scene.
        conflict_phase: Conflict phase at time of generation.
        ranveer_phase: Ranveer's phase at time of generation.
        summary: 2-3 sentence summary of what happened in the scene.
            This is what gets embedded for retrieval.
            Should be written by the caller after reading the prose.
        prose: The full rendered prose of the scene.
            Stored for reference but not embedded directly.
        voice_samples: Dialogue and behavioral samples extracted
            from the prose for each character present.
        threads_introduced: New narrative threads introduced in this scene.
        threads_resolved: Thread IDs resolved in this scene.
    """

    scene_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step: int
    location_name: str
    characters_present: list[str]
    active_flags: list[str]
    conflict_phase: str
    ranveer_phase: str
    summary: str
    prose: str
    voice_samples: list[dict[str, Any]] = Field(default_factory=list)
    threads_introduced: list[dict[str, Any]] = Field(default_factory=list)
    threads_resolved: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# MemoryInput — what the caller provides after rendering a scene
# ---------------------------------------------------------------------------


class MemoryInput(BaseModel):
    """Structured input for ``MemorySystem.store_scene()``.

    The caller (inference pipeline) provides this after the LLM
    renders a scene. The memory system does not extract voice samples
    or threads automatically — that requires reading the prose,
    which the caller is better positioned to do.

    Args:
        scene_id: Unique scene identifier.
        step: Engine step.
        location_name: LocationName.name string.
        characters_present: Lowercase character names.
        active_flags: WorldFlag.name strings active during scene.
        conflict_phase: ConflictPhase.name.
        ranveer_phase: RanveerPhase.name.
        summary: 2-3 sentence factual summary of what happened.
            This is what gets retrieved. Be specific.
            Include who did what, what was not said,
            what was established between characters.
        prose: Full rendered prose.
        voice_samples: Per-character voice samples extracted from prose.
        threads_introduced: New unresolved threads introduced in this scene.
        threads_resolved: IDs of threads resolved in this scene.
    """

    scene_id: str
    step: int
    location_name: str
    characters_present: list[str]
    active_flags: list[str]
    conflict_phase: str
    ranveer_phase: str
    summary: str
    prose: str
    voice_samples: list[VoiceSample] = Field(default_factory=list)
    threads_introduced: list[NarrativeThread] = Field(default_factory=list)
    threads_resolved: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# MemoryEnrichment — what the memory system adds to a scene brief
# ---------------------------------------------------------------------------


class MemoryEnrichment(BaseModel):
    """Structured memory context injected into a scene brief.

    The SceneBriefGenerator uses this to populate the memory
    fields of the brief before it reaches the LLM.

    Args:
        relevant_prior_scenes: 2-4 scene summaries most relevant
            to the current scene. Retrieved by vector similarity.
        character_voice_samples: Per-character voice samples from
            their most recent appearances. Keyed by character name.
        open_threads: Unresolved narrative threads that should be
            honored or acknowledged in the current scene.
        prior_incident_descriptions: Last 3 incident descriptions
            from the engine's incident log (already in the brief,
            duplicated here for convenience).
    """

    relevant_prior_scenes: list[str]
    character_voice_samples: dict[str, list[dict[str, Any]]]
    open_threads: list[str]
    prior_incident_descriptions: list[str]


# ---------------------------------------------------------------------------
# MemorySystem
# ---------------------------------------------------------------------------


class MemorySystem:
    """Three-layer memory for the story engine inference pipeline.

    Layer 1 (world facts): Handled by WorldState — not stored here.
    Layer 2 (prose continuity): ChromaDB vector store of scene summaries.
    Layer 3 (character voice): In-memory rolling cache of voice samples.
    Layer 4 (threads): In-memory open thread tracker.

    Usage pattern::

        memory = MemorySystem()
        # Before generating each scene:
        enrichment = memory.retrieve(characters_present, flags, location)
        # Inject enrichment into scene brief (see enrich_brief)
        # After LLM renders the scene:
        memory.store_scene(memory_input)

    Args:
        persist_directory: Directory for ChromaDB persistence.
            ``None`` means in-memory (testing / ephemeral use).
        collection_name: ChromaDB collection name.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str = "story_scenes",
    ) -> None:
        # Layer 2: ChromaDB vector store
        if persist_directory is not None:
            self._chroma: chromadb.ClientAPI = chromadb.PersistentClient(
                path=persist_directory
            )
        else:
            self._chroma = chromadb.EphemeralClient()

        self._collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Layer 3: Character voice cache
        # Dict of character_name → list of VoiceSample (most recent first)
        self._voice_cache: dict[str, list[VoiceSample]] = {}

        # Layer 4: Open thread tracker
        # Dict of thread_id → NarrativeThread
        self._open_threads: dict[str, NarrativeThread] = {}

        # Scene registry: scene_id → SceneRecord (for reference)
        self._scene_registry: dict[str, SceneRecord] = {}

        # Step counter for ordering
        self._current_step: int = 0

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        characters_present: list[str],
        active_flags: list[str],
        location_name: str,
        n_results: int = _RETRIEVAL_COUNT,
    ) -> MemoryEnrichment:
        """Retrieve memory context for the next scene.

        Queries the vector store for relevant prior scenes,
        retrieves voice samples for characters present,
        and returns open narrative threads.

        Args:
            characters_present: Characters who will be in the scene.
            active_flags: WorldFlag names currently active.
            location_name: LocationName.name for the scene location.
            n_results: Number of prior scene summaries to retrieve.

        Returns:
            MemoryEnrichment ready to inject into the scene brief.
        """
        prior_scenes = self._retrieve_relevant_scenes(
            characters_present=characters_present,
            active_flags=active_flags,
            location_name=location_name,
            n_results=n_results,
        )

        voice_samples = self._retrieve_voice_samples(characters_present)
        open_threads = self._get_open_thread_descriptions(characters_present)

        return MemoryEnrichment(
            relevant_prior_scenes=prior_scenes,
            character_voice_samples=voice_samples,
            open_threads=open_threads,
            prior_incident_descriptions=[],  # populated by engine
        )

    def store_scene(self, memory_input: MemoryInput) -> None:
        """Store a rendered scene in all memory layers.

        Call this after the LLM has rendered the scene and the
        caller has extracted voice samples and thread information.

        Args:
            memory_input: Structured scene data with prose, summary,
                voice samples, and thread information.
        """
        self._current_step = memory_input.step

        # Layer 2: Store scene summary in vector store
        self._store_scene_summary(memory_input)

        # Layer 3: Update voice cache for each character
        for sample in memory_input.voice_samples:
            self._update_voice_cache(sample)

        # Layer 4: Update thread tracker
        for thread in memory_input.threads_introduced:
            self._add_thread(thread)
        for thread_id in memory_input.threads_resolved:
            self._resolve_thread(thread_id)

        # Scene registry
        self._scene_registry[memory_input.scene_id] = SceneRecord(
            scene_id=memory_input.scene_id,
            step=memory_input.step,
            location_name=memory_input.location_name,
            characters_present=memory_input.characters_present,
            active_flags=memory_input.active_flags,
            conflict_phase=memory_input.conflict_phase,
            ranveer_phase=memory_input.ranveer_phase,
            summary=memory_input.summary,
            prose=memory_input.prose,
            voice_samples=[s.to_dict() for s in memory_input.voice_samples],
            threads_introduced=[t.to_dict() for t in memory_input.threads_introduced],
            threads_resolved=memory_input.threads_resolved,
        )

    def enrich_brief(
        self,
        brief: SceneBrief,
        enrichment: MemoryEnrichment,
    ) -> dict[str, Any]:
        """Merge a SceneBrief with memory enrichment into a prompt-ready dict.

        The result is a JSON-serialisable dict that includes all the
        fields from SceneBrief plus memory context.
        This is what gets serialised and sent to the LLM.

        Args:
            brief: SceneBrief from the engine.
            enrichment: MemoryEnrichment from retrieve().

        Returns:
            Complete scene context dict ready for LLM prompting.
        """
        brief_dict = brief.model_dump()

        brief_dict["memory"] = {
            "relevant_prior_scenes": enrichment.relevant_prior_scenes,
            "character_voice_samples": enrichment.character_voice_samples,
            "open_narrative_threads": enrichment.open_threads,
        }

        return brief_dict

    # ------------------------------------------------------------------
    # Thread management (public for external use)
    # ------------------------------------------------------------------

    def add_thread(
        self,
        description: str,
        scene_id: str,
        characters_involved: list[str],
    ) -> NarrativeThread:
        """Manually add a narrative thread outside of store_scene.

        Use this when a thread is identified after reviewing prose
        rather than being pre-identified by the caller.

        Args:
            description: What the thread is.
            scene_id: Scene where it was introduced.
            characters_involved: Characters involved.

        Returns:
            The created NarrativeThread.
        """
        thread = NarrativeThread(
            thread_id=str(uuid.uuid4()),
            description=description,
            introduced_in_scene=scene_id,
            introduced_at_step=self._current_step,
            characters_involved=characters_involved,
        )
        self._add_thread(thread)
        return thread

    def resolve_thread(self, thread_id: str, scene_id: str) -> bool:
        """Mark a thread as resolved.

        Args:
            thread_id: ID of the thread to resolve.
            scene_id: Scene where it was resolved.

        Returns:
            True if the thread existed and was resolved; False if not found.
        """
        if thread_id not in self._open_threads:
            return False
        self._open_threads[thread_id].resolved = True
        self._open_threads[thread_id].resolved_in_scene = scene_id
        return True

    def get_open_threads(self) -> list[NarrativeThread]:
        """Return all currently unresolved narrative threads.

        Returns:
            List of unresolved NarrativeThread instances.
        """
        return [t for t in self._open_threads.values() if not t.resolved]

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    def get_scene(self, scene_id: str) -> SceneRecord | None:
        """Return a stored SceneRecord by ID, or None if not found."""
        return self._scene_registry.get(scene_id)

    def get_voice_samples(self, character: str) -> list[VoiceSample]:
        """Return stored voice samples for a character."""
        return self._voice_cache.get(character, [])

    def scene_count(self) -> int:
        """Return number of scenes stored in the vector store."""
        return self._collection.count()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Persist the in-memory layers (voice cache and threads) to disk.

        The vector store is persisted by ChromaDB automatically
        when using PersistentClient. This method handles the
        in-memory layers that ChromaDB does not manage.

        Args:
            path: File path to write JSON state.
        """
        state = {
            "voice_cache": {
                char: [s.to_dict() for s in samples]
                for char, samples in self._voice_cache.items()
            },
            "open_threads": {tid: t.to_dict() for tid, t in self._open_threads.items()},
            "current_step": self._current_step,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str) -> None:
        """Load in-memory layers from a file saved by ``save_state()``.

        Args:
            path: File path produced by ``save_state()``.
        """
        with open(path, encoding="utf-8") as f:
            state = json.load(f)

        self._voice_cache = {
            char: [VoiceSample.from_dict(s) for s in samples]
            for char, samples in state.get("voice_cache", {}).items()
        }
        self._open_threads = {
            tid: NarrativeThread.from_dict(t)
            for tid, t in state.get("open_threads", {}).items()
        }
        self._current_step = state.get("current_step", 0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve_relevant_scenes(
        self,
        characters_present: list[str],
        active_flags: list[str],
        location_name: str,
        n_results: int,
    ) -> list[str]:
        """Query vector store for scene summaries relevant to the current scene."""
        if self._collection.count() == 0:
            return ["No prior scenes on record."]

        # Characters present are the strongest signal.
        # Location and flags add context.
        query = (
            f"scene with characters: {', '.join(characters_present)}. "
            f"location: {location_name}. "
            f"flags: {', '.join(active_flags)}."
        )

        actual_n = min(n_results, self._collection.count())

        results = self._collection.query(
            query_texts=[query],
            n_results=actual_n,
            include=["documents", "metadatas"],
        )

        summaries: list[str] = results.get("documents", [[]])[0]
        if not summaries:
            return ["No relevant prior scenes found."]
        return summaries

    def _store_scene_summary(self, memory_input: MemoryInput) -> None:
        """Embed and store a scene summary in the vector store."""
        metadata: dict[str, Any] = {
            "step": memory_input.step,
            "location": memory_input.location_name,
            "conflict_phase": memory_input.conflict_phase,
            "ranveer_phase": memory_input.ranveer_phase,
            "characters": json.dumps(memory_input.characters_present),
            "flags": json.dumps(memory_input.active_flags),
        }

        self._collection.add(
            documents=[memory_input.summary],
            metadatas=[metadata],
            ids=[memory_input.scene_id],
        )

    def _update_voice_cache(self, sample: VoiceSample) -> None:
        """Add a voice sample to the character's cache, evicting old ones."""
        char = sample.character
        if char not in self._voice_cache:
            self._voice_cache[char] = []

        # Prepend — most recent first
        self._voice_cache[char].insert(0, sample)

        # Evict oldest if over limit
        if len(self._voice_cache[char]) > _MAX_VOICE_SAMPLES_PER_CHARACTER:
            self._voice_cache[char] = self._voice_cache[char][
                :_MAX_VOICE_SAMPLES_PER_CHARACTER
            ]

    def _retrieve_voice_samples(
        self, characters_present: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Return voice samples for all characters present in the scene."""
        result: dict[str, list[dict[str, Any]]] = {}
        for char in characters_present:
            samples = self._voice_cache.get(char, [])
            if samples:
                # Return at most 3 samples per character
                result[char] = [s.to_dict() for s in samples[:3]]
        return result

    def _add_thread(self, thread: NarrativeThread) -> None:
        """Add a thread to the tracker, evicting oldest resolved thread if over limit."""
        self._open_threads[thread.thread_id] = thread

        if len(self._open_threads) > _MAX_OPEN_THREADS:
            resolved = [(tid, t) for tid, t in self._open_threads.items() if t.resolved]
            if resolved:
                oldest_resolved = min(resolved, key=lambda x: x[1].introduced_at_step)
                del self._open_threads[oldest_resolved[0]]

    def _resolve_thread(self, thread_id: str) -> None:
        """Mark a thread resolved internally."""
        if thread_id in self._open_threads:
            self._open_threads[thread_id].resolved = True

    def _get_open_thread_descriptions(self, characters_present: list[str]) -> list[str]:
        """Return descriptions of open threads involving the current characters."""
        open_threads = [t for t in self._open_threads.values() if not t.resolved]

        # Filter to threads involving at least one character in the scene
        relevant = [
            t
            for t in open_threads
            if any(c in t.characters_involved for c in characters_present)
        ]

        # Fall back to all open threads if none involve current characters
        if not relevant:
            relevant = open_threads

        # Return descriptions, most recent first
        relevant_sorted = sorted(
            relevant, key=lambda t: t.introduced_at_step, reverse=True
        )
        return [t.description for t in relevant_sorted[:5]]
