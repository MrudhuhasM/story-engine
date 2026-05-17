"""Tests for memory.py — MemorySystem, VoiceSample, NarrativeThread, helpers."""

from __future__ import annotations

import json
import pathlib
import uuid

from story_engine.memory import (
    MemoryEnrichment,
    MemoryInput,
    MemorySystem,
    NarrativeThread,
    VoiceSample,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _memory() -> MemorySystem:
    """Return an ephemeral (in-memory) MemorySystem with a unique collection.

    Each call gets an isolated ChromaDB collection so tests do not bleed
    state into each other even though they share the same in-process client.
    """
    return MemorySystem(collection_name=f"test_{uuid.uuid4().hex}")


def _voice_sample(
    character: str = "vikram",
    step: int = 0,
    scene_id: str | None = None,
) -> VoiceSample:
    return VoiceSample(
        scene_id=scene_id or str(uuid.uuid4()),
        character=character,
        dialogue="He said nothing.",
        behavior="Ate slowly. Did not look toward Ranveer's corner.",
        step=step,
    )


def _thread(
    description: str = "Dhruv was absent.",
    scene_id: str | None = None,
    step: int = 0,
    characters: list[str] | None = None,
) -> NarrativeThread:
    return NarrativeThread(
        thread_id=str(uuid.uuid4()),
        description=description,
        introduced_in_scene=scene_id or str(uuid.uuid4()),
        introduced_at_step=step,
        characters_involved=characters or ["dhruv"],
    )


def _memory_input(
    scene_id: str | None = None,
    step: int = 0,
    voice_samples: list[VoiceSample] | None = None,
    threads_introduced: list[NarrativeThread] | None = None,
    threads_resolved: list[str] | None = None,
) -> MemoryInput:
    sid = scene_id or str(uuid.uuid4())
    return MemoryInput(
        scene_id=sid,
        step=step,
        location_name="MAIN_CANTEEN",
        characters_present=["vikram", "ranveer"],
        active_flags=["SEMESTER_OPENING"],
        conflict_phase="FRICTION",
        ranveer_phase="COLD",
        summary="Ranveer watched Vikram eat alone. Neither acknowledged the other.",
        prose="Full prose goes here.",
        voice_samples=voice_samples or [],
        threads_introduced=threads_introduced or [],
        threads_resolved=threads_resolved or [],
    )


# ---------------------------------------------------------------------------
# VoiceSample
# ---------------------------------------------------------------------------


class TestVoiceSample:
    def test_to_dict_round_trip(self) -> None:
        sample = _voice_sample()
        d = sample.to_dict()
        restored = VoiceSample.from_dict(d)
        assert restored.character == sample.character
        assert restored.dialogue == sample.dialogue
        assert restored.behavior == sample.behavior
        assert restored.step == sample.step

    def test_none_dialogue_survives_round_trip(self) -> None:
        sample = VoiceSample(
            scene_id="s1",
            character="rajan",
            dialogue=None,
            behavior="Stood very still.",
            step=1,
        )
        restored = VoiceSample.from_dict(sample.to_dict())
        assert restored.dialogue is None
        assert restored.behavior == "Stood very still."

    def test_none_behavior_survives_round_trip(self) -> None:
        sample = VoiceSample(
            scene_id="s2",
            character="neel",
            dialogue="Nothing personal.",
            behavior=None,
            step=2,
        )
        restored = VoiceSample.from_dict(sample.to_dict())
        assert restored.behavior is None
        assert restored.dialogue == "Nothing personal."


# ---------------------------------------------------------------------------
# NarrativeThread
# ---------------------------------------------------------------------------


class TestNarrativeThread:
    def test_to_dict_round_trip(self) -> None:
        t = _thread()
        restored = NarrativeThread.from_dict(t.to_dict())
        assert restored.description == t.description
        assert restored.characters_involved == t.characters_involved
        assert restored.resolved is False

    def test_resolved_false_by_default(self) -> None:
        t = _thread()
        assert t.resolved is False
        assert t.resolved_in_scene is None

    def test_resolved_scene_survives_round_trip(self) -> None:
        t = _thread()
        t.resolved = True
        t.resolved_in_scene = "scene-abc"
        restored = NarrativeThread.from_dict(t.to_dict())
        assert restored.resolved is True
        assert restored.resolved_in_scene == "scene-abc"


# ---------------------------------------------------------------------------
# Layer 2: retrieve before any scenes stored
# ---------------------------------------------------------------------------


class TestRetrieveEmpty:
    def test_returns_no_prior_scenes_sentinel_when_empty(self) -> None:
        memory = _memory()
        enrichment = memory.retrieve(
            characters_present=["vikram"],
            active_flags=["SEMESTER_OPENING"],
            location_name="MAIN_CANTEEN",
        )
        assert enrichment.relevant_prior_scenes == ["No prior scenes on record."]

    def test_voice_samples_empty_when_no_scenes(self) -> None:
        memory = _memory()
        enrichment = memory.retrieve(["vikram", "ranveer"], [], "MAIN_CANTEEN")
        assert enrichment.character_voice_samples == {}

    def test_open_threads_empty_when_none_added(self) -> None:
        memory = _memory()
        enrichment = memory.retrieve(["vikram"], [], "MAIN_CANTEEN")
        assert enrichment.open_threads == []

    def test_returns_memory_enrichment_type(self) -> None:
        memory = _memory()
        enrichment = memory.retrieve(["vikram"], [], "MAIN_CANTEEN")
        assert isinstance(enrichment, MemoryEnrichment)


# ---------------------------------------------------------------------------
# Layer 2: store and retrieve
# ---------------------------------------------------------------------------


class TestStoreAndRetrieve:
    def test_collection_count_increments_on_store(self) -> None:
        memory = _memory()
        assert memory.scene_count() == 0
        memory.store_scene(_memory_input())
        assert memory.scene_count() == 1

    def test_store_two_scenes_increments_count_to_two(self) -> None:
        memory = _memory()
        memory.store_scene(_memory_input())
        memory.store_scene(_memory_input())
        assert memory.scene_count() == 2

    def test_stored_summary_is_retrievable(self) -> None:
        memory = _memory()
        memory.store_scene(_memory_input(step=1))
        enrichment = memory.retrieve(
            characters_present=["vikram", "ranveer"],
            active_flags=["SEMESTER_OPENING"],
            location_name="MAIN_CANTEEN",
        )
        # After storing one scene, retrieve must not return the empty sentinel.
        assert enrichment.relevant_prior_scenes != ["No prior scenes on record."]
        # The stored summary must appear in the results.
        assert any("Ranveer" in s for s in enrichment.relevant_prior_scenes)

    def test_get_scene_returns_stored_record(self) -> None:
        memory = _memory()
        sid = str(uuid.uuid4())
        memory.store_scene(_memory_input(scene_id=sid, step=2))
        record = memory.get_scene(sid)
        assert record is not None
        assert record.scene_id == sid
        assert record.step == 2

    def test_get_scene_returns_none_for_unknown_id(self) -> None:
        memory = _memory()
        assert memory.get_scene("does-not-exist") is None


# ---------------------------------------------------------------------------
# Layer 3: voice cache
# ---------------------------------------------------------------------------


class TestVoiceCache:
    def test_voice_cache_populated_after_store(self) -> None:
        memory = _memory()
        sample = _voice_sample(character="vikram", step=1)
        memory.store_scene(_memory_input(voice_samples=[sample]))
        samples = memory.get_voice_samples("vikram")
        assert len(samples) == 1
        assert samples[0].character == "vikram"

    def test_voice_cache_retrieves_most_recent_first(self) -> None:
        memory = _memory()
        s1 = _voice_sample(character="vikram", step=1)
        s2 = _voice_sample(character="vikram", step=2)
        memory.store_scene(_memory_input(step=1, voice_samples=[s1]))
        memory.store_scene(_memory_input(step=2, voice_samples=[s2]))
        samples = memory.get_voice_samples("vikram")
        assert samples[0].step == 2

    def test_voice_cache_evicts_oldest_at_limit(self) -> None:
        memory = _memory()
        # Store 6 samples — limit is 5
        for i in range(6):
            s = _voice_sample(character="vikram", step=i)
            memory.store_scene(_memory_input(step=i, voice_samples=[s]))
        samples = memory.get_voice_samples("vikram")
        assert len(samples) == 5
        # Most recent should be step 5 (last inserted)
        assert samples[0].step == 5

    def test_voice_cache_returns_empty_for_unknown_character(self) -> None:
        memory = _memory()
        assert memory.get_voice_samples("unknown_character") == []

    def test_retrieve_includes_voice_samples_for_present_characters(self) -> None:
        memory = _memory()
        sample = _voice_sample(character="vikram", step=0)
        memory.store_scene(_memory_input(voice_samples=[sample]))
        enrichment = memory.retrieve(
            characters_present=["vikram", "ranveer"],
            active_flags=[],
            location_name="MAIN_CANTEEN",
        )
        assert "vikram" in enrichment.character_voice_samples
        assert len(enrichment.character_voice_samples["vikram"]) == 1

    def test_retrieve_excludes_voice_samples_for_absent_characters(self) -> None:
        memory = _memory()
        sample = _voice_sample(character="karan", step=0)
        memory.store_scene(_memory_input(voice_samples=[sample]))
        enrichment = memory.retrieve(
            characters_present=["vikram"],  # karan not present
            active_flags=[],
            location_name="MAIN_CANTEEN",
        )
        assert "karan" not in enrichment.character_voice_samples


# ---------------------------------------------------------------------------
# Layer 4: thread tracker
# ---------------------------------------------------------------------------


class TestThreadTracker:
    def test_thread_added_and_returned_as_open(self) -> None:
        memory = _memory()
        thread = _thread(description="Dhruv was absent at the canteen.")
        memory.store_scene(_memory_input(threads_introduced=[thread]))
        open_threads = memory.get_open_threads()
        assert len(open_threads) == 1
        assert open_threads[0].description == "Dhruv was absent at the canteen."

    def test_thread_resolved_not_returned_as_open(self) -> None:
        memory = _memory()
        thread = _thread()
        memory.store_scene(_memory_input(threads_introduced=[thread]))
        # Resolve it in a second scene
        memory.store_scene(_memory_input(threads_resolved=[thread.thread_id]))
        open_threads = memory.get_open_threads()
        assert len(open_threads) == 0

    def test_threads_filtered_by_characters_present(self) -> None:
        memory = _memory()
        t_dhruv = _thread(description="Dhruv thread.", characters=["dhruv"], step=0)
        t_surya = _thread(description="Surya thread.", characters=["surya"], step=0)
        memory.store_scene(_memory_input(threads_introduced=[t_dhruv, t_surya]))
        enrichment = memory.retrieve(
            characters_present=["vikram", "dhruv"],  # surya not present
            active_flags=[],
            location_name="MAIN_CANTEEN",
        )
        # Dhruv thread should appear; Surya thread may appear as fallback
        # but Dhruv thread must be present
        assert "Dhruv thread." in enrichment.open_threads

    def test_add_thread_manually(self) -> None:
        memory = _memory()
        sid = str(uuid.uuid4())
        thread = memory.add_thread(
            description="A look that went unexplained.",
            scene_id=sid,
            characters_involved=["vikram", "meera"],
        )
        assert thread.thread_id in [t.thread_id for t in memory.get_open_threads()]

    def test_resolve_thread_returns_true_for_known_thread(self) -> None:
        memory = _memory()
        thread = memory.add_thread("desc", "s1", ["vikram"])
        result = memory.resolve_thread(thread.thread_id, "s2")
        assert result is True

    def test_resolve_thread_returns_false_for_unknown_thread(self) -> None:
        memory = _memory()
        result = memory.resolve_thread("nonexistent-id", "s2")
        assert result is False

    def test_resolve_thread_removes_it_from_open_list(self) -> None:
        memory = _memory()
        thread = memory.add_thread("A debt.", "s1", ["ranveer"])
        memory.resolve_thread(thread.thread_id, "s2")
        assert len(memory.get_open_threads()) == 0


# ---------------------------------------------------------------------------
# enrich_brief
# ---------------------------------------------------------------------------


class TestEnrichBrief:
    def test_enrich_brief_produces_complete_dict(self) -> None:
        from story_engine.engine import StoryEngine, StoryInitParams
        from story_engine.locations import LocationName

        engine = StoryEngine()
        engine.initialize_story(
            StoryInitParams(
                active_flags=["SEMESTER_OPENING"],
                ranveer_phase_start="COLD",
                surya_true_state="WITH_VIKRAM",
                dhruv_cost_start=0.0,
                resolution_type="R1_VISIBLE_DEFEAT",
            )
        )
        brief = engine.generate_scene_brief(LocationName.MAIN_CANTEEN)

        memory = _memory()
        enrichment = memory.retrieve(["vikram"], ["SEMESTER_OPENING"], "MAIN_CANTEEN")
        result = memory.enrich_brief(brief, enrichment)

        assert "memory" in result
        assert "relevant_prior_scenes" in result["memory"]
        assert "character_voice_samples" in result["memory"]
        assert "open_narrative_threads" in result["memory"]

    def test_enrich_brief_contains_all_scene_brief_fields(self) -> None:
        from story_engine.engine import StoryEngine, StoryInitParams
        from story_engine.locations import LocationName

        engine = StoryEngine()
        engine.initialize_story(
            StoryInitParams(
                active_flags=["SEMESTER_OPENING"],
                ranveer_phase_start="COLD",
                surya_true_state="WITH_VIKRAM",
                dhruv_cost_start=0.0,
                resolution_type="R1_VISIBLE_DEFEAT",
            )
        )
        brief = engine.generate_scene_brief(LocationName.MAIN_CANTEEN)
        memory = _memory()
        enrichment = memory.retrieve([], [], "MAIN_CANTEEN")
        result = memory.enrich_brief(brief, enrichment)

        assert "world_state" in result
        assert "location" in result
        assert "scene_goal" in result
        assert "emotional_arc" in result


# ---------------------------------------------------------------------------
# Persistence: save_state / load_state
# ---------------------------------------------------------------------------


class TestSaveLoadState:
    def test_save_and_load_state_round_trip(self, tmp_path: pathlib.Path) -> None:
        path = str(tmp_path / "memory_state.json")
        memory = _memory()

        sample = _voice_sample(character="vikram", step=3)
        thread = _thread(description="Unresolved look.", step=3)
        memory.store_scene(
            _memory_input(step=3, voice_samples=[sample], threads_introduced=[thread])
        )
        memory.save_state(path)

        # Load into a fresh instance
        memory2 = _memory()
        memory2.load_state(path)

        assert len(memory2.get_voice_samples("vikram")) == 1
        assert memory2.get_voice_samples("vikram")[0].step == 3
        assert len(memory2.get_open_threads()) == 1
        assert memory2.get_open_threads()[0].description == "Unresolved look."

    def test_save_state_is_valid_json(self, tmp_path: pathlib.Path) -> None:
        path = str(tmp_path / "state.json")
        memory = _memory()
        memory.store_scene(_memory_input(voice_samples=[_voice_sample()]))
        memory.save_state(path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        assert "voice_cache" in data
        assert "open_threads" in data
        assert "current_step" in data

    def test_load_state_restores_step_counter(self, tmp_path: pathlib.Path) -> None:
        path = str(tmp_path / "state.json")
        memory = _memory()
        memory.store_scene(_memory_input(step=7))
        memory.save_state(path)

        memory2 = _memory()
        memory2.load_state(path)
        assert memory2._current_step == 7
