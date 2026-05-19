"""print_prompt.py — inspect the exact prompt sent to the LLM.

Run before generating to see what the model actually receives.
This is the fastest way to diagnose prompt quality issues.

Usage:
  uv run src/story_engine/print_prompt.py
"""

from __future__ import annotations

from story_engine.engine import StoryEngine, StoryInitParams
from story_engine.locations import LocationName
from story_engine.memory import MemorySystem
from story_engine.prompt_builder import PromptBuilder
from story_engine.triggers import make_vikram_refusal, make_public_humiliation
from story_engine.brief_generator import SceneBriefGenerator
from story_engine.memory import MemoryEnrichment

_SEP = "─" * 68


def show_prompt(label: str, system: str, user: str) -> None:
    print(f"\n{'═' * 68}")
    print(f"  {label}")
    print(f"{'═' * 68}")
    print(f"\n{_SEP}")
    print("SYSTEM PROMPT")
    print(_SEP)
    print(system)
    print(f"\n{_SEP}")
    print("USER PROMPT")
    print(_SEP)
    print(user)
    print(f"\n{_SEP}")
    est_tokens = (len(system) + len(user)) // 4
    print(f"Estimated tokens: ~{est_tokens} (system+user)")
    print(_SEP)


def main() -> None:
    params = StoryInitParams(
        active_flags=["SEMESTER_OPENING"],
        ranveer_phase_start="COLD",
        surya_true_state="WITH_VIKRAM",
        dhruv_cost_start=0.0,
        resolution_type="R1_VISIBLE_DEFEAT",
        time_of_day="MORNING",
        initial_conflict_phase="COLD_EQUILIBRIUM",
    )
    engine = StoryEngine()
    engine.initialize_story(params)
    memory = MemorySystem()
    builder = PromptBuilder()

    # --- Scene 0: quiet step ---
    brief0 = engine.generate_scene_brief(LocationName.MAIN_CANTEEN)
    enrichment0 = memory.retrieve(
        characters_present=[c.name for c in brief0.characters_in_scene],
        active_flags=brief0.world_state.active_flags,
        location_name=LocationName.MAIN_CANTEEN.name,
    )
    enriched0 = memory.enrich_brief(brief0, enrichment0)
    system0, user0 = builder.build(enriched0)
    show_prompt("SCENE 0 — QUIET STEP — MAIN_CANTEEN", system0, user0)

    # Advance and fire a trigger for scene 1
    engine.advance_state()
    trigger1 = make_vikram_refusal(
        LocationName.MAIN_CANTEEN,
        "Ranveer's gang arrived at the corner table. Savar made the demand implicit. Vikram did not move.",
    )
    engine.fire_trigger(trigger1)
    brief1 = engine.generate_scene_brief(LocationName.MAIN_CANTEEN, trigger=trigger1)
    enrichment1 = memory.retrieve(
        characters_present=[c.name for c in brief1.characters_in_scene],
        active_flags=brief1.world_state.active_flags,
        location_name=LocationName.MAIN_CANTEEN.name,
    )
    enriched1 = memory.enrich_brief(brief1, enrichment1)
    system1, user1 = builder.build(enriched1)
    show_prompt("SCENE 1 — VIKRAM REFUSAL — MAIN_CANTEEN", system1, user1)


if __name__ == "__main__":
    main()