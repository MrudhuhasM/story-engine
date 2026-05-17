"""run_story.py — end-to-end story generation with thinking display.

Thinking tokens are shown in dim gray.
Prose tokens are shown in normal white.
Both are visible — you can follow the model's reasoning live.

Run:
  uv run src/story_engine/run_story.py

llama-server must be running:
  .\\build\\bin\\Release\\llama-server.exe -m <model_path> -c 8096
"""

from __future__ import annotations

from story_engine.engine import StoryEngine, StoryInitParams
from story_engine.inference import InferencePipeline, RenderedScene, SceneInput
from story_engine.llm_interface import LLMClient
from story_engine.locations import LocationName
from story_engine.memory import MemorySystem
from story_engine.triggers import (
    make_physical_confrontation,
    make_public_humiliation,
    make_notice_board_move,
    make_vikram_refusal,
)


# ---------------------------------------------------------------------------
# ANSI display helpers
# ---------------------------------------------------------------------------

# These work in Windows Terminal, PowerShell 7+, and most modern terminals.
# If you see garbage characters, set FORCE_PLAIN = True.
FORCE_PLAIN = False

_DIM_GRAY  = "\033[2;90m"  # dim + dark gray — for thinking
_RESET     = "\033[0m"
_BOLD      = "\033[1m"
_CYAN      = "\033[36m"
_YELLOW    = "\033[33m"
_GREEN     = "\033[32m"


def _c(code: str, text: str) -> str:
    """Apply ANSI color code, or return plain text if FORCE_PLAIN."""
    if FORCE_PLAIN:
        return text
    return f"{code}{text}{_RESET}"


def print_thinking_token(token: str) -> None:
    """Print a thinking token in dim gray, no newline."""
    print(_c(_DIM_GRAY, token), end="", flush=True)


def print_prose_token(token: str) -> None:
    """Print a prose token in normal white, no newline."""
    print(token, end="", flush=True)


def print_header(text: str) -> None:
    print(_c(_BOLD + _CYAN, text))


def print_label(text: str) -> None:
    print(_c(_YELLOW, text))


def print_ok(text: str) -> None:
    print(_c(_GREEN, text))


# ---------------------------------------------------------------------------
# Story configuration
# ---------------------------------------------------------------------------

STORY_PARAMS = StoryInitParams(
    active_flags=["SEMESTER_OPENING"],
    ranveer_phase_start="COLD",
    surya_true_state="WITH_VIKRAM",
    dhruv_cost_start=0.0,
    resolution_type="R1_VISIBLE_DEFEAT",
    arjun_acts_in_window=False,
    time_of_day="MORNING",
    initial_conflict_phase="COLD_EQUILIBRIUM",
)

SCENE_SCHEDULE: list[SceneInput] = [
    # Step 0 — Opening. No trigger. Establish the space.
    SceneInput(
        location=LocationName.MAIN_CANTEEN,
        trigger=None,
    ),
    # Step 1 — First direct challenge. Vikram refuses.
    SceneInput(
        location=LocationName.MAIN_CANTEEN,
        trigger=make_vikram_refusal(
            LocationName.MAIN_CANTEEN,
            "Ranveer's gang arrived at the corner table Vikram occupied. "
            "Savar made the unspoken demand clear. Vikram did not move.",
        ),
    ),
    # Step 2 — Ranveer escalates publicly.
    SceneInput(
        location=LocationName.MAIN_CANTEEN,
        trigger=make_public_humiliation(
            LocationName.MAIN_CANTEEN,
            "ranveer",
            "vikram",
            "Ranveer arrived late and greeted the table by name — everyone "
            "except Vikram. Spoke as if Vikram were not sitting there.",
        ),
    ),
    # Step 3 — Neel moves through the institution.
    SceneInput(
        location=LocationName.NOTICE_BOARD_CLUSTER,
        trigger=make_notice_board_move(
            "A typed list of inter-hostel cricket selections was posted. "
            "Vikram's name was absent despite his known selection. "
            "No author. Neel was not near the board when it was found.",
        ),
    ),
    # Step 4 — Karan. Physical.
    SceneInput(
        location=LocationName.DEAD_PATHS,
        trigger=make_physical_confrontation(
            LocationName.DEAD_PATHS,
            "karan",
            "vikram",
            "Karan was already on the dead path when Vikram took the shortcut. "
            "Two of Karan's second-years were behind him. No campus witnesses.",
        ),
        dhruv_event_cost=-1.5,
    ),
]

# Pre-written summaries for automated store_scene calls.
# In real use: write these after reading each scene's prose.
SUMMARIES = [
    "Opening. The canteen before anyone has moved. Vikram at the corner table alone. "
    "Ranveer's gang has not arrived. The space is quiet. The tension is in the knowing.",

    "Vikram did not move when Savar made the demand implicit. He occupied the table "
    "through the meal. Ranveer's gang sat elsewhere. Dhruv watched. Rajan was absent.",

    "Ranveer arrived and addressed every name at the table except Vikram's. Vikram "
    "finished eating without acknowledging the omission. Left. The canteen registered it.",

    "The cricket selection list appeared without Vikram's name. Arjun confirmed the "
    "original list had included him. No author was traceable. Neel was not near the board.",

    "Karan was on the dead path with two second-years. He moved to block. Vikram did "
    "not slow. They were close enough. Nothing happened. Vikram came out the other end.",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run() -> None:
    engine = StoryEngine()
    engine.initialize_story(STORY_PARAMS)
    memory = MemorySystem()
    llm = LLMClient(
        base_url="http://127.0.0.1:8080/v1",
        api_key="local",
        timeout=300.0,  # thinking models take longer; 5 min is safe
    )
    pipeline = InferencePipeline(
        engine,
        memory,
        llm,
        max_tokens=3000,   # thinking (~1000) + prose (~1500) fits in 8096 context
        temperature=0.88,
    )

    print_header("=" * 70)
    print_header("STORY ENGINE — INFERENCE PIPELINE")
    print_header("=" * 70)
    print(_c(_DIM_GRAY, "  thinking tokens shown in dim gray"))
    print("  prose tokens shown in normal white")
    print()

    for step_idx, scene_input in enumerate(SCENE_SCHEDULE):
        print_header(f"\n{'─' * 70}")
        loc = scene_input.location.name
        trig = (
            scene_input.trigger.description[:72] + "…"
            if scene_input.trigger and len(scene_input.trigger.description) > 72
            else (scene_input.trigger.description if scene_input.trigger else "QUIET STEP")
        )
        print_header(f"STEP {step_idx} — {loc}")
        print_label(f"Trigger: {trig}")
        print_header("─" * 70)
        print()

        # --- generate (streaming) ---
        metadata, token_stream = pipeline.run_scene_stream(scene_input)

        prose_parts: list[str] = []
        thinking_parts: list[str] = []
        in_thinking = False

        for chunk in token_stream:
            if chunk.is_thinking:
                if not in_thinking:
                    # First thinking token — print label
                    print_label("[thinking]")
                    in_thinking = True
                print_thinking_token(chunk.token)
                thinking_parts.append(chunk.token)
            else:
                if in_thinking:
                    # Switched from thinking to prose
                    print()  # newline after thinking block
                    print()
                    print_label("[prose]")
                    in_thinking = False
                print_prose_token(chunk.token)
                prose_parts.append(chunk.token)

        prose = "".join(prose_parts)
        thinking = "".join(thinking_parts) if thinking_parts else None

        print("\n")  # spacing after prose

        # --- state snapshot ---
        state = metadata["state_snapshot"]
        print_ok(
            f"[conflict={state['conflict_phase']} | "
            f"ranveer={state['ranveer']['phase']} | "
            f"dhruv={state['dhruv']['drift_state']} | "
            f"neel={state['neel']['effective_capacity']:.0%}]"
        )

        # --- store ---
        summary = SUMMARIES[step_idx] if step_idx < len(SUMMARIES) else prose[:200]

        scene = RenderedScene(
            step=metadata["step"],
            trigger=scene_input.trigger,
            location_name=scene_input.location,
            brief=metadata["brief"],
            prose=prose,
            thinking=thinking,
            tokens_used=None,
            world_state_snapshot=metadata["state_snapshot"],
            enriched_brief=metadata["enriched_brief"],
        )
        pipeline.store_scene(scene, summary)

        # --- resolution check ---
        resolution = pipeline.check_resolution()
        if resolution:
            print_ok(f"\n*** RESOLUTION REACHED: {resolution} ***")
            break

    print()
    print_header("=" * 70)
    print_header("RUN COMPLETE")
    print_header("=" * 70)


if __name__ == "__main__":
    run()