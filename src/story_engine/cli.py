"""cli.py — interactive story generation loop.

Scene streams live → auto-summary → world state display →
trigger menu → location picker → description → next scene.

Run:
  uv run src/story_engine/cli.py

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
    Trigger,
    TriggerType,
    TriggerVariant,
    make_academic_threat,
    make_administrative_action,
    make_dhruv_contact,
    make_election_positioning,
    make_gang_member_acts_alone,
    make_information_surface,
    make_meera_intersection,
    make_notice_board_move,
    make_opportunity_denial,
    make_physical_confrontation,
    make_public_callout,
    make_public_humiliation,
    make_vikram_refusal,
)


# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------

FORCE_PLAIN = False  # set True if terminal shows garbage characters

_R = "\033[0m"
_DIM    = "\033[2;90m"
_BOLD   = "\033[1m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_GRAY   = "\033[90m"


def c(code: str, text: str) -> str:
    return text if FORCE_PLAIN else f"{code}{text}{_R}"


def header(text: str) -> None:
    print(c(_BOLD + _CYAN, text))


def label(text: str) -> None:
    print(c(_YELLOW, text))


def ok(text: str) -> None:
    print(c(_GREEN, text))


def dim(text: str) -> None:
    print(c(_GRAY, text), end="", flush=True)


def warn(text: str) -> None:
    print(c(_RED, text))


# ---------------------------------------------------------------------------
# Location menu
# ---------------------------------------------------------------------------

# Grouped for display — shown when user picks a location
_LOCATIONS: list[tuple[LocationName, str]] = [
    (LocationName.MAIN_CANTEEN,           "Main canteen       [HIGH visibility, contested]"),
    (LocationName.MAIN_GROUND,            "Main ground        [MAX visibility, whoever mobilizes]"),
    (LocationName.NOTICE_BOARD_CLUSTER,   "Notice board       [HIGH visibility, Neel controls]"),
    (LocationName.STUDENT_UNION_BUILDING, "Student union      [HIGH visibility, Neel controls]"),
    (LocationName.BOYS_HOSTEL_BLOCKS,     "Hostel blocks      [MEDIUM visibility, split control]"),
    (LocationName.KAVYA_DEPT_CORRIDOR,    "Kavya's corridor   [MEDIUM visibility]"),
    (LocationName.SECONDARY_CANTEEN,      "Secondary canteen  [LOW visibility, neutral]"),
    (LocationName.DEAD_PATHS,             "Dead paths         [NO visibility, Karan's territory]"),
    (LocationName.HOSTEL_ROOF,            "Hostel roof        [NO visibility, no rules]"),
    (LocationName.OLD_BANYAN_TREE,        "Old banyan tree    [LOW visibility, neutral]"),
    (LocationName.GIRLS_HOSTEL,           "Girls hostel       [LOW visibility, Meera's space]"),
    (LocationName.ADMINISTRATION_BUILDING,"Administration     [INSTITUTIONAL, Ranveer family reach]"),
    (LocationName.FACULTY_BUILDINGS,      "Faculty buildings  [LOW-INTERNAL visibility]"),
    (LocationName.MAIN_GATE_AREA,         "Main gate          [HIGH visibility, Ranveer's watch]"),
    (LocationName.FACULTY_QUARTERS,       "Faculty quarters   [PRIVATE, Kavya's home]"),
]


def pick_location() -> LocationName:
    print()
    label("Where does this happen?")
    for i, (loc, desc) in enumerate(_LOCATIONS, 1):
        print(f"  {i:>2}. {desc}")
    while True:
        raw = input(c(_CYAN, "\n  Location [1-15]: ")).strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(_LOCATIONS):
                return _LOCATIONS[idx][0]
        except ValueError:
            pass
        warn("  Enter a number between 1 and 15.")


# ---------------------------------------------------------------------------
# Trigger menu
# ---------------------------------------------------------------------------

_TRIGGER_MENU: list[tuple[str, str]] = [
    # (display label, internal key)
    ("Vikram refuses / does not move",                  "vikram_refusal"),
    ("Ranveer publicly humiliates Vikram",               "public_humiliation"),
    ("Vikram publicly calls out Ranveer's gang",         "public_callout"),
    ("Physical confrontation (Karan)",                   "physical"),
    ("Academic threat — marks, attendance, assessment",  "academic"),
    ("Administrative action — summons, misconduct record","admin"),
    ("Notice board move — list, announcement",           "notice_board"),
    ("Opportunity denial — selection, event access",     "opportunity"),
    ("Election positioning",                             "election"),
    ("Neel contacts Dhruv directly",                     "dhruv_contact"),
    ("Meera intersects with the conflict",               "meera"),
    ("Gang member acts without Vikram (Rajan/Savar/Dhruv)", "gang_alone"),
    ("Information surfaces (Surya)",                     "surya"),
    ("Quiet step — no trigger, ambient drift only",      "quiet"),
]


def pick_trigger() -> tuple[str, str] | tuple[None, None]:
    """Return (trigger_key, description) or (None, None) for quiet step."""
    print()
    label("What happens next?")
    print()

    header("  DIRECT CHALLENGE")
    for i in range(4):
        print(f"  {i+1:>2}. {_TRIGGER_MENU[i][0]}")

    header("\n  INSTITUTIONAL MOVE")
    for i in range(4, 8):
        print(f"  {i+1:>2}. {_TRIGGER_MENU[i][0]}")

    header("\n  POLITICAL MOVE")
    for i in range(8, 10):
        print(f"  {i+1:>2}. {_TRIGGER_MENU[i][0]}")

    header("\n  AMBIENT")
    for i in range(10, 13):
        print(f"  {i+1:>2}. {_TRIGGER_MENU[i][0]}")

    print()
    print(f"   0. {_TRIGGER_MENU[13][0]}")
    print(f"   q. Quit")
    print()

    while True:
        raw = input(c(_CYAN, "  Choice: ")).strip().lower()
        if raw == "q":
            return None, None
        if raw == "0":
            return "quiet", ""
        try:
            idx = int(raw) - 1
            if 0 <= idx < 13:
                key = _TRIGGER_MENU[idx][1]
                desc = get_description(key)
                return key, desc
        except ValueError:
            pass
        warn("  Enter a number (0–13) or q to quit.")


def get_description(key: str) -> str:
    """Prompt user for a specific description of what happened."""
    prompts = {
        "vikram_refusal":    "What did Vikram refuse, and how did it look?",
        "public_humiliation":"What did Ranveer do, and who was watching?",
        "public_callout":    "What did Vikram call out, and where?",
        "physical":          "What did Karan do, exactly?",
        "academic":          "What was the academic threat, specifically?",
        "admin":             "What administrative action was taken?",
        "notice_board":      "What was posted, and what damage does it do?",
        "opportunity":       "What opportunity was blocked, and how was it discovered?",
        "election":          "What positioning move was made?",
        "dhruv_contact":     "What did Neel offer or communicate to Dhruv?",
        "meera":             "What did Meera do that touched the conflict?",
        "gang_alone":        "Which gang member, and what did they do?",
        "surya":             "What information surfaced, and from where?",
    }
    question = prompts.get(key, "Describe what happened:")
    print()
    label(f"  {question}")
    return input(c(_CYAN, "  > ")).strip()


def get_gang_member() -> str:
    """Ask which gang member acted alone."""
    print()
    label("  Which gang member?")
    print("    1. Rajan")
    print("    2. Savar")
    print("    3. Dhruv")
    while True:
        raw = input(c(_CYAN, "  Choice [1-3]: ")).strip()
        if raw == "1":
            return "rajan"
        if raw == "2":
            return "savar"
        if raw == "3":
            return "dhruv"
        warn("  Enter 1, 2, or 3.")


def build_trigger(key: str, description: str) -> tuple[Trigger | None, LocationName]:
    """Build a Trigger from key + description. Returns (trigger, location)."""
    if key == "quiet":
        # Still need a location for the brief
        loc = pick_location()
        return None, loc

    loc = pick_location()

    if key == "vikram_refusal":
        return make_vikram_refusal(loc, description), loc

    if key == "public_humiliation":
        return make_public_humiliation(loc, "ranveer", "vikram", description), loc

    if key == "public_callout":
        return make_public_callout(loc, description), loc

    if key == "physical":
        return make_physical_confrontation(loc, "karan", "vikram", description), loc

    if key == "academic":
        return make_academic_threat(description, location=loc), loc

    if key == "admin":
        return make_administrative_action(description), LocationName.ADMINISTRATION_BUILDING

    if key == "notice_board":
        return make_notice_board_move(description), LocationName.NOTICE_BOARD_CLUSTER

    if key == "opportunity":
        return make_opportunity_denial(description, location=loc), loc

    if key == "election":
        return make_election_positioning(description), LocationName.STUDENT_UNION_BUILDING

    if key == "dhruv_contact":
        return make_dhruv_contact(description), LocationName.SECONDARY_CANTEEN

    if key == "meera":
        return make_meera_intersection(loc, description), loc

    if key == "gang_alone":
        member = get_gang_member()
        is_pub = loc in {
            LocationName.MAIN_CANTEEN, LocationName.MAIN_GROUND,
            LocationName.NOTICE_BOARD_CLUSTER, LocationName.STUDENT_UNION_BUILDING,
        }
        return make_gang_member_acts_alone(member, loc, description, is_public=is_pub), loc

    if key == "surya":
        return make_information_surface(loc, description), loc

    return None, loc


# ---------------------------------------------------------------------------
# State display
# ---------------------------------------------------------------------------

def display_state(state: dict) -> None:
    """Print a compact world state summary."""
    print()
    header("WORLD STATE")
    conflict = state.get("conflict_phase", "?")
    ranveer  = state.get("ranveer", {}).get("phase", "?")
    dhruv    = state.get("dhruv", {}).get("drift_state", "?")
    neel     = state.get("neel", {}).get("effective_capacity", 1.0)
    karan    = state.get("karan", {}).get("is_activated", False)
    kavya    = state.get("kavya", {}).get("is_active", False)
    surya    = state.get("surya", {}).get("is_revealed", False)
    step     = state.get("step", 0)
    flags    = ", ".join(state.get("active_flags", [])) or "none"

    print(f"  Step:     {step}")
    print(f"  Conflict: {c(_YELLOW, conflict)}")
    print(f"  Ranveer:  {c(_YELLOW, ranveer)}")
    print(f"  Dhruv:    {c(_YELLOW, dhruv)}")
    print(f"  Neel:     {neel:.0%} capacity")
    print(f"  Karan:    {'ACTIVATED' if karan else 'peripheral'}")
    print(f"  Kavya:    {'ACTIVE' if kavya else 'passive'}")
    print(f"  Surya:    {'REVEALED' if surya else 'opaque'}")
    print(f"  Flags:    {flags}")
    print()


# ---------------------------------------------------------------------------
# Scene display
# ---------------------------------------------------------------------------

def stream_scene(metadata: dict, token_stream) -> tuple[str, str | None]:
    """Stream a scene to stdout. Returns (prose, thinking)."""
    prose_parts: list[str] = []
    thinking_parts: list[str] = []
    in_thinking = False

    for chunk in token_stream:
        if chunk.is_thinking:
            if not in_thinking:
                print()
                label("[thinking]")
                in_thinking = True
            print(c(_DIM, chunk.token), end="", flush=True)
            thinking_parts.append(chunk.token)
        else:
            if in_thinking:
                print("\n")
                label("[prose]")
                in_thinking = False
            print(chunk.token, end="", flush=True)
            prose_parts.append(chunk.token)

    print("\n")
    prose = "".join(prose_parts)
    thinking = "".join(thinking_parts) if thinking_parts else None
    return prose, thinking


# ---------------------------------------------------------------------------
# Story setup
# ---------------------------------------------------------------------------

def setup_story() -> StoryInitParams:
    """Interactive story setup — or use defaults for fast start."""
    print()
    header("STORY SETUP")
    print()
    print("  Press Enter to use defaults, or type a value.")
    print()

    def ask(prompt: str, default: str) -> str:
        val = input(f"  {prompt} [{default}]: ").strip()
        return val if val else default

    ranveer_phase = ask(
        "Ranveer starting phase (COLD/IRRITATED/OBSESSED/PERSONAL)", "COLD"
    ).upper()

    surya_state = ask(
        "Surya's true allegiance (WITH_VIKRAM/RANVEER_PLANT/OWN_AGENDA/DRIFTER)",
        "WITH_VIKRAM"
    ).upper()

    resolution = ask(
        "Resolution type (R1_VISIBLE_DEFEAT/R2_VISIBLE_WIN/R3_PYRRHIC/R4_SUSPENDED/R5_STRUCTURAL)",
        "R1_VISIBLE_DEFEAT"
    ).upper()

    return StoryInitParams(
        active_flags=["SEMESTER_OPENING"],
        ranveer_phase_start=ranveer_phase,
        surya_true_state=surya_state,
        dhruv_cost_start=0.0,
        resolution_type=resolution,
        arjun_acts_in_window=False,
        time_of_day="MORNING",
        initial_conflict_phase="COLD_EQUILIBRIUM",
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    header("=" * 68)
    header("  STORY ENGINE — INTERACTIVE GENERATOR")
    header("=" * 68)
    print(c(_DIM, "  thinking shown in gray  |  prose in white"))
    print()

    # Setup
    params = setup_story()

    engine = StoryEngine()
    engine.initialize_story(params)
    memory = MemorySystem()
    llm = LLMClient(
        base_url="http://127.0.0.1:8080/v1",
        api_key="local",
        timeout=300.0,
    )
    pipeline = InferencePipeline(engine, memory, llm, max_tokens=3000, temperature=0.88)

    print()
    ok("Engine initialised. Starting with a quiet opening scene.")
    print()

    # First scene is always a quiet step — establish the space
    first_input = SceneInput(location=LocationName.MAIN_CANTEEN)
    step = 0

    while True:
        step_num = pipeline.engine.state.step

        # Determine scene input
        if step == 0:
            scene_input = first_input
        else:
            # Trigger menu
            key, description = pick_trigger()
            if key is None:  # user quit
                break
            trigger, location = build_trigger(key, description)
            scene_input = SceneInput(location=location, trigger=trigger)

        # Scene header
        print()
        header("─" * 68)
        trig_label = (
            scene_input.trigger.trigger_type.name
            if scene_input.trigger else "QUIET STEP"
        )
        loc_label = scene_input.location.name
        header(f"  STEP {step_num} — {loc_label} — {trig_label}")
        header("─" * 68)
        print()

        # Stream the scene
        metadata, token_stream = pipeline.run_scene_stream(scene_input)
        prose, thinking = stream_scene(metadata, token_stream)

        if not prose.strip():
            warn("No prose was generated. Check the server and try again.")
            step += 1
            continue

        # Auto-summarize
        print()
        label("Auto-summarizing scene for memory...")
        summary = pipeline.auto_summarize(prose)

        print()
        label("Summary:")
        print(summary[:30]) #debug: print first 30 chars to confirm summary is working
        print(f"  {c(_GRAY, summary)}")
        print()

        # Allow editing the summary
        edit = input(c(_CYAN, "  Edit summary? [Enter to accept, or type replacement]: ")).strip()
        if edit:
            summary = edit

        # Build and store the scene
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

        # Show world state
        display_state(pipeline.get_state())

        # Check resolution
        resolution = pipeline.check_resolution()
        if resolution:
            print()
            ok(f"★ RESOLUTION REACHED: {resolution} ★")
            print()
            cont = input(c(_CYAN, "  Continue past resolution? [y/N]: ")).strip().lower()
            if cont != "y":
                break

        step += 1

    print()
    header("=" * 68)
    header("  SESSION ENDED")
    header("=" * 68)
    final = pipeline.get_state()
    print(f"  Total steps: {final.get('step', 0)}")
    print(f"  Scenes in memory: {pipeline.memory.scene_count()}")
    print()


if __name__ == "__main__":
    run()