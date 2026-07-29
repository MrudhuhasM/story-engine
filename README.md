# Story Engine

An experimental narrative simulation engine that separates deterministic world-state transitions from LLM-generated prose. Structured triggers update characters, relationships, locations, conflict phases, and narrative memory; an OpenAI-compatible local model then renders the resulting scene.

The goal is to explore world-model-style state generation without asking the language model to own the simulation rules.

## Architecture

```text
Player or scripted trigger
        │
        ▼
StoryEngine ── deterministic state transition and resolution checks
        │
        ▼
SceneBriefGenerator ── structured characters, location, conflict, and scene goal
        │
        ├── MemorySystem ── relevant scenes, voice samples, and open threads
        │
        ▼
PromptBuilder
        │
        ▼
OpenAI-compatible local LLM
        │
        ▼
Rendered prose + updated narrative memory
```

## Highlights

- Typed world state for characters, relationships, flags, incidents, and conflict phases
- Trigger factories for direct, institutional, political, and ambient events
- Location-aware consequence and visibility rules
- Deterministic state transitions separated from prose generation
- Scene briefs and prompt construction from structured state
- Layered narrative memory:
  - recent scene records;
  - ChromaDB retrieval over scene summaries;
  - character voice samples and unresolved narrative threads
- Streaming separation of model reasoning and prose when supported by the server
- Interactive and scripted generation flows
- Simulation utilities for testing state evolution without an LLM

## Setup

Requirements:

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/)
- An OpenAI-compatible model server; the included runners default to llama.cpp at `http://127.0.0.1:8080/v1`

```bash
uv sync --dev
```

Start a local llama.cpp server with a model and an 8K context, for example:

```powershell
.\build\bin\Release\llama-server.exe -m <model-path> -c 8096
```

The server path and model are external to this repository.

## Run

Interactive story loop:

```bash
uv run python src/story_engine/cli.py
```

Scripted five-scene demonstration:

```bash
uv run python src/story_engine/run_story.py
```

Inspect the prompts without running a full session:

```bash
uv run python src/story_engine/print_prompt.py
```

The interactive flow lets you choose a trigger, location, and event description, then streams the scene, summarizes it for memory, displays the new world state, and checks whether a configured resolution has been reached.

## Test

```bash
uv run pytest
```

The test suite covers state initialization and serialization, trigger behavior, relationship changes, location rules, memory retrieval, simulation, prompt construction, LLM boundaries, and resolution conditions.

## Repository layout

```text
src/story_engine/
├── engine.py           # orchestration and deterministic state transitions
├── world_state.py      # serializable story state
├── characters.py       # character-specific state and phases
├── triggers.py         # typed events and trigger factories
├── locations.py        # visibility, control, and consequence rules
├── brief_generator.py  # structured scene briefs
├── prompt_builder.py   # LLM prompt construction
├── llm_interface.py    # OpenAI-compatible client
├── inference.py        # generation and memory pipeline
├── memory.py           # ChromaDB and continuity memory
├── simulation.py       # LLM-free state simulation
├── cli.py              # interactive runner
└── run_story.py        # scripted runner
```

## Current scope

This is a local experimental engine rather than a general-purpose authoring product. The included story configuration and character model are specific to the current scenario, while the architecture is being used to test broader ideas around deterministic simulation, retrieval-backed continuity, and LLM rendering.
