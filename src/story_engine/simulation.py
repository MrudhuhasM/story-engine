"""SimulationRunner — orchestrates N steps, collects SceneBriefs, checks resolution.

The runner is the outermost layer of the engine stack. It ties together:
1. A ``StoryEngine`` (pre-initialised by the caller)
2. A sequence of ``SimulationStep`` entries — each one fires a trigger and
   requests a brief from a named location
3. Per-step collection of state snapshots and SceneBriefs
4. Automatic early-exit when the target resolution condition is met

No LLM calls live here. The runner produces a ``SimulationResult`` — a
fully serialisable Pydantic model — which the LLM prose renderer (or any
downstream consumer) can process independently.

Import contract
~~~~~~~~~~~~~~~
simulation.py → engine.py → brief_generator.py → world_state.py → characters.py → flags.py
simulation.py → triggers.py → locations.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict

from story_engine.brief_generator import SceneBrief
from story_engine.engine import StoryEngine, StoryInitParams
from story_engine.locations import LocationName
from story_engine.triggers import Trigger


# ---------------------------------------------------------------------------
# SimulationStep — one entry in the schedule (plain dataclass, not Pydantic,
# because Trigger is a frozen dataclass that Pydantic does not own)
# ---------------------------------------------------------------------------


@dataclass
class SimulationStep:
    """One scheduled step in a simulation run.

    Attributes:
        trigger: The trigger to fire at this step. ``None`` means advance
            state and generate a brief without firing a trigger (useful for
            "quiet" steps that capture ambient drift without a discrete event).
        brief_location: The location at which the SceneBrief is generated
            after this step's trigger is applied.
        dhruv_event_cost: Override the engine's default Dhruv cost for this
            trigger. Positive = benefit; negative = cost. ``None`` uses the
            engine default.
    """

    brief_location: LocationName
    trigger: Trigger | None = None
    dhruv_event_cost: float | None = None


# ---------------------------------------------------------------------------
# Output models (Pydantic — external boundary)
# ---------------------------------------------------------------------------


class StepRecord(BaseModel):
    """Snapshot of world state and prose brief captured after one simulation step.

    All enum values are stored as their ``.name`` strings for JSON portability.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step: int
    """The engine step counter value at the time this record was captured."""

    trigger_type: str | None
    """TriggerType.name for the trigger fired this step, or None for quiet steps."""

    trigger_description: str | None
    """Human-readable description from the Trigger, or None for quiet steps."""

    location_name: str
    """LocationName.name used as the brief generation anchor for this step."""

    conflict_phase_after: str
    """ConflictPhase.name after this step's chain rules have been applied."""

    ranveer_phase_after: str
    """RanveerPhase.name after this step."""

    dhruv_drift_after: str
    """DhruvDriftState.name after this step."""

    neel_capacity_after: float
    """Neel's effective_capacity (0.0–1.0) after this step."""

    resolution_met: str | None
    """ResolutionType.name if resolution condition was met this step, else None."""

    brief: SceneBrief
    """Full SceneBrief generated at brief_location after trigger application."""


class SimulationResult(BaseModel):
    """Complete output of a ``SimulationRunner.run()`` call.

    This is the primary serialisable artefact. Pass it to the LLM prose
    renderer or write it to disk via ``to_json()``.

    All enum values are stored as ``.name`` strings. ``final_state`` is a
    plain-Python dict produced by ``WorldState.to_dict()``.
    """

    total_steps_run: int
    """Number of times ``advance_state()`` was called (engine step counter)."""

    steps_scheduled: int
    """Number of SimulationStep entries provided to ``run()``."""

    steps_executed: int
    """Number of steps actually executed (≤ steps_scheduled if stopped early)."""

    resolution_met: str | None
    """ResolutionType.name if resolution was reached during the run, else None."""

    resolution_step: int | None
    """Engine step counter value at which resolution was first detected, or None."""

    final_state: dict[str, Any]
    """``WorldState.to_dict()`` snapshot at the end of the run."""

    step_records: list[StepRecord]
    """Ordered list of per-step records, one per executed SimulationStep."""

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise to a JSON string.

        Uses Pydantic's ``model_dump_json`` so all nested models (including
        SceneBrief) are included.

        Args:
            indent: JSON indentation level. Defaults to 2.

        Returns:
            A JSON string representation of this SimulationResult.
        """
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, raw: str) -> "SimulationResult":
        """Deserialise from a JSON string produced by ``to_json()``.

        Args:
            raw: A JSON string previously produced by ``to_json()``.

        Returns:
            A reconstructed SimulationResult instance.
        """
        return cls.model_validate_json(raw)


# ---------------------------------------------------------------------------
# SimulationRunner
# ---------------------------------------------------------------------------


class SimulationRunner:
    """Runs a pre-initialised StoryEngine through a sequence of SimulationSteps.

    Each step:
    1. Fires the scheduled trigger (if any) via ``engine.fire_trigger()``.
    2. Generates a ``SceneBrief`` at ``step.brief_location``.
    3. Checks the resolution condition.
    4. Records a ``StepRecord``.
    5. Calls ``engine.advance_state()`` to increment the step counter.

    If ``stop_on_resolution=True`` (default) and the resolution condition is
    met at any step, the runner stops after recording that step — it does
    not execute further scheduled steps.

    The runner never re-initialises the engine. Call
    ``engine.initialize_story()`` before passing the engine to the runner.

    Args:
        engine: A ``StoryEngine`` instance. Must have been initialised via
            ``initialize_story()`` before ``run()`` is called.
    """

    def __init__(self, engine: StoryEngine) -> None:
        self._engine = engine

    # ------------------------------------------------------------------
    # Alternative constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_params(cls, params: StoryInitParams) -> "SimulationRunner":
        """Create a SimulationRunner with a freshly initialised engine.

        Convenience factory that constructs and initialises a StoryEngine
        from ``params``, then wraps it in a SimulationRunner.

        Args:
            params: Validated StoryInitParams for this story.

        Returns:
            A SimulationRunner ready to call ``run()`` on.
        """
        engine = StoryEngine()
        engine.initialize_story(params)
        return cls(engine)

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------

    def run(
        self,
        scheduled_steps: list[SimulationStep],
        *,
        stop_on_resolution: bool = True,
    ) -> SimulationResult:
        """Execute the simulation through the scheduled steps.

        For each SimulationStep:
        - If ``step.trigger`` is not None: fire it via ``engine.fire_trigger()``.
        - Generate a SceneBrief at ``step.brief_location``.
        - Check the resolution condition.
        - Append a StepRecord.
        - Call ``engine.advance_state()``.

        If ``stop_on_resolution=True`` and resolution is detected, the runner
        records the current step and exits the loop without processing further
        scheduled steps.

        Args:
            scheduled_steps: Ordered sequence of steps to execute. The runner
                processes them in list order.
            stop_on_resolution: If True (default), halt after the first step
                at which ``check_resolution_condition()`` returns non-None.

        Returns:
            A fully populated SimulationResult.
        """
        records: list[StepRecord] = []
        resolution_met: str | None = None
        resolution_step: int | None = None
        steps_executed = 0

        for scheduled in scheduled_steps:
            # 1. Fire trigger (if scheduled)
            if scheduled.trigger is not None:
                self._engine.fire_trigger(
                    scheduled.trigger,
                    dhruv_event_cost=scheduled.dhruv_event_cost,
                )

            # 2. Generate brief
            brief = self._engine.generate_scene_brief(scheduled.brief_location)

            # 3. Snapshot state
            state = self._engine.state
            res = self._engine.check_resolution_condition()
            res_name = res.name if res is not None else None

            # 4. Record
            records.append(
                StepRecord(
                    step=state.step,
                    trigger_type=(
                        scheduled.trigger.trigger_type.name
                        if scheduled.trigger is not None
                        else None
                    ),
                    trigger_description=(
                        scheduled.trigger.description
                        if scheduled.trigger is not None
                        else None
                    ),
                    location_name=scheduled.brief_location.name,
                    conflict_phase_after=state.conflict_phase.name,
                    ranveer_phase_after=state.ranveer.phase.name,
                    dhruv_drift_after=state.dhruv.drift_state.name,
                    neel_capacity_after=state.neel.effective_capacity,
                    resolution_met=res_name,
                    brief=brief,
                )
            )

            steps_executed += 1

            # 5. Track resolution
            if res is not None and resolution_met is None:
                resolution_met = res_name
                resolution_step = state.step

            # 6. Advance state (increments engine step counter)
            self._engine.advance_state()

            # 7. Early exit
            if res is not None and stop_on_resolution:
                break

        return SimulationResult(
            total_steps_run=self._engine.state.step,
            steps_scheduled=len(scheduled_steps),
            steps_executed=steps_executed,
            resolution_met=resolution_met,
            resolution_step=resolution_step,
            final_state=self._engine.get_current_state(),
            step_records=records,
        )
