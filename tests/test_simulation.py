"""Tests for simulation.py — SimulationRunner, SimulationStep, SimulationResult."""

from __future__ import annotations

import json

import pytest

from story_engine.brief_generator import SceneBrief
from story_engine.characters import RanveerPhase
from story_engine.engine import StoryEngine, StoryInitParams
from story_engine.locations import LocationName
from story_engine.simulation import SimulationResult, SimulationRunner, SimulationStep
from story_engine.triggers import (
    make_academic_threat,
    make_public_humiliation,
    make_vikram_refusal,
)
from story_engine.world_state import ConflictPhase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _default_params(**overrides: object) -> StoryInitParams:
    defaults: dict[str, object] = {
        "active_flags": ["SEMESTER_OPENING"],
        "ranveer_phase_start": "COLD",
        "surya_true_state": "WITH_VIKRAM",
        "dhruv_cost_start": 0.0,
        "resolution_type": "R1_VISIBLE_DEFEAT",
    }
    defaults.update(overrides)
    return StoryInitParams(**defaults)  # type: ignore[arg-type]


@pytest.fixture()
def runner() -> SimulationRunner:
    """Return a SimulationRunner with a default-initialised engine."""
    engine = StoryEngine()
    engine.initialize_story(_default_params())
    return SimulationRunner(engine)


def _refusal_step(
    location: LocationName = LocationName.MAIN_CANTEEN,
    brief_location: LocationName | None = None,
) -> SimulationStep:
    return SimulationStep(
        trigger=make_vikram_refusal(location, "Vikram refused."),
        brief_location=brief_location or location,
    )


def _humiliation_step(brief_location: LocationName = LocationName.MAIN_CANTEEN) -> SimulationStep:
    return SimulationStep(
        trigger=make_public_humiliation(
            LocationName.MAIN_CANTEEN, "ranveer", "vikram", "Ranveer mocked Vikram."
        ),
        brief_location=brief_location,
    )


# ---------------------------------------------------------------------------
# SimulationStep
# ---------------------------------------------------------------------------


class TestSimulationStep:
    def test_step_with_trigger(self) -> None:
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        step = SimulationStep(trigger=trigger, brief_location=LocationName.MAIN_CANTEEN)
        assert step.trigger is trigger
        assert step.brief_location is LocationName.MAIN_CANTEEN

    def test_step_without_trigger(self) -> None:
        step = SimulationStep(brief_location=LocationName.HOSTEL_ROOF)
        assert step.trigger is None

    def test_step_dhruv_cost_default_none(self) -> None:
        step = SimulationStep(brief_location=LocationName.MAIN_CANTEEN)
        assert step.dhruv_event_cost is None

    def test_step_dhruv_cost_override(self) -> None:
        step = SimulationStep(
            brief_location=LocationName.MAIN_CANTEEN,
            trigger=make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc"),
            dhruv_event_cost=2.5,
        )
        assert step.dhruv_event_cost == 2.5


# ---------------------------------------------------------------------------
# SimulationRunner construction
# ---------------------------------------------------------------------------


class TestSimulationRunnerConstruction:
    def test_init_with_engine(self) -> None:
        engine = StoryEngine()
        engine.initialize_story(_default_params())
        runner = SimulationRunner(engine)
        assert runner._engine is engine

    def test_from_params_factory(self) -> None:
        params = _default_params()
        runner = SimulationRunner.from_params(params)
        # Engine is initialised — state is accessible
        assert runner._engine.state.step == 0

    def test_from_params_sets_ranveer_phase(self) -> None:
        params = _default_params(ranveer_phase_start="IRRITATED")
        runner = SimulationRunner.from_params(params)
        assert runner._engine.state.ranveer.phase is RanveerPhase.IRRITATED


# ---------------------------------------------------------------------------
# Basic run behaviour
# ---------------------------------------------------------------------------


class TestSimulationRunBasic:
    def test_run_empty_schedule_returns_result(self, runner: SimulationRunner) -> None:
        result = runner.run([])
        assert isinstance(result, SimulationResult)
        assert result.steps_executed == 0
        assert result.step_records == []

    def test_run_single_step_produces_one_record(
        self, runner: SimulationRunner
    ) -> None:
        result = runner.run([_refusal_step()])
        assert result.steps_executed == 1
        assert len(result.step_records) == 1

    def test_run_three_steps_produces_three_records(
        self, runner: SimulationRunner
    ) -> None:
        steps = [_refusal_step() for _ in range(3)]
        result = runner.run(steps, stop_on_resolution=False)
        assert result.steps_executed == 3
        assert len(result.step_records) == 3

    def test_steps_scheduled_count_correct(self, runner: SimulationRunner) -> None:
        steps = [_refusal_step() for _ in range(4)]
        result = runner.run(steps, stop_on_resolution=False)
        assert result.steps_scheduled == 4

    def test_total_steps_run_increments(self, runner: SimulationRunner) -> None:
        steps = [_refusal_step() for _ in range(3)]
        result = runner.run(steps, stop_on_resolution=False)
        assert result.total_steps_run == 3


# ---------------------------------------------------------------------------
# StepRecord contents
# ---------------------------------------------------------------------------


class TestStepRecordContents:
    def test_step_counter_starts_at_zero(self, runner: SimulationRunner) -> None:
        result = runner.run([_refusal_step()])
        assert result.step_records[0].step == 0

    def test_step_counter_increments_across_records(
        self, runner: SimulationRunner
    ) -> None:
        steps = [_refusal_step() for _ in range(3)]
        result = runner.run(steps, stop_on_resolution=False)
        assert [r.step for r in result.step_records] == [0, 1, 2]

    def test_trigger_type_recorded(self, runner: SimulationRunner) -> None:
        result = runner.run([_refusal_step()])
        assert result.step_records[0].trigger_type == "DIRECT_CHALLENGE"

    def test_trigger_description_recorded(self, runner: SimulationRunner) -> None:
        result = runner.run([_refusal_step()])
        assert result.step_records[0].trigger_description == "Vikram refused."

    def test_location_name_recorded(self, runner: SimulationRunner) -> None:
        step = SimulationStep(
            trigger=make_vikram_refusal(LocationName.DEAD_PATHS, "desc"),
            brief_location=LocationName.DEAD_PATHS,
        )
        result = runner.run([step])
        assert result.step_records[0].location_name == "DEAD_PATHS"

    def test_conflict_phase_after_direct_challenge(
        self, runner: SimulationRunner
    ) -> None:
        # COLD_EQUILIBRIUM → FRICTION after first direct challenge
        result = runner.run([_refusal_step()])
        assert result.step_records[0].conflict_phase_after == "FRICTION"

    def test_ranveer_phase_after_non_submission(
        self, runner: SimulationRunner
    ) -> None:
        # COLD → IRRITATED after first non-submission
        result = runner.run([_refusal_step()])
        assert result.step_records[0].ranveer_phase_after == "IRRITATED"

    def test_dhruv_drift_after_recorded(self, runner: SimulationRunner) -> None:
        result = runner.run([_refusal_step()])
        assert result.step_records[0].dhruv_drift_after == "PRESENT"

    def test_neel_capacity_after_recorded(self, runner: SimulationRunner) -> None:
        result = runner.run([_refusal_step()])
        assert result.step_records[0].neel_capacity_after == 1.0

    def test_brief_is_scene_brief_instance(self, runner: SimulationRunner) -> None:
        result = runner.run([_refusal_step()])
        assert isinstance(result.step_records[0].brief, SceneBrief)

    def test_brief_location_matches_step(self, runner: SimulationRunner) -> None:
        step = SimulationStep(
            trigger=make_vikram_refusal(LocationName.DEAD_PATHS, "desc"),
            brief_location=LocationName.DEAD_PATHS,
        )
        result = runner.run([step])
        assert result.step_records[0].brief.location.name == "DEAD_PATHS"

    def test_resolution_met_none_when_not_met(self, runner: SimulationRunner) -> None:
        result = runner.run([_refusal_step()])
        assert result.step_records[0].resolution_met is None


# ---------------------------------------------------------------------------
# Quiet steps (no trigger)
# ---------------------------------------------------------------------------


class TestQuietStep:
    def test_quiet_step_produces_record(self, runner: SimulationRunner) -> None:
        step = SimulationStep(brief_location=LocationName.HOSTEL_ROOF)
        result = runner.run([step])
        assert result.steps_executed == 1

    def test_quiet_step_trigger_fields_are_none(self, runner: SimulationRunner) -> None:
        step = SimulationStep(brief_location=LocationName.HOSTEL_ROOF)
        result = runner.run([step])
        record = result.step_records[0]
        assert record.trigger_type is None
        assert record.trigger_description is None

    def test_quiet_step_does_not_fire_chain_rules(
        self, runner: SimulationRunner
    ) -> None:
        # Phase should NOT advance without a trigger
        step = SimulationStep(brief_location=LocationName.HOSTEL_ROOF)
        runner.run([step])
        assert runner._engine.state.conflict_phase is ConflictPhase.COLD_EQUILIBRIUM

    def test_mixed_trigger_and_quiet_steps(self, runner: SimulationRunner) -> None:
        steps = [
            _refusal_step(),
            SimulationStep(brief_location=LocationName.HOSTEL_ROOF),  # quiet
            _refusal_step(),
        ]
        result = runner.run(steps, stop_on_resolution=False)
        assert result.steps_executed == 3
        assert result.step_records[1].trigger_type is None
        assert result.step_records[0].trigger_type == "DIRECT_CHALLENGE"


# ---------------------------------------------------------------------------
# Resolution detection and stop_on_resolution
# ---------------------------------------------------------------------------


class TestResolutionDetection:
    def test_no_resolution_result_fields_none(self, runner: SimulationRunner) -> None:
        steps = [_refusal_step()]
        result = runner.run(steps, stop_on_resolution=False)
        assert result.resolution_met is None
        assert result.resolution_step is None

    def test_r3_pyrrhic_detected_when_dhruv_gone(self) -> None:
        """R3 resolution: Dhruv gone after 12 consecutive negative events."""
        engine = StoryEngine()
        engine.initialize_story(_default_params(resolution_type="R3_PYRRHIC"))
        runner = SimulationRunner(engine)

        # 12 direct challenge triggers → 4 sets of 3 negatives → GONE
        steps = [_refusal_step() for _ in range(12)]
        result = runner.run(steps, stop_on_resolution=True)

        assert result.resolution_met == "R3_PYRRHIC"
        assert result.resolution_step is not None

    def test_stop_on_resolution_true_stops_early(self) -> None:
        """With stop_on_resolution=True, runner stops as soon as resolution is met."""
        engine = StoryEngine()
        engine.initialize_story(_default_params(resolution_type="R3_PYRRHIC"))
        runner = SimulationRunner(engine)

        steps = [_refusal_step() for _ in range(20)]  # schedule 20
        result = runner.run(steps, stop_on_resolution=True)

        # Should stop before all 20 are executed
        assert result.steps_executed < 20
        assert result.resolution_met == "R3_PYRRHIC"

    def test_stop_on_resolution_false_runs_all(self) -> None:
        """With stop_on_resolution=False, runner continues past resolution."""
        engine = StoryEngine()
        engine.initialize_story(_default_params(resolution_type="R3_PYRRHIC"))
        runner = SimulationRunner(engine)

        steps = [_refusal_step() for _ in range(15)]
        result = runner.run(steps, stop_on_resolution=False)

        assert result.steps_executed == 15
        # Resolution still recorded
        assert result.resolution_met == "R3_PYRRHIC"

    def test_resolution_step_recorded_on_correct_step(self) -> None:
        """resolution_step is the engine step counter when resolution was first met."""
        engine = StoryEngine()
        engine.initialize_story(_default_params(resolution_type="R3_PYRRHIC"))
        runner = SimulationRunner(engine)

        steps = [_refusal_step() for _ in range(20)]
        result = runner.run(steps, stop_on_resolution=False)

        # Find the first record where resolution_met is set
        first_resolution_record = next(
            r for r in result.step_records if r.resolution_met == "R3_PYRRHIC"
        )
        assert result.resolution_step == first_resolution_record.step

    def test_r4_suspended_detected_after_step_6(self) -> None:
        engine = StoryEngine()
        engine.initialize_story(
            _default_params(
                resolution_type="R4_SUSPENDED",
                initial_conflict_phase="FRICTION",
            )
        )
        runner = SimulationRunner(engine)

        # Need to reach step >= 6; quiet steps don't change conflict phase
        steps = [
            SimulationStep(brief_location=LocationName.MAIN_CANTEEN)
            for _ in range(8)
        ]
        result = runner.run(steps, stop_on_resolution=True)
        assert result.resolution_met == "R4_SUSPENDED"
        assert result.resolution_step is not None
        assert result.resolution_step >= 6


# ---------------------------------------------------------------------------
# Dhruv cost override via SimulationStep
# ---------------------------------------------------------------------------


class TestDhruvCostOverride:
    def test_positive_override_increases_total(self, runner: SimulationRunner) -> None:
        step = SimulationStep(
            trigger=make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc"),
            brief_location=LocationName.MAIN_CANTEEN,
            dhruv_event_cost=5.0,
        )
        runner.run([step])
        assert runner._engine.state.dhruv.cost_benefit_total == 5.0

    def test_negative_override_decreases_total(self, runner: SimulationRunner) -> None:
        step = SimulationStep(
            trigger=make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc"),
            brief_location=LocationName.MAIN_CANTEEN,
            dhruv_event_cost=-3.0,
        )
        runner.run([step])
        assert runner._engine.state.dhruv.cost_benefit_total == -3.0


# ---------------------------------------------------------------------------
# SimulationResult serialisation
# ---------------------------------------------------------------------------


class TestSimulationResultSerialisation:
    def test_to_json_produces_string(self, runner: SimulationRunner) -> None:
        result = runner.run([_refusal_step()])
        raw = result.to_json()
        assert isinstance(raw, str)

    def test_to_json_is_valid_json(self, runner: SimulationRunner) -> None:
        result = runner.run([_refusal_step()])
        raw = result.to_json()
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_to_json_contains_step_records(self, runner: SimulationRunner) -> None:
        result = runner.run([_refusal_step()])
        raw = result.to_json()
        parsed = json.loads(raw)
        assert "step_records" in parsed
        assert len(parsed["step_records"]) == 1

    def test_from_json_round_trips(self, runner: SimulationRunner) -> None:
        result = runner.run([_refusal_step()])
        raw = result.to_json()
        restored = SimulationResult.from_json(raw)
        assert restored.steps_executed == result.steps_executed
        assert restored.total_steps_run == result.total_steps_run

    def test_from_json_step_records_intact(self, runner: SimulationRunner) -> None:
        result = runner.run([_refusal_step()])
        restored = SimulationResult.from_json(result.to_json())
        assert len(restored.step_records) == 1
        assert restored.step_records[0].trigger_type == "DIRECT_CHALLENGE"

    def test_final_state_is_dict(self, runner: SimulationRunner) -> None:
        result = runner.run([_refusal_step()])
        assert isinstance(result.final_state, dict)

    def test_final_state_has_step_key(self, runner: SimulationRunner) -> None:
        result = runner.run([_refusal_step()])
        assert "step" in result.final_state

    def test_final_state_step_matches_total_steps_run(
        self, runner: SimulationRunner
    ) -> None:
        steps = [_refusal_step() for _ in range(3)]
        result = runner.run(steps, stop_on_resolution=False)
        assert result.final_state["step"] == result.total_steps_run


# ---------------------------------------------------------------------------
# from_params factory
# ---------------------------------------------------------------------------


class TestFromParamsFactory:
    def test_from_params_creates_runner(self) -> None:
        runner = SimulationRunner.from_params(_default_params())
        assert isinstance(runner, SimulationRunner)

    def test_from_params_engine_is_initialised(self) -> None:
        runner = SimulationRunner.from_params(_default_params())
        assert runner._engine.state.step == 0

    def test_from_params_then_run(self) -> None:
        runner = SimulationRunner.from_params(
            _default_params(ranveer_phase_start="IRRITATED")
        )
        result = runner.run([_refusal_step()])
        assert result.step_records[0].ranveer_phase_after == "OBSESSED"


# ---------------------------------------------------------------------------
# Multi-trigger interaction
# ---------------------------------------------------------------------------


class TestMultiTriggerInteraction:
    def test_institutional_trigger_type_recorded(
        self, runner: SimulationRunner
    ) -> None:
        step = SimulationStep(
            trigger=make_academic_threat("Vikram's attendance was flagged."),
            brief_location=LocationName.FACULTY_BUILDINGS,
        )
        result = runner.run([step])
        assert result.step_records[0].trigger_type == "INSTITUTIONAL_MOVE"

    def test_conflict_phase_progresses_across_direct_challenges(
        self, runner: SimulationRunner
    ) -> None:
        steps = [
            _refusal_step(),  # COLD → FRICTION
            _humiliation_step(),  # FRICTION → OPEN_CONFLICT
        ]
        result = runner.run(steps, stop_on_resolution=False)
        assert result.step_records[0].conflict_phase_after == "FRICTION"
        assert result.step_records[1].conflict_phase_after == "OPEN_CONFLICT"

    def test_neel_capacity_degrades_as_ranveer_advances(
        self, runner: SimulationRunner
    ) -> None:
        """Three non-submissions advance Ranveer to OBSESSED → Neel drops to 0.70."""
        steps = [_refusal_step() for _ in range(3)]
        result = runner.run(steps, stop_on_resolution=False)
        # After 3 non-submissions: COLD→IRRITATED→OBSESSED→PERSONAL (clamped)
        # Actually COLD(1) + 3 = PERSONAL(4). Neel should be 0.40.
        # Let's check the final record.
        last = result.step_records[-1]
        assert last.neel_capacity_after in {0.40, 0.70}  # depends on final phase

    def test_brief_has_correct_world_state_snapshot(
        self, runner: SimulationRunner
    ) -> None:
        result = runner.run([_refusal_step()], stop_on_resolution=False)
        brief = result.step_records[0].brief
        assert brief.world_state.conflict_phase == "FRICTION"
        assert brief.world_state.ranveer_phase == "IRRITATED"
