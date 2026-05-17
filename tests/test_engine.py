"""Tests for engine.py — StoryEngine chain rules, main API, and integration."""

from __future__ import annotations

import pytest

from story_engine.characters import DhruvDriftState, RanveerPhase, SuryaAllegiance
from story_engine.engine import EngineStateError, StoryEngine, StoryInitParams
from story_engine.flags import WorldFlag
from story_engine.locations import LocationName
from story_engine.triggers import (
    TriggerVariant,
    make_academic_threat,
    make_dhruv_contact,
    make_election_positioning,
    make_gang_member_acts_alone,
    make_meera_intersection,
    make_notice_board_move,
    make_physical_confrontation,
    make_public_humiliation,
    make_vikram_refusal,
)
from story_engine.world_state import ConflictPhase, ResolutionType, TimeOfDay


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _default_params(**overrides: object) -> StoryInitParams:
    """Build a minimal valid StoryInitParams, overridable via kwargs."""
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
def engine() -> StoryEngine:
    """Return an uninitialised StoryEngine."""
    return StoryEngine()


@pytest.fixture()
def init_engine() -> StoryEngine:
    """Return a StoryEngine initialised with default parameters."""
    e = StoryEngine()
    e.initialize_story(_default_params())
    return e


# ---------------------------------------------------------------------------
# EngineStateError guard
# ---------------------------------------------------------------------------


class TestEngineStateGuard:
    def test_state_property_raises_before_init(self, engine: StoryEngine) -> None:
        with pytest.raises(EngineStateError):
            _ = engine.state

    def test_fire_trigger_raises_before_init(self, engine: StoryEngine) -> None:
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        with pytest.raises(EngineStateError):
            engine.fire_trigger(trigger)

    def test_get_current_state_raises_before_init(self, engine: StoryEngine) -> None:
        with pytest.raises(EngineStateError):
            engine.get_current_state()

    def test_advance_state_raises_before_init(self, engine: StoryEngine) -> None:
        with pytest.raises(EngineStateError):
            engine.advance_state()

    def test_check_resolution_raises_before_init(self, engine: StoryEngine) -> None:
        with pytest.raises(EngineStateError):
            engine.check_resolution_condition()


# ---------------------------------------------------------------------------
# StoryInitParams validation
# ---------------------------------------------------------------------------


class TestStoryInitParams:
    def test_valid_params_construct(self) -> None:
        p = _default_params()
        assert p.ranveer_phase_start == "COLD"
        assert p.surya_true_state == "WITH_VIKRAM"

    def test_invalid_flag_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown WorldFlag"):
            _default_params(active_flags=["NOT_A_FLAG"])

    def test_invalid_ranveer_phase_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown RanveerPhase"):
            _default_params(ranveer_phase_start="FURIOUS")

    def test_invalid_surya_state_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown SuryaState"):
            _default_params(surya_true_state="DOUBLE_AGENT")

    def test_invalid_resolution_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown ResolutionType"):
            _default_params(resolution_type="R9_TOTAL_VICTORY")

    def test_invalid_time_of_day_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown TimeOfDay"):
            _default_params(time_of_day="MIDNIGHT")

    def test_invalid_conflict_phase_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown ConflictPhase"):
            _default_params(initial_conflict_phase="WAR")

    def test_defaults_applied(self) -> None:
        p = _default_params()
        assert p.time_of_day == "MORNING"
        assert p.initial_conflict_phase == "COLD_EQUILIBRIUM"
        assert p.arjun_acts_in_window is False


# ---------------------------------------------------------------------------
# initialize_story
# ---------------------------------------------------------------------------


class TestInitializeStory:
    def test_returns_world_state(self, engine: StoryEngine) -> None:
        state = engine.initialize_story(_default_params())
        assert state is engine.state

    def test_step_starts_at_zero(self, init_engine: StoryEngine) -> None:
        assert init_engine.state.step == 0

    def test_active_flags_set(self, engine: StoryEngine) -> None:
        engine.initialize_story(_default_params(active_flags=["ELECTION_SEASON"]))
        assert WorldFlag.ELECTION_SEASON in engine.state.active_flags.flags

    def test_ranveer_phase_set(self, engine: StoryEngine) -> None:
        engine.initialize_story(_default_params(ranveer_phase_start="IRRITATED"))
        assert engine.state.ranveer.phase is RanveerPhase.IRRITATED

    def test_surya_true_state_set(self, engine: StoryEngine) -> None:
        engine.initialize_story(_default_params(surya_true_state="RANVEER_PLANT"))
        assert engine.state.surya.true_state is SuryaAllegiance.RANVEER_PLANT

    def test_dhruv_cost_start_set(self, engine: StoryEngine) -> None:
        engine.initialize_story(_default_params(dhruv_cost_start=-3.5))
        assert engine.state.dhruv.cost_benefit_total == -3.5

    def test_resolution_type_set(self, engine: StoryEngine) -> None:
        engine.initialize_story(_default_params(resolution_type="R3_PYRRHIC"))
        assert engine.state.resolution_type is ResolutionType.R3_PYRRHIC

    def test_relationship_graph_populated(self, engine: StoryEngine) -> None:
        from story_engine.engine import RelationshipStateParam

        engine.initialize_story(
            _default_params(
                active_relationship_states={
                    "vikram|ranveer": RelationshipStateParam(tension=8, trust=0)
                }
            )
        )
        assert ("vikram", "ranveer") in engine.state.relationship_graph

    def test_arjun_acts_in_window_propagated(self, engine: StoryEngine) -> None:
        engine.initialize_story(_default_params(arjun_acts_in_window=True))
        assert engine.state.arjun.arjun_acts_in_window is True

    def test_neel_capacity_derived_after_init(self, engine: StoryEngine) -> None:
        """Neel management threshold is applied during init."""
        engine.initialize_story(_default_params(ranveer_phase_start="COLD"))
        assert engine.state.neel.effective_capacity == 1.0

    def test_reinitialise_resets_state(self, init_engine: StoryEngine) -> None:
        init_engine.advance_state()
        init_engine.initialize_story(_default_params())
        assert init_engine.state.step == 0


# ---------------------------------------------------------------------------
# Chain rule: apply_pride_ratchet (RULE_01)
# ---------------------------------------------------------------------------


class TestApplyPrideRatchet:
    def test_default_advances_phase_by_one(self, init_engine: StoryEngine) -> None:
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        phase = init_engine.apply_pride_ratchet(trigger)
        assert phase is RanveerPhase.IRRITATED

    def test_consecutive_non_submissions_tracked(self, init_engine: StoryEngine) -> None:
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        init_engine.apply_pride_ratchet(trigger)
        assert init_engine.state.ranveer.consecutive_unacknowledged_non_submissions == 1

    def test_apparent_weakness_regresses_phase(self, init_engine: StoryEngine) -> None:
        # Advance to IRRITATED first
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        init_engine.apply_pride_ratchet(trigger)
        assert init_engine.state.ranveer.phase is RanveerPhase.IRRITATED

        phase = init_engine.apply_pride_ratchet(trigger, is_apparent_weakness=True)
        assert phase is RanveerPhase.COLD

    def test_apparent_weakness_sets_strategy_flag(self, init_engine: StoryEngine) -> None:
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        init_engine.apply_pride_ratchet(trigger, is_apparent_weakness=True)
        assert init_engine.state.ranveer.last_weakness_was_strategy is True

    def test_non_submission_clears_strategy_flag(self, init_engine: StoryEngine) -> None:
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        # Set the flag first
        init_engine.apply_pride_ratchet(trigger, is_apparent_weakness=True)
        assert init_engine.state.ranveer.last_weakness_was_strategy is True
        # Normal non-submission should clear it
        init_engine.apply_pride_ratchet(trigger)
        assert init_engine.state.ranveer.last_weakness_was_strategy is False

    def test_weakness_revealed_as_strategy_jumps_plus_two(
        self, engine: StoryEngine
    ) -> None:
        engine.initialize_story(_default_params(ranveer_phase_start="COLD"))
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        # Advance to IRRITATED
        engine.apply_pride_ratchet(trigger)
        assert engine.state.ranveer.phase is RanveerPhase.IRRITATED
        # Apparent weakness → COLD
        engine.apply_pride_ratchet(trigger, is_apparent_weakness=True)
        assert engine.state.ranveer.phase is RanveerPhase.COLD
        # Revealed as strategy from COLD → COLD + 2 = OBSESSED
        phase = engine.apply_pride_ratchet(trigger, weakness_revealed_as_strategy=True)
        assert phase is RanveerPhase.OBSESSED

    def test_phase_clamped_at_personal(self, engine: StoryEngine) -> None:
        engine.initialize_story(_default_params(ranveer_phase_start="PERSONAL"))
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        phase = engine.apply_pride_ratchet(trigger)
        assert phase is RanveerPhase.PERSONAL

    def test_phase_clamped_at_cold_for_weakness(self, engine: StoryEngine) -> None:
        engine.initialize_story(_default_params(ranveer_phase_start="COLD"))
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        phase = engine.apply_pride_ratchet(trigger, is_apparent_weakness=True)
        assert phase is RanveerPhase.COLD


# ---------------------------------------------------------------------------
# Chain rule: apply_visibility_multiplier (RULE_02)
# ---------------------------------------------------------------------------


class TestApplyVisibilityMultiplier:
    def test_non_public_trigger_always_one(self, init_engine: StoryEngine) -> None:
        from story_engine.locations import get_location

        trigger = make_physical_confrontation(
            LocationName.MAIN_GROUND, "karan", "vikram", "desc", is_public=False
        )
        loc = get_location(LocationName.MAIN_GROUND)
        assert init_engine.apply_visibility_multiplier(trigger, loc) == 1.0

    def test_public_trigger_at_main_canteen_doubles(self, init_engine: StoryEngine) -> None:
        from story_engine.locations import get_location

        trigger = make_public_humiliation(
            LocationName.MAIN_CANTEEN, "ranveer", "vikram", "desc"
        )
        loc = get_location(LocationName.MAIN_CANTEEN)
        assert init_engine.apply_visibility_multiplier(trigger, loc) == 2.0

    def test_public_trigger_at_dead_paths_is_one(self, init_engine: StoryEngine) -> None:
        from story_engine.locations import get_location

        # DEAD_PATHS visibility is NONE → multiplier = 1.0
        trigger = make_physical_confrontation(
            LocationName.DEAD_PATHS, "karan", "vikram", "desc", is_public=True
        )
        loc = get_location(LocationName.DEAD_PATHS)
        assert init_engine.apply_visibility_multiplier(trigger, loc) == 1.0


# ---------------------------------------------------------------------------
# Chain rule: apply_dhruv_drift (RULE_03)
# ---------------------------------------------------------------------------


class TestApplyDhruvDrift:
    def test_negative_event_increments_counter(self, init_engine: StoryEngine) -> None:
        init_engine.apply_dhruv_drift(-1.0)
        assert init_engine.state.dhruv.consecutive_negative_events == 1

    def test_positive_event_resets_counter(self, init_engine: StoryEngine) -> None:
        init_engine.apply_dhruv_drift(-1.0)
        init_engine.apply_dhruv_drift(-1.0)
        init_engine.apply_dhruv_drift(0.5)  # positive resets
        assert init_engine.state.dhruv.consecutive_negative_events == 0

    def test_cost_accumulates(self, init_engine: StoryEngine) -> None:
        init_engine.apply_dhruv_drift(-2.0)
        init_engine.apply_dhruv_drift(1.0)
        assert init_engine.state.dhruv.cost_benefit_total == -1.0

    def test_three_consecutive_negatives_advance_drift(
        self, init_engine: StoryEngine
    ) -> None:
        init_engine.apply_dhruv_drift(-1.0)
        init_engine.apply_dhruv_drift(-1.0)
        state = init_engine.apply_dhruv_drift(-1.0)
        assert state is DhruvDriftState.LESS_AVAILABLE

    def test_counter_resets_after_advancement(self, init_engine: StoryEngine) -> None:
        init_engine.apply_dhruv_drift(-1.0)
        init_engine.apply_dhruv_drift(-1.0)
        init_engine.apply_dhruv_drift(-1.0)
        assert init_engine.state.dhruv.consecutive_negative_events == 0

    def test_six_negatives_advance_two_steps(self, init_engine: StoryEngine) -> None:
        for _ in range(6):
            init_engine.apply_dhruv_drift(-1.0)
        assert init_engine.state.dhruv.drift_state is DhruvDriftState.PRESENT_BUT_UNINVESTED

    def test_gone_is_terminal(self, engine: StoryEngine) -> None:
        engine.initialize_story(_default_params())
        # Manually advance to MAKING_EXIT_ARRANGEMENTS
        from story_engine.characters import DhruvDriftState as DDS

        engine.state.dhruv.drift_state = DDS.MAKING_EXIT_ARRANGEMENTS
        for _ in range(3):
            engine.apply_dhruv_drift(-1.0)
        assert engine.state.dhruv.drift_state is DhruvDriftState.GONE
        # Another 3 negatives: state stays GONE
        for _ in range(3):
            engine.apply_dhruv_drift(-1.0)
        assert engine.state.dhruv.drift_state is DhruvDriftState.GONE


# ---------------------------------------------------------------------------
# Chain rule: apply_savar_inversion (RULE_04)
# ---------------------------------------------------------------------------


class TestApplySavarInversion:
    @pytest.mark.parametrize(
        ("visibility", "expected_health"),
        [(1, 5), (2, 4), (3, 3), (4, 2), (5, 1)],
    )
    def test_inversion_formula(
        self, init_engine: StoryEngine, visibility: int, expected_health: int
    ) -> None:
        init_engine.state.savar.visibility_level = visibility
        assert init_engine.apply_savar_inversion() == expected_health


# ---------------------------------------------------------------------------
# Chain rule: apply_neel_management_threshold (RULE_05)
# ---------------------------------------------------------------------------


class TestApplyNeelManagementThreshold:
    def test_cold_phase_full_capacity(self, init_engine: StoryEngine) -> None:
        init_engine.state.ranveer.phase = RanveerPhase.COLD
        capacity = init_engine.apply_neel_management_threshold()
        assert capacity == 1.0

    def test_irritated_phase_full_capacity(self, init_engine: StoryEngine) -> None:
        init_engine.state.ranveer.phase = RanveerPhase.IRRITATED
        capacity = init_engine.apply_neel_management_threshold()
        assert capacity == 1.0

    def test_obsessed_phase_reduced_capacity(self, init_engine: StoryEngine) -> None:
        init_engine.state.ranveer.phase = RanveerPhase.OBSESSED
        capacity = init_engine.apply_neel_management_threshold()
        assert capacity == pytest.approx(0.70)

    def test_personal_phase_heavily_reduced(self, init_engine: StoryEngine) -> None:
        init_engine.state.ranveer.phase = RanveerPhase.PERSONAL
        capacity = init_engine.apply_neel_management_threshold()
        assert capacity == pytest.approx(0.40)

    def test_capacity_written_to_neel_state(self, init_engine: StoryEngine) -> None:
        init_engine.state.ranveer.phase = RanveerPhase.OBSESSED
        init_engine.apply_neel_management_threshold()
        assert init_engine.state.neel.effective_capacity == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# Chain rule: apply_kavya_threshold (RULE_06)
# ---------------------------------------------------------------------------


class TestApplyKavyaThreshold:
    def test_both_conditions_activates_kavya(self, init_engine: StoryEngine) -> None:
        result = init_engine.apply_kavya_threshold(True, True)
        assert result is True
        assert init_engine.state.kavya.is_active is True

    def test_only_condition_a_not_active(self, init_engine: StoryEngine) -> None:
        result = init_engine.apply_kavya_threshold(True, False)
        assert result is False
        assert init_engine.state.kavya.is_active is False

    def test_only_condition_b_not_active(self, init_engine: StoryEngine) -> None:
        result = init_engine.apply_kavya_threshold(False, True)
        assert result is False

    def test_neither_condition_not_active(self, init_engine: StoryEngine) -> None:
        result = init_engine.apply_kavya_threshold(False, False)
        assert result is False

    def test_conditions_written_to_kavya_state(self, init_engine: StoryEngine) -> None:
        init_engine.apply_kavya_threshold(True, False)
        assert init_engine.state.kavya.condition_a_met is True
        assert init_engine.state.kavya.condition_b_met is False

    def test_kavya_can_deactivate(self, init_engine: StoryEngine) -> None:
        init_engine.apply_kavya_threshold(True, True)
        assert init_engine.state.kavya.is_active is True
        init_engine.apply_kavya_threshold(False, False)
        assert init_engine.state.kavya.is_active is False


# ---------------------------------------------------------------------------
# Chain rule: apply_meera_transformation (RULE_07)
# ---------------------------------------------------------------------------


class TestApplyMeeraTransformation:
    def test_flag_added_to_lived_through(self, init_engine: StoryEngine) -> None:
        init_engine.apply_meera_transformation(WorldFlag.CULTURAL_FEST)
        assert WorldFlag.CULTURAL_FEST in init_engine.state.meera.flags_lived_through

    def test_responses_expanded(self, init_engine: StoryEngine) -> None:
        before = len(init_engine.state.meera.response_set)
        init_engine.apply_meera_transformation(WorldFlag.CULTURAL_FEST)
        assert len(init_engine.state.meera.response_set) > before

    def test_specific_responses_unlocked(self, init_engine: StoryEngine) -> None:
        init_engine.apply_meera_transformation(WorldFlag.CULTURAL_FEST)
        assert "exposed" in init_engine.state.meera.response_set

    def test_responses_accumulate_across_flags(self, init_engine: StoryEngine) -> None:
        init_engine.apply_meera_transformation(WorldFlag.CULTURAL_FEST)
        init_engine.apply_meera_transformation(WorldFlag.INCIDENT_AFTERMATH)
        responses = init_engine.state.meera.response_set
        assert "exposed" in responses
        assert "pragmatic" in responses

    def test_same_flag_twice_is_idempotent(self, init_engine: StoryEngine) -> None:
        init_engine.apply_meera_transformation(WorldFlag.EXAM_SEASON)
        size_after_first = len(init_engine.state.meera.response_set)
        init_engine.apply_meera_transformation(WorldFlag.EXAM_SEASON)
        assert len(init_engine.state.meera.response_set) == size_after_first

    def test_returns_updated_response_set(self, init_engine: StoryEngine) -> None:
        result = init_engine.apply_meera_transformation(WorldFlag.ELECTION_SEASON)
        assert "politically_legible" in result


# ---------------------------------------------------------------------------
# Chain rule: apply_rajan_constant (RULE_08)
# ---------------------------------------------------------------------------


class TestApplyRajanConstant:
    def test_no_direction_returns_true(self, init_engine: StoryEngine) -> None:
        init_engine.state.rajan.has_direction_from_vikram = False
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        assert init_engine.apply_rajan_constant(trigger) is True

    def test_has_direction_returns_false(self, init_engine: StoryEngine) -> None:
        init_engine.state.rajan.has_direction_from_vikram = True
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        assert init_engine.apply_rajan_constant(trigger) is False


# ---------------------------------------------------------------------------
# Chain rule: check_surya_reveal (RULE_09)
# ---------------------------------------------------------------------------


class TestCheckSuryaReveal:
    def test_crisis_phase_reveals(self, init_engine: StoryEngine) -> None:
        result = init_engine.check_surya_reveal(
            ConflictPhase.CRISIS, confronted=False, operationally_necessary=False
        )
        assert result is SuryaAllegiance.WITH_VIKRAM

    def test_confronted_reveals(self, init_engine: StoryEngine) -> None:
        result = init_engine.check_surya_reveal(
            ConflictPhase.FRICTION, confronted=True, operationally_necessary=False
        )
        assert result is SuryaAllegiance.WITH_VIKRAM

    def test_operationally_necessary_reveals(self, init_engine: StoryEngine) -> None:
        result = init_engine.check_surya_reveal(
            ConflictPhase.FRICTION, confronted=False, operationally_necessary=True
        )
        assert result is SuryaAllegiance.WITH_VIKRAM

    def test_no_conditions_returns_none(self, init_engine: StoryEngine) -> None:
        result = init_engine.check_surya_reveal(
            ConflictPhase.FRICTION, confronted=False, operationally_necessary=False
        )
        assert result is None

    def test_reveal_sets_is_revealed_flag(self, init_engine: StoryEngine) -> None:
        init_engine.check_surya_reveal(
            ConflictPhase.CRISIS, confronted=False, operationally_necessary=False
        )
        assert init_engine.state.surya.is_revealed is True

    def test_no_reveal_leaves_flag_false(self, init_engine: StoryEngine) -> None:
        init_engine.check_surya_reveal(
            ConflictPhase.FRICTION, confronted=False, operationally_necessary=False
        )
        assert init_engine.state.surya.is_revealed is False

    def test_reveals_correct_true_state(self, engine: StoryEngine) -> None:
        engine.initialize_story(_default_params(surya_true_state="RANVEER_PLANT"))
        result = engine.check_surya_reveal(
            ConflictPhase.CRISIS, confronted=False, operationally_necessary=False
        )
        assert result is SuryaAllegiance.RANVEER_PLANT


# ---------------------------------------------------------------------------
# Chain rule: check_arjun_window (RULE_10)
# ---------------------------------------------------------------------------


class TestCheckArjunWindow:
    def test_obsessed_with_acts_true_opens_window(self, engine: StoryEngine) -> None:
        engine.initialize_story(
            _default_params(ranveer_phase_start="OBSESSED", arjun_acts_in_window=True)
        )
        assert engine.check_arjun_window() is True
        assert engine.state.arjun.window_is_open is True

    def test_obsessed_with_acts_false_window_not_active(
        self, engine: StoryEngine
    ) -> None:
        engine.initialize_story(
            _default_params(ranveer_phase_start="OBSESSED", arjun_acts_in_window=False)
        )
        assert engine.check_arjun_window() is False
        # But the window IS open (Arjun just doesn't act in it)
        assert engine.state.arjun.window_is_open is True

    def test_personal_phase_closes_window(self, engine: StoryEngine) -> None:
        engine.initialize_story(
            _default_params(ranveer_phase_start="PERSONAL", arjun_acts_in_window=True)
        )
        assert engine.check_arjun_window() is False
        assert engine.state.arjun.window_is_open is False

    def test_cold_phase_window_closed(self, init_engine: StoryEngine) -> None:
        assert init_engine.check_arjun_window() is False


# ---------------------------------------------------------------------------
# fire_trigger
# ---------------------------------------------------------------------------


class TestFireTrigger:
    def test_logs_incident(self, init_engine: StoryEngine) -> None:
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "Vikram refused.")
        init_engine.fire_trigger(trigger)
        assert len(init_engine.state.incident_log) == 1

    def test_incident_has_correct_description(self, init_engine: StoryEngine) -> None:
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "Vikram refused.")
        init_engine.fire_trigger(trigger)
        entry = init_engine.state.incident_log[0]
        assert entry.description == "Vikram refused."

    def test_direct_challenge_advances_conflict_phase(
        self, init_engine: StoryEngine
    ) -> None:
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        init_engine.fire_trigger(trigger)
        assert init_engine.state.conflict_phase is ConflictPhase.FRICTION

    def test_two_direct_challenges_advance_to_open_conflict(
        self, init_engine: StoryEngine
    ) -> None:
        t1 = make_vikram_refusal(LocationName.MAIN_CANTEEN, "first refusal")
        t2 = make_public_humiliation(
            LocationName.MAIN_CANTEEN, "ranveer", "vikram", "humiliation"
        )
        init_engine.fire_trigger(t1)
        init_engine.fire_trigger(t2)
        assert init_engine.state.conflict_phase is ConflictPhase.OPEN_CONFLICT

    def test_physical_confrontation_activates_karan(
        self, init_engine: StoryEngine
    ) -> None:
        trigger = make_physical_confrontation(
            LocationName.DEAD_PATHS, "karan", "vikram", "Karan cornered Vikram."
        )
        init_engine.fire_trigger(trigger)
        assert init_engine.state.karan.is_activated is True

    def test_dhruv_contact_applies_positive_cost(self, init_engine: StoryEngine) -> None:
        before = init_engine.state.dhruv.cost_benefit_total
        trigger = make_dhruv_contact("Neel offered Dhruv something.")
        init_engine.fire_trigger(trigger)
        assert init_engine.state.dhruv.cost_benefit_total > before

    def test_caller_override_dhruv_cost(self, init_engine: StoryEngine) -> None:
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        init_engine.fire_trigger(trigger, dhruv_event_cost=2.0)
        assert init_engine.state.dhruv.cost_benefit_total == 2.0

    def test_vikram_last_trigger_type_updated(self, init_engine: StoryEngine) -> None:
        trigger = make_notice_board_move("A list was posted.")
        init_engine.fire_trigger(trigger)
        assert init_engine.state.vikram.last_trigger_type == "INSTITUTIONAL_MOVE"

    def test_savar_alone_bumps_visibility(self, init_engine: StoryEngine) -> None:
        before = init_engine.state.savar.visibility_level
        trigger = make_gang_member_acts_alone(
            "savar", LocationName.MAIN_CANTEEN, "Savar said something irreversible."
        )
        init_engine.fire_trigger(trigger)
        assert init_engine.state.savar.visibility_level == before + 1

    def test_incident_is_public_set_correctly(self, init_engine: StoryEngine) -> None:
        trigger = make_public_humiliation(
            LocationName.MAIN_CANTEEN, "ranveer", "vikram", "Public mock."
        )
        init_engine.fire_trigger(trigger)
        assert init_engine.state.incident_log[0].is_public is True

    def test_incident_step_is_current_step(self, init_engine: StoryEngine) -> None:
        init_engine.advance_state()  # step = 1
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        init_engine.fire_trigger(trigger)
        assert init_engine.state.incident_log[0].step == 1


# ---------------------------------------------------------------------------
# get_current_state
# ---------------------------------------------------------------------------


class TestGetCurrentState:
    def test_returns_dict(self, init_engine: StoryEngine) -> None:
        snapshot = init_engine.get_current_state()
        assert isinstance(snapshot, dict)

    def test_contains_step_key(self, init_engine: StoryEngine) -> None:
        snapshot = init_engine.get_current_state()
        assert "step" in snapshot

    def test_step_value_correct(self, init_engine: StoryEngine) -> None:
        init_engine.advance_state()
        snapshot = init_engine.get_current_state()
        assert snapshot["step"] == 1


# ---------------------------------------------------------------------------
# advance_state
# ---------------------------------------------------------------------------


class TestAdvanceState:
    def test_increments_step(self, init_engine: StoryEngine) -> None:
        init_engine.advance_state()
        assert init_engine.state.step == 1

    def test_increments_step_repeatedly(self, init_engine: StoryEngine) -> None:
        for _ in range(5):
            init_engine.advance_state()
        assert init_engine.state.step == 5

    def test_time_of_day_set_when_provided(self, init_engine: StoryEngine) -> None:
        init_engine.advance_state(time_of_day=TimeOfDay.NIGHT)
        assert init_engine.state.time_of_day is TimeOfDay.NIGHT

    def test_time_of_day_unchanged_when_not_provided(
        self, init_engine: StoryEngine
    ) -> None:
        original = init_engine.state.time_of_day
        init_engine.advance_state()
        assert init_engine.state.time_of_day is original

    def test_neel_threshold_recomputed(self, init_engine: StoryEngine) -> None:
        init_engine.state.ranveer.phase = RanveerPhase.OBSESSED
        init_engine.advance_state()
        assert init_engine.state.neel.effective_capacity == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# check_resolution_condition
# ---------------------------------------------------------------------------


class TestCheckResolutionCondition:
    def test_r3_pyrrhic_met_when_dhruv_gone(self, engine: StoryEngine) -> None:
        engine.initialize_story(_default_params(resolution_type="R3_PYRRHIC"))
        from story_engine.characters import DhruvDriftState as DDS

        engine.state.dhruv.drift_state = DDS.GONE
        assert engine.check_resolution_condition() is ResolutionType.R3_PYRRHIC

    def test_r3_not_met_when_dhruv_present(self, engine: StoryEngine) -> None:
        engine.initialize_story(_default_params(resolution_type="R3_PYRRHIC"))
        assert engine.check_resolution_condition() is None

    def test_r1_not_met_at_cold_equilibrium(self, init_engine: StoryEngine) -> None:
        assert init_engine.check_resolution_condition() is None

    def test_r4_met_at_step_6_in_friction(self, engine: StoryEngine) -> None:
        engine.initialize_story(
            _default_params(
                resolution_type="R4_SUSPENDED",
                initial_conflict_phase="FRICTION",
            )
        )
        engine.state.step = 6
        assert engine.check_resolution_condition() is ResolutionType.R4_SUSPENDED

    def test_r4_not_met_before_step_6(self, engine: StoryEngine) -> None:
        engine.initialize_story(
            _default_params(
                resolution_type="R4_SUSPENDED",
                initial_conflict_phase="FRICTION",
            )
        )
        engine.state.step = 5
        assert engine.check_resolution_condition() is None

    def test_r2_met_when_ranveer_personal_and_neel_depleted(
        self, engine: StoryEngine
    ) -> None:
        engine.initialize_story(
            _default_params(
                resolution_type="R2_VISIBLE_WIN",
                ranveer_phase_start="PERSONAL",
                initial_conflict_phase="OPEN_CONFLICT",
            )
        )
        engine.state.neel.effective_capacity = 0.40
        assert engine.check_resolution_condition() is ResolutionType.R2_VISIBLE_WIN

    def test_r5_met_when_resolution_one_side_up_and_karan_unfinished(
        self, engine: StoryEngine
    ) -> None:
        engine.initialize_story(
            _default_params(
                resolution_type="R5_STRUCTURAL",
                initial_conflict_phase="RESOLUTION_ONE_SIDE_UP",
            )
        )
        engine.state.karan.unfinished_feeling = True
        assert engine.check_resolution_condition() is ResolutionType.R5_STRUCTURAL


# ---------------------------------------------------------------------------
# generate_scene_brief
# ---------------------------------------------------------------------------


class TestGenerateSceneBrief:
    def test_returns_scene_brief(self, init_engine: StoryEngine) -> None:
        from story_engine.brief_generator import SceneBrief

        brief = init_engine.generate_scene_brief(LocationName.MAIN_CANTEEN)
        assert isinstance(brief, SceneBrief)

    def test_brief_conflict_phase_correct(self, init_engine: StoryEngine) -> None:
        brief = init_engine.generate_scene_brief(LocationName.MAIN_CANTEEN)
        assert brief.world_state.conflict_phase == "COLD_EQUILIBRIUM"

    def test_brief_location_name_correct(self, init_engine: StoryEngine) -> None:
        brief = init_engine.generate_scene_brief(LocationName.DEAD_PATHS)
        assert brief.location.name == "DEAD_PATHS"

    def test_brief_has_characters(self, init_engine: StoryEngine) -> None:
        brief = init_engine.generate_scene_brief(LocationName.MAIN_CANTEEN)
        assert len(brief.characters_in_scene) > 0

    def test_brief_vikram_always_present(self, init_engine: StoryEngine) -> None:
        brief = init_engine.generate_scene_brief(LocationName.MAIN_CANTEEN)
        names = [c.name for c in brief.characters_in_scene]
        assert "vikram" in names

    def test_brief_no_empty_required_fields(self, init_engine: StoryEngine) -> None:
        brief = init_engine.generate_scene_brief(LocationName.MAIN_CANTEEN)
        assert brief.scene_goal != ""
        assert len(brief.emotional_arc) > 0
        assert len(brief.what_must_be_shown_not_told) > 0
        assert len(brief.prior_context) > 0
        assert len(brief.what_must_not_happen) > 0


# ---------------------------------------------------------------------------
# FIX 2: Auto-CRISIS sequencing — pride ratchet must not skip CRISIS in one step
# ---------------------------------------------------------------------------


class TestCrisisSequencing:
    """RULE_01: CRISIS requires Ranveer to have been at PERSONAL before the trigger.

    A single trigger must not promote Ranveer from OBSESSED to PERSONAL *and*
    simultaneously advance the phase to CRISIS. The snapshot-before-ratchet
    pattern ensures this.
    """

    def test_obsessed_plus_open_conflict_does_not_immediately_become_crisis(
        self,
    ) -> None:
        """OBSESSED → fires trigger → should land at PERSONAL, NOT CRISIS."""
        engine = StoryEngine()
        engine.initialize_story(
            _default_params(
                ranveer_phase_start="OBSESSED",
                initial_conflict_phase="OPEN_CONFLICT",
            )
        )
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "Vikram refuses.")
        engine.fire_trigger(trigger)

        # Phase advances to PERSONAL via the pride ratchet — that is correct.
        assert engine.state.ranveer.phase is RanveerPhase.PERSONAL
        # But a single trigger must NOT skip to CRISIS.
        assert engine.state.conflict_phase is not ConflictPhase.CRISIS

    def test_personal_plus_open_conflict_does_become_crisis(self) -> None:
        """PERSONAL (pre-existing) + OPEN_CONFLICT → direct challenge → CRISIS."""
        engine = StoryEngine()
        engine.initialize_story(
            _default_params(
                ranveer_phase_start="PERSONAL",
                initial_conflict_phase="OPEN_CONFLICT",
            )
        )
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "Vikram refuses again.")
        engine.fire_trigger(trigger)

        # Now CRISIS is valid because Ranveer was ALREADY at PERSONAL.
        assert engine.state.conflict_phase is ConflictPhase.CRISIS

    def test_cold_equilibrium_direct_challenge_moves_to_friction_not_crisis(
        self,
    ) -> None:
        """COLD_EQUILIBRIUM should step to FRICTION, not jump to CRISIS."""
        engine = StoryEngine()
        engine.initialize_story(
            _default_params(
                ranveer_phase_start="PERSONAL",
                initial_conflict_phase="COLD_EQUILIBRIUM",
            )
        )
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "First move.")
        engine.fire_trigger(trigger)
        assert engine.state.conflict_phase is ConflictPhase.FRICTION

    def test_force_crisis_bypasses_conditions(self) -> None:
        """force_crisis() must unconditionally set CRISIS regardless of phase."""
        engine = StoryEngine()
        engine.initialize_story(_default_params())
        assert engine.state.conflict_phase is ConflictPhase.COLD_EQUILIBRIUM
        engine.force_crisis("External irreversible event.")
        assert engine.state.conflict_phase is ConflictPhase.CRISIS


# ---------------------------------------------------------------------------
# FIX 1: Trigger participants appear in SceneBrief characters_in_scene
# ---------------------------------------------------------------------------


class TestRanveerPresenceInBrief:
    """Trigger initiator and target must appear in characters_in_scene.

    Before this fix, Ranveer could appear in the incident log as the initiator
    of a public humiliation, yet be absent from the scene brief because
    _who_is_present() only used location-based heuristics.
    """

    def test_trigger_initiator_appears_in_brief(self) -> None:
        """Ranveer as trigger initiator must be in characters_in_scene."""
        engine = StoryEngine()
        engine.initialize_story(_default_params())
        trigger = make_public_humiliation(
            LocationName.MAIN_CANTEEN, "ranveer", "vikram", "Public mock."
        )
        brief = engine.generate_scene_brief(LocationName.MAIN_CANTEEN, trigger)
        names = [c.name for c in brief.characters_in_scene]
        assert "ranveer" in names

    def test_trigger_target_appears_in_brief(self) -> None:
        """When dhruv is targeted, he must appear in characters_in_scene."""
        engine = StoryEngine()
        engine.initialize_story(_default_params())
        trigger = make_dhruv_contact("Neel offered Dhruv something.")
        # trigger.target == "dhruv", location is SECONDARY_CANTEEN
        brief = engine.generate_scene_brief(LocationName.SECONDARY_CANTEEN, trigger)
        names = [c.name for c in brief.characters_in_scene]
        assert "dhruv" in names

    def test_no_trigger_does_not_crash(self) -> None:
        """generate_scene_brief(location, trigger=None) must work unchanged."""
        engine = StoryEngine()
        engine.initialize_story(_default_params())
        brief = engine.generate_scene_brief(LocationName.MAIN_CANTEEN, None)
        names = [c.name for c in brief.characters_in_scene]
        assert "vikram" in names

    def test_no_duplicate_characters_when_location_overlaps_trigger(self) -> None:
        """If trigger initiator is also naturally at the location, no duplicate."""
        engine = StoryEngine()
        engine.initialize_story(_default_params())
        # neel is naturally at STUDENT_UNION_BUILDING (Neel controls it)
        trigger = make_election_positioning("Neel positioned a candidate.")
        brief = engine.generate_scene_brief(LocationName.STUDENT_UNION_BUILDING, trigger)
        names = [c.name for c in brief.characters_in_scene]
        neel_count = names.count("neel")
        assert neel_count == 1


# ---------------------------------------------------------------------------
# FIX 3: Trigger-variant-aware scene goal and emotional arc
# ---------------------------------------------------------------------------


class TestTriggerAwareSceneGoal:
    """SceneBrief scene_goal and emotional_arc must vary by trigger variant.

    The same conflict phase must produce a different goal when a specific
    trigger variant is known versus the phase-level fallback.
    """

    def test_physical_confrontation_gives_specific_scene_goal(self) -> None:
        """DIRECT_CHALLENGE_PHYSICAL_CONFRONTATION → Karan-specific goal."""
        engine = StoryEngine()
        engine.initialize_story(_default_params())
        trigger = make_physical_confrontation(
            LocationName.DEAD_PATHS, "karan", "vikram", "Karan cornered Vikram."
        )
        brief = engine.generate_scene_brief(LocationName.DEAD_PATHS, trigger)
        assert "physical" in brief.scene_goal.lower()
        assert "karan" in brief.scene_goal.lower()

    def test_academic_threat_gives_specific_scene_goal(self) -> None:
        """INSTITUTIONAL_ACADEMIC_THREAT → invisible-move goal."""
        engine = StoryEngine()
        engine.initialize_story(_default_params())
        trigger = make_academic_threat("Marks sheet compromised.", target="vikram")
        brief = engine.generate_scene_brief(LocationName.FACULTY_BUILDINGS, trigger)
        assert "invisible" in brief.scene_goal.lower()

    def test_meera_intersection_gives_specific_scene_goal(self) -> None:
        """AMBIENT_MEERA_INTERSECTION → undefined-response goal."""
        engine = StoryEngine()
        engine.initialize_story(_default_params())
        trigger = make_meera_intersection(
            LocationName.GIRLS_HOSTEL, "Meera spoke at the wrong moment."
        )
        brief = engine.generate_scene_brief(LocationName.GIRLS_HOSTEL, trigger)
        assert "meera" in brief.scene_goal.lower()

    def test_no_trigger_falls_back_to_phase_goal(self) -> None:
        """Without a trigger, goal is derived from phase only."""
        engine = StoryEngine()
        engine.initialize_story(_default_params())
        brief = engine.generate_scene_brief(LocationName.MAIN_CANTEEN)
        # COLD_EQUILIBRIUM phase fallback should mention tension / knowing
        goal = brief.scene_goal.lower()
        assert "tension" in goal or "knowing" in goal or "action" in goal

    def test_physical_confrontation_gives_four_beat_arc(self) -> None:
        """Physical confrontation must produce a 4-beat emotional arc."""
        engine = StoryEngine()
        engine.initialize_story(_default_params())
        trigger = make_physical_confrontation(
            LocationName.DEAD_PATHS, "karan", "vikram", "Karan cornered Vikram."
        )
        brief = engine.generate_scene_brief(LocationName.DEAD_PATHS, trigger)
        assert len(brief.emotional_arc) == 4

    def test_no_trigger_gives_three_beat_arc(self) -> None:
        """Without a trigger at non-CRISIS phase, arc is 3 beats."""
        engine = StoryEngine()
        engine.initialize_story(_default_params())
        brief = engine.generate_scene_brief(LocationName.MAIN_CANTEEN)
        assert len(brief.emotional_arc) == 3

    def test_crisis_phase_gives_four_beat_arc(self) -> None:
        """CRISIS phase (no trigger) must also produce a 4-beat arc."""
        engine = StoryEngine()
        engine.initialize_story(_default_params(initial_conflict_phase="CRISIS"))
        brief = engine.generate_scene_brief(LocationName.MAIN_GROUND)
        assert len(brief.emotional_arc) == 4


# ---------------------------------------------------------------------------
# Incident structured fields (initiator / target / variant)
# ---------------------------------------------------------------------------


class TestIncidentStructuredFields:
    """IncidentEntry must capture initiator, target, and variant from the trigger."""

    def test_incident_initiator_recorded(self, init_engine: StoryEngine) -> None:
        trigger = make_public_humiliation(
            LocationName.MAIN_CANTEEN, "ranveer", "vikram", "desc"
        )
        init_engine.fire_trigger(trigger)
        assert init_engine.state.incident_log[0].initiator == "ranveer"

    def test_incident_target_recorded(self, init_engine: StoryEngine) -> None:
        trigger = make_public_humiliation(
            LocationName.MAIN_CANTEEN, "ranveer", "vikram", "desc"
        )
        init_engine.fire_trigger(trigger)
        assert init_engine.state.incident_log[0].target == "vikram"

    def test_incident_variant_recorded(self, init_engine: StoryEngine) -> None:
        trigger = make_public_humiliation(
            LocationName.MAIN_CANTEEN, "ranveer", "vikram", "desc"
        )
        init_engine.fire_trigger(trigger)
        entry = init_engine.state.incident_log[0]
        assert entry.variant == TriggerVariant.DIRECT_CHALLENGE_PUBLIC_HUMILIATION.name


# ---------------------------------------------------------------------------
# Dhruv drift: 3 consecutive net-negative events trigger drift advance
# ---------------------------------------------------------------------------


class TestDhruvDriftIntegration:
    """Three consecutive negative dhruv_event_cost values advance DhruvDriftState."""

    def test_three_negative_events_advance_drift(self) -> None:
        engine = StoryEngine()
        engine.initialize_story(_default_params(dhruv_cost_start=0.0))
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        for _ in range(3):
            engine.fire_trigger(trigger, dhruv_event_cost=-1.0)
        assert engine.state.dhruv.drift_state is DhruvDriftState.LESS_AVAILABLE

    def test_positive_event_resets_consecutive_negative_counter(self) -> None:
        """Two negatives, then a positive, must not advance drift."""
        engine = StoryEngine()
        engine.initialize_story(_default_params(dhruv_cost_start=0.0))
        trigger = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        engine.fire_trigger(trigger, dhruv_event_cost=-1.0)
        engine.fire_trigger(trigger, dhruv_event_cost=-1.0)
        engine.fire_trigger(trigger, dhruv_event_cost=+5.0)  # breaks streak
        from story_engine.characters import DhruvDriftState as DDS

        assert engine.state.dhruv.drift_state is DDS.PRESENT
