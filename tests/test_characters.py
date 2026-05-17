"""Tests for characters.py — enums, dataclasses, and validation."""

from __future__ import annotations

import pytest

from story_engine.characters import (
    ArjunState,
    CoreTrait,
    DhruvDriftState,
    DhruvState,
    KaranState,
    KavyaState,
    MeeraState,
    NeelState,
    RajanState,
    RanveerPhase,
    RanveerState,
    SavarState,
    SuryaState,
    SuryaAllegiance,
    VikramState,
    Year,
)
from story_engine.flags import WorldFlag


# ---------------------------------------------------------------------------
# Enum sanity checks
# ---------------------------------------------------------------------------


class TestRanveerPhase:
    def test_integer_values_ordered(self) -> None:
        phases = [
            RanveerPhase.COLD,
            RanveerPhase.IRRITATED,
            RanveerPhase.OBSESSED,
            RanveerPhase.PERSONAL,
        ]
        values = [p.value for p in phases]
        assert values == sorted(values)

    def test_arithmetic_advance(self) -> None:
        """Engine uses integer arithmetic on phase values."""
        phase = RanveerPhase.COLD
        next_val = phase.value + 1
        assert RanveerPhase(next_val) is RanveerPhase.IRRITATED

    def test_plus_two_jump(self) -> None:
        """Apparent weakness revealed as strategy → +2."""
        phase = RanveerPhase.IRRITATED
        jumped = RanveerPhase(min(phase.value + 2, RanveerPhase.PERSONAL.value))
        assert jumped is RanveerPhase.PERSONAL

    def test_phase_clamped_at_personal(self) -> None:
        """Phase cannot exceed PERSONAL."""
        phase = RanveerPhase.PERSONAL
        clamped = RanveerPhase(min(phase.value + 1, RanveerPhase.PERSONAL.value))
        assert clamped is RanveerPhase.PERSONAL


class TestDhruvDriftState:
    def test_all_states_present(self) -> None:
        names = {s.name for s in DhruvDriftState}
        assert names == {
            "PRESENT",
            "LESS_AVAILABLE",
            "PRESENT_BUT_UNINVESTED",
            "MAKING_EXIT_ARRANGEMENTS",
            "GONE",
        }

    def test_gone_is_last(self) -> None:
        states = list(DhruvDriftState)
        assert states[-1] is DhruvDriftState.GONE


class TestSuryaState:
    def test_all_variants_present(self) -> None:
        names = {s.name for s in SuryaAllegiance}
        assert names == {"WITH_VIKRAM", "RANVEER_PLANT", "OWN_AGENDA", "DRIFTER"}


class TestWorldFlagInCharacters:
    def test_world_flag_importable(self) -> None:
        assert WorldFlag.SEMESTER_OPENING is WorldFlag.SEMESTER_OPENING


# ---------------------------------------------------------------------------
# Default construction — all character states
# ---------------------------------------------------------------------------


class TestDefaultConstruction:
    def test_vikram_defaults(self) -> None:
        v = VikramState()
        assert v.year is Year.SECOND
        assert v.core_trait is CoreTrait.PRIDE
        assert v.last_trigger_type is None

    def test_ranveer_defaults(self) -> None:
        r = RanveerState()
        assert r.year is Year.THIRD
        assert r.phase is RanveerPhase.COLD
        assert r.consecutive_unacknowledged_non_submissions == 0
        assert r.last_weakness_was_strategy is False

    def test_karan_defaults(self) -> None:
        k = KaranState()
        assert k.year is Year.THIRD
        assert k.unfinished_feeling is True
        assert k.is_activated is False

    def test_neel_defaults(self) -> None:
        n = NeelState()
        assert n.year is Year.THIRD
        assert n.effective_capacity == 1.0

    def test_arjun_defaults(self) -> None:
        a = ArjunState()
        assert a.year is Year.SECOND
        assert a.arjun_acts_in_window is False
        assert a.window_is_open is False

    def test_savar_defaults(self) -> None:
        s = SavarState()
        assert s.year is Year.SECOND
        assert s.visibility_level == 1

    def test_dhruv_defaults(self) -> None:
        d = DhruvState()
        assert d.year is Year.SECOND
        assert d.cost_benefit_total == 0.0
        assert d.consecutive_negative_events == 0
        assert d.drift_state is DhruvDriftState.PRESENT

    def test_rajan_defaults(self) -> None:
        r = RajanState()
        assert r.year is Year.SECOND
        assert r.has_direction_from_vikram is False

    def test_surya_defaults(self) -> None:
        s = SuryaState()
        assert s.year is Year.SECOND
        assert s.true_state is SuryaAllegiance.WITH_VIKRAM
        assert s.is_revealed is False

    def test_kavya_defaults(self) -> None:
        k = KavyaState()
        assert k.is_active is False
        assert k.condition_a_met is False
        assert k.condition_b_met is False

    def test_meera_defaults(self) -> None:
        m = MeeraState()
        assert m.year is Year.FIRST
        assert m.core_trait is CoreTrait.BECOMING
        assert m.flags_lived_through == frozenset()
        assert m.response_set == frozenset()


# ---------------------------------------------------------------------------
# SavarState validation
# ---------------------------------------------------------------------------


class TestSavarValidation:
    def test_valid_range_boundaries(self) -> None:
        assert SavarState(visibility_level=1).visibility_level == 1
        assert SavarState(visibility_level=5).visibility_level == 5

    def test_invalid_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="visibility_level"):
            SavarState(visibility_level=0)

    def test_invalid_six_raises(self) -> None:
        with pytest.raises(ValueError, match="visibility_level"):
            SavarState(visibility_level=6)


# ---------------------------------------------------------------------------
# MeeraState flag accumulation (simulated engine step)
# ---------------------------------------------------------------------------


class TestMeeraFlagAccumulation:
    def test_flags_accumulate_as_frozenset(self) -> None:
        m = MeeraState()
        # Simulate engine: add a flag
        m.flags_lived_through = m.flags_lived_through | {WorldFlag.CULTURAL_FEST}
        assert WorldFlag.CULTURAL_FEST in m.flags_lived_through

    def test_duplicate_flag_does_not_grow_set(self) -> None:
        m = MeeraState()
        m.flags_lived_through = m.flags_lived_through | {WorldFlag.CULTURAL_FEST}
        m.flags_lived_through = m.flags_lived_through | {WorldFlag.CULTURAL_FEST}
        assert len(m.flags_lived_through) == 1

    def test_multiple_flags_tracked(self) -> None:
        m = MeeraState()
        for flag in [
            WorldFlag.SEMESTER_OPENING,
            WorldFlag.CULTURAL_FEST,
            WorldFlag.INCIDENT_AFTERMATH,
        ]:
            m.flags_lived_through = m.flags_lived_through | {flag}
        assert len(m.flags_lived_through) == 3


# ---------------------------------------------------------------------------
# SuryaCharacter reveal logic (state mutation)
# ---------------------------------------------------------------------------


class TestSuryaReveal:
    def test_not_revealed_by_default(self) -> None:
        s = SuryaState(true_state=SuryaAllegiance.RANVEER_PLANT)
        assert s.is_revealed is False

    def test_reveal_exposes_true_state(self) -> None:
        s = SuryaState(true_state=SuryaAllegiance.RANVEER_PLANT)
        s.is_revealed = True
        assert s.true_state is SuryaAllegiance.RANVEER_PLANT


# ---------------------------------------------------------------------------
# NeelState capacity
# ---------------------------------------------------------------------------


class TestNeelCapacity:
    def test_full_capacity_default(self) -> None:
        n = NeelState()
        assert n.effective_capacity == 1.0

    def test_capacity_can_be_reduced(self) -> None:
        """Simulates what apply_neel_management_threshold will set."""
        n = NeelState()
        n.effective_capacity = 0.70
        assert n.effective_capacity == pytest.approx(0.70)
