"""Tests for world_state.py — enums, dataclasses, and JSON round-trip."""

from __future__ import annotations

import json

import pytest

from story_engine.characters import DhruvDriftState, RanveerPhase, SuryaAllegiance
from story_engine.flags import FlagSet, WorldFlag
from story_engine.world_state import (
    ConflictPhase,
    IncidentEntry,
    RelationshipState,
    ResolutionType,
    TimeOfDay,
    WorldState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_state() -> WorldState:
    """Construct a WorldState with all required fields and sensible defaults."""
    return WorldState(
        active_flags=FlagSet(flags=frozenset({WorldFlag.SEMESTER_OPENING})),
        conflict_phase=ConflictPhase.COLD_EQUILIBRIUM,
        resolution_type=ResolutionType.R1_VISIBLE_DEFEAT,
        time_of_day=TimeOfDay.AFTERNOON,
    )


def _rich_state() -> WorldState:
    """Construct a WorldState with non-default character values for round-trip tests."""
    state = _minimal_state()
    state.step = 3
    state.conflict_phase = ConflictPhase.OPEN_CONFLICT
    state.ranveer.phase = RanveerPhase.OBSESSED
    state.ranveer.consecutive_unacknowledged_non_submissions = 4
    state.dhruv.drift_state = DhruvDriftState.LESS_AVAILABLE
    state.dhruv.cost_benefit_total = -2.5
    state.surya.true_state = SuryaAllegiance.RANVEER_PLANT
    state.meera.flags_lived_through = frozenset(
        {WorldFlag.CULTURAL_FEST, WorldFlag.INCIDENT_AFTERMATH}
    )
    state.meera.response_set = frozenset({"withdrawn", "watchful"})
    state.relationship_graph[("vikram", "ranveer")] = RelationshipState(
        tension=8,
        trust=0,
        history_notes=("Year-1 confrontation — Vikram did not break.",),
        is_public=True,
    )
    state.incident_log.append(
        IncidentEntry(
            step=1,
            trigger_type="DIRECT_CHALLENGE",
            variant="DIRECT_CHALLENGE_PUBLIC_CALLOUT",
            location_name="MAIN_CANTEEN",
            initiator="ranveer",
            target="vikram",
            description="Ranveer publicly challenged Vikram over a seat.",
            consequence_notes=("Vikram responded immediately.", "Gang morale visible."),
            is_public=True,
        )
    )
    return state


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestConflictPhase:
    def test_all_phases_present(self) -> None:
        names = {p.name for p in ConflictPhase}
        assert names == {
            "COLD_EQUILIBRIUM",
            "FRICTION",
            "OPEN_CONFLICT",
            "RESOLUTION_ONE_SIDE_UP",
            "PYRRHIC",
            "CRISIS",
        }

    def test_crisis_is_a_member(self) -> None:
        assert ConflictPhase["CRISIS"] is ConflictPhase.CRISIS


class TestResolutionType:
    def test_all_types_present(self) -> None:
        names = {r.name for r in ResolutionType}
        assert names == {
            "R1_VISIBLE_DEFEAT",
            "R2_VISIBLE_WIN",
            "R3_PYRRHIC",
            "R4_SUSPENDED",
            "R5_STRUCTURAL",
        }


class TestTimeOfDay:
    def test_all_times_present(self) -> None:
        names = {t.name for t in TimeOfDay}
        assert names == {"EARLY_MORNING", "MORNING", "AFTERNOON", "EVENING", "NIGHT"}


# ---------------------------------------------------------------------------
# RelationshipState
# ---------------------------------------------------------------------------


class TestRelationshipState:
    def test_defaults(self) -> None:
        r = RelationshipState()
        assert r.tension == 0
        assert r.trust == 5
        assert r.history_notes == ()
        assert r.is_public is False

    def test_boundary_values_accepted(self) -> None:
        RelationshipState(tension=0, trust=0)
        RelationshipState(tension=10, trust=10)

    def test_tension_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="tension"):
            RelationshipState(tension=11)

    def test_trust_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="trust"):
            RelationshipState(trust=-1)

    def test_history_notes_stored(self) -> None:
        r = RelationshipState(history_notes=("Year-1 confrontation.",))
        assert r.history_notes == ("Year-1 confrontation.",)


# ---------------------------------------------------------------------------
# IncidentEntry
# ---------------------------------------------------------------------------


class TestIncidentEntry:
    def test_construction(self) -> None:
        e = IncidentEntry(
            step=2,
            trigger_type="INSTITUTIONAL_MOVE",
            variant="INSTITUTIONAL_ACADEMIC_THREAT",
            location_name="ADMINISTRATION_BUILDING",
            initiator="neel",
            target="vikram",
            description="Attendance records flagged.",
        )
        assert e.step == 2
        assert e.consequence_notes == ()
        assert e.is_public is False

    def test_with_consequences(self) -> None:
        e = IncidentEntry(
            step=1,
            trigger_type="DIRECT_CHALLENGE",
            variant="DIRECT_CHALLENGE_PUBLIC_CALLOUT",
            location_name="MAIN_CANTEEN",
            initiator="ranveer",
            target="vikram",
            description="Public challenge.",
            consequence_notes=("Ranveer phase advances.",),
            is_public=True,
        )
        assert len(e.consequence_notes) == 1
        assert e.is_public is True


# ---------------------------------------------------------------------------
# WorldState construction
# ---------------------------------------------------------------------------


class TestWorldStateConstruction:
    def test_minimal_construction(self) -> None:
        state = _minimal_state()
        assert state.step == 0
        assert state.conflict_phase is ConflictPhase.COLD_EQUILIBRIUM
        assert state.resolution_type is ResolutionType.R1_VISIBLE_DEFEAT
        assert state.time_of_day is TimeOfDay.AFTERNOON
        assert WorldFlag.SEMESTER_OPENING in state.active_flags.flags

    def test_character_states_default_constructed(self) -> None:
        state = _minimal_state()
        assert state.ranveer.phase is RanveerPhase.COLD
        assert state.dhruv.drift_state is DhruvDriftState.PRESENT
        assert state.surya.is_revealed is False
        assert state.meera.flags_lived_through == frozenset()

    def test_relationship_graph_starts_empty(self) -> None:
        state = _minimal_state()
        assert state.relationship_graph == {}

    def test_incident_log_starts_empty(self) -> None:
        state = _minimal_state()
        assert state.incident_log == []

    def test_relationship_graph_can_be_populated(self) -> None:
        state = _minimal_state()
        state.relationship_graph[("vikram", "ranveer")] = RelationshipState(tension=7)
        assert state.relationship_graph[("vikram", "ranveer")].tension == 7

    def test_incident_log_can_be_appended(self) -> None:
        state = _minimal_state()
        state.incident_log.append(
            IncidentEntry(
                step=0,
                trigger_type="DIRECT_CHALLENGE",
                variant="DIRECT_CHALLENGE_REFUSAL",
                location_name="MAIN_CANTEEN",
                initiator="vikram",
                target="__diffuse__",
                description="Test incident.",
            )
        )
        assert len(state.incident_log) == 1


# ---------------------------------------------------------------------------
# FlagSet integration via WorldState
# ---------------------------------------------------------------------------


class TestFlagSetIntegration:
    def test_flag_active(self) -> None:
        state = _minimal_state()
        assert state.active_flags.is_active(WorldFlag.SEMESTER_OPENING)

    def test_add_flag_returns_new_flagset(self) -> None:
        state = _minimal_state()
        new_flags = state.active_flags.with_flag(WorldFlag.ELECTION_SEASON)
        assert new_flags.is_active(WorldFlag.ELECTION_SEASON)
        # Original unchanged
        assert not state.active_flags.is_active(WorldFlag.ELECTION_SEASON)

    def test_texture_note_for_overlap(self) -> None:
        fs = FlagSet(
            flags=frozenset({WorldFlag.ELECTION_SEASON, WorldFlag.POLITICAL_AGITATION})
        )
        note = fs.texture_note()
        assert note is not None
        assert "volatile" in note.lower()

    def test_texture_note_none_for_single_flag(self) -> None:
        fs = FlagSet(flags=frozenset({WorldFlag.EXAM_SEASON}))
        assert fs.texture_note() is None


# ---------------------------------------------------------------------------
# Serialisation round-trip: to_dict / from_dict
# ---------------------------------------------------------------------------


class TestSerialisationRoundTrip:
    def test_minimal_to_dict_keys(self) -> None:
        d = _minimal_state().to_dict()
        expected_keys = {
            "active_flags",
            "conflict_phase",
            "resolution_type",
            "time_of_day",
            "step",
            "vikram",
            "ranveer",
            "karan",
            "neel",
            "arjun",
            "savar",
            "dhruv",
            "rajan",
            "surya",
            "kavya",
            "meera",
            "relationship_graph",
            "incident_log",
        }
        assert set(d.keys()) == expected_keys

    def test_enums_serialised_as_names(self) -> None:
        d = _minimal_state().to_dict()
        assert d["conflict_phase"] == "COLD_EQUILIBRIUM"
        assert d["resolution_type"] == "R1_VISIBLE_DEFEAT"
        assert d["time_of_day"] == "AFTERNOON"
        assert "SEMESTER_OPENING" in d["active_flags"]

    def test_minimal_round_trip(self) -> None:
        original = _minimal_state()
        reconstructed = WorldState.from_dict(original.to_dict())
        assert reconstructed.conflict_phase is original.conflict_phase
        assert reconstructed.resolution_type is original.resolution_type
        assert reconstructed.time_of_day is original.time_of_day
        assert reconstructed.step == original.step
        assert reconstructed.active_flags.flags == original.active_flags.flags

    def test_rich_round_trip_conflict_phase(self) -> None:
        original = _rich_state()
        reconstructed = WorldState.from_dict(original.to_dict())
        assert reconstructed.conflict_phase is ConflictPhase.OPEN_CONFLICT

    def test_rich_round_trip_ranveer_phase(self) -> None:
        original = _rich_state()
        reconstructed = WorldState.from_dict(original.to_dict())
        assert reconstructed.ranveer.phase is RanveerPhase.OBSESSED
        assert reconstructed.ranveer.consecutive_unacknowledged_non_submissions == 4

    def test_rich_round_trip_dhruv(self) -> None:
        original = _rich_state()
        reconstructed = WorldState.from_dict(original.to_dict())
        assert reconstructed.dhruv.drift_state is DhruvDriftState.LESS_AVAILABLE
        assert reconstructed.dhruv.cost_benefit_total == pytest.approx(-2.5)

    def test_rich_round_trip_surya_hidden_state(self) -> None:
        original = _rich_state()
        reconstructed = WorldState.from_dict(original.to_dict())
        assert reconstructed.surya.true_state is SuryaAllegiance.RANVEER_PLANT

    def test_rich_round_trip_meera_flags(self) -> None:
        original = _rich_state()
        reconstructed = WorldState.from_dict(original.to_dict())
        assert WorldFlag.CULTURAL_FEST in reconstructed.meera.flags_lived_through
        assert WorldFlag.INCIDENT_AFTERMATH in reconstructed.meera.flags_lived_through
        assert reconstructed.meera.response_set == frozenset({"withdrawn", "watchful"})

    def test_rich_round_trip_relationship_graph(self) -> None:
        original = _rich_state()
        reconstructed = WorldState.from_dict(original.to_dict())
        rel = reconstructed.relationship_graph[("vikram", "ranveer")]
        assert rel.tension == 8
        assert rel.trust == 0
        assert rel.is_public is True
        assert "Year-1 confrontation" in rel.history_notes[0]

    def test_rich_round_trip_incident_log(self) -> None:
        original = _rich_state()
        reconstructed = WorldState.from_dict(original.to_dict())
        assert len(reconstructed.incident_log) == 1
        entry = reconstructed.incident_log[0]
        assert entry.step == 1
        assert entry.trigger_type == "DIRECT_CHALLENGE"
        assert entry.is_public is True
        assert len(entry.consequence_notes) == 2

    def test_step_counter_survives_round_trip(self) -> None:
        original = _rich_state()
        reconstructed = WorldState.from_dict(original.to_dict())
        assert reconstructed.step == 3


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_to_json_produces_valid_json(self) -> None:
        s = _minimal_state().to_json()
        parsed = json.loads(s)
        assert isinstance(parsed, dict)

    def test_json_round_trip_minimal(self) -> None:
        original = _minimal_state()
        reconstructed = WorldState.from_json(original.to_json())
        assert reconstructed.conflict_phase is original.conflict_phase
        assert reconstructed.active_flags.flags == original.active_flags.flags

    def test_json_round_trip_rich(self) -> None:
        original = _rich_state()
        reconstructed = WorldState.from_json(original.to_json())
        assert reconstructed.ranveer.phase is RanveerPhase.OBSESSED
        assert reconstructed.surya.true_state is SuryaAllegiance.RANVEER_PLANT
        assert len(reconstructed.incident_log) == 1
