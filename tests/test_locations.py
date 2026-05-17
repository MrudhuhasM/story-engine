"""Tests for locations.py — enums, registry completeness, and rule logic."""

from __future__ import annotations

import pytest

from story_engine.locations import (
    LOCATIONS,
    ControlType,
    Location,
    LocationName,
    VisibilityLevel,
    get_location,
)


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


class TestRegistryCompleteness:
    def test_every_location_name_in_registry(self) -> None:
        """Every LocationName enum value must have a canonical entry."""
        for name in LocationName:
            assert name in LOCATIONS, f"{name} missing from LOCATIONS registry"

    def test_registry_count_matches_enum(self) -> None:
        assert len(LOCATIONS) == len(list(LocationName))

    def test_location_name_field_matches_key(self) -> None:
        """The Location.name field must match its registry key."""
        for key, loc in LOCATIONS.items():
            assert loc.name is key, f"Location.name mismatch for key {key}"


# ---------------------------------------------------------------------------
# Zone assignments
# ---------------------------------------------------------------------------


class TestZoneAssignments:
    def test_zone_1_entrance(self) -> None:
        assert get_location(LocationName.MAIN_GATE_AREA).zone == 1

    def test_zone_2_academic(self) -> None:
        assert get_location(LocationName.FACULTY_BUILDINGS).zone == 2
        assert get_location(LocationName.KAVYA_DEPT_CORRIDOR).zone == 2

    def test_zone_3_union(self) -> None:
        assert get_location(LocationName.STUDENT_UNION_BUILDING).zone == 3
        assert get_location(LocationName.NOTICE_BOARD_CLUSTER).zone == 3

    def test_zone_4_canteen(self) -> None:
        assert get_location(LocationName.MAIN_CANTEEN).zone == 4
        assert get_location(LocationName.SECONDARY_CANTEEN).zone == 4

    def test_zone_5_hostel(self) -> None:
        assert get_location(LocationName.BOYS_HOSTEL_BLOCKS).zone == 5
        assert get_location(LocationName.HOSTEL_ROOF).zone == 5
        assert get_location(LocationName.GIRLS_HOSTEL).zone == 5

    def test_zone_6_open_campus(self) -> None:
        assert get_location(LocationName.MAIN_GROUND).zone == 6
        assert get_location(LocationName.DEAD_PATHS).zone == 6
        assert get_location(LocationName.OLD_BANYAN_TREE).zone == 6

    def test_zone_7_admin(self) -> None:
        assert get_location(LocationName.ADMINISTRATION_BUILDING).zone == 7
        assert get_location(LocationName.FACULTY_QUARTERS).zone == 7

    def test_all_zones_in_range(self) -> None:
        for loc in LOCATIONS.values():
            assert 1 <= loc.zone <= 7, f"{loc.name} has zone {loc.zone} outside [1,7]"


# ---------------------------------------------------------------------------
# Control type assignments (from mode instructions spec)
# ---------------------------------------------------------------------------


class TestControlTypes:
    def test_main_gate_ranveer_informal(self) -> None:
        assert (
            get_location(LocationName.MAIN_GATE_AREA).control
            is ControlType.RANVEER_INFORMAL
        )

    def test_faculty_buildings_neutral(self) -> None:
        assert (
            get_location(LocationName.FACULTY_BUILDINGS).control is ControlType.NEUTRAL
        )

    def test_kavya_dept_corridor_kavya(self) -> None:
        assert (
            get_location(LocationName.KAVYA_DEPT_CORRIDOR).control is ControlType.KAVYA
        )

    def test_student_union_neel(self) -> None:
        assert (
            get_location(LocationName.STUDENT_UNION_BUILDING).control
            is ControlType.NEEL
        )

    def test_notice_board_neel(self) -> None:
        assert (
            get_location(LocationName.NOTICE_BOARD_CLUSTER).control is ControlType.NEEL
        )

    def test_main_canteen_contested(self) -> None:
        assert get_location(LocationName.MAIN_CANTEEN).control is ControlType.CONTESTED

    def test_secondary_canteen_neutral(self) -> None:
        assert (
            get_location(LocationName.SECONDARY_CANTEEN).control is ControlType.NEUTRAL
        )

    def test_boys_hostel_split(self) -> None:
        assert (
            get_location(LocationName.BOYS_HOSTEL_BLOCKS).control is ControlType.SPLIT
        )

    def test_hostel_roof_none(self) -> None:
        assert get_location(LocationName.HOSTEL_ROOF).control is ControlType.NONE

    def test_girls_hostel_meera(self) -> None:
        assert get_location(LocationName.GIRLS_HOSTEL).control is ControlType.MEERA

    def test_main_ground_whoever_mobilizes(self) -> None:
        assert (
            get_location(LocationName.MAIN_GROUND).control
            is ControlType.WHOEVER_MOBILIZES
        )

    def test_dead_paths_karan(self) -> None:
        assert get_location(LocationName.DEAD_PATHS).control is ControlType.KARAN

    def test_old_banyan_tree_none(self) -> None:
        assert get_location(LocationName.OLD_BANYAN_TREE).control is ControlType.NONE

    def test_administration_ranveer_family(self) -> None:
        assert (
            get_location(LocationName.ADMINISTRATION_BUILDING).control
            is ControlType.RANVEER_FAMILY
        )

    def test_faculty_quarters_kavya(self) -> None:
        assert get_location(LocationName.FACULTY_QUARTERS).control is ControlType.KAVYA


# ---------------------------------------------------------------------------
# Visibility level assignments (from mode instructions spec)
# ---------------------------------------------------------------------------


class TestVisibilityLevels:
    def test_main_gate_high(self) -> None:
        assert (
            get_location(LocationName.MAIN_GATE_AREA).visibility is VisibilityLevel.HIGH
        )

    def test_faculty_buildings_low_internal(self) -> None:
        assert (
            get_location(LocationName.FACULTY_BUILDINGS).visibility
            is VisibilityLevel.LOW_INTERNAL
        )

    def test_kavya_dept_corridor_medium(self) -> None:
        assert (
            get_location(LocationName.KAVYA_DEPT_CORRIDOR).visibility
            is VisibilityLevel.MEDIUM
        )

    def test_student_union_high(self) -> None:
        assert (
            get_location(LocationName.STUDENT_UNION_BUILDING).visibility
            is VisibilityLevel.HIGH
        )

    def test_notice_board_high(self) -> None:
        assert (
            get_location(LocationName.NOTICE_BOARD_CLUSTER).visibility
            is VisibilityLevel.HIGH
        )

    def test_main_canteen_high(self) -> None:
        assert (
            get_location(LocationName.MAIN_CANTEEN).visibility is VisibilityLevel.HIGH
        )

    def test_secondary_canteen_low(self) -> None:
        assert (
            get_location(LocationName.SECONDARY_CANTEEN).visibility
            is VisibilityLevel.LOW
        )

    def test_boys_hostel_medium(self) -> None:
        assert (
            get_location(LocationName.BOYS_HOSTEL_BLOCKS).visibility
            is VisibilityLevel.MEDIUM
        )

    def test_hostel_roof_none(self) -> None:
        assert get_location(LocationName.HOSTEL_ROOF).visibility is VisibilityLevel.NONE

    def test_girls_hostel_low(self) -> None:
        assert get_location(LocationName.GIRLS_HOSTEL).visibility is VisibilityLevel.LOW

    def test_main_ground_maximum(self) -> None:
        assert (
            get_location(LocationName.MAIN_GROUND).visibility is VisibilityLevel.MAXIMUM
        )

    def test_dead_paths_none(self) -> None:
        assert get_location(LocationName.DEAD_PATHS).visibility is VisibilityLevel.NONE

    def test_old_banyan_tree_low(self) -> None:
        assert (
            get_location(LocationName.OLD_BANYAN_TREE).visibility is VisibilityLevel.LOW
        )

    def test_administration_institutional(self) -> None:
        assert (
            get_location(LocationName.ADMINISTRATION_BUILDING).visibility
            is VisibilityLevel.INSTITUTIONAL
        )

    def test_faculty_quarters_private(self) -> None:
        assert (
            get_location(LocationName.FACULTY_QUARTERS).visibility
            is VisibilityLevel.PRIVATE
        )


# ---------------------------------------------------------------------------
# RULE_02: consequence_multiplier
# ---------------------------------------------------------------------------


class TestConsequenceMultiplier:
    """Verifies RULE_02_VISIBILITY_MULTIPLIER encoding on Location."""

    def test_maximum_visibility_doubles(self) -> None:
        loc = get_location(LocationName.MAIN_GROUND)
        assert loc.consequence_multiplier() == pytest.approx(2.0)

    def test_high_visibility_doubles(self) -> None:
        for name in (
            LocationName.MAIN_CANTEEN,
            LocationName.STUDENT_UNION_BUILDING,
            LocationName.NOTICE_BOARD_CLUSTER,
            LocationName.MAIN_GATE_AREA,
        ):
            assert get_location(name).consequence_multiplier() == pytest.approx(2.0), (
                name
            )

    def test_medium_visibility_one_point_five(self) -> None:
        for name in (LocationName.KAVYA_DEPT_CORRIDOR, LocationName.BOYS_HOSTEL_BLOCKS):
            assert get_location(name).consequence_multiplier() == pytest.approx(1.5), (
                name
            )

    def test_none_visibility_no_multiplier(self) -> None:
        for name in (LocationName.DEAD_PATHS, LocationName.HOSTEL_ROOF):
            assert get_location(name).consequence_multiplier() == pytest.approx(1.0), (
                name
            )

    def test_low_visibility_no_multiplier(self) -> None:
        for name in (
            LocationName.SECONDARY_CANTEEN,
            LocationName.GIRLS_HOSTEL,
            LocationName.OLD_BANYAN_TREE,
        ):
            assert get_location(name).consequence_multiplier() == pytest.approx(1.0), (
                name
            )

    def test_institutional_no_multiplier(self) -> None:
        assert get_location(
            LocationName.ADMINISTRATION_BUILDING
        ).consequence_multiplier() == pytest.approx(1.0)

    def test_private_no_multiplier(self) -> None:
        assert get_location(
            LocationName.FACULTY_QUARTERS
        ).consequence_multiplier() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# is_witnessed_by_student_body
# ---------------------------------------------------------------------------


class TestIsWitnessed:
    def test_high_and_maximum_are_witnessed(self) -> None:
        for name in (
            LocationName.MAIN_CANTEEN,
            LocationName.MAIN_GROUND,
            LocationName.STUDENT_UNION_BUILDING,
        ):
            assert get_location(name).is_witnessed_by_student_body() is True, name

    def test_medium_is_witnessed(self) -> None:
        assert (
            get_location(LocationName.BOYS_HOSTEL_BLOCKS).is_witnessed_by_student_body()
            is True
        )

    def test_none_is_not_witnessed(self) -> None:
        assert (
            get_location(LocationName.DEAD_PATHS).is_witnessed_by_student_body()
            is False
        )
        assert (
            get_location(LocationName.HOSTEL_ROOF).is_witnessed_by_student_body()
            is False
        )

    def test_private_is_not_witnessed(self) -> None:
        assert (
            get_location(LocationName.FACULTY_QUARTERS).is_witnessed_by_student_body()
            is False
        )


# ---------------------------------------------------------------------------
# Immutability (frozen=True)
# ---------------------------------------------------------------------------


class TestLocationImmutability:
    def test_location_is_frozen(self) -> None:
        loc = get_location(LocationName.MAIN_CANTEEN)
        with pytest.raises((AttributeError, TypeError)):
            loc.zone = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# get_location helper
# ---------------------------------------------------------------------------


class TestGetLocation:
    def test_returns_correct_instance(self) -> None:
        loc = get_location(LocationName.DEAD_PATHS)
        assert loc is LOCATIONS[LocationName.DEAD_PATHS]

    def test_all_names_resolve(self) -> None:
        for name in LocationName:
            loc = get_location(name)
            assert isinstance(loc, Location)

    def test_dramatic_function_non_empty(self) -> None:
        for name in LocationName:
            loc = get_location(name)
            assert loc.dramatic_function.strip(), f"{name} has empty dramatic_function"
