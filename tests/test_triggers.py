"""Tests for triggers.py — enums, Trigger dataclass, and factory functions."""

from __future__ import annotations

import pytest

from story_engine.locations import LocationName
from story_engine.triggers import (
    NO_TARGET,
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
# Enum completeness
# ---------------------------------------------------------------------------


class TestTriggerTypeEnum:
    def test_four_types_present(self) -> None:
        names = {t.name for t in TriggerType}
        assert names == {
            "DIRECT_CHALLENGE",
            "INSTITUTIONAL_MOVE",
            "POLITICAL_MOVE",
            "AMBIENT_TRIGGER",
        }


class TestTriggerVariantEnum:
    def test_seventeen_variants_present(self) -> None:
        assert len(list(TriggerVariant)) == 17

    def test_direct_challenge_variants(self) -> None:
        names = {
            v.name for v in TriggerVariant if v.name.startswith("DIRECT_CHALLENGE")
        }
        assert names == {
            "DIRECT_CHALLENGE_PUBLIC_HUMILIATION",
            "DIRECT_CHALLENGE_PHYSICAL_CONFRONTATION",
            "DIRECT_CHALLENGE_REFUSAL",
            "DIRECT_CHALLENGE_PUBLIC_CALLOUT",
        }

    def test_institutional_variants(self) -> None:
        names = {v.name for v in TriggerVariant if v.name.startswith("INSTITUTIONAL")}
        assert names == {
            "INSTITUTIONAL_ACADEMIC_THREAT",
            "INSTITUTIONAL_ADMINISTRATIVE_ACTION",
            "INSTITUTIONAL_NOTICE_BOARD",
            "INSTITUTIONAL_OPPORTUNITY_DENIAL",
        }

    def test_political_variants(self) -> None:
        names = {v.name for v in TriggerVariant if v.name.startswith("POLITICAL")}
        assert names == {
            "POLITICAL_ELECTION_POSITIONING",
            "POLITICAL_FACTION_APPROACH",
            "POLITICAL_AGITATION_SHAPING",
            "POLITICAL_DHRUV_CONTACT",
        }

    def test_ambient_variants(self) -> None:
        names = {v.name for v in TriggerVariant if v.name.startswith("AMBIENT")}
        assert names == {
            "AMBIENT_MEERA_INTERSECTION",
            "AMBIENT_KAVYA_EXPOSED",
            "AMBIENT_OUTSIDE_ACTOR",
            "AMBIENT_INFORMATION_SURFACE",
            "AMBIENT_GANG_MEMBER_ACTS_ALONE",
        }


# ---------------------------------------------------------------------------
# Trigger dataclass
# ---------------------------------------------------------------------------


class TestTriggerDataclass:
    def _make_minimal(self) -> Trigger:
        return Trigger(
            trigger_type=TriggerType.DIRECT_CHALLENGE,
            variant=TriggerVariant.DIRECT_CHALLENGE_REFUSAL,
            location=LocationName.MAIN_CANTEEN,
            initiator="vikram",
            target="ranveer",
            is_public=True,
            description="Vikram refused to move from the corner table.",
        )

    def test_construction(self) -> None:
        t = self._make_minimal()
        assert t.trigger_type is TriggerType.DIRECT_CHALLENGE
        assert t.variant is TriggerVariant.DIRECT_CHALLENGE_REFUSAL
        assert t.location is LocationName.MAIN_CANTEEN
        assert t.initiator == "vikram"
        assert t.target == "ranveer"
        assert t.is_public is True

    def test_is_frozen(self) -> None:
        t = self._make_minimal()
        with pytest.raises((AttributeError, TypeError)):
            t.is_public = False  # type: ignore[misc]

    def test_is_direct_true_for_type01(self) -> None:
        t = self._make_minimal()
        assert t.is_direct() is True

    def test_is_direct_false_for_institutional(self) -> None:
        t = make_academic_threat("Attendance flagged.")
        assert t.is_direct() is False

    def test_has_target_true(self) -> None:
        t = self._make_minimal()
        assert t.has_target() is True

    def test_has_target_false_for_diffuse(self) -> None:
        t = make_meera_intersection(
            LocationName.MAIN_CANTEEN, "Meera sat at the wrong table."
        )
        assert t.has_target() is False

    def test_involves_character_initiator(self) -> None:
        t = self._make_minimal()
        assert t.involves_character("vikram") is True

    def test_involves_character_target(self) -> None:
        t = self._make_minimal()
        assert t.involves_character("ranveer") is True

    def test_involves_character_unrelated(self) -> None:
        t = self._make_minimal()
        assert t.involves_character("neel") is False


# ---------------------------------------------------------------------------
# Factory: make_public_humiliation (01-A)
# ---------------------------------------------------------------------------


class TestMakePublicHumiliation:
    def test_type_and_variant(self) -> None:
        t = make_public_humiliation(
            LocationName.MAIN_CANTEEN,
            "ranveer",
            "vikram",
            "Ranveer mocked Vikram's seat.",
        )
        assert t.trigger_type is TriggerType.DIRECT_CHALLENGE
        assert t.variant is TriggerVariant.DIRECT_CHALLENGE_PUBLIC_HUMILIATION

    def test_is_public_always_true(self) -> None:
        t = make_public_humiliation(
            LocationName.MAIN_CANTEEN, "ranveer", "vikram", "desc"
        )
        assert t.is_public is True

    def test_initiator_and_target(self) -> None:
        t = make_public_humiliation(
            LocationName.MAIN_CANTEEN, "ranveer", "vikram", "desc"
        )
        assert t.initiator == "ranveer"
        assert t.target == "vikram"


# ---------------------------------------------------------------------------
# Factory: make_physical_confrontation (01-B)
# ---------------------------------------------------------------------------


class TestMakePhysicalConfrontation:
    def test_type_and_variant(self) -> None:
        t = make_physical_confrontation(
            LocationName.DEAD_PATHS, "karan", "vikram", "Karan cornered Vikram."
        )
        assert t.trigger_type is TriggerType.DIRECT_CHALLENGE
        assert t.variant is TriggerVariant.DIRECT_CHALLENGE_PHYSICAL_CONFRONTATION

    def test_default_not_public(self) -> None:
        t = make_physical_confrontation(
            LocationName.DEAD_PATHS, "karan", "vikram", "desc"
        )
        assert t.is_public is False

    def test_explicit_is_public(self) -> None:
        t = make_physical_confrontation(
            LocationName.MAIN_CANTEEN, "karan", "vikram", "desc", is_public=True
        )
        assert t.is_public is True


# ---------------------------------------------------------------------------
# Factory: make_vikram_refusal (01-C)
# ---------------------------------------------------------------------------


class TestMakeVikramRefusal:
    def test_type_and_variant(self) -> None:
        t = make_vikram_refusal(LocationName.MAIN_CANTEEN, "Vikram refused to leave.")
        assert t.trigger_type is TriggerType.DIRECT_CHALLENGE
        assert t.variant is TriggerVariant.DIRECT_CHALLENGE_REFUSAL

    def test_initiator_always_vikram(self) -> None:
        t = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        assert t.initiator == "vikram"

    def test_default_target_ranveer(self) -> None:
        t = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        assert t.target == "ranveer"

    def test_is_public_always_true(self) -> None:
        t = make_vikram_refusal(LocationName.MAIN_CANTEEN, "desc")
        assert t.is_public is True


# ---------------------------------------------------------------------------
# Factory: make_public_callout (01-D)
# ---------------------------------------------------------------------------


class TestMakePublicCallout:
    def test_type_and_variant(self) -> None:
        t = make_public_callout(LocationName.MAIN_GROUND, "Vikram named what Neel did.")
        assert t.trigger_type is TriggerType.DIRECT_CHALLENGE
        assert t.variant is TriggerVariant.DIRECT_CHALLENGE_PUBLIC_CALLOUT

    def test_initiator_always_vikram(self) -> None:
        t = make_public_callout(LocationName.MAIN_GROUND, "desc")
        assert t.initiator == "vikram"

    def test_is_public_always_true(self) -> None:
        t = make_public_callout(LocationName.MAIN_GROUND, "desc")
        assert t.is_public is True


# ---------------------------------------------------------------------------
# Factory: make_academic_threat (02-A)
# ---------------------------------------------------------------------------


class TestMakeAcademicThreat:
    def test_type_and_variant(self) -> None:
        t = make_academic_threat("Vikram's attendance record was flagged.")
        assert t.trigger_type is TriggerType.INSTITUTIONAL_MOVE
        assert t.variant is TriggerVariant.INSTITUTIONAL_ACADEMIC_THREAT

    def test_not_public(self) -> None:
        t = make_academic_threat("desc")
        assert t.is_public is False

    def test_default_initiator_neel(self) -> None:
        t = make_academic_threat("desc")
        assert t.initiator == "neel"

    def test_default_target_vikram(self) -> None:
        t = make_academic_threat("desc")
        assert t.target == "vikram"


# ---------------------------------------------------------------------------
# Factory: make_administrative_action (02-B)
# ---------------------------------------------------------------------------


class TestMakeAdministrativeAction:
    def test_type_and_variant(self) -> None:
        t = make_administrative_action("Disciplinary summons issued.")
        assert t.trigger_type is TriggerType.INSTITUTIONAL_MOVE
        assert t.variant is TriggerVariant.INSTITUTIONAL_ADMINISTRATIVE_ACTION

    def test_location_administration_building(self) -> None:
        t = make_administrative_action("desc")
        assert t.location is LocationName.ADMINISTRATION_BUILDING

    def test_not_public(self) -> None:
        t = make_administrative_action("desc")
        assert t.is_public is False


# ---------------------------------------------------------------------------
# Factory: make_notice_board_move (02-C)
# ---------------------------------------------------------------------------


class TestMakeNoticeBoardMove:
    def test_type_and_variant(self) -> None:
        t = make_notice_board_move("A list excluding Vikram was posted.")
        assert t.trigger_type is TriggerType.INSTITUTIONAL_MOVE
        assert t.variant is TriggerVariant.INSTITUTIONAL_NOTICE_BOARD

    def test_location_notice_board_cluster(self) -> None:
        t = make_notice_board_move("desc")
        assert t.location is LocationName.NOTICE_BOARD_CLUSTER

    def test_is_public_true(self) -> None:
        """Notice board is public — the posting is visible even if authorship isn't."""
        t = make_notice_board_move("desc")
        assert t.is_public is True


# ---------------------------------------------------------------------------
# Factory: make_opportunity_denial (02-D)
# ---------------------------------------------------------------------------


class TestMakeOpportunityDenial:
    def test_type_and_variant(self) -> None:
        t = make_opportunity_denial("Vikram was blocked from the selection list.")
        assert t.trigger_type is TriggerType.INSTITUTIONAL_MOVE
        assert t.variant is TriggerVariant.INSTITUTIONAL_OPPORTUNITY_DENIAL

    def test_not_public(self) -> None:
        t = make_opportunity_denial("desc")
        assert t.is_public is False


# ---------------------------------------------------------------------------
# Factory: make_election_positioning (03-A)
# ---------------------------------------------------------------------------


class TestMakeElectionPositioning:
    def test_type_and_variant(self) -> None:
        t = make_election_positioning("A candidate opposed to Savar was put forward.")
        assert t.trigger_type is TriggerType.POLITICAL_MOVE
        assert t.variant is TriggerVariant.POLITICAL_ELECTION_POSITIONING

    def test_location_student_union(self) -> None:
        t = make_election_positioning("desc")
        assert t.location is LocationName.STUDENT_UNION_BUILDING


# ---------------------------------------------------------------------------
# Factory: make_dhruv_contact (03-D)
# ---------------------------------------------------------------------------


class TestMakeDhruvContact:
    def test_type_and_variant(self) -> None:
        t = make_dhruv_contact("Neel offered Dhruv a placement connection.")
        assert t.trigger_type is TriggerType.POLITICAL_MOVE
        assert t.variant is TriggerVariant.POLITICAL_DHRUV_CONTACT

    def test_target_always_dhruv(self) -> None:
        t = make_dhruv_contact("desc")
        assert t.target == "dhruv"

    def test_location_secondary_canteen(self) -> None:
        t = make_dhruv_contact("desc")
        assert t.location is LocationName.SECONDARY_CANTEEN

    def test_not_public(self) -> None:
        t = make_dhruv_contact("desc")
        assert t.is_public is False


# ---------------------------------------------------------------------------
# Factory: make_meera_intersection (04-A)
# ---------------------------------------------------------------------------


class TestMakeMeeraIntersection:
    def test_type_and_variant(self) -> None:
        t = make_meera_intersection(
            LocationName.MAIN_CANTEEN, "Meera sat at Ranveer's usual table."
        )
        assert t.trigger_type is TriggerType.AMBIENT_TRIGGER
        assert t.variant is TriggerVariant.AMBIENT_MEERA_INTERSECTION

    def test_target_is_no_target(self) -> None:
        t = make_meera_intersection(LocationName.MAIN_CANTEEN, "desc")
        assert t.target == NO_TARGET
        assert not t.has_target()

    def test_default_initiator_meera(self) -> None:
        t = make_meera_intersection(LocationName.MAIN_CANTEEN, "desc")
        assert t.initiator == "meera"


# ---------------------------------------------------------------------------
# Factory: make_information_surface (04-D)
# ---------------------------------------------------------------------------


class TestMakeInformationSurface:
    def test_type_and_variant(self) -> None:
        t = make_information_surface(
            LocationName.HOSTEL_ROOF,
            "Surya knew about the notice before it was posted.",
        )
        assert t.trigger_type is TriggerType.AMBIENT_TRIGGER
        assert t.variant is TriggerVariant.AMBIENT_INFORMATION_SURFACE

    def test_default_initiator_surya(self) -> None:
        t = make_information_surface(LocationName.HOSTEL_ROOF, "desc")
        assert t.initiator == "surya"

    def test_no_specific_target(self) -> None:
        t = make_information_surface(LocationName.HOSTEL_ROOF, "desc")
        assert t.target == NO_TARGET


# ---------------------------------------------------------------------------
# Factory: make_gang_member_acts_alone (04-E)
# ---------------------------------------------------------------------------


class TestMakeGangMemberActsAlone:
    def test_type_and_variant(self) -> None:
        t = make_gang_member_acts_alone(
            "rajan",
            LocationName.DEAD_PATHS,
            "Rajan escalated without Vikram's instruction.",
        )
        assert t.trigger_type is TriggerType.AMBIENT_TRIGGER
        assert t.variant is TriggerVariant.AMBIENT_GANG_MEMBER_ACTS_ALONE

    def test_default_not_public(self) -> None:
        t = make_gang_member_acts_alone("rajan", LocationName.DEAD_PATHS, "desc")
        assert t.is_public is False

    def test_explicit_public(self) -> None:
        t = make_gang_member_acts_alone(
            "savar",
            LocationName.MAIN_CANTEEN,
            "Savar said something irreversible.",
            is_public=True,
        )
        assert t.is_public is True

    def test_no_specific_target(self) -> None:
        t = make_gang_member_acts_alone("dhruv", LocationName.SECONDARY_CANTEEN, "desc")
        assert t.target == NO_TARGET

    @pytest.mark.parametrize("member", ["rajan", "dhruv", "savar"])
    def test_any_gang_member(self, member: str) -> None:
        t = make_gang_member_acts_alone(member, LocationName.BOYS_HOSTEL_BLOCKS, "desc")
        assert t.initiator == member
