"""Location dataclasses, enums, and canonical registry for the story engine.

Every location in the world is an immutable ``Location`` instance stored in
the ``LOCATIONS`` registry. The engine looks up locations by ``LocationName``
enum value — no string literals in chain-rule logic.

RULE_02_VISIBILITY_MULTIPLIER is encoded as ``Location.consequence_multiplier()``:
  HIGH or MAXIMUM visibility → 2.0 × consequence weight
  MEDIUM visibility          → 1.5 × consequence weight
  All other visibility levels → 1.0 × consequence weight

Import contract
~~~~~~~~~~~~~~~
locations.py has no imports from other engine modules.
world_state.py, triggers.py, and engine.py all import from locations.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


# ---------------------------------------------------------------------------
# ControlType
# ---------------------------------------------------------------------------


class ControlType(Enum):
    """Who controls a location — informally, institutionally, or not at all.

    Control determines whose information network is active, which characters
    are advantaged on entry, and what social cost is incurred by presence.
    """

    RANVEER_INFORMAL = auto()
    """Ranveer's gang watches through third parties (shopkeepers, informants).
    Not visible control — accumulated obligation."""

    RANVEER_FAMILY = auto()
    """Ranveer's family's institutional reach. Decisions bend without being pushed."""

    NEEL = auto()
    """Neel's operational territory. He knows every lever and who stands beside it."""

    KAVYA = auto()
    """Kavya's professional space. Her invisibility-as-strategy is sharpest here."""

    KARAN = auto()
    """Karan operates most comfortably here — consequence-with-deniability territory."""

    MEERA = auto()
    """Meera's space; the only campus territory genuinely hers and not filtered
    through Vikram or Kavya."""

    CONTESTED = auto()
    """Neutral by tradition but practically dominated by proximity and occupation.
    No one formally owns it; no one formally gives it up."""

    SPLIT = auto()
    """Divided between Ranveer's gang (one block) and Vikram's (another).
    The corridor between them is the most charged physical space on campus."""

    WHOEVER_MOBILIZES = auto()
    """Dynamic control — whoever fills the space first controls it in that moment.
    Neel's student union machinery is what fills it most reliably."""

    NEUTRAL = auto()
    """Formally neutral; no single faction's home ground."""

    NONE = auto()
    """Genuinely uncontrolled. Predates every current conflict or is too
    marginal to contest."""


# ---------------------------------------------------------------------------
# VisibilityLevel
# ---------------------------------------------------------------------------


class VisibilityLevel(Enum):
    """How visible events at this location are to the general student body.

    Drives ``Location.consequence_multiplier()`` per RULE_02_VISIBILITY_MULTIPLIER.
    """

    MAXIMUM = auto()
    """Entire campus as potential witness. Highest emotional and social stakes."""

    HIGH = auto()
    """General student body present; information spreads campus-wide within hours."""

    MEDIUM = auto()
    """Partial audience — specific groups, those who live or pass through here."""

    LOW = auto()
    """Few witnesses by default. Conversations here are chosen for their privacy."""

    LOW_INTERNAL = auto()
    """Low external visibility; each department has its own internal audience.
    Information stays within the faculty building's social network."""

    INSTITUTIONAL = auto()
    """Formal bureaucratic visibility. What happens here produces paper, not rumour.
    Consequences are structural rather than social."""

    PRIVATE = auto()
    """Home-inside-institution. The most intimate visibility — only those who
    belong here see what happens."""

    NONE = auto()
    """No witnesses unless deliberately arranged. What happens here does not exist
    officially until someone chooses to surface it."""


# ---------------------------------------------------------------------------
# LocationName
# ---------------------------------------------------------------------------


class LocationName(Enum):
    """Canonical identifier for every named location in the world model.

    Used as keys in ``LOCATIONS`` and as the ``location`` field of ``Trigger``
    and ``IncidentEntry``. Never use raw strings for location identity.
    """

    MAIN_GATE_AREA = auto()
    FACULTY_BUILDINGS = auto()
    KAVYA_DEPT_CORRIDOR = auto()
    STUDENT_UNION_BUILDING = auto()
    NOTICE_BOARD_CLUSTER = auto()
    MAIN_CANTEEN = auto()
    SECONDARY_CANTEEN = auto()
    BOYS_HOSTEL_BLOCKS = auto()
    HOSTEL_ROOF = auto()
    GIRLS_HOSTEL = auto()
    MAIN_GROUND = auto()
    DEAD_PATHS = auto()
    OLD_BANYAN_TREE = auto()
    ADMINISTRATION_BUILDING = auto()
    FACULTY_QUARTERS = auto()


# ---------------------------------------------------------------------------
# Location dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Location:
    """Immutable descriptor for a single campus location.

    Instances live in ``LOCATIONS``; the engine never constructs them ad-hoc.

    Args:
        name: Canonical ``LocationName`` enum value for this location.
        control: Who controls or watches this space.
        visibility: How visible events here are to the general student body.
        zone: Campus zone number (1–7) as defined in the world model.
        dramatic_function: One-line summary of this location's story role.
    """

    name: LocationName
    control: ControlType
    visibility: VisibilityLevel
    zone: int
    dramatic_function: str

    def consequence_multiplier(self) -> float:
        """Return the consequence weight multiplier for events at this location.

        Implements RULE_02_VISIBILITY_MULTIPLIER:
          - MAXIMUM or HIGH visibility → 2.0 (event witnessed by general student body)
          - MEDIUM visibility          → 1.5 (partial audience)
          - All other levels           → 1.0 (no amplification)

        Returns:
            Float multiplier to be applied to event consequence weight.
        """
        if self.visibility in (VisibilityLevel.MAXIMUM, VisibilityLevel.HIGH):
            return 2.0
        if self.visibility is VisibilityLevel.MEDIUM:
            return 1.5
        return 1.0

    def is_witnessed_by_student_body(self) -> bool:
        """Return True if events here are seen by the general student body.

        Convenience predicate used by engine visibility checks.
        Equivalent to ``consequence_multiplier() > 1.0``.
        """
        return self.visibility in (
            VisibilityLevel.MAXIMUM,
            VisibilityLevel.HIGH,
            VisibilityLevel.MEDIUM,
        )


# ---------------------------------------------------------------------------
# Canonical location registry
# ---------------------------------------------------------------------------


LOCATIONS: dict[LocationName, Location] = {
    LocationName.MAIN_GATE_AREA: Location(
        name=LocationName.MAIN_GATE_AREA,
        control=ControlType.RANVEER_INFORMAL,
        visibility=VisibilityLevel.HIGH,
        zone=1,
        dramatic_function=(
            "Arrivals, departures, deniable confrontations; "
            "Ranveer's informal watch through shopkeeper debts."
        ),
    ),
    LocationName.FACULTY_BUILDINGS: Location(
        name=LocationName.FACULTY_BUILDINGS,
        control=ControlType.NEUTRAL,
        visibility=VisibilityLevel.LOW_INTERNAL,
        zone=2,
        dramatic_function=(
            "Where institutional weapons are loaded before firing; "
            "Kavya navigates invisibility in the same space Neel has mapped."
        ),
    ),
    LocationName.KAVYA_DEPT_CORRIDOR: Location(
        name=LocationName.KAVYA_DEPT_CORRIDOR,
        control=ControlType.KAVYA,
        visibility=VisibilityLevel.MEDIUM,
        zone=2,
        dramatic_function=(
            "Collision zone; Vikram and Kavya's non-interaction "
            "is itself a scene students read and interpret."
        ),
    ),
    LocationName.STUDENT_UNION_BUILDING: Location(
        name=LocationName.STUDENT_UNION_BUILDING,
        control=ControlType.NEEL,
        visibility=VisibilityLevel.HIGH,
        zone=3,
        dramatic_function=(
            "Real campus power centre; institutional memory; "
            "any institutional move requires passing through or around Neel here."
        ),
    ),
    LocationName.NOTICE_BOARD_CLUSTER: Location(
        name=LocationName.NOTICE_BOARD_CLUSTER,
        control=ControlType.NEEL,
        visibility=VisibilityLevel.HIGH,
        zone=3,
        dramatic_function=(
            "Campus nervous system made physical; "
            "what is posted here is official — manufactured notices become real."
        ),
    ),
    LocationName.MAIN_CANTEEN: Location(
        name=LocationName.MAIN_CANTEEN,
        control=ControlType.CONTESTED,
        visibility=VisibilityLevel.HIGH,
        zone=4,
        dramatic_function=(
            "Maximum witnesses; the social map is most legible here — "
            "who sits where tells you the current state of everything."
        ),
    ),
    LocationName.SECONDARY_CANTEEN: Location(
        name=LocationName.SECONDARY_CANTEEN,
        control=ControlType.NEUTRAL,
        visibility=VisibilityLevel.LOW,
        zone=4,
        dramatic_function=(
            "Private-in-public; where conversations too dangerous "
            "for the main canteen happen. Dhruv and Neel's three exchanges occurred here."
        ),
    ),
    LocationName.BOYS_HOSTEL_BLOCKS: Location(
        name=LocationName.BOYS_HOSTEL_BLOCKS,
        control=ControlType.SPLIT,
        visibility=VisibilityLevel.MEDIUM,
        zone=5,
        dramatic_function=(
            "Physical hierarchy; after midnight guards drop and "
            "the corridor between Ranveer's block and Vikram's is the most charged space on campus."
        ),
    ),
    LocationName.HOSTEL_ROOF: Location(
        name=LocationName.HOSTEL_ROOF,
        control=ControlType.NONE,
        visibility=VisibilityLevel.NONE,
        zone=5,
        dramatic_function=(
            "The one space outside campus rules; no performance required. "
            "Where the war pauses without either side formally pausing it."
        ),
    ),
    LocationName.GIRLS_HOSTEL: Location(
        name=LocationName.GIRLS_HOSTEL,
        control=ControlType.MEERA,
        visibility=VisibilityLevel.LOW,
        zone=5,
        dramatic_function=(
            "Meera's territory — the only campus space genuinely hers. "
            "Vikram waiting at the boundary is visible in a specific way."
        ),
    ),
    LocationName.MAIN_GROUND: Location(
        name=LocationName.MAIN_GROUND,
        control=ControlType.WHOEVER_MOBILIZES,
        visibility=VisibilityLevel.MAXIMUM,
        zone=6,
        dramatic_function=(
            "Highest-stakes public space; entire campus as witness. "
            "What happens here is the campus record."
        ),
    ),
    LocationName.DEAD_PATHS: Location(
        name=LocationName.DEAD_PATHS,
        control=ControlType.KARAN,
        visibility=VisibilityLevel.NONE,
        zone=6,
        dramatic_function=(
            "Campus blind spots — not on any map; "
            "where public consequences become private violence with full deniability."
        ),
    ),
    LocationName.OLD_BANYAN_TREE: Location(
        name=LocationName.OLD_BANYAN_TREE,
        control=ControlType.NONE,
        visibility=VisibilityLevel.LOW,
        zone=6,
        dramatic_function=(
            "Genuinely neutral; predates every current conflict and will outlast all of them. "
            "The most honest scenes happen here."
        ),
    ),
    LocationName.ADMINISTRATION_BUILDING: Location(
        name=LocationName.ADMINISTRATION_BUILDING,
        control=ControlType.RANVEER_FAMILY,
        visibility=VisibilityLevel.INSTITUTIONAL,
        zone=7,
        dramatic_function=(
            "Ranveer family's deepest reach; the administration bends without being visibly pushed. "
            "Being summoned here means the conflict has moved from social to structural."
        ),
    ),
    LocationName.FACULTY_QUARTERS: Location(
        name=LocationName.FACULTY_QUARTERS,
        control=ControlType.KAVYA,
        visibility=VisibilityLevel.PRIVATE,
        zone=7,
        dramatic_function=(
            "Home inside the institution; no decompression space between Vikram's roles. "
            "Scenes here carry automatic weight — the space itself is a contradiction."
        ),
    ),
}


def get_location(name: LocationName) -> Location:
    """Look up a location by its canonical name.

    Args:
        name: The ``LocationName`` enum value to look up.

    Returns:
        The immutable ``Location`` instance for this name.

    Raises:
        KeyError: If *name* is not present in the registry (should never happen
            in practice since the registry covers all ``LocationName`` values).
    """
    return LOCATIONS[name]
