"""
CPT Query Service — Public Contract (v1.0.0)
==============================================

This module defines the full external contract for a location + depth based
CPT (cone penetration test) query service. It specifies shapes, constraints,
validation order, and error modes. It intentionally does NOT expose model
architecture, training details, interpolation methods, or internal
computation — only externally observable behavior at the boundary.

Design summary
--------------
location + depth selection + optional site context
    -> ordered CPT samples containing depth, qc, and fs

Key contract decisions (see inline docstrings/comments for rationale):

1. Depth selection is mandatory. Exactly one of `depth_m` / `depth_range`
   must be supplied to `query()`. There is no implicit full-profile
   default — use `profile()` for that instead.
2. The service returns values on a fixed 0.05 m grid. Off-grid requests
   are REJECTED, not snapped or interpolated, subject to a floating-point
   tolerance (see `_is_on_grid`).
3. Site context elevations each carry their own datum (`Elevation`),
   making datum consistency a structural property rather than a
   "should normally" suggestion.
4. Coverage is modeled as possibly-discontinuous vertical bands, since
   real subsurface data can have gaps.
5. A complete exception hierarchy exists, with an explicit, documented
   validation order so that a request failing multiple checks always
   raises the same, predictable error.
6. Uncertainty is represented as structural `DataQuality` metadata
   (support distance, observation count, coverage class) rather than an
   opaque confidence number.
7. Every result echoes its originating query for traceability.
8. Batch queries (`query_many`) fail independently per item.
9. A `schema_version` field is reserved on results for future evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CPT_MIN_DEPTH_M = 0.05
CPT_MAX_DEPTH_M = 20.0
CPT_DEPTH_INCREMENT_M = 0.05

# Tolerance for floating-point grid-alignment comparisons. 0.05 is not
# exactly representable in IEEE-754 binary floating point, so a caller
# computing e.g. 1.00 + 0.05 * 4 may land on 1.2000000000000002 rather
# than exactly 1.20. Such values are still treated as on-grid.
_GRID_EPSILON_M = 1e-9

SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VerticalDatum(StrEnum):
    NAVD88 = "NAVD88"
    NGVD29 = "NGVD29"
    LOCAL = "LOCAL"


class CoverageStatus(StrEnum):
    COVERED = "covered"
    NOT_COVERED = "not_covered"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Elevation:
    value_m: float
    datum: VerticalDatum


@dataclass(frozen=True, slots=True)
class SiteContext:
    ground_surface: Elevation | None = None
    groundwater_surface: Elevation | None = None

    def validate(self) -> None:
        """
        Enforces datum consistency structurally. Deliberately does NOT
        reject groundwater_surface above ground_surface — that condition
        is physically real in artesian settings, and the API preserves
        whatever values are supplied without judgment.
        """
        if self.ground_surface is not None and self.groundwater_surface is not None:
            if self.ground_surface.datum != self.groundwater_surface.datum:
                raise InvalidSiteContextError(
                    "ground_surface and groundwater_surface must use the "
                    "same vertical datum within a single request; "
                    "cross-datum support is not implemented in this "
                    "version."
                )


@dataclass(frozen=True, slots=True)
class DepthRange:
    start_m: float
    end_m: float

    def validate(self) -> None:
        if not (self.start_m < self.end_m):
            raise InvalidDepthRangeError(
                f"start_m ({self.start_m}) must be strictly less than "
                f"end_m ({self.end_m})."
            )


# ---------------------------------------------------------------------------
# Query / Result objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CPTQuery:
    latitude_deg: float
    longitude_deg: float
    depth_m: float | None = None
    depth_range: DepthRange | None = None
    context: SiteContext | None = None
    allow_partial: bool = False
    # Optional caller-supplied identifier used purely to correlate a
    # CPTQueryFailure back to its originating request in query_many. It
    # is NOT part of the scientific query and does not affect samples
    # returned; two queries differing only in request_id are otherwise
    # identical requests.
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class DataQuality:
    """
    Structural uncertainty metadata. Deliberately avoids a single opaque
    confidence score, which can be misinterpreted unless rigorously
    calibrated. All fields are optional and may remain unset in v1 while
    the type reserves a coherent path for later population.
    """
    support_distance_m: float | None = None
    nearby_observation_count: int | None = None
    coverage_class: str | None = None


@dataclass(frozen=True, slots=True)
class CPTSample:
    depth_m: float
    qc_mpa: float
    fs_kpa: float
    quality: DataQuality | None = None


@dataclass(frozen=True, slots=True)
class CPTResult:
    query: CPTQuery
    samples: tuple[CPTSample, ...]
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class CoverageBand:
    """A single contiguous depth interval with reliable data."""
    minimum_depth_m: float
    maximum_depth_m: float


@dataclass(frozen=True, slots=True)
class Coverage:
    """
    Vertical coverage may be discontinuous — e.g. reliable data from
    0-4m, a data gap from 4-6m, then reliable data resuming 6-8m. Bands
    are ordered and non-overlapping. An empty `bands` tuple means no
    vertical coverage at this location.
    """
    horizontal_status: CoverageStatus
    bands: tuple[CoverageBand, ...] = ()


@dataclass(frozen=True, slots=True)
class CPTQueryFailure:
    query: CPTQuery
    error: "CPTError"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
#
# Validation order (applied top to bottom; the first failing layer is the
# one that raises, even if a request also violates a later layer):
#
#   1. Structural validity  -> InvalidCoordinateError
#                               AmbiguousDepthQueryError
#                               InvalidDepthRangeError
#                               InvalidSiteContextError
#   2. Global bounds         -> InvalidDepthError
#   3. Grid alignment        -> DepthGridAlignmentError
#   4. Coverage              -> OutOfCoverageError
#                               DepthOutOfCoverageError
#   5. Fulfillment           -> PartialResultError
#
# Example: depth_m=25.03 is both off-grid and beyond CPT_MAX_DEPTH_M.
# Global bounds (layer 2) is checked before grid alignment (layer 3), so
# InvalidDepthError is raised, not DepthGridAlignmentError.

class CPTError(Exception):
    """Base exception for the CPT interface."""


class InvalidCoordinateError(CPTError):
    """Latitude or longitude is outside its valid domain."""


class AmbiguousDepthQueryError(CPTError):
    """Both depth forms were supplied, or neither was supplied."""


class InvalidDepthRangeError(CPTError):
    """The range is reversed, empty, or otherwise malformed."""


class InvalidSiteContextError(CPTError):
    """Optional context is structurally inconsistent."""


class InvalidDepthError(CPTError):
    """A depth is outside the global supported bounds
    [CPT_MIN_DEPTH_M, CPT_MAX_DEPTH_M]."""


class DepthGridAlignmentError(CPTError):
    """A requested depth does not lie on the CPT_DEPTH_INCREMENT_M grid,
    within floating-point tolerance."""


class OutOfCoverageError(CPTError):
    """The geographic location is outside model coverage."""


class DepthOutOfCoverageError(CPTError):
    """The requested depth, or some portion of a requested range,
    exceeds reliable local vertical coverage (including falling inside
    a coverage gap between two bands)."""


class PartialResultError(CPTError):
    """An atomic request could only be partially fulfilled and
    allow_partial was not set on the query."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
# These functions are part of the observable contract: any conforming
# implementation of CPTService must produce the same accept/reject
# decisions and the same exception types as these helpers would.

def _validate_coordinates(latitude_deg: float, longitude_deg: float) -> None:
    if not (-90.0 <= latitude_deg <= 90.0):
        raise InvalidCoordinateError(
            f"latitude_deg must be in [-90, 90], got {latitude_deg}."
        )
    if not (-180.0 <= longitude_deg <= 180.0):
        raise InvalidCoordinateError(
            f"longitude_deg must be in [-180, 180], got {longitude_deg}."
        )


def _validate_depth_selection(query: CPTQuery) -> None:
    if (query.depth_m is None) == (query.depth_range is None):
        raise AmbiguousDepthQueryError(
            "Exactly one of depth_m or depth_range must be supplied "
            "(both or neither is invalid). Use CPTService.profile() for "
            "an implicit full-depth sweep."
        )
    if query.depth_range is not None:
        query.depth_range.validate()
    if query.context is not None:
        query.context.validate()


def _validate_global_bounds(depth_m: float) -> None:
    if not (CPT_MIN_DEPTH_M <= depth_m <= CPT_MAX_DEPTH_M):
        raise InvalidDepthError(
            f"depth {depth_m} m is outside the supported global bounds "
            f"[{CPT_MIN_DEPTH_M}, {CPT_MAX_DEPTH_M}] m."
        )


def _is_on_grid(depth_m: float) -> bool:
    quotient = depth_m / CPT_DEPTH_INCREMENT_M
    nearest = round(quotient)
    return abs(quotient - nearest) * CPT_DEPTH_INCREMENT_M < _GRID_EPSILON_M


def _validate_grid_alignment(depth_m: float) -> None:
    if not _is_on_grid(depth_m):
        raise DepthGridAlignmentError(
            f"depth {depth_m} m does not lie on the "
            f"{CPT_DEPTH_INCREMENT_M} m grid. Values are rejected rather "
            f"than snapped or interpolated at the interface boundary."
        )


def _snap_to_grid(depth_m: float) -> float:
    """
    Used only AFTER _validate_grid_alignment has confirmed depth_m is
    within tolerance of a grid point, purely to normalize floating-point
    noise (e.g. 1.2000000000000002 -> 1.20) before echoing or comparing
    depths. Never used to accept off-grid input.
    """
    steps = round(depth_m / CPT_DEPTH_INCREMENT_M)
    return round(steps * CPT_DEPTH_INCREMENT_M, 10)


def _generate_grid_depths(depth_range: DepthRange) -> tuple[float, ...]:
    """Both endpoints must already be validated as on-grid before this
    is called."""
    start_steps = round(depth_range.start_m / CPT_DEPTH_INCREMENT_M)
    end_steps = round(depth_range.end_m / CPT_DEPTH_INCREMENT_M)
    return tuple(
        _snap_to_grid(step * CPT_DEPTH_INCREMENT_M)
        for step in range(start_steps, end_steps + 1)
    )


def _validate_coverage_for_depths(
    depths: tuple[float, ...],
    coverage: Coverage,
    allow_partial: bool,
) -> tuple[float, ...]:
    """
    Returns the subset of `depths` that are covered. Raises
    OutOfCoverageError / DepthOutOfCoverageError / PartialResultError
    according to allow_partial semantics.
    """
    if coverage.horizontal_status != CoverageStatus.COVERED:
        raise OutOfCoverageError("Location is outside horizontal model coverage.")

    covered = tuple(
        depth_m
        for depth_m in depths
        if any(
            band.minimum_depth_m <= depth_m <= band.maximum_depth_m
            for band in coverage.bands
        )
    )

    if len(covered) == len(depths):
        return covered

    if not allow_partial:
        if not covered:
            raise DepthOutOfCoverageError(
                "Requested depth(s) exceed reliable local vertical "
                "coverage at this location."
            )
        raise PartialResultError(
            "Requested range only partially falls within reliable local "
            "vertical coverage (e.g. it spans a coverage gap). Set "
            "allow_partial=True to receive the covered subset instead of "
            "an error."
        )

    return covered


# ---------------------------------------------------------------------------
# Service protocol
# ---------------------------------------------------------------------------

class CPTService(Protocol):
    def query(self, query: CPTQuery) -> CPTResult:
        """
        Resolve a single CPTQuery.

        Raises, in validation order (see module-level comment above the
        error classes):
            InvalidCoordinateError
            AmbiguousDepthQueryError
            InvalidDepthRangeError
            InvalidSiteContextError
            InvalidDepthError
            DepthGridAlignmentError
            OutOfCoverageError
            DepthOutOfCoverageError
            PartialResultError   (only when allow_partial is False and
                                   the request is only partially coverable)
        """
        ...

    def query_many(
        self, queries: list[CPTQuery]
    ) -> tuple[Union[CPTResult, CPTQueryFailure], ...]:
        """
        Resolve multiple queries independently. One query's failure does
        not prevent others from succeeding or being attempted. The
        returned tuple corresponds positionally to the input list;
        CPTQueryFailure.query is the original query object (use
        request_id to disambiguate structurally identical queries).
        """
        ...

    def profile(
        self,
        *,
        latitude_deg: float,
        longitude_deg: float,
        context: SiteContext | None = None,
        allow_partial: bool = False,
    ) -> CPTResult:
        """
        Convenience method returning the full available depth sweep
        (CPT_MIN_DEPTH_M through CPT_MAX_DEPTH_M) at a location. This is
        the only supported way to request an implicit full-profile
        result — CPTQuery/query() never defaults to a full sweep, since
        depth selection is mandatory there.

        If allow_partial is False and vertical coverage does not span
        the full range without gaps, raises DepthOutOfCoverageError or
        PartialResultError per the same rules as query().
        """
        ...

    def coverage_at(
        self, *, latitude_deg: float, longitude_deg: float
    ) -> Coverage:
        """
        Report horizontal and (possibly discontinuous) vertical coverage
        at a location. Never raises OutOfCoverageError itself; instead
        returns CoverageStatus.NOT_COVERED with an empty bands tuple.
        """
        ...


# ---------------------------------------------------------------------------
# Usage examples (illustrative — not executable without a concrete
# CPTService implementation)
# ---------------------------------------------------------------------------

def _examples(model: CPTService) -> None:
    # --- Single depth ---
    result = model.query(
        CPTQuery(
            latitude_deg=30.001842,
            longitude_deg=-91.203715,
            depth_m=3.50,
        )
    )
    assert len(result.samples) == 1

    # --- Depth range with site context (matching datums) ---
    query = CPTQuery(
        latitude_deg=30.001842,
        longitude_deg=-91.203715,
        depth_range=DepthRange(start_m=1.00, end_m=1.20),
        context=SiteContext(
            ground_surface=Elevation(value_m=12.40, datum=VerticalDatum.NAVD88),
            groundwater_surface=Elevation(value_m=9.70, datum=VerticalDatum.NAVD88),
        ),
    )
    result = model.query(query)
    assert result.query == query  # echoed request

    # --- Ambiguous depth selection ---
    try:
        model.query(
            CPTQuery(latitude_deg=30.0, longitude_deg=-91.0)  # neither depth form
        )
    except AmbiguousDepthQueryError:
        pass

    # --- Off-grid rejection ---
    try:
        model.query(
            CPTQuery(latitude_deg=30.0, longitude_deg=-91.0, depth_m=3.47)
        )
    except DepthGridAlignmentError:
        pass

    # --- Coverage check before querying ---
    coverage = model.coverage_at(latitude_deg=30.001842, longitude_deg=-91.203715)
    if coverage.horizontal_status == CoverageStatus.COVERED:
        for band in coverage.bands:
            pass  # inspect reliable depth bands, possibly discontinuous

    # --- Full profile convenience method ---
    profile_result = model.profile(
        latitude_deg=30.001842,
        longitude_deg=-91.203715,
        allow_partial=True,
    )

    # --- Batch query with independent failure ---
    batch = model.query_many(
        [
            CPTQuery(
                latitude_deg=30.001, longitude_deg=-91.201, depth_m=2.00,
                request_id="a",
            ),
            CPTQuery(
                latitude_deg=99.000, longitude_deg=-91.208,  # invalid latitude
                depth_range=DepthRange(1.00, 5.00),
                request_id="b",
            ),
        ]
    )
    for item in batch:
        if isinstance(item, CPTQueryFailure):
            _ = (item.query.request_id, item.error)
        else:
            _ = item.samples
