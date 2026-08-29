"""
Reference implementation of cptfm.query_service.CPTService, backed
directly by observed USGS soundings rather than the trained model.

This is a nearest-site adapter: `query()` resolves a (lat, lon) to the
closest sounding in the corpus (great-circle distance) and serves that
site's own measured qc/fs, interpolated onto CPT_DEPTH_INCREMENT_M steps.
`DataQuality.support_distance_m` reports the real distance to that
sounding, so a caller can see when "coverage" means "there happens to be
a sounding half a kilometre away" rather than a dense local grid. It does
not run cptfm.model.CPTMaskedAutoencoder — this is meant for exercising
and validating the query_service.py contract against ground truth, and
as a fallback/baseline the trained model can later be compared against.

A max_distance_deg cutoff controls what counts as "covered" at all;
the default (0.05 deg, roughly 5 km at mid-latitudes) is a placeholder,
not a validated geotechnical support radius — CPT properties can vary
significantly over tens of metres, so treat any query that only
succeeds by way of a multi-kilometre nearest site with real skepticism.
"""

from __future__ import annotations

import numpy as np

from .query_service import (
    CPTQuery, CPTQueryFailure, CPTResult, CPTSample, Coverage, CoverageBand,
    CoverageStatus, DataQuality, DepthRange, Elevation, VerticalDatum,
    CPT_DEPTH_INCREMENT_M, CPT_MAX_DEPTH_M, CPT_MIN_DEPTH_M,
    CPTError, DepthGridAlignmentError, InvalidDepthError,
    _generate_grid_depths, _is_on_grid, _snap_to_grid,
    _validate_coordinates, _validate_coverage_for_depths,
    _validate_depth_selection, _validate_global_bounds,
)
from .sources import usgs as usgs_reader

# The FGDC record for this corpus states elevations assume a "uniform
# NAD27 vertical datum" — but NAD27 is a horizontal datum. USGS data from
# that era conventionally paired NAD27 (horizontal) with NGVD29
# (vertical), so NGVD29 is used here as the least-wrong default rather
# than inventing a NAD27 member on VerticalDatum. The FGDC text itself
# says precise per-site datum requires checking individual legacy files,
# so this is a corpus-wide approximation, not a verified per-site fact.
_ASSUMED_ELEVATION_DATUM = VerticalDatum.NGVD29


class USGSObservedCPTService:
    def __init__(self, cpt_csv: str, max_distance_deg: float = 0.05):
        self._max_distance_deg = max_distance_deg
        records = usgs_reader.load(cpt_csv)

        self._sites: list[str] = []
        self._lat = np.empty(len(records), dtype=np.float64)
        self._lon = np.empty(len(records), dtype=np.float64)
        self._by_site: dict[str, object] = {}

        for i, rec in enumerate(records):
            self._sites.append(rec.site)
            self._lat[i] = rec.lat
            self._lon[i] = rec.lon
            self._by_site[rec.site] = rec

    # ---------------------------------------------------------------- #
    # Nearest-site resolution
    # ---------------------------------------------------------------- #

    def _nearest(self, latitude_deg: float, longitude_deg: float):
        # Plane distance in degrees, not great-circle — adequate for
        # ranking nearest-site candidates within the CONUS footprint of
        # this corpus, not a general-purpose geodesic distance.
        d = np.hypot(self._lat - latitude_deg, self._lon - longitude_deg)
        i = int(np.argmin(d))
        return self._sites[i], float(d[i])

    def _coverage_band_for_record(self, rec) -> tuple[CoverageBand, ...]:
        lo = max(float(rec.depth.min()), CPT_MIN_DEPTH_M)
        hi = min(float(rec.depth.max()), CPT_MAX_DEPTH_M)
        if lo >= hi:
            return ()
        # Snap to the query grid so band edges are directly comparable
        # against grid-aligned query depths.
        lo = _snap_to_grid(np.ceil(lo / CPT_DEPTH_INCREMENT_M) * CPT_DEPTH_INCREMENT_M)
        hi = _snap_to_grid(np.floor(hi / CPT_DEPTH_INCREMENT_M) * CPT_DEPTH_INCREMENT_M)
        if lo > hi:
            return ()
        return (CoverageBand(minimum_depth_m=lo, maximum_depth_m=hi),)

    # ---------------------------------------------------------------- #
    # CPTService protocol
    # ---------------------------------------------------------------- #

    def coverage_at(self, *, latitude_deg: float, longitude_deg: float) -> Coverage:
        site, dist = self._nearest(latitude_deg, longitude_deg)
        if dist > self._max_distance_deg:
            return Coverage(horizontal_status=CoverageStatus.NOT_COVERED, bands=())
        bands = self._coverage_band_for_record(self._by_site[site])
        if not bands:
            return Coverage(horizontal_status=CoverageStatus.NOT_COVERED, bands=())
        return Coverage(horizontal_status=CoverageStatus.COVERED, bands=bands)

    def query(self, query: CPTQuery) -> CPTResult:
        _validate_coordinates(query.latitude_deg, query.longitude_deg)
        _validate_depth_selection(query)

        if query.depth_m is not None:
            candidate_depths = (query.depth_m,)
        else:
            for d in (query.depth_range.start_m, query.depth_range.end_m):
                _validate_global_bounds(d)
            for d in (query.depth_range.start_m, query.depth_range.end_m):
                if not _is_on_grid(d):
                    raise DepthGridAlignmentError(
                        f"depth_range endpoint {d} m does not lie on the "
                        f"{CPT_DEPTH_INCREMENT_M} m grid."
                    )
            candidate_depths = _generate_grid_depths(query.depth_range)

        if query.depth_m is not None:
            _validate_global_bounds(query.depth_m)
            if not _is_on_grid(query.depth_m):
                raise DepthGridAlignmentError(
                    f"depth_m {query.depth_m} m does not lie on the "
                    f"{CPT_DEPTH_INCREMENT_M} m grid."
                )
            candidate_depths = (_snap_to_grid(query.depth_m),)

        site, dist = self._nearest(query.latitude_deg, query.longitude_deg)
        coverage = self.coverage_at(
            latitude_deg=query.latitude_deg, longitude_deg=query.longitude_deg
        )
        covered_depths = _validate_coverage_for_depths(
            candidate_depths, coverage, query.allow_partial
        )

        rec = self._by_site[site]
        qc = np.interp(covered_depths, rec.depth, rec.qc)
        fs = np.interp(covered_depths, rec.depth, rec.fs)

        quality = DataQuality(
            support_distance_m=dist * 111_000.0,  # rough deg->m, mid-latitude
            nearby_observation_count=1,
            coverage_class="nearest_site",
        )
        samples = tuple(
            CPTSample(depth_m=d, qc_mpa=float(q), fs_kpa=float(f), quality=quality)
            for d, q, f in zip(covered_depths, qc, fs)
        )
        return CPTResult(query=query, samples=samples)

    def query_many(self, queries: list[CPTQuery]):
        results = []
        for q in queries:
            try:
                results.append(self.query(q))
            except CPTError as e:
                results.append(CPTQueryFailure(query=q, error=e))
        return tuple(results)

    def profile(
        self, *, latitude_deg: float, longitude_deg: float,
        context=None, allow_partial: bool = False,
    ) -> CPTResult:
        return self.query(CPTQuery(
            latitude_deg=latitude_deg, longitude_deg=longitude_deg,
            depth_range=DepthRange(start_m=CPT_MIN_DEPTH_M, end_m=CPT_MAX_DEPTH_M),
            context=context, allow_partial=allow_partial,
        ))

    def ground_surface_elevation(self, site: str) -> Elevation | None:
        """Not part of the CPTService protocol — exposed here because the
        contract's SiteContext has no query-side way to ask "what is the
        elevation at this site", only to supply one as caller-provided
        context."""
        rec = self._by_site.get(site)
        if rec is None or rec.elevation_m is None:
            return None
        return Elevation(value_m=rec.elevation_m, datum=_ASSUMED_ELEVATION_DATUM)
