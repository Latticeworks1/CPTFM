from dataclasses import dataclass
import numpy as np


@dataclass
class CPTRecord:
    site:  str
    lon:   float
    lat:   float
    depth: np.ndarray   # (M,) float32 — metres below surface
    qc:    np.ndarray   # (M,) float32 — cone tip resistance, MPa
    fs:    np.ndarray   # (M,) float32 — sleeve friction, kPa

    # Site-level metadata, absent for sources that don't provide it (e.g. BRO).
    # None means "not recorded at the source", distinct from a measured 0.
    elevation_m:         float | None = None   # ground-surface elevation, m (site vertical datum)
    water_table_depth_m: float | None = None   # measured depth to water table, m below surface
    water_table_notes:   str   | None = None   # qualitative note when depth wasn't measured numerically
    date:                str   | None = None   # collection date, source-native format
    county:              str   | None = None
    state:               str   | None = None
    operator:            str   | None = None
    cone:                str   | None = None
