"""
Robertson (1990/2010) Soil Behaviour Type classification from CPT data.

Reference: Robertson, P.K. (2010). Soil behaviour type from the CPT: an update.
           2nd International Symposium on Cone Penetration Testing, CPT'10.
           Robertson, P.K. (1990). Soil classification using the CPT.
           Canadian Geotechnical Journal, 27(1), 151-158.

All pressures in kPa, depths in metres, qc in MPa, fs in kPa — matching the
conventions used everywhere else in this codebase.

No pore pressure (u2) measurement is assumed available, so qt = qc throughout.
Water table depth defaults to the ground surface, which is appropriate for the
coastal and deltaic settings that dominate both the USGS and BRO corpora.
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
_Pa      = 100.0   # atmospheric pressure, kPa
_GAMMA_W = 9.81    # unit weight of water, kN/m³ = kPa/m

# Robertson (1990) 9-zone SBTn names, keyed by integer zone number
ZONE_NAMES = {
    1: "Sensitive fine-grained",
    2: "Clay - organic soil",
    3: "Clays: clay to silty clay",
    4: "Silt mixtures: clayey silt & silty clay",
    5: "Sand mixtures: silty sand to sandy silt",
    6: "Sands: clean sands to silty sands",
    7: "Dense sand to gravelly sand",
    8: "Stiff sand to clayey sand",   # overconsolidated / cemented
    9: "Stiff fine-grained",          # overconsolidated / cemented
}

# Ic thresholds separating zones 2–7 on the Robertson (1990) SBTn chart.
# Values from Figure 2 of the Rocscience CPT Theory Manual (2025).
# Applied in ascending order so each larger threshold correctly overwrites the
# smaller one; the default (no threshold exceeded) is zone 7.
_IC_THRESHOLDS = [
    (1.31, 6),
    (2.05, 5),
    (2.60, 4),
    (2.95, 3),
    (3.60, 2),
]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def compute(
    depth:          np.ndarray,
    qc_mpa:         np.ndarray,
    fs_kpa:         np.ndarray,
    water_table_m:  float = 0.0,
    n_iter:         int   = 3,
) -> dict:
    """
    Compute Robertson (1990) SBTn Ic profile from raw CPT measurements.

    Parameters
    ----------
    depth          : (N,) metres below ground surface, monotonically increasing
    qc_mpa         : (N,) cone tip resistance in MPa
    fs_kpa         : (N,) sleeve friction in kPa
    water_table_m  : depth to water table in metres (default: ground surface)
    n_iter         : iterations for the Qtn stress-exponent refinement (§2.15)

    Returns
    -------
    dict with keys:
        Ic              (N,) soil behaviour type index
        zone            (N,) int in 1–9, Robertson (1990) SBTn zone
        Qt              (N,) normalised cone resistance (Qt, stress-exponent n=1)
        Qtn             (N,) stress-exponent-normalised cone resistance
        Fr              (N,) normalised friction ratio, %
        gamma_ratio     (N,) γ/γw from Robertson (2010) unit weight correlation
        sigma_v0        (N,) total overburden stress, kPa
        sigma_v0_eff    (N,) effective overburden stress, kPa
    """
    depth  = np.asarray(depth,  dtype=np.float64)
    qc_mpa = np.asarray(qc_mpa, dtype=np.float64)
    fs_kpa = np.asarray(fs_kpa, dtype=np.float64)
    qt_kpa = qc_mpa * 1000.0   # qt = qc (no u2 correction)

    # ------------------------------------------------------------------ #
    # 1. Unit weight from Robertson (2010) §2.3
    #    γ/γw = 0.27 log(Rf) + 0.36 log(qt/Pa) + 1.236
    #    Rf here is the simple friction ratio fs/qt × 100% (§2.2)
    # ------------------------------------------------------------------ #
    Rf = np.clip(fs_kpa / np.maximum(qt_kpa, 1e-3) * 100.0, 0.1, 20.0)
    gamma_ratio = (
        0.27 * np.log10(Rf)
        + 0.36 * np.log10(np.maximum(qt_kpa / _Pa, 1e-3))
        + 1.236
    )
    gamma_ratio = np.clip(gamma_ratio, 1.2, 2.4)   # 12–24 kN/m³
    gamma = gamma_ratio * _GAMMA_W                  # kPa/m

    # ------------------------------------------------------------------ #
    # 2. Overburden stresses §2.4
    #    σv0 = Σ(zi · γi)  integrated from surface
    # ------------------------------------------------------------------ #
    dz         = np.empty_like(depth)
    dz[0]      = depth[0]                    # surface to first measurement
    dz[1:]     = np.diff(depth)
    sigma_v0   = np.cumsum(gamma * dz)       # kPa

    u_hydro    = np.where(
        depth > water_table_m,
        _GAMMA_W * (depth - water_table_m),
        0.0,
    )
    sigma_v0_eff = np.maximum(sigma_v0 - u_hydro, 1.0)   # clamp ≥ 1 kPa

    # ------------------------------------------------------------------ #
    # 3. Qt and Fr §2.6, §2.9
    # ------------------------------------------------------------------ #
    net_cone = np.maximum(qt_kpa - sigma_v0, 1.0)   # kPa
    Qt = net_cone / sigma_v0_eff
    Fr = np.clip(fs_kpa / net_cone * 100.0, 0.01, 20.0)

    # ------------------------------------------------------------------ #
    # 4. Ic §2.10 — formula uses Qt (n=1) as stated in the manual
    # ------------------------------------------------------------------ #
    Ic = _ic(Qt, Fr)

    # ------------------------------------------------------------------ #
    # 5. Qtn with iterated stress exponent n §2.15
    #    Reported for downstream use; Ic and zone use Qt per the manual.
    # ------------------------------------------------------------------ #
    Qtn = Qt.copy()
    for _ in range(n_iter):
        n   = np.clip(0.381 * Ic + 0.05 * (sigma_v0_eff / _Pa) - 0.15, 0.5, 1.0)
        Qtn = (net_cone / _Pa) * (_Pa / sigma_v0_eff) ** n
        Qtn = np.maximum(Qtn, 0.01)

    # ------------------------------------------------------------------ #
    # 6. Zone assignment — Ic from Qt, zone 8/9 criterion also from Qt
    # ------------------------------------------------------------------ #
    zone = _zone_from_ic(Ic, Qt, Fr)

    return {
        "Ic":           Ic,
        "zone":         zone,
        "Qt":           Qt,
        "Qtn":          Qtn,
        "Fr":           Fr,
        "gamma_ratio":  gamma_ratio,
        "sigma_v0":     sigma_v0,
        "sigma_v0_eff": sigma_v0_eff,
    }


def zone_name(zone: int) -> str:
    return ZONE_NAMES.get(int(zone), "Unknown")


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _ic(Q: np.ndarray, Fr: np.ndarray) -> np.ndarray:
    return np.sqrt(
        (3.47 - np.log10(np.maximum(Q, 0.01))) ** 2
        + (np.log10(np.maximum(Fr, 0.01)) + 1.22) ** 2
    )


def _zone_from_ic(
    Ic: np.ndarray,
    Qt: np.ndarray,
    Fr: np.ndarray,
) -> np.ndarray:
    # Default: zone 7 (dense sand/gravel, Ic ≤ 1.31).
    # Each threshold overwrites in ascending order so the highest applicable
    # zone number (softest material) always wins.
    zone = np.full(len(Ic), 7, dtype=np.int8)
    for threshold, z in _IC_THRESHOLDS:
        zone[Ic > threshold] = z

    # Zone 1 override: sensitive fine-grained — lower-left of the SBTn chart,
    # low normalised resistance and very low friction ratio.
    zone[(Qt < 12) & (Fr < 0.5)] = 1

    # Zones 8 and 9 (OC/cemented) are not assigned here.  They sit above the
    # main Ic arc pattern on the Robertson chart and require geological context
    # (stratigraphy, known preconsolidation history) beyond what qc/fs alone
    # can supply at the shallow depths (0–25 m) of this corpus.

    return zone
