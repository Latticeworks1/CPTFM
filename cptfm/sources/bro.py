import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

from . import CPTRecord

_NS = {
    "b":   "http://www.broservices.nl/xsd/dscpt/1.1",
    "bc":  "http://www.broservices.nl/xsd/brocommon/3.0",
    "cpt": "http://www.broservices.nl/xsd/cptcommon/1.1",
    "gml": "http://www.opengis.net/gml/3.2",
}

_BRO_BASE    = "https://publiek.broservices.nl/sr/cpt/v1"
_SENTINEL    = -999999.0
_MIN_DEPTH_M = 1.0
_QC_MIN      = 0.01
_QC_MAX      = 120.0
_FS_MIN      = 0.0
_FS_MAX      = 1500.0
_RF_MAX      = 0.20
_MIN_ROWS    = 10

_GML_POS  = "{http://www.opengis.net/gml/3.2}pos"
_CPT_VALS = "{http://www.broservices.nl/xsd/cptcommon/1.1}values"

_TILE_RADIUS  = 1.0
# 5km grid across the Netherlands (0.045° lat ≈ 5km, 0.072° lon ≈ 5km at 52°N).
# Per-tile cap in fetch() ensures geographic spread even where density is high.
import numpy as _np
_TILE_CENTERS = [
    (round(float(la), 4), round(float(lo), 4))
    for la in _np.arange(51.20, 53.55, 0.045)
    for lo in _np.arange(3.50,  7.25, 0.072)
]
del _np


def load(csv_path: str) -> list[CPTRecord]:
    """Load CPT records from a BRO CSV previously written by fetch()."""
    df = pd.read_csv(csv_path)
    records = []
    for site_id, grp in df.groupby("site"):
        grp = grp.sort_values("depth")
        d   = grp["depth"].values.astype(np.float32)
        qc  = grp["tip"].values.astype(np.float32)
        fs  = grp["sleeve"].values.astype(np.float32)
        lon = float(grp["lon"].iloc[0])
        lat = float(grp["lat"].iloc[0])
        keep = np.isfinite(qc) & np.isfinite(fs) & (d > 0)
        if keep.sum() < 2:
            continue
        records.append(CPTRecord(
            site=str(site_id), lon=lon, lat=lat,
            depth=d[keep], qc=qc[keep], fs=fs[keep],
        ))
    return records


def _search_ids(session, lat, lon, radius):
    body = {"area": {"enclosingCircle": {"center": {"lat": lat, "lon": lon}, "radius": radius}}}
    try:
        r = session.post(f"{_BRO_BASE}/characteristics/searches", json=body, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        return [el.text for el in root.findall(".//bc:broId", _NS)]
    except Exception as e:
        print(f"  search ({lat:.2f},{lon:.2f}) failed: {e}", flush=True)
        return []


def _fetch_one(session, bro_id) -> CPTRecord | None:
    try:
        r = session.get(f"{_BRO_BASE}/objects/{bro_id}", timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
    except Exception as e:
        print(f"  fetch {bro_id} failed: {e}", flush=True)
        return None

    pos_els = list(root.iter(_GML_POS))
    if not pos_els:
        return None
    parts = pos_els[0].text.strip().split()
    if len(parts) < 2:
        return None
    lat, lon = float(parts[0]), float(parts[1])

    values_el = next(root.iter(_CPT_VALS), None)
    if values_el is None or not values_el.text:
        return None

    rows = []
    for record in values_el.text.strip().split(";"):
        record = record.strip()
        if not record:
            continue
        cols = record.split(",")
        if len(cols) < 4:
            continue
        try:
            depth  = float(cols[1])
            fs_kpa = float(cols[2])
            qc_mpa = float(cols[3])
        except ValueError:
            continue
        if abs(depth - _SENTINEL) < 1 or abs(fs_kpa - _SENTINEL) < 1 or abs(qc_mpa - _SENTINEL) < 1:
            continue
        if depth <= 0:
            continue
        if not (_QC_MIN <= qc_mpa <= _QC_MAX) or not (_FS_MIN <= fs_kpa <= _FS_MAX):
            continue
        if qc_mpa > 0 and (fs_kpa / (qc_mpa * 1000.0)) > _RF_MAX:
            continue
        rows.append((depth, qc_mpa, fs_kpa))

    if len(rows) < _MIN_ROWS or max(r[0] for r in rows) < _MIN_DEPTH_M:
        return None

    return CPTRecord(
        site=bro_id, lon=lon, lat=lat,
        depth=np.array([r[0] for r in rows], dtype=np.float32),
        qc=np.array([r[1] for r in rows], dtype=np.float32),
        fs=np.array([r[2] for r in rows], dtype=np.float32),
    )


def fetch(max_soundings: int = 3000, rate_delay: float = 0.15) -> list[CPTRecord]:
    """Fetch CPT soundings from the BRO public REST API."""
    session = requests.Session()
    session.headers["User-Agent"] = "CPTFM-research/1.0"

    seen     = set()
    per_tile = max(1, max_soundings // len(_TILE_CENTERS))
    id_list  = []

    print("Phase 1: collecting BRO IDs across Netherlands grid ...", flush=True)
    for i, (lat, lon) in enumerate(_TILE_CENTERS):
        ids     = _search_ids(session, lat, lon, _TILE_RADIUS)
        new     = [x for x in ids if x not in seen]
        sampled = new[:per_tile]
        seen.update(sampled)
        id_list.extend(sampled)
        print(f"  tile {i+1}/{len(_TILE_CENTERS)}  ({lat:.2f},{lon:.2f})  "
              f"found {len(ids)}  new {len(new)}  kept {len(sampled)}  "
              f"total {len(id_list)}", flush=True)
        time.sleep(0.5)
        if len(id_list) >= max_soundings:
            break

    id_list = id_list[:max_soundings]
    print(f"\nPhase 1 complete: {len(id_list)} unique IDs\n", flush=True)

    print("Phase 2: fetching soundings ...", flush=True)
    records = []
    failed  = 0
    for i, bro_id in enumerate(id_list):
        rec = _fetch_one(session, bro_id)
        if rec is None:
            failed += 1
        else:
            records.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(id_list)}  ok={len(records)}  failed={failed}", flush=True)
        time.sleep(rate_delay)

    print(f"\nPhase 2 complete: {len(records)} soundings  {failed} failures\n", flush=True)
    return records
