#!/usr/bin/env python3
"""
Append SoilGrids Q0.05 / Q0.95 uncertainty bounds to the selected-sites CSV,
reading ISRIC's hosted VRTs directly over the network (no bulk download).

WHY THIS APPROACH
-----------------
  * The SoilGrids REST API is currently paused by ISRIC.
  * The Q0.05 / Q0.95 layers are not in Earth Engine (the community
    projects/soilgrids-isric/* assets serve only the _mean layers).
  * The WCS service caps one response at 16384 x 16384 px, so the tropical
    strip can't be fetched whole, and the 5-calls/min fair-use limit makes
    many small WCS requests impractical.
So instead we open each hosted .vrt with GDAL's /vsicurl/ virtual filesystem
and sample it at the site points. GDAL fetches ONLY the tiles the points
actually touch — kilobytes, not gigabytes — so there is nothing to download
or manage locally.

WHAT IT PRODUCES
----------------
Four new columns appended to a copy of the input CSV, in RAW SoilGrids units
(integers as stored — directly comparable to the existing ocs / clay columns,
which come from the _mean assets at the same scaling):

    ocs_q05,  ocs_q95    — 0-30 cm, taken directly (ocs is natively 0-30 cm)
    clay_q05, clay_q95   — 0-30 cm, DEPTH-WEIGHTED from the three layers:
                           (Q_0-5 * 5 + Q_5-15 * 10 + Q_15-30 * 15) / 30

CAVEAT ON THE CLAY QUANTILES
----------------------------
Depth-weighting the per-layer quantiles gives the depth-weighted average of
the layer-wise 5th/95th percentiles. That is NOT the 5th/95th percentile of
the depth-weighted mean: it implicitly assumes the layers' errors are
perfectly correlated, so the interval is conservative (wider than a properly
propagated one). SoilGrids does not publish the layer covariances needed to
do it exactly. It remains a reasonable uncertainty indicator, and it is
consistent with how the clay MEAN is depth-weighted in the pipeline.

PROJECTION NOTE
---------------
SoilGrids rasters are in Homolosine (EPSG:152160) with coordinates in METRES,
not lon/lat (confirmed via WCS DescribeCoverage: axisLabels "x y", uomLabels
"m m"). The script reads each raster's declared CRS and reprojects the site
coordinates into it before sampling.

USAGE
-----
    pip install rasterio pandas numpy pyproj
    python append_soilgrids_quantiles.py sites.csv sites_with_quantiles.csv

Needs a working connection to files.isric.org at run time. For 500 points it
typically takes a couple of minutes. If ISRIC's file server is slow or down,
this will hang/fail — rerun when it is available, or fall back to downloading
the VRTs + tiles locally and opening those paths instead.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.env import Env
from pyproj import Transformer

# ---------------------------------------------------------------------------
# Hosted VRT locations. SoilGrids serves one VRT per (property, depth, stat)
# under .../data/<property>/<coverage>.vrt. /vsicurl/ lets GDAL read them
# remotely, pulling only the tiles the sample points fall on.
# ---------------------------------------------------------------------------
BASE = "https://files.isric.org/soilgrids/latest/data"

def vrt_url(prop: str, coverage: str) -> str:
    return f"/vsicurl/{BASE}/{prop}/{coverage}.vrt"

# (internal handle) -> (property, coverage name)
COVERAGES = {
    "ocs_q05":        ("ocs",  "ocs_0-30cm_Q0.05"),
    "ocs_q95":        ("ocs",  "ocs_0-30cm_Q0.95"),
    "clay_q05_0_5":   ("clay", "clay_0-5cm_Q0.05"),
    "clay_q05_5_15":  ("clay", "clay_5-15cm_Q0.05"),
    "clay_q05_15_30": ("clay", "clay_15-30cm_Q0.05"),
    "clay_q95_0_5":   ("clay", "clay_0-5cm_Q0.95"),
    "clay_q95_5_15":  ("clay", "clay_5-15cm_Q0.95"),
    "clay_q95_15_30": ("clay", "clay_15-30cm_Q0.95"),
}

# Thickness (cm) of each clay depth layer, for the 0-30 cm depth weighting.
DEPTH_WEIGHTS = {"0_5": 5, "5_15": 10, "15_30": 15}
TOTAL_DEPTH = 30

# GDAL /vsicurl/ tuning: allow partial-range reads so only needed tiles are
# fetched, and don't choke on the directory listing.
GDAL_OPTS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".vrt,.tif",
    "VSI_CACHE": "TRUE",
    "GDAL_HTTP_TIMEOUT": "120",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "5",
}


def sample_remote(prop: str, coverage: str, lons, lats):
    """Open a hosted VRT via /vsicurl/, reproject points, sample. NaN=nodata."""
    url = vrt_url(prop, coverage)
    with rasterio.open(url) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = transformer.transform(list(lons), list(lats))
        nodata = src.nodata
        out = []
        for val in src.sample(list(zip(xs, ys))):
            v = val[0]
            out.append(np.nan if (nodata is not None and v == nodata)
                       else float(v))
        return out


def main():
    ap = argparse.ArgumentParser(
        description="Append SoilGrids Q0.05/Q0.95 columns to a sites CSV "
                    "(reads ISRIC's hosted VRTs directly).")
    ap.add_argument("in_csv")
    ap.add_argument("out_csv")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    for col in ("lat", "lon"):
        if col not in df.columns:
            sys.exit(f"ERROR: input CSV has no '{col}' column. "
                     f"Found: {list(df.columns)}")
    lons, lats = df["lon"].tolist(), df["lat"].tolist()
    print(f"Loaded {len(df)} sites from {args.in_csv}")

    sampled = {}
    print("\nSampling ISRIC hosted VRTs via /vsicurl/ ...")
    with Env(**GDAL_OPTS):
        for key, (prop, coverage) in COVERAGES.items():
            try:
                sampled[key] = sample_remote(prop, coverage, lons, lats)
                print(f"  ok  {coverage}")
            except Exception as e:
                sys.exit(
                    f"\nFailed reading {coverage} from ISRIC.\n  {e}\n"
                    f"If the file server is down/slow, rerun later, or download "
                    f"the .vrt + tile folders from {BASE}/{prop}/ and adapt the "
                    f"script to open the local paths."
                )

    # ocs: natively 0-30 cm, taken directly.
    df["ocs_q05"] = sampled["ocs_q05"]
    df["ocs_q95"] = sampled["ocs_q95"]

    # clay: depth-weighted 0-30 cm, matching the MEAN's weighting. NaN in any
    # contributing layer -> NaN, mirroring Earth Engine mask propagation.
    for q in ("q05", "q95"):
        parts = [
            np.array(sampled[f"clay_{q}_{d}"], dtype=float) * w
            for d, w in DEPTH_WEIGHTS.items()
        ]
        df[f"clay_{q}"] = np.sum(parts, axis=0) / TOTAL_DEPTH

    new_cols = ["ocs_q05", "ocs_q95", "clay_q05", "clay_q95"]
    print("\nNull counts in the new columns:")
    for c in new_cols:
        print(f"  {c}: {int(df[c].isna().sum())} null of {len(df)}")
    if df[new_cols].isna().any().any():
        print("\nNOTE: nulls mean a site fell on SoilGrids nodata in at least "
              "one contributing layer. The GEE eligibility mask only requires "
              "the MEAN layers to be present — not the QUANTILE layers. "
              "Inspect those rows before use.")

    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {len(df)} rows with quantile columns to {args.out_csv}")


if __name__ == "__main__":
    main()