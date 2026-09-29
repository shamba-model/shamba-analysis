import csv
import glob
import os
from pathlib import Path

# Mirrors get_first_pos_def() in shamba/model/soil_models/roth_c/roth_c.py:
# RothC logs "EVAPORATION ALWAYS EXCEED RAINFALL" when no month in the
# 12-month climatology has rain exceeding evaporation (deficit = rain - evap
# never positive). This scans each Morris batch site's raw climate_cover_data
# csv and flags which ones would trigger that warning.

THIS_FILE = Path(__file__).resolve()
SITE_DIR = THIS_FILE.parent / "inputs" / "climate_cover_data"
OUT_PATH = THIS_FILE.parent / "results" / "morris_batch" / "evaporation_exceeds_rainfall_sites.csv"

rows = []

for path in sorted(glob.glob(os.path.join(SITE_DIR, "site_*_climate_cover_data.csv"))):
    site_id = os.path.basename(path).split("_")[1]
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rain = []
        evap = []
        for r in reader:
            rain.append(float(r["rain"]))
            evap.append(float(r["evap"]))

    deficit = [ra - ev for ra, ev in zip(rain, evap)]
    flagged = not any(d > 0 for d in deficit)
    flagged_evap_never_exceeds = not any(d < 0 for d in deficit)

    rows.append(
        {
            "site_id": site_id,
            "flagged_evap_always_exceeds_rainfall": flagged,
            "flagged_evap_never_exceeds_rainfall": flagged_evap_never_exceeds,
            "max_deficit_rain_minus_evap": max(deficit),
            "annual_rain_total": sum(rain),
            "annual_evap_total": sum(evap),
        }
    )

with open(OUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

n_flagged = sum(1 for r in rows if r["flagged_evap_always_exceeds_rainfall"])
n_flagged_never = sum(1 for r in rows if r["flagged_evap_never_exceeds_rainfall"])
print(f"Total sites: {len(rows)}")
print(f"Flagged sites (evap always exceeds rainfall): {n_flagged}")
print(f"Flagged sites (evap never exceeds rainfall): {n_flagged_never}")
print(f"Written to: {OUT_PATH}")
