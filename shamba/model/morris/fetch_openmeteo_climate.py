"""
Bulk-fetch ERA5 historical daily climate data from Open-Meteo for many
locations, and save one split-file climate_cover_data.csv per location,
ready for use as SHAMBA split-file input (see io_handler.py's
STEP 1 / Option 2 documentation and model/climate.py's from_vectors()).

INPUT
-----
A CSV with columns: lat, lon (plus any other stratification columns,
which are ignored). An "id" column is optional; if absent, sites are
numbered site_0000, site_0001, ...

OUTPUT
------
model/morris/inputs/climate_cover_data/{id}_climate_cover_data.csv
    -- 12 rows of monthly-climatology temp/rain/evap plus base_cover/proj_cover,
       in the column layout SHAMBA's split-file pathway expects
       (eg projects/examples/UG_TS_2016/input/WL_climate_cover_data.csv).
       base_cover and proj_cover are both broadcast to 1 (full cover).
       Always written next to this script (not cwd-relative).

       Also includes temp_std/rain_std/evap_std columns: the inter-annual
       monthly std across the fetched START_DATE-END_DATE record, computed by
       model/climate.py's from_vectors(); the same function the main model
       uses to derive climate uncertainty from a multi-year split-file input.
       run_morris_direct.py reads these columns back in and uses them to scale
       the temp_ci_delta/rain_ci_delta/evap_ci_delta parameters. NB.
       run_mc_direct.py and shamba_command_line.py build ClimateData from a plain
       (temp, rain, evap) tuple and wouldn't consume these columns.)

Run from the shamba/ directory:
    poetry run python -m model.morris.fetch_openmeteo_climate

Why batching:
Open-Meteo's archive API accepts multiple locations in a single request
via comma-separated lat/lon params, returning a JSON array with one
response per location in the same order. Batching locations per call
cuts down the number of requests, but the API also enforces a hard
per-call data-volume cap (locations x days x variables), so BATCH_SIZE
can't just be maximised; see the comment by BATCH_SIZE below.
"""

import time
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import requests

from model.climate import from_vectors, ClimateDataSchema
from model.common.data_sources.climate import aggregate_daily_to_monthly

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
INPUT_CSV = "Stratified_TestSites_byKGzoneCropland.csv" # must have columns: lat, lon (id optional)
START_DATE = "1995-01-01"
END_DATE = "2024-12-31"
# Matches the variable set used by model/common/data_sources/climate.py's
# get_climate_data(), so aggregate_daily_to_monthly() below rolls these up
# the same way the live climate-API path does.
DAILY_VARS = [
    "temperature_2m_mean",
    "rain_sum",
    "et0_fao_evapotranspiration",
]
# Open-Meteo enforces a hard per-call data-volume cap (locations x days x
# variables). Empirically, for this 30-year/3-variable request shape, 20
# locations/call succeeds and 50 fails with a "too much data" 400 --
# even on a fully fresh quota. 15 keeps a safety margin below that boundary.
BATCH_SIZE = 15                      # locations per API request
SLEEP_BETWEEN_BATCHES = 2.0          # seconds, fair use of Open-Meteo's free API
# Anchored to this script's own directory (model/morris/), not the cwd the
# container happened to be launched from. docker-compose.yaml only bind-mounts
# ./shamba/model:/app/model back to the host, so a cwd-relative "inputs" dir
# (e.g. under /app or /app/server) is written inside the container's ephemeral
# filesystem and never appears on the host.
OUTPUT_DIR = Path(__file__).parent / "inputs"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# How to aggregate each daily variable when rolling up to monthly, passed
# straight to aggregate_daily_to_monthly().
MONTHLY_AGG = {
    "temperature_2m_mean": np.mean,
    "rain_sum": np.sum,
    "et0_fao_evapotranspiration": np.sum,
}

RETRIES = 5
RETRY_BACKOFF = 0.5  # seconds, doubled after each retry
RETRY_STATUS_CODES = {500, 502, 503, 504}  # transient server errors, worth retrying

SESSION = requests.Session()


class QuotaExceededError(Exception):
    """Open-Meteo's free-tier fair-use quota (calls and/or data volume,
    tracked per source IP over an hour) has been exhausted. Retrying
    immediately won't help, the caller should stop and try again later.
    """


def _get_with_retry(url, params):
    """GET with an exponential-backoff retry on transient failures.
    No response caching: if you re-run the script it re-fetches, which is
    fine at this batch size.

    429 and the "too much data" 400 both mean the fair-use quota is spent
    for now (empirically: 50-location/30-year/3-variable batches succeed
    individually but a run of ~8-9 of them back-to-back trips this) --
    raised immediately as QuotaExceededError rather than retried, since a
    few seconds of backoff won't free up an hourly quota.
    """
    backoff = RETRY_BACKOFF
    for attempt in range(RETRIES + 1):
        try:
            response = SESSION.get(url, params=params, timeout=60)
        except requests.exceptions.RequestException:
            if attempt == RETRIES:
                raise
        else:
            if response.status_code == 429 or (
                response.status_code == 400 and "too much data" in response.text.lower()
            ):
                reason = response.json().get("reason", response.text)
                raise QuotaExceededError(reason)
            if response.status_code not in RETRY_STATUS_CODES:
                response.raise_for_status()
                return response
            if attempt == RETRIES:
                response.raise_for_status()
        time.sleep(backoff)
        backoff *= 2


def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def load_locations(input_csv):
    """Load and normalise the site list: lat/lon -> latitude/longitude,
    and generate an id column if the input doesn't have one."""
    df = pd.read_csv(Path(__file__).parent / "inputs" / input_csv)
    df = df.rename(columns={"lat": "latitude", "lon": "longitude"})
    if "id" not in df.columns:
        df.insert(0, "id", [f"site_{i:04d}" for i in range(len(df))])
    return df.to_dict("records")


def fetch_batch(locations_batch):
    """locations_batch: list of dicts with id, latitude, longitude.
    Returns list of (loc, daily_dataframe) tuples.

    Open-Meteo's archive API accepts comma-separated latitude/longitude
    lists and returns a JSON array (one object per location, same order)
    when more than one location is requested, or a single JSON object
    for one location.
    """
    params = {
        "latitude": ",".join(str(loc["latitude"]) for loc in locations_batch),
        "longitude": ",".join(str(loc["longitude"]) for loc in locations_batch),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARS),
        "timezone": "UTC",
    }

    response = _get_with_retry(ARCHIVE_URL, params=params)
    payload = response.json()
    responses = payload if isinstance(payload, list) else [payload]

    results = []
    for loc, resp in zip(locations_batch, responses):
        daily = resp["daily"]
        df = pd.DataFrame({"date": daily["time"], **{var: daily[var] for var in DAILY_VARS}})
        df.insert(0, "longitude", resp["longitude"])
        df.insert(0, "latitude", resp["latitude"])
        df.insert(0, "location_id", loc["id"])
        results.append((loc, df))
    return results


def to_climate_data(daily_df):
    """Roll a location's daily dataframe up to a ClimateData object, using
    the same aggregate_daily_to_monthly() the live climate-API path uses,
    and the same PET -> open-pan evaporation conversion (/0.75) applied in
    model/climate.py's from_location() for et0_fao_evapotranspiration.
    """
    date_strings = daily_df["date"].tolist()

    temp = aggregate_daily_to_monthly(
        daily_df["temperature_2m_mean"].to_numpy(), date_strings, MONTHLY_AGG["temperature_2m_mean"]
    )
    rain = aggregate_daily_to_monthly(
        daily_df["rain_sum"].to_numpy(), date_strings, MONTHLY_AGG["rain_sum"]
    )
    et0 = aggregate_daily_to_monthly(
        daily_df["et0_fao_evapotranspiration"].to_numpy(), date_strings, MONTHLY_AGG["et0_fao_evapotranspiration"]
    )
    evap = et0 / 0.75

    return from_vectors(temperature=temp, rain=rain, evaporation=evap)


def write_climate_cover_csv(loc, climate, out_dir):
    """Write a {id}_climate_cover_data.csv matching SHAMBA's split-file
    format (see projects/examples/UG_TS_2016/input/WL_climate_cover_data.csv):
    12 rows of temp/rain/evap climatology plus base_cover/proj_cover, plus
    temp_std/rain_std/evap_std: the inter-annual monthly std over the
    fetched record, computed by from_vectors() (see to_climate_data() above).

    ASSUMPTION: base_cover and proj_cover are broadcast to 1 (full cover) here.
    """
    errors = ClimateDataSchema().validate({
        "temperature": climate.temperature.tolist(),
        "rain": climate.rain.tolist(),
        "evaporation": climate.evaporation.tolist(),
    })
    if errors:
        print(f"  WARNING: skipping {loc['id']}, invalid climate data: {errors}", file=sys.stderr)
        return

    out_df = pd.DataFrame({
        "temp": climate.temperature,
        "rain": climate.rain,
        "evap": climate.evaporation,
        "temp_std": climate.temperature_std,
        "rain_std": climate.rain_std,
        "evap_std": climate.evaporation_std,
        "base_cover": 1,
        "proj_cover": 1,
    })
    out_df.to_csv(out_dir / f"{loc['id']}_climate_cover_data.csv", index=False)


def main():
    locations = load_locations(INPUT_CSV)
    climate_cover_dir = OUTPUT_DIR / "climate_cover_data"
    climate_cover_dir.mkdir(parents=True, exist_ok=True)

    # Resumable: skip sites already written by a previous (possibly
    # quota-cut-short) run, so re-running after the quota resets picks up
    # where it left off instead of re-fetching everything.
    already_done = {p.name.removesuffix("_climate_cover_data.csv") for p in climate_cover_dir.glob("*_climate_cover_data.csv")}
    locations = [loc for loc in locations if loc["id"] not in already_done]
    print(f"Loaded {len(locations)} locations to fetch from {INPUT_CSV} ({len(already_done)} already done, skipped)")

    batches = list(chunk(locations, BATCH_SIZE))
    n_written = 0

    for batch_num, batch in enumerate(batches, 1):
        print(f"Batch {batch_num}/{len(batches)} ({len(batch)} locations)...")
        try:
            results = fetch_batch(batch)
        except QuotaExceededError as e:
            remaining = len(locations) - n_written
            print(
                f"Open-Meteo quota exhausted: {e}\n"
                f"Stopping -- wrote {n_written} sites this run, {remaining} remaining. "
                f"Re-run the script later (e.g. after an hour) to continue.",
                file=sys.stderr,
            )
            break
        except Exception as e:
            print(f"  ERROR on batch {batch_num}: {e}", file=sys.stderr)
            continue

        for loc, daily_df in results:
            climate = to_climate_data(daily_df)
            write_climate_cover_csv(loc, climate, climate_cover_dir)
            n_written += 1

        if batch_num < len(batches):
            time.sleep(SLEEP_BETWEEN_BATCHES)

    print(f"Done. Wrote {n_written} climate_cover_data.csv files to {climate_cover_dir}")


if __name__ == "__main__":
    main()
