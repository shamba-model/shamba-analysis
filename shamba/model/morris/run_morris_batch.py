"""Run the Morris sensitivity batch locally, one site at a time.

Sites are fully independent, and each site's own Morris run already fans its 
evaluations across every core via runner.py's ProcessPoolExecutor. 
So run sites serially here.

For each site, this script:
  1. Stages projects/morris_batch_XXXX/input/ (climate, soil-info, common
     inputs, and the {}_* templates renamed to site_XXXX_*).
  2. Runs run_morris_direct.py as a subprocess (each site gets its own process,
     so configuration's module-level SAVE_DIR/INPUT_DIR globals never collide).
  3. Collects the outputs and appends a row to batch_log.csv.

Progress and a rolling ETA are printed after every site. Use --resume to skip
sites whose collected results already exist, so an interrupted run continues.

When the batch finishes it collapses each site's per-year morris_results CSV
down to one total_mu_star (and total_mu_star_conf) per parameter — the Morris
indices for the total emissions difference summed over the whole project
horizon — and writes one row per (site_index, parameter) to
morris_results_all_sites.csv. Pass --aggregate-only to rebuild that table
from existing results without rerunning.

Usage (inside Docker; host shamba/ is mounted as /app, so no "shamba/" prefix):
    poetry run python /app/model/morris/run_morris_batch.py
    poetry run python /app/model/morris/run_morris_batch.py --start 0 --end 49
    poetry run python /app/model/morris/run_morris_batch.py --aggregate-only
"""

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


# --- Repo layout, resolved relative to this file so CWD doesn't matter ---
THIS_FILE = Path(__file__).resolve()
MORRIS_DIR = THIS_FILE.parent                       # shamba/model/morris
SHAMBA_DIR = MORRIS_DIR.parents[1]                  # shamba
REPO_ROOT = SHAMBA_DIR.parent                       # repo root

INPUTS_DIR = MORRIS_DIR / "inputs"
COMMON_INPUTS_DIR = INPUTS_DIR / "common_inputs"
CLIMATE_COVER_DIR = INPUTS_DIR / "climate_cover_data"
PROJECTS_DIR = SHAMBA_DIR / "projects"
RUN_MORRIS_DIRECT = SHAMBA_DIR / "run_morris_direct.py"

# Common input files copied verbatim into every site's input folder. The three
# species-lookup tables are read from INPUT_DIR by run_morris_direct.py, so they
# must be present per-site.
COMMON_COPY_FILES = (
    "biomass_pool_params.csv",
    "crop_params.csv",
    "tree_params.csv",
)
# Templates named "{}_*.csv": the "{}" becomes the site prefix (site_XXXX).
TEMPLATE_GLOB = "{}_*.csv"

# TestSites_withSoilGridsQs.csv's clay columns are raw SoilGrids units; divide
# by this to match the 0-100 percentage read_soil_table() expects - the same
# factor convert_units_in_api_response() applies on the live API path (see
# UNIT_CONVERSIONS[PROPORTION_OF_CLAY_IN_FINE_FRACTION] in data_sources/soil.py).
CLAY_UNIT_CONVERSION_FACTOR = 10

# TestSites columns, in file order. soil-info.csv needs a subset of these.
TESTSITES_COLUMNS = (
    "lat", "lon", "kg_class", "ttc", "landcover_crop_frac", "koppen_geiger",
    "ocs", "clay", "ocs_q05", "ocs_q95", "clay_q05", "clay_q95",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--testsites", default=str(INPUTS_DIR / "TestSites_withSoilGridsQs.csv"),
                   help="TestSites CSV; one data row per site (default: TestSites_withSoilGridsQs.csv)")
    p.add_argument("--bounds-file", default=str(COMMON_INPUTS_DIR / "bounds_file.csv"),
                   help="Morris bounds CSV passed through to run_morris_direct.py")
    p.add_argument("--start", type=int, default=0,
                   help="First site index, inclusive (default: 0)")
    p.add_argument("--end", type=int, default=None,
                   help="Last site index, inclusive (default: last row of the TestSites file)")
    p.add_argument("--n-proj-cohorts", type=int, default=1,
                   help="Passed to run_morris_direct.py (default: 1)")
    p.add_argument("--n-base-cohorts", type=int, default=1,
                   help="Passed to run_morris_direct.py (default: 1)")
    p.add_argument("--n-trajectories", type=int, default=40,
                   help="Morris trajectories N, passed through (default: 40)")
    p.add_argument("--num-levels", type=int, default=8,
                   help="Morris grid levels, passed through (default: 8)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed, passed through (default: 42)")
    p.add_argument("--results-dir", default=str(MORRIS_DIR / "results" / "morris_batch"),
                   help="Where collected per-site outputs are copied")
    p.add_argument("--resume", action="store_true",
                   help="Skip sites whose collected morris_results CSV already exists")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Abort the batch on the first site that fails (default: keep going)")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip the batch; just rebuild morris_results_all_sites.csv from existing results")
    p.add_argument("--no-aggregate", action="store_true",
                   help="Don't build the combined morris_results_all_sites.csv after the batch")
    p.add_argument("--dry-run", action="store_true",
                   help="Stage inputs and print the command for each site without running it")
    return p.parse_args()


def read_testsites(path: Path) -> list[dict]:
    """Return one dict per site row, keyed by TESTSITES_COLUMNS."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in TESTSITES_COLUMNS if c not in header]
        if missing:
            raise ValueError(
                f"{path.name} is missing expected column(s): {', '.join(missing)}.\n"
                f"Found columns: {', '.join(reader.fieldnames or [])}"
            )
        return list(reader)


def aggregate_results(results_dir: Path) -> Path | None:
    """
    Reads results_dir/morris_results_morris_batch_XXXX.csv (each with columns
    parameter,year,mu,mu_star,sigma,mu_star_conf), takes the year="total" row
    per parameter — the Morris indices for the total emissions difference
    summed over the whole project horizon, not an average across per-year
    indices — and writes one row per (site_index, project_name, parameter) to
    results_dir/morris_results_all_sites.csv. Returns the output path, or None
    if no per-site files were found.
    """
    file_prefix = "morris_results_morris_batch_"
    out_path = results_dir / "morris_results_all_sites.csv"
    # The aggregate itself doesn't match file_prefix, so it's never re-folded in.
    result_files = sorted(results_dir.glob(f"{file_prefix}*.csv"))
    if not result_files:
        return None

    site_tables = []
    for path in result_files:
        project_name = path.stem[len("morris_results_"):]   # morris_batch_0007
        site_index = int(path.stem[len(file_prefix):])      # 7
        df = pd.read_csv(path)
        totals = (
            df[df["year"] == "total"][["parameter", "mu_star", "mu_star_conf"]]
            .rename(columns={"mu_star": "total_mu_star", "mu_star_conf": "total_mu_star_conf"})
            .reset_index(drop=True)
        )
        totals.insert(0, "project_name", project_name)
        totals.insert(0, "site_index", site_index)
        site_tables.append(totals)

    pd.concat(site_tables, ignore_index=True).to_csv(out_path, index=False)
    return out_path


def stage_site(site_index: int, site_row: dict, bounds_file: Path) -> tuple[str, str, Path]:
    """Create projects/morris_batch_XXXX/input/ and populate it.

    Returns (project_name, prefix, input_dir).
    """
    padded = f"{site_index:04d}"
    project_name = f"morris_batch_{padded}"
    prefix = f"site_{padded}"

    project_dir = PROJECTS_DIR / project_name
    input_dir = project_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Climate/cover data: source is already named site_XXXX_climate_cover_data.csv,
    # which is exactly the split-file name the reader expects.
    climate_src = CLIMATE_COVER_DIR / f"{prefix}_climate_cover_data.csv"
    if not climate_src.exists():
        raise FileNotFoundError(f"missing climate file for site {site_index}: {climate_src}")
    shutil.copyfile(climate_src, input_dir / climate_src.name)

    # soil-info.csv, built from this site's TestSites row. plot_name=0 matches the
    # constant used in the shared {}_plot_data.csv template. Columns per
    # read_soil_table() in data_sources/soil.py.
    #
    # TestSites_withSoilGridsQs.csv's clay/clay_q05/clay_q95 columns are raw
    # SoilGrids units (see append_soilgrids_quantiles.py's docstring), not a
    # 0-100 percentage. read_soil_table() applies no conversion (it expects an
    # already-correct percentage, unlike the live API path, which divides by
    # CLAY_UNIT_CONVERSION_FACTOR via convert_units_in_api_response() in
    # data_sources/soil.py) - so it must be applied here before writing.
    soil_info_path = input_dir / "soil-info.csv"
    with soil_info_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["plot_name", "Cy0", "clay", "Cy0_q05", "Cy0_q95", "clay_q05", "clay_q95"])
        writer.writerow([
            "0",
            site_row["ocs"], float(site_row["clay"]) / CLAY_UNIT_CONVERSION_FACTOR,
            site_row["ocs_q05"], site_row["ocs_q95"],
            float(site_row["clay_q05"]) / CLAY_UNIT_CONVERSION_FACTOR,
            float(site_row["clay_q95"]) / CLAY_UNIT_CONVERSION_FACTOR,
        ])

    # Species-lookup tables and the bounds file, copied verbatim so the project
    # folder is self-contained and reproducible.
    for name in COMMON_COPY_FILES:
        shutil.copyfile(COMMON_INPUTS_DIR / name, input_dir / name)
    shutil.copyfile(bounds_file, input_dir / "bounds_file.csv")

    # {}_*.csv templates: "{}" -> prefix (e.g. {}_plot_data.csv -> site_0007_plot_data.csv).
    for template in COMMON_INPUTS_DIR.glob(TEMPLATE_GLOB):
        suffix = template.name[len("{}"):]          # e.g. "_plot_data.csv"
        shutil.copyfile(template, input_dir / f"{prefix}{suffix}")

    return project_name, prefix, input_dir


def collect_outputs(project_name: str, results_dir: Path) -> bool:
    """Copy a site's outputs into results_dir. Returns True if all were found."""
    src = PROJECTS_DIR / project_name / "output" / "plot_1"
    wanted = ("morris_results.csv", "morris_design_X.npy", "morris_outputs_Y.npy")
    all_found = True
    for name in wanted:
        src_path = src / name
        if not src_path.exists():
            all_found = False
            continue
        stem, ext = name.rsplit(".", 1)
        shutil.copyfile(src_path, results_dir / f"{stem}_{project_name}.{ext}")
    return all_found


def collected_results_path(project_name: str, results_dir: Path) -> Path:
    return results_dir / f"morris_results_{project_name}.csv"


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def run_site(project_name: str, prefix: str, args: argparse.Namespace,
             log_path: Path) -> int:
    """Run run_morris_direct.py for one site. Returns the subprocess return code."""
    cmd = [
        sys.executable, str(RUN_MORRIS_DIRECT),
        "--project-name", project_name,
        "--prefix", prefix,
        "--n-proj-cohorts", str(args.n_proj_cohorts),
        "--n-base-cohorts", str(args.n_base_cohorts),
        "--n-trajectories", str(args.n_trajectories),
        "--num-levels", str(args.num_levels),
        "--seed", str(args.seed),
        "--bounds-file", str(Path(args.bounds_file).resolve()),
        "--no-soil-api", "--no-climate-api",
    ]
    if args.dry_run:
        print("    would run:", " ".join(cmd))
        return 0

    # Capture each site's stdout+stderr to its own log so failures are diagnosable
    # without drowning the batch's progress output.
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log,
                              stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def report_aggregation(results_dir: Path) -> None:
    """Build the combined table and print where it went (or that there was nothing)."""
    out_path = aggregate_results(results_dir)
    if out_path is None:
        print(f"No per-site result files found in {results_dir} to aggregate.")
    else:
        print(f"Combined results written to {out_path}")


def main() -> None:
    args = parse_args()

    if args.aggregate_only:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            raise FileNotFoundError(f"results dir not found: {results_dir}")
        report_aggregation(results_dir)
        return

    testsites_path = Path(args.testsites)
    bounds_path = Path(args.bounds_file)
    for label, path in (("TestSites", testsites_path), ("bounds", bounds_path)):
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")

    sites = read_testsites(testsites_path)
    end = args.end if args.end is not None else len(sites) - 1
    if not (0 <= args.start <= end < len(sites)):
        raise ValueError(
            f"site range {args.start}..{end} is out of bounds for "
            f"{len(sites)} sites (valid indices 0..{len(sites) - 1})"
        )
    indices = range(args.start, end + 1)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_files_dir = results_dir / "log_files"
    log_files_dir.mkdir(exist_ok=True)
    batch_log_path = results_dir / "batch_log.csv"
    write_header = not batch_log_path.exists()
    batch_log = batch_log_path.open("a", newline="")
    log_writer = csv.writer(batch_log)
    if write_header:
        log_writer.writerow(["site_index", "project_name", "status", "returncode", "seconds"])

    total = len(indices)
    print(f"Morris batch: {total} sites ({args.start}..{end}), serial, "
          f"N={args.n_trajectories} trajectories each.")
    print(f"  bounds file : {bounds_path}")
    print(f"  results dir : {results_dir}")
    print(f"  per-site log: {PROJECTS_DIR / 'morris_batch_XXXX' / 'morris_run.log'}")
    print()

    batch_start = time.monotonic()
    durations: list[float] = []
    n_done = n_failed = n_skipped = 0

    for site_index in indices:
        padded = f"{site_index:04d}"
        project_name = f"morris_batch_{padded}"

        if args.resume and collected_results_path(project_name, results_dir).exists():
            n_skipped += 1
            n_done += 1
            print(f"[{n_done}/{total}] {project_name} skipped (already done)")
            continue

        try:
            _, prefix, _ = stage_site(site_index, sites[site_index], bounds_path)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            n_done += 1
            n_failed += 1
            log_writer.writerow([site_index, project_name, "stage_error", "", ""])
            batch_log.flush()
            print(f"[{n_done}/{total}] {project_name} STAGE FAILED: {exc}")
            if args.stop_on_error:
                break
            continue

        log_path = PROJECTS_DIR / project_name / "morris_run.log"
        site_start = time.monotonic()
        returncode = run_site(project_name, prefix, args, log_path)
        elapsed = time.monotonic() - site_start

        # Copy the per-site log alongside the other collected outputs, so
        # they can be inspected, even if the site failed and the project 
        # folder is later deleted or overwritten.
        if not args.dry_run and log_path.exists():
            shutil.copyfile(log_path, log_files_dir / f"morris_run_{project_name}.log")

        outputs_ok = args.dry_run or collect_outputs(project_name, results_dir)
        ok = (returncode == 0) and outputs_ok
        status = "ok" if ok else ("bad_output" if returncode == 0 else "run_error")

        n_done += 1
        if not ok:
            n_failed += 1
        durations.append(elapsed)
        log_writer.writerow([site_index, project_name, status, returncode, f"{elapsed:.1f}"])
        batch_log.flush()

        mean = sum(durations) / len(durations)
        remaining = total - n_done
        eta = fmt_duration(mean * remaining)
        batch_elapsed = fmt_duration(time.monotonic() - batch_start)
        flag = "" if ok else f"  <-- {status} (see {log_path})"
        print(f"[{n_done}/{total}] {project_name} {status} {elapsed:.1f}s | "
              f"mean {mean:.1f}s | failed {n_failed} | "
              f"elapsed {batch_elapsed} | eta {eta}{flag}")

    batch_log.close()
    total_elapsed = fmt_duration(time.monotonic() - batch_start)
    print()
    print(f"Done. {n_done} processed ({n_failed} failed, {n_skipped} skipped) "
          f"in {total_elapsed}.")
    print(f"Per-site log rows: {batch_log_path}")

    # Aggregate before the failure exit so a partially-failed batch still yields
    # a combined table over the sites that did succeed.
    if not args.no_aggregate and not args.dry_run:
        report_aggregation(results_dir)

    if n_failed:
        print("Some sites failed; check their morris_run.log and the batch_log.csv status column.")
        sys.exit(1)


if __name__ == "__main__":
    main()
