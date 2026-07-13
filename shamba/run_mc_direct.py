"""
Direct Monte Carlo runner — bypasses the CLI for sensitivity analysis.

Usage (inside Docker):
  docker exec -it <container-id> bash
  cd ..
  poetry run python run_mc_direct.py --project-name <name> --prefix <PREFIX> --n-cohorts <N> [options]

Example:
  poetry run python run_mc_direct.py --project-name examples/UG_TS_2016 --prefix WL --n-cohorts 3 --n-samples 100

Input files are read from:  projects/<project-name>/input/
Output files are written to: projects/<project-name>/output/
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

from model.common import data_handler
from model.common.calculate_emissions import get_location
from model import configuration
from model.monte_carlo import distribution_handler
from model.monte_carlo.runner import (
    MCSummaries,
    run_monte_carlo,
    summarise_mc_results,
    write_mc_summary_csv,
    write_mc_metadata,
)
import model.climate as Climate
import model.soil_params as SoilParams
import model.soil_models.forward_soil_model as ForwardSoilModule
import model.soil_models.inverse_soil_model as InverseSoilModule
import model.common.constants as CONSTANTS
import model.tree_params as TreeParams
import model.crop_params as CropParams
import model.tree_model as TreeModel


def parse_args():
    p = argparse.ArgumentParser(description="Run SHAMBA Monte Carlo directly.")
    p.add_argument("--project-name", required=True,
                   help="Project folder under projects/, e.g. 'examples/UG_TS_2016'.")
    p.add_argument("--prefix", required=True,
                   help="File prefix, e.g. 'WL' for WL_plot_data.csv etc.")
    p.add_argument("--n-proj-cohorts", type=int, required=True,
                   help="Number of project tree cohorts.")
    p.add_argument("--n-base-cohorts", type=int, default=1,
                   help="Number of baseline tree cohorts (default: 1).")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-climate-api", action="store_true",
                   help="Fetch climate data from API instead of CSV.")
    p.add_argument("--use-soil-api", action="store_true",
                   help="Fetch soil data from API instead of CSV.")
    p.add_argument("--sample-emission-factors", action="store_true")
    p.add_argument("--distribution-file", default=None,
                   help="Path to distributions CSV. Defaults to <prefix>_distributions.csv in the input dir if it exists.")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="Write a checkpoint file every N samples. Checkpoint files are named checkpoint_<N>.csv and contain the summary of differences up to that point.")
    return p.parse_args()


def main():
    args = parse_args()

    # Mirror how the CLI resolves project paths via configuration
    configuration.SAVE_DIR = os.path.join(configuration.PROJECT_DIR, args.project_name)
    configuration.INPUT_DIR = os.path.join(configuration.SAVE_DIR, "input")
    configuration.OUTPUT_DIR = os.path.join(configuration.SAVE_DIR, "output")

    input_dir = configuration.INPUT_DIR
    prefix = args.prefix
    output_dir = Path(configuration.OUTPUT_DIR) / "plot_1" ## FIXME: use name from input file instead
    output_dir.mkdir(parents=True, exist_ok=True)

    def log_checkpoint(n_done: int, summary: MCSummaries) -> None:
        # Write a MC summary file at a checkpoint, useful for assessing convergence of MC results.
        print(f"{n_done} samples complete")
        write_mc_summary_csv(summary.diff, str(output_dir / f"checkpoint_{n_done}.csv"))

    # --- Load split-file inputs ---------------------------------------------
    scalar_input_data = data_handler.read_and_validate_timeseries_by_header(
        file_path=os.path.join(input_dir, f"{prefix}_plot_data.csv"),
        permitted_vector_lengths=[1],
        target_vector_length=1,
    )
    n_years = int(np.atleast_1d(scalar_input_data["yrs_proj"])[0])

    mgmt_input_data = data_handler.read_and_validate_timeseries_by_header(
        file_path=os.path.join(input_dir, f"{prefix}_mgmt_data.csv"),
        permitted_vector_lengths=[1, n_years, n_years + 1],
        target_vector_length=n_years,
    )
    tree_size_data = data_handler.read_and_validate_timeseries_by_header(
        file_path=os.path.join(input_dir, f"{prefix}_tree_size_data.csv"),
        permitted_vector_lengths=list(range(5, n_years + 1)),
        target_vector_length=None,
    )
    climate_cover_data = data_handler.read_and_validate_timeseries_by_header(
        file_path=os.path.join(input_dir, f"{prefix}_climate_cover_data.csv"),
        permitted_vector_lengths=[1] + [i * 12 for i in range(1, n_years + 1)],
        target_vector_length=12 * n_years,
    )
    if "temp" in climate_cover_data:
        climate_cover_data = data_handler.resolve_evap_pet(climate_cover_data)

    input_dict = scalar_input_data | mgmt_input_data | tree_size_data | climate_cover_data

    # --- Validate -----------------------------------------------------------
    errors = (
        data_handler.validate_all_grouped_headers(input_dict)
        + data_handler.validate_species_data(input_dict)
        + data_handler.validate_required_mgmt_keys(input_dict)
    )
    # Load the three species-lookup tables once here, at the shared import/validate
    # step, rather than lazily inside calculate_emissions.py. This ensures a malformed
    # tree_params.csv/crop_params.csv/biomass_pool_params.csv is reported alongside the
    # other input errors above, instead of surfacing as a mid-calculation crash.
    tree_species_data = crop_species_data = pool_species_data = None
    try:
        tree_species_data = TreeParams.load_tree_species_data()
    except ValueError as e:
        errors.append(str(e))
    try:
        crop_species_data = CropParams.load_crop_species_data()
    except ValueError as e:
        errors.append(str(e))
    try:
        pool_species_data = TreeModel.load_biomass_pool_species_data()
    except ValueError as e:
        errors.append(str(e))

    if errors:
        raise ValueError("\n".join(errors))

    # --- Soil model factories -----------------------------------------------
    ForwardSoilModel = ForwardSoilModule.get_soil_model(ForwardSoilModule.SoilModelType.ROTH_C)
    InverseSoilModel = InverseSoilModule.get_soil_model(InverseSoilModule.SoilModelType.ROTH_C)

    # --- Climate and soil ---------------------------------------------------
    location = get_location(input_dict)
    plot_id = input_dict.get("plot_name")

    climate_vectors = None
    if "temp" in input_dict:
        climate_vectors = (input_dict["temp"], input_dict["rain"], input_dict["evap"])

    climate = Climate.from_location(
        location=location,
        use_climate_api=args.use_climate_api,
        climate_vectors=climate_vectors,
    )
    soil_params = SoilParams.get_soil_params(
        location=location,
        use_soil_api=args.use_soil_api,
        plot_id=plot_id,
        plot_index=0,
    )

    # --- Distribution file (optional) ---------------------------------------
    dist_file = args.distribution_file
    if dist_file is None:
        default = os.path.join(input_dir, f"{prefix}_distributions.csv")
        if os.path.exists(default):
            dist_file = default

    distribution_dict = None
    if dist_file:
        distribution_dict = distribution_handler.load_distributions(dist_file, input_dict)

    # project_allometry.py in the input dir is loaded by tree_growth.py via
    # importlib.import_module — the directory must be on sys.path first.
    sys.path.insert(0, input_dir)

    # --- Run Monte Carlo ----------------------------------------------------
    mc_results = run_monte_carlo(
        base_input_dict=input_dict,
        soil_params=soil_params,
        climate=climate,
        n_samples=args.n_samples,
        tree_species_data=tree_species_data,
        crop_species_data=crop_species_data,
        pool_species_data=pool_species_data,
        create_forward_soil_model=ForwardSoilModel.create,
        create_inverse_soil_model=InverseSoilModel.create,
        n_proj_cohorts=args.n_proj_cohorts,
        n_base_cohorts=args.n_base_cohorts,
        plot_index=0,
        sample_emission_factors=args.sample_emission_factors,
        distribution_dict=distribution_dict,
        allometry=[CONSTANTS.DEFAULT_ALLOMORPHY] * (args.n_base_cohorts + args.n_proj_cohorts),
        gwp=CONSTANTS.GWP_list[CONSTANTS.DEFAULT_GWP],
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        on_checkpoint=log_checkpoint if args.checkpoint_every > 0 else None,
    )

    # --- Save outputs -------------------------------------------------------

    mc_summary = summarise_mc_results(mc_results)
    for scenario, label in [
        (mc_summary.base,    "baseline"),
        (mc_summary.project, "project"),
        (mc_summary.diff,    "diff"),
    ]:
        write_mc_summary_csv(scenario, str(output_dir / f"plot_1_mc_{label}.csv"))

    write_mc_metadata(
        output_path=str(output_dir / "mc_run_metadata.txt"),
        n_samples=args.n_samples,
        seed=args.seed,
        soil_params=soil_params,
        climate=climate,
        distribution_dict=distribution_dict,
        sample_emission_factors=args.sample_emission_factors,
    )

    emit_diffs = [r.emit_project_emissions - r.emit_base_emissions for r in mc_results]
    total_diffs = np.array([float(np.sum(d)) for d in emit_diffs])
    print(
        f"\nMonte Carlo complete: {len(mc_results)} samples\n"
        f"  Summaries written to: {output_dir}\n"
        f"  Total emission difference — mean: {total_diffs.mean():.4f} t CO2 ha^-1  "
        f"  std: {total_diffs.std():.4f} t CO2 ha^-1"
    )


if __name__ == "__main__":
    main()
