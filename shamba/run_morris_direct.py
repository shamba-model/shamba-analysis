"""Morris sensitivity analysis entry point.

Loads split-file project inputs, generates a Morris trajectory design via SALib,
evaluates handle_intervention() for each design row, and writes sensitivity indices
to CSV. Follows the same data-loading pattern as shamba_command_line.py.

Usage (inside Docker):
    poetry run python run_morris_direct.py \\
        --project-name examples/UG_TS_2016 \\
        --prefix WL \\
        --n-proj-cohorts 3 \\
        --n-trajectories 10 \\
        --seed 42
"""

import argparse
import os
import sys
from pathlib import Path
import re

import numpy as np
from SALib.sample import morris as morris_sample

from model import configuration
from model.common import data_handler
from model.common.calculate_emissions import get_location
import model.climate as Climate
import model.soil_params as SoilParams
import model.soil_models.forward_soil_model as ForwardSoilModule
import model.soil_models.inverse_soil_model as InverseSoilModule
import model.common.constants as CONSTANTS
from model.emit import EmissionFactors
from model.morris.parameter_space import (
    default_bounds,
    build_salib_problem,
    load_bounds_override,
)
from model.morris.runner import (
    run_morris,
    compute_morris_indices,
    write_morris_results,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Morris sensitivity analysis for a SHAMBA project.")
    p.add_argument("--project-name", required=True,
                   help="Project directory name under projects/ (e.g. examples/UG_TS_2016)")
    p.add_argument("--prefix", required=True,
                   help="Split-file prefix (e.g. WL) used for _plot_data.csv etc.")
    p.add_argument("--n-proj-cohorts", type=int, required=True,
                   help="Number of project tree cohorts")
    p.add_argument("--n-base-cohorts", type=int, default=1,
                   help="Number of baseline tree cohorts (default: 1)")
    p.add_argument("--n-trajectories", type=int, default=50,
                   help="Number of Morris trajectories N (default: 50)")
    p.add_argument("--num-levels", type=int, default=4,
                   help="Number of grid levels for Morris sampling (default: 4)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--bounds-file", default=None,
                   help="Optional CSV with columns parameter,min,max to override default bounds")
    p.add_argument("--no-climate-api", action="store_true",
                   help="Skip the climate API and use local data only")
    p.add_argument("--no-soil-api", action="store_true",
                   help="Skip the soil API and use local data only")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Directory setup ---
    configuration.SAVE_DIR = os.path.join(configuration.PROJECT_DIR, args.project_name)
    configuration.INPUT_DIR = os.path.join(configuration.SAVE_DIR, "input")
    configuration.OUTPUT_DIR = os.path.join(configuration.SAVE_DIR, "output")

    input_dir = configuration.INPUT_DIR
    prefix = args.prefix

    ForwardSoilModel = ForwardSoilModule.get_soil_model(ForwardSoilModule.SoilModelType.ROTH_C)
    InverseSoilModel = InverseSoilModule.get_soil_model(InverseSoilModule.SoilModelType.ROTH_C)

    # --- Load split-file inputs ---
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

    vector_input_data = scalar_input_data | mgmt_input_data | tree_size_data | climate_cover_data

    validation_errors = (
        data_handler.validate_all_grouped_headers(vector_input_data)
        + data_handler.validate_species_data(vector_input_data)
        + data_handler.validate_required_mgmt_keys(vector_input_data)
    )
    if validation_errors:
        raise ValueError("\n".join(validation_errors))

    # --- Resolve climate and soil ---
    climate_vectors = None
    if "temp" in vector_input_data:
        climate_vectors = (
            vector_input_data["temp"],
            vector_input_data["rain"],
            vector_input_data["evap"],
        )
    use_climate_api = not args.no_climate_api
    use_soil_api = not args.no_soil_api

    location = get_location(vector_input_data)
    climate = Climate.from_location(
        location=location,
        use_climate_api=use_climate_api,
        climate_vectors=climate_vectors,
    )
    plot_id = vector_input_data.get("plot_name", None)
    soil_params = SoilParams.get_soil_params(
        location=location,
        use_soil_api=use_soil_api,
        plot_id=plot_id,
        plot_index=0,
    )

    # --- Build parameter bounds ---
    bounds_dict = default_bounds(vector_input_data, soil_params)

    if args.bounds_file is not None:
        overrides = load_bounds_override(args.bounds_file)
        bounds_dict.update(overrides)

    # The next few blocks remove parameters from bounds_dict if they are not relevant to the input data.

    # tree_biomass_scale: skip if no trees are present (i.e. no proj_plant_dens1 key in the input).
    has_trees = any(
        k.startswith("proj_plant_dens") or k.startswith("base_plant_dens")
        for k in vector_input_data
        )
    if not has_trees:
        bounds_dict.pop("tree_biomass_scale", None)

    # Filter out management parameters that have no effect because the
    # underlying quantity is zero everywhere (e.g. no fertiliser/litter
    # applied). These keys are always present per REQUIRED_MGMT_KEYS — a
    # missing key is a validation error, not a "not applicable" signal — so
    # the check is against the values, not key presence.
    def _any_nonzero(*data_keys):
        return any(
            np.any(np.asarray(vector_input_data[k], dtype=float) != 0.0)
            for k in data_keys
        )

    _qty_dependent = {
        "base_sf_qty_scale": ("base_sf_qty1",),
        "proj_sf_qty_scale": ("proj_sf_qty1",),
        "base_sf_n1": ("base_sf_qty1",),
        "proj_sf_n1": ("proj_sf_qty1",),
        "base_lit_qty_scale": ("base_lit_qty1",),
        "proj_lit_qty_scale": ("proj_lit_qty1",),
        # Emission factors below are shared across base/project (not split
        # like the scales above), so they're only dead if *neither* side
        # ever applies the relevant input — see emit.py:fert_emit().
        "volatile_frac_organic_fertiliser": ("base_lit_qty1", "proj_lit_qty1"),
        "volatile_frac_synthetic_fertiliser": ("base_sf_qty1", "proj_sf_qty1"),
    }
    for scale_key, data_keys in _qty_dependent.items():
        if not _any_nonzero(*data_keys):
            bounds_dict.pop(scale_key, None)

    # Fire-related emission factors: dead if fire never occurs. Crop residues
    # can be burned on-farm (fire_on) or off-farm (fire_off via burn_off);
    # trees are only burned on-farm — see emit.py:fire_emit().
    if not _any_nonzero("fire_on_base", "fire_on_proj", "fire_off_base", "fire_off_proj"):
        bounds_dict.pop("ef_burn_crop_N2O", None)
        bounds_dict.pop("ef_burn_crop_CH4", None)
        bounds_dict.pop("combustion_factor_crop", None)
    if not _any_nonzero("fire_on_base", "fire_on_proj"):
        bounds_dict.pop("ef_burn_tree_N2O", None)
        bounds_dict.pop("ef_burn_tree_CH4", None)
        bounds_dict.pop("combustion_factor_tree", None)

    # Remove thinning/mortality scales if no non-zero values are present.
    for scale_key, pattern in (
        ("proj_thinning_scale", r"^thin_proj_cohort\d+$"),
        ("base_thinning_scale", r"^thin_base_cohort\d+$"),
        ("proj_mortality_scale", r"^mort_proj_cohort\d+$"),
        ("base_mortality_scale", r"^mort_base_cohort\d+$"),
    ):
        matching = [k for k in vector_input_data if re.match(pattern, k)]
        has_nonzero = any(
            np.any(np.asarray(vector_input_data[k], dtype=float) != 0.0)
            for k in matching
        )
        if not has_nonzero:
            bounds_dict.pop(scale_key, None)

    param_names = list(bounds_dict.keys())
    bounds_list = [bounds_dict[n] for n in param_names]
    problem = build_salib_problem(param_names, bounds_list)

    print(f"Morris analysis: {len(param_names)} parameters, {args.n_trajectories} trajectories")
    print(f"  Parameters: {param_names}")

    # project_allometry.py in the input dir is loaded by tree_growth.py via
    # importlib.import_module — insert before spawning worker processes.
    sys.path.insert(0, input_dir)

    # --- Generate design matrix ---
    X = morris_sample.sample(
        problem,
        N=args.n_trajectories,
        num_levels=args.num_levels,
        seed=args.seed,
    )
    print(f"  Design matrix shape: {X.shape}")

    # --- Run Morris ---
    Y = run_morris(
        X=X,
        param_names=param_names,
        base_input=vector_input_data,
        base_soil=soil_params,
        base_climate=climate,
        create_forward_soil_model=ForwardSoilModel.create,
        create_inverse_soil_model=InverseSoilModel.create,
        n_proj_cohorts=args.n_proj_cohorts,
        n_base_cohorts=args.n_base_cohorts,
        plot_index=0,
        base_emission_factors=EmissionFactors(),
        allometry=[CONSTANTS.DEFAULT_ALLOMORPHY] * (args.n_base_cohorts + args.n_proj_cohorts),
        gwp=CONSTANTS.GWP_list[CONSTANTS.DEFAULT_GWP],
        on_progress=lambda done, total: print(f"  {done}/{total} runs complete"),
    )
    print(f"  Output Y shape: {Y.shape}")

    # --- Output directory ---
    out_dir = Path(configuration.OUTPUT_DIR) / "plot_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Save raw arrays ---
    np.save(str(out_dir / "morris_design_X.npy"), X)
    np.save(str(out_dir / "morris_outputs_Y.npy"), Y)
    print(f"  Raw arrays saved to {out_dir}")

    # --- Compute and write indices ---
    si_df = compute_morris_indices(problem, X, Y, seed=args.seed)
    results_path = str(out_dir / "morris_results.csv")
    write_morris_results(si_df, results_path)
    print(f"  Results written to {results_path}")

    # --- Print top-10 by mu_star (averaged over years) ---
    mean_mu_star = (
        si_df.groupby("parameter")["mu_star"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\nTop parameters by mean μ* across all years:")
    print(f"  {'Parameter':<40} {'Mean μ*':>12}")
    print("  " + "-" * 54)
    for name, val in mean_mu_star.head(10).items():
        print(f"  {name:<40} {val:>12.6f}")


if __name__ == "__main__":
    main()
