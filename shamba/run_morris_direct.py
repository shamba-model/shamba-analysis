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

import numpy as np
from SALib.sample import morris as morris_sample

from model import configuration
import model.common.constants as CONSTANTS
from model.emit import EmissionFactors
from model.soil_models.soil_model_params import RothCParams
from model.morris.parameter_space import build_salib_problem
from model.morris.site_loader import load_site_context
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

    ctx = load_site_context(
        input_dir=input_dir,
        prefix=args.prefix,
        bounds_file=args.bounds_file,
        use_climate_api=not args.no_climate_api,
        use_soil_api=not args.no_soil_api,
    )
    vector_input_data = ctx.vector_input_data
    soil_params = ctx.soil_params
    climate = ctx.climate
    tree_species_data = ctx.tree_species_data
    crop_species_data = ctx.crop_species_data
    pool_species_data = ctx.pool_species_data
    ForwardSoilModel = ctx.ForwardSoilModel
    InverseSoilModel = ctx.InverseSoilModel

    param_names = ctx.param_names
    bounds_list = [ctx.bounds_dict[n] for n in param_names]
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
        tree_species_data=tree_species_data,
        crop_species_data=crop_species_data,
        pool_species_data=pool_species_data,
        create_forward_soil_model=ForwardSoilModel.create,
        create_inverse_soil_model=InverseSoilModel.create,
        n_proj_cohorts=args.n_proj_cohorts,
        n_base_cohorts=args.n_base_cohorts,
        plot_index=0,
        base_emission_factors=EmissionFactors(),
        base_soil_model_params=RothCParams(),
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

    # --- Print top-10 by mu_star (averaged over years, excluding the total row) ---
    mean_mu_star = (
        si_df[si_df["year"] != "total"]
        .groupby("parameter")["mu_star"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\nTop parameters by mean μ* across all years:")
    print(f"  {'Parameter':<40} {'Mean μ*':>12}")
    print("  " + "-" * 54)
    for name, val in mean_mu_star.head(10).items():
        print(f"  {name:<40} {val:>12.6f}")

    # --- Print top-10 by mu_star for the total emissions difference ---
    total_mu_star = (
        si_df[si_df["year"] == "total"]
        .set_index("parameter")["mu_star"]
        .sort_values(ascending=False)
    )
    print("\nTop parameters by μ* for total emissions difference:")
    print(f"  {'Parameter':<40} {'Total μ*':>12}")
    print("  " + "-" * 54)
    for name, val in total_mu_star.head(10).items():
        print(f"  {name:<40} {val:>12.6f}")


if __name__ == "__main__":
    main()
