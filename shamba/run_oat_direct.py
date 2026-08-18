import argparse
import os
import sys
import numpy as np
import pandas as pd
from typing import List
from model import configuration
import model.common.constants as CONSTANTS
from model.morris.runner import _MorrisSampleArgs, run_oat
from model.emit import EmissionFactors
from model.morris.site_loader import SiteContext, load_site_context
from model.soil_models.soil_model_params import RothCParams

def build_oat_sample_args(
    ctx: SiteContext,
    n_proj_cohorts: int,
    n_base_cohorts: int,
    plot_index: int,
    allometry: List[str],
    gwp: dict,
) -> List[_MorrisSampleArgs]:
    """
    Build a list of _MorrisSampleArgs for the OAT analysis.

    Args:
        ctx (SiteContext): The site context containing site-specific information.
        n_proj_cohorts (int): Number of project cohorts.
        n_base_cohorts (int): Number of base cohorts.
        plot_index (int): ID of the plot.
        allometry (List[str]): List of allometric equations per cohort.
        gwp (dict): Global warming potential values.

    Returns:
        List[_MorrisSampleArgs]: A list of _MorrisSampleArgs for the OAT analysis.
    """

    run_list = []
    common_dict = {
        "base_input": ctx.vector_input_data,
        "base_soil": ctx.soil_params,
        "base_climate": ctx.climate,
        "tree_species_data": ctx.tree_species_data,
        "crop_species_data": ctx.crop_species_data,
        "pool_species_data": ctx.pool_species_data,
        "create_forward_soil_model": ctx.ForwardSoilModel.create,
        "create_inverse_soil_model": ctx.InverseSoilModel.create,
        "n_proj_cohorts": n_proj_cohorts,
        "n_base_cohorts": n_base_cohorts,
        "plot_index": plot_index,
        "base_emission_factors": EmissionFactors(),
        "base_soil_model_params": RothCParams(),
        "allometry": allometry,
        "gwp": gwp,
    }
    baseline_run = _MorrisSampleArgs(
        x=np.array([]),
        param_names=[],
        **common_dict
    )
    run_list.append(baseline_run)

    for param in ctx.param_names:
        param_names = [param]
        min_run = _MorrisSampleArgs(
            x=np.array([ctx.bounds_dict[param][0]]),
            param_names=param_names,
            **common_dict
        )
        max_run = _MorrisSampleArgs(
            x=np.array([ctx.bounds_dict[param][1]]),
            param_names=param_names,
            **common_dict
        )
        run_list.append(min_run)
        run_list.append(max_run)
    return run_list

def build_oat_results_df(Y: np.ndarray, param_list: List[str]) -> pd.DataFrame:

    # check that the number of rows in Y matches the expected number of runs
    expected_rows = 1 + 2 * len(param_list)
    if Y.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in Y, but got {Y.shape[0]}.")
    n_years = Y.shape[1]

    param_list = param_list + ["baseline"]
    records = []

    # Loop through each parameter and year to build the results DataFrame
    for param in param_list:
        if param == "baseline":
            direction_strings = ["baseline"]
        else:
            direction_strings = ["min", "max"]
        for j in range(len(direction_strings)):
            for i in range(0,n_years):
                row_index = 1 + 2 * param_list.index(param) + j if param != "baseline" else 0
                records.append({
                    "parameter": param,
                    "direction": direction_strings[j],
                    "year": i + 1,
                    "emissions_diff": float(Y[row_index, i]),
                    "effect_vs_baseline": float(Y[row_index, i] - Y[0, i]) if param != "baseline" else np.nan,
                })
            records.append({
                "parameter": param,
                "direction": direction_strings[j],
                "year": "total",
                "emissions_diff": float(np.sum(Y[row_index, :])),
                "effect_vs_baseline": float(np.sum(Y[row_index, :]) - np.sum(Y[0, :])) if param != "baseline" else np.nan,
            })

    return pd.DataFrame(records, columns = ["parameter", "direction", "year", "emissions_diff", "effect_vs_baseline" ])



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Morris sensitivity analysis for a SHAMBA project.")
    p.add_argument("--project-name", required=True,
                   help="Project directory name under projects/ (e.g. examples/UG_TS_2016)")
    p.add_argument("--prefix", required=True,
                   help="Split-file prefix (e.g. WL) used for _plot_data.csv etc.")
    p.add_argument("--n-proj-cohorts", type=int, required=True,
                   help="Number of project tree cohorts")
    p.add_argument("--n-base-cohorts", type=int, default=0,
                   help="Number of baseline tree cohorts (default: 0)")
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

    # project_allometry.py in the input dir is loaded by tree_growth.py via
    # importlib.import_module — insert before spawning worker processes.
    sys.path.insert(0, input_dir)

    run_list = build_oat_sample_args(
        ctx=ctx,
        n_proj_cohorts=args.n_proj_cohorts,
        n_base_cohorts=args.n_base_cohorts,
        plot_index=0,  # Assuming a single plot for OAT analysis
        allometry=[CONSTANTS.DEFAULT_ALLOMORPHY] * (args.n_proj_cohorts + args.n_base_cohorts),
        gwp=CONSTANTS.GWP_list[CONSTANTS.DEFAULT_GWP],
    )

    Y = run_oat(run_list)