import concurrent.futures
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyze

import model.common.constants as CONSTANTS
from model.common.calculate_emissions import handle_intervention
from model.emit import EmissionFactors
from model.climate import ClimateData
from model.soil_params import SoilParamsData
from model.soil_models.soil_model_params import SoilModelParams, RothCParams
from model.morris.parameter_space import apply_design_row


class _MorrisSampleArgs(NamedTuple):
    x: np.ndarray
    param_names: List[str]
    base_input: Dict[str, Any]
    base_soil: SoilParamsData
    base_climate: ClimateData
    base_emission_factors: EmissionFactors
    base_soil_model_params: SoilModelParams
    tree_species_data: Dict[int, Dict]
    crop_species_data: Dict[int, Dict]
    pool_species_data: Dict[int, Dict]
    create_forward_soil_model: Callable
    create_inverse_soil_model: Callable
    n_proj_cohorts: int
    n_base_cohorts: int
    plot_index: int
    allometry: List[str]
    gwp: dict


def _run_single_morris(args: _MorrisSampleArgs) -> np.ndarray:
    design = apply_design_row(
        x=args.x,
        param_names=args.param_names,
        base_input=args.base_input,
        base_soil=args.base_soil,
        base_climate=args.base_climate,
        base_emission_factors=args.base_emission_factors,
        base_tree_species_data=args.tree_species_data,
        base_crop_species_data=args.crop_species_data,
        base_pool_species_data=args.pool_species_data,
        base_soil_model_params=args.base_soil_model_params,
    )
    result = handle_intervention(
        intervention_input=design.input_dict,
        climate=design.climate,
        soil=design.soil,
        create_forward_soil_model=args.create_forward_soil_model,
        create_inverse_soil_model=args.create_inverse_soil_model,
        n_proj_cohorts=args.n_proj_cohorts,
        n_base_cohorts=args.n_base_cohorts,
        plot_index=args.plot_index,
        tree_species_data=design.tree_species_data,
        crop_species_data=design.crop_species_data,
        pool_species_data=design.pool_species_data,
        allometry=args.allometry,
        gwp=args.gwp,
        emission_factors=design.emission_factors,
        soil_model_params=design.soil_model_params,
        tree_root_in_top_30=design.tree_root_in_top_30,
        crop_root_in_top_30=design.crop_root_in_top_30,
        litter_carbon=design.litter_carbon,
        litter_nitrogen=design.litter_nitrogen,
    )
    base_emissions = result.emit_base_emissions
    project_emissions = result.emit_project_emissions
    emissions_diff = project_emissions - base_emissions

    return emissions_diff                   # shape (N_YEARS,)


def run_morris(
    X: np.ndarray,
    param_names: List[str],
    base_input: Dict[str, Any],
    base_soil: SoilParamsData,
    base_climate: ClimateData,
    create_forward_soil_model: Callable,
    create_inverse_soil_model: Callable,
    n_proj_cohorts: int,
    n_base_cohorts: int,
    plot_index: int,
    tree_species_data: Dict[int, Dict],
    crop_species_data: Dict[int, Dict],
    pool_species_data: Dict[int, Dict],
    base_emission_factors: EmissionFactors = EmissionFactors(),
    base_soil_model_params: SoilModelParams = RothCParams(),
    allometry: Optional[List[str]] = None,
    gwp: dict = CONSTANTS.GWP_list[CONSTANTS.DEFAULT_GWP],
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Run handle_intervention() for each row in X.

    Returns Y array of shape (N*(k+1), N_YEARS) where Y[run, i] is the
    annual SOC increment (t C ha⁻¹ yr⁻¹) for year i+1 under the project scenario.
    """
    n_runs = X.shape[0]
    if allometry is None:
        allometry = [CONSTANTS.DEFAULT_ALLOMORPHY] * (n_base_cohorts + n_proj_cohorts)

    sample_args = [
        _MorrisSampleArgs(
            x=X[i],
            param_names=param_names,
            base_input=base_input,
            base_soil=base_soil,
            base_climate=base_climate,
            base_emission_factors=base_emission_factors,
            base_soil_model_params=base_soil_model_params,
            tree_species_data=tree_species_data,
            crop_species_data=crop_species_data,
            pool_species_data=pool_species_data,
            create_forward_soil_model=create_forward_soil_model,
            create_inverse_soil_model=create_inverse_soil_model,
            n_proj_cohorts=n_proj_cohorts,
            n_base_cohorts=n_base_cohorts,
            plot_index=plot_index,
            allometry=allometry,
            gwp=gwp,
        )
        for i in range(n_runs)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_run_single_morris, sample_args))

    if on_progress is not None:
        on_progress(n_runs, n_runs)

    return np.array(results)  # shape (n_runs, N_YEARS)


def compute_morris_indices(
    problem: dict,
    X: np.ndarray,
    Y: np.ndarray,
    num_resamples: int = 100,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Compute Morris indices for each output year, plus a total-difference summary.

    Runs SALib morris.analyze() once per year column of Y, then once more on
    Y summed over years (the total emissions difference across the whole
    project horizon, per run). Assembles a long-format DataFrame with one row
    per (parameter, year) combination, plus one row per parameter with
    year="total" for the summed-output analysis.

    Columns: parameter, year, mu, mu_star, sigma, mu_star_conf.
    """
    n_years = Y.shape[1]
    records = []

    for year_idx in range(n_years):
        si = morris_analyze.analyze(
            problem,
            X,
            Y[:, year_idx],
            num_resamples=num_resamples,
            seed=seed,
            print_to_console=False,
        )
        for j, name in enumerate(problem["names"]):
            records.append({
                "parameter": name,
                "year": year_idx + 1,
                "mu": float(si["mu"][j]),
                "mu_star": float(si["mu_star"][j]),
                "sigma": float(si["sigma"][j]),
                "mu_star_conf": float(si["mu_star_conf"][j]),
            })

    si_total = morris_analyze.analyze(
        problem,
        X,
        Y.sum(axis=1),
        num_resamples=num_resamples,
        seed=seed,
        print_to_console=False,
    )
    for j, name in enumerate(problem["names"]):
        records.append({
            "parameter": name,
            "year": "total",
            "mu": float(si_total["mu"][j]),
            "mu_star": float(si_total["mu_star"][j]),
            "sigma": float(si_total["sigma"][j]),
            "mu_star_conf": float(si_total["mu_star_conf"][j]),
        })

    return pd.DataFrame(records, columns=["parameter", "year", "mu", "mu_star", "sigma", "mu_star_conf"])


def write_morris_results(df: pd.DataFrame, output_path: str) -> None:
    """Write the long-format Morris results DataFrame to CSV."""
    df.to_csv(output_path, index=False)
