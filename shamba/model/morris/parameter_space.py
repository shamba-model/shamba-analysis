import re
import csv
import numpy as np
from typing import Dict, List, NamedTuple, Tuple

from model.soil_params import SoilParamsData
from model.climate import ClimateData
from model.emit import EmissionFactors
from model.monte_carlo.model_parameter_distributions import MODEL_PARAMETER_DISTRIBUTIONS
from model.common.constants import (
    ef_burn_default,
    ef_N_inputs_default,
    combustion_factor_default,
    volatile_frac_organic_fertiliser_default,
    volatile_frac_synthetic_fertiliser_default,
)


class MorrisParameter(NamedTuple):
    name: str
    bounds: Tuple[float, float]


def build_salib_problem(
    param_names: List[str],
    bounds: List[Tuple[float, float]],
) -> dict:
    return {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": list(bounds),
    }


def _ef_bounds(base_value: float, spec) -> Tuple[float, float]:
    """Compute ±2σ bounds for an emission factor using its distribution spread."""
    lo = max(base_value * (1.0 - 2.0 * spec.spread_lower), 0.0)
    hi = base_value * (1.0 + 2.0 * spec.spread_upper)
    return (lo, hi)


def default_bounds(
    base_input: dict,
    soil_params: SoilParamsData,
) -> Dict[str, Tuple[float, float]]:
    """Compute default parameter bounds for Morris screening.

    Returns a dict mapping parameter name → (min, max).
    """
    bounds: Dict[str, Tuple[float, float]] = {}

    # --- Soil ---
    # ASSUMPTION: if the quantile bounds are valid, use them; otherwise, 
    # use ±10 around the mean (clamped to non-negative).
    if soil_params.Cy0_q05 < soil_params.Cy0_q95:
        bounds["cy0"] = (soil_params.Cy0_q05, soil_params.Cy0_q95)
    else:
        bounds["cy0"] = (max(soil_params.Cy0 - 10.0, 0.0), soil_params.Cy0 + 10.0)
    
    # ASSUMPTION: if the quantile bounds are valid, use them; otherwise, 
    # use a default range of 5–70%.
    if soil_params.clay_q05 < soil_params.clay_q95:
        bounds["clay"] = (soil_params.clay_q05, soil_params.clay_q95)
    else:
        bounds["clay"] = (5.0, 70.0)

    # --- Climate ---
    # ASSUMPTION: use a default range for climate perturbations: 
    # ±2°C for temperature, ±50% for rain and evaporation.
    bounds["temp_delta"] = (-2.0, 2.0)
    bounds["rain_scale"] = (0.5, 1.5)
    bounds["evap_scale"] = (0.5, 1.5)

    # --- Emission factors (base ± 2 × relative spread) ---
    ef_specs = MODEL_PARAMETER_DISTRIBUTIONS
    bounds["ef_burn_crop_N2O"] = _ef_bounds(ef_burn_default["crop_N2O"], ef_specs["ef_burn_crop_N2O"])
    bounds["ef_burn_crop_CH4"] = _ef_bounds(ef_burn_default["crop_CH4"], ef_specs["ef_burn_crop_CH4"])
    bounds["ef_burn_tree_N2O"] = _ef_bounds(ef_burn_default["tree_N2O"], ef_specs["ef_burn_tree_N2O"])
    bounds["ef_burn_tree_CH4"] = _ef_bounds(ef_burn_default["tree_CH4"], ef_specs["ef_burn_tree_CH4"])
    bounds["ef_N_inputs"] = _ef_bounds(ef_N_inputs_default, ef_specs["ef_N_inputs"])
    bounds["combustion_factor_crop"] = _ef_bounds(combustion_factor_default["crop"], ef_specs["combustion_factor_crop"])
    bounds["combustion_factor_tree"] = _ef_bounds(combustion_factor_default["tree"], ef_specs["combustion_factor_tree"])
    bounds["volatile_frac_organic_fertiliser"] = (
        max(volatile_frac_organic_fertiliser_default * (1.0 - 2.0 * ef_specs["volatile_frac_organic_fertiliser"].spread_lower), 0.0),
        volatile_frac_organic_fertiliser_default * (1.0 + 2.0 * ef_specs["volatile_frac_organic_fertiliser"].spread_upper),
    )
    bounds["volatile_frac_synthetic_fertiliser"] = (
        max(volatile_frac_synthetic_fertiliser_default * (1.0 - 2.0 * ef_specs["volatile_frac_synthetic_fertiliser"].spread_lower), 0.0),
        volatile_frac_synthetic_fertiliser_default * (1.0 + 2.0 * ef_specs["volatile_frac_synthetic_fertiliser"].spread_upper),
    )

    # --- Management: direct scalars (N fraction of synthetic fertiliser) ---
    for key in ("base_sf_n1", "proj_sf_n1"):
        if key in base_input:
            v = float(np.atleast_1d(base_input[key])[0])
            # ASSUMPTION: use ±30% around the base value, clamped to [0, 1]; 
            # if that range is invalid, use ±0.05.
            lo = max(v * 0.7, 0.0)
            hi = min(v * 1.3, 1.0)
            if lo >= hi:
                lo = max(v - 0.05, 0.0)
                hi = min(v + 0.05, 1.0)
            bounds[key] = (lo, hi)

    # --- Management: multiplicative scales ---
    for key in (
        "base_sf_qty_scale", "proj_sf_qty_scale",
        "base_lit_qty_scale", "proj_lit_qty_scale",
        "base_thinning_scale", "proj_thinning_scale",
        "base_mortality_scale", "proj_mortality_scale",
        "tree_biomass_scale",
    ):
        bounds[key] = (0.5, 2.0)

    return bounds


def load_bounds_override(path: str) -> Dict[str, Tuple[float, float]]:
    """Load a Morris bounds override CSV.

    Expected format: three columns with header row `parameter,min,max`.
    Returns a dict mapping parameter name → (min, max).
    """
    overrides: Dict[str, Tuple[float, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["parameter"].strip()
            lo = float(row["min"])
            hi = float(row["max"])
            overrides[name] = (lo, hi)
    return overrides


def apply_design_row(
    x: np.ndarray,
    param_names: List[str],
    base_input: Dict,
    base_soil: SoilParamsData,
    base_climate: ClimateData,
    base_emission_factors: EmissionFactors,
) -> Tuple[Dict, SoilParamsData, ClimateData, EmissionFactors]:
    """Map one row of the SALib design matrix to SHAMBA typed inputs.

    Returns perturbed (input_dict, soil, climate, emission_factors).
    Soil Ceq and iom are recomputed from drawn Cy0, matching SoilParams.create() logic.
    Rain and evaporation scales are clamped to zero from below.
    Thinning and mortality proportions are clamped to [0, 1] after scaling.
    """
    vals = {param_names[i]: float(x[i]) for i in range(len(param_names))}

    input_dict = dict(base_input)

    # --- Soil ---
    cy0 = vals.get("cy0", base_soil.Cy0)
    clay = vals.get("clay", base_soil.clay)
    ceq = 1.25 * cy0
    iom = 0.049 * ceq ** 1.139
    # Quantile fields are MC metadata; set them equal to drawn values so the
    # RothC schema invariant (q05 <= mean <= q95) always holds for design points.
    soil = SoilParamsData(
        Cy0=cy0,
        clay=clay,
        depth=base_soil.depth,
        Ceq=ceq,
        iom=iom,
        Cy0_q05=cy0,
        Cy0_q95=cy0,
        clay_q05=clay,
        clay_q95=clay,
    )

    # --- Climate ---
    temp_delta = vals.get("temp_delta", 0.0)
    rain_scale = vals.get("rain_scale", 1.0)
    evap_scale = vals.get("evap_scale", 1.0)
    climate = ClimateData(
        temperature=base_climate.temperature + temp_delta,
        rain=np.clip(base_climate.rain * rain_scale, 0.0, None),
        evaporation=np.clip(base_climate.evaporation * evap_scale, 0.0, None),
        temperature_std=base_climate.temperature_std,
        rain_std=base_climate.rain_std,
        evaporation_std=base_climate.evaporation_std,
    )

    # --- Emission factors ---
    ef_burn = dict(base_emission_factors.ef_burn)
    combustion_factor = dict(base_emission_factors.combustion_factor)
    ef_N_inputs = base_emission_factors.ef_N_inputs
    vol_org = base_emission_factors.volatile_frac_organic_fertiliser
    vol_syn = base_emission_factors.volatile_frac_synthetic_fertiliser

    _ef_map = {
        "ef_burn_crop_N2O": ("ef_burn", "crop_N2O"),
        "ef_burn_crop_CH4": ("ef_burn", "crop_CH4"),
        "ef_burn_tree_N2O": ("ef_burn", "tree_N2O"),
        "ef_burn_tree_CH4": ("ef_burn", "tree_CH4"),
        "combustion_factor_crop": ("combustion_factor", "crop"),
        "combustion_factor_tree": ("combustion_factor", "tree"),
    }
    for name, v in vals.items():
        if name in _ef_map:
            container_name, sub_key = _ef_map[name]
            if container_name == "ef_burn":
                ef_burn[sub_key] = v
            else:
                combustion_factor[sub_key] = v

    if "ef_N_inputs" in vals:
        ef_N_inputs = vals["ef_N_inputs"]
    if "volatile_frac_organic_fertiliser" in vals:
        vol_org = vals["volatile_frac_organic_fertiliser"]
    if "volatile_frac_synthetic_fertiliser" in vals:
        vol_syn = vals["volatile_frac_synthetic_fertiliser"]

    emission_factors = EmissionFactors(
        ef_burn=ef_burn,
        ef_N_inputs=ef_N_inputs,
        combustion_factor=combustion_factor,
        volatile_frac_organic_fertiliser=vol_org,
        volatile_frac_synthetic_fertiliser=vol_syn,
    )

    # --- Management: direct scalar replacement (N fraction) ---
    for key in ("base_sf_n1", "proj_sf_n1"):
        if key in vals and key in input_dict:
            input_dict[key] = np.clip(vals[key], 0.0, 1.0)

    # --- Management: quantity scale multipliers ---
    for scale_key, data_key in (
        ("base_sf_qty_scale", "base_sf_qty1"),
        ("proj_sf_qty_scale", "proj_sf_qty1"),
        ("base_lit_qty_scale", "base_lit_qty1"),
        ("proj_lit_qty_scale", "proj_lit_qty1"),
    ):
        if scale_key in vals and data_key in input_dict:
            base_arr = np.asarray(base_input[data_key], dtype=float)
            input_dict[data_key] = np.clip(base_arr * vals[scale_key], 0.0, None)

    # --- Management: thinning scale multipliers ---
    for scale_key, key_pattern in (
        ("proj_thinning_scale", r"^thin_proj_cohort\d+$"),
        ("base_thinning_scale", r"^thin_base_cohort\d+$"),
    ):
        if scale_key in vals:
            s = vals[scale_key]
            for k in list(input_dict.keys()):
                if re.match(key_pattern, k):
                    base_arr = np.asarray(base_input[k], dtype=float)
                    input_dict[k] = np.clip(base_arr * s, 0.0, 1.0)

    # --- Management: mortality scale multipliers ---
    for scale_key, key_pattern in (
        ("proj_mortality_scale", r"^mort_proj_cohort\d+$"),
        ("base_mortality_scale", r"^mort_base_cohort\d+$"),
    ):
        if scale_key in vals:
            s = vals[scale_key]
            for k in list(input_dict.keys()):
                if re.match(key_pattern, k):
                    base_arr = np.asarray(base_input[k], dtype=float)
                    input_dict[k] = np.clip(base_arr * s, 0.0, 1.0)

    # --- Tree biomass: scale all planting densities (proxy for overall biomass uncertainty) ---
    if "tree_biomass_scale" in vals:
        s = vals["tree_biomass_scale"]
        for k in list(input_dict.keys()):
            if re.match(r"^(base|proj)_plant_dens\d+$", k):
                base_val = np.atleast_1d(np.asarray(base_input[k], dtype=float))
                input_dict[k] = np.clip(base_val * s, 0.0, None)

    return input_dict, soil, climate, emission_factors
