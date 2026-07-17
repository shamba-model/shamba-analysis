import re
import csv
import numpy as np
from typing import Dict, List, NamedTuple, Tuple

from model.soil_params import SoilParamsData
from model.climate import ClimateData
from model.emit import EmissionFactors
from model.soil_models.soil_model_params import SoilModelParams, RothCParams
from model.monte_carlo.model_parameter_distributions import MODEL_PARAMETER_DISTRIBUTIONS
from model.common.constants import (
    ef_burn_default,
    ef_N_inputs_default,
    combustion_factor_default,
    volatile_frac_organic_fertiliser_default,
    volatile_frac_synthetic_fertiliser_default,
    TREE_ROOT_IN_TOP_30,
    CROP_ROOT_IN_TOP_30,
)
from model.tree_params import TREE_SPECIES_DIST_KEY_PATTERN
from model.crop_params import CROP_SPECIES_DIST_KEY_PATTERN
from model.tree_model import BIOMASS_POOL_DIST_KEY_PATTERN
from model.morris.parameter_registry import (
    MorrisSpeciesContext,
    TREE_SPECIES_ELEMENT_DIST_KEY_PATTERN,
    BIOMASS_POOL_ELEMENT_DIST_KEY_PATTERN,
)

# Pool ordering shared with tree_model.py's biomass-pool arrays and
# parameter_registry.py's element patterns
_BIOMASS_POOLS = ("leaf", "branch", "stem", "croot", "froot")
_BIOMASS_POOL_INDEX = {pool: i for i, pool in enumerate(_BIOMASS_POOLS)}

# Biomass-pool fields with a per-species element form (thinning_fraction/
# mortality_fraction are perturbed via flat scale names instead 
# so they're deliberately excluded here).
_POOL_ELEMENT_FIELDS = frozenset({"turnover", "alloc"})

# thinning_fraction/mortality_fraction scale names: maps each field to the
# mgmt-input column prefix it perturbs, and each pool to its actual column
# token (br/st abbreviate branch/stem; other pools use their full name).
_FRACTION_MGMT_PREFIX = {"thinning_fraction": "thin", "mortality_fraction": "mort"}
_POOL_COLUMN_TOKEN = {"leaf": "leaf", "branch": "br", "stem": "st", "croot": "croot", "froot": "froot"}


class MorrisParameter(NamedTuple):
    name: str
    bounds: Tuple[float, float]


class MorrisDesignRowResult(NamedTuple):
    input_dict: Dict
    soil: SoilParamsData
    climate: ClimateData
    emission_factors: EmissionFactors
    soil_model_params: SoilModelParams
    tree_species_data: Dict[int, Dict]
    crop_species_data: Dict[int, Dict]
    pool_species_data: Dict[int, Dict]
    tree_root_in_top_30: float
    crop_root_in_top_30: float


def build_salib_problem(
    param_names: List[str],
    bounds: List[Tuple[float, float]],
) -> dict:
    return {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": list(bounds),
    }


_Z_95 = 1.96  # two-tailed 95% CI z-score — the convention for Morris spreads here.
_Z_90_HALF = 1.645  # z-score for the 5th/95th percentile, used to infer sigma from
                     # a Q05/Q95 pair — matches monte_carlo/sampler.py's sample_soil_params().


def _ef_bounds(base_value: float, spec) -> Tuple[float, float]:
    """Compute ±95% CI bounds for an emission factor using its distribution spread."""
    lo = max(base_value * (1.0 - _Z_95 * spec.spread_lower), 0.0)
    hi = base_value * (1.0 + _Z_95 * spec.spread_upper)
    return (lo, hi)


def default_bounds(
    base_input: dict,
    soil_params: SoilParamsData,
    species_ctx: MorrisSpeciesContext,
) -> Dict[str, Tuple[float, float]]:
    """Compute default parameter bounds for Morris screening.

    Returns a dict mapping parameter name → (min, max). --bounds-file
    overrides merge into this dict, so every key returned here is a real,
    functioning default that gets screened whenever --bounds-file doesn't
    override it.

    WARNING: these bounds are currently a mix of illustrative and 
    intended to be used. Tidy this up once Morris design final.
    """
    bounds: Dict[str, Tuple[float, float]] = {}

    # --- Soil ---
    # ASSUMPTION: if the quantile bounds are valid, infer sigma from the Q05/Q95
    # pair and use a 95% CI around the mean (matching the EF bounds' convention);
    # otherwise, use ±10 around the mean (clamped to non-negative).
    if soil_params.Cy0_q05 < soil_params.Cy0_q95:
        cy0_sigma = (soil_params.Cy0_q95 - soil_params.Cy0_q05) / (2.0 * _Z_90_HALF)
        bounds["cy0"] = (
            max(soil_params.Cy0 - _Z_95 * cy0_sigma, 0.0),
            soil_params.Cy0 + _Z_95 * cy0_sigma,
        )
    else:
        bounds["cy0"] = (max(soil_params.Cy0 - 10.0, 0.0), soil_params.Cy0 + 10.0)

    # ASSUMPTION: if the quantile bounds are valid, infer sigma from the Q05/Q95
    # pair and use a 95% CI around the mean (clamped to [0, 100]); otherwise, use
    # a default range of 5–70%.
    if soil_params.clay_q05 < soil_params.clay_q95:
        clay_sigma = (soil_params.clay_q95 - soil_params.clay_q05) / (2.0 * _Z_90_HALF)
        bounds["clay"] = (
            max(soil_params.clay - _Z_95 * clay_sigma, 0.0),
            min(soil_params.clay + _Z_95 * clay_sigma, 100.0),
        )
    else:
        bounds["clay"] = (5.0, 70.0)

    # --- Climate ---
    # ASSUMPTION: use a default range for climate perturbations:
    # ±2°C for temperature, ±50% for rain and evaporation.
    bounds["temp_delta"] = (-2.0, 2.0)
    bounds["rain_scale"] = (0.5, 1.5)
    bounds["evap_scale"] = (0.5, 1.5)

    # --- Emission factors (base ± 95% CI, from relative spread) ---
    ef_specs = MODEL_PARAMETER_DISTRIBUTIONS
    bounds["ef_burn_crop_N2O"] = _ef_bounds(ef_burn_default["crop_N2O"], ef_specs["ef_burn_crop_N2O"])
    bounds["ef_burn_crop_CH4"] = _ef_bounds(ef_burn_default["crop_CH4"], ef_specs["ef_burn_crop_CH4"])
    bounds["ef_burn_tree_N2O"] = _ef_bounds(ef_burn_default["tree_N2O"], ef_specs["ef_burn_tree_N2O"])
    bounds["ef_burn_tree_CH4"] = _ef_bounds(ef_burn_default["tree_CH4"], ef_specs["ef_burn_tree_CH4"])
    bounds["ef_N_inputs"] = _ef_bounds(ef_N_inputs_default, ef_specs["ef_N_inputs"])
    bounds["combustion_factor_crop"] = _ef_bounds(combustion_factor_default["crop"], ef_specs["combustion_factor_crop"])
    bounds["combustion_factor_tree"] = _ef_bounds(combustion_factor_default["tree"], ef_specs["combustion_factor_tree"])
    bounds["volatile_frac_organic_fertiliser"] = (
        max(volatile_frac_organic_fertiliser_default * (1.0 - _Z_95 * ef_specs["volatile_frac_organic_fertiliser"].spread_lower), 0.0),
        volatile_frac_organic_fertiliser_default * (1.0 + _Z_95 * ef_specs["volatile_frac_organic_fertiliser"].spread_upper),
    )
    bounds["volatile_frac_synthetic_fertiliser"] = (
        max(volatile_frac_synthetic_fertiliser_default * (1.0 - _Z_95 * ef_specs["volatile_frac_synthetic_fertiliser"].spread_lower), 0.0),
        volatile_frac_synthetic_fertiliser_default * (1.0 + _Z_95 * ef_specs["volatile_frac_synthetic_fertiliser"].spread_upper),
    )

    # --- Management: multiplicative scales ---
    for key in (
        "base_sf_n_scale", "proj_sf_n_scale",
        "base_sf_qty_scale", "proj_sf_qty_scale",
        "base_lit_qty_scale", "proj_lit_qty_scale",
        "base_thinning_scale", "proj_thinning_scale",
        "base_mortality_scale", "proj_mortality_scale",
        "base_stand_density_scale", "proj_stand_density_scale",
    ):
        bounds[key] = (0.5, 2.0)

    # --- Species/pool families: not auto-bounded, one example only ---
    # RothC/tree/crop/pool fields are recognised by apply_design_row() but have
    # no real default range here yet — real coverage of these families is still
    # TODO. This single example (±20% around the first available tree species'
    # wood_dens) exists only to show the naming convention when no --bounds-file
    # is given; it is not a placeholder for the rest of the family.
    if species_ctx.tree_species_data:
        sc = min(species_ctx.tree_species_data)
        wood_dens = species_ctx.tree_species_data[sc]["wood_dens"]
        bounds[f"tree_wood_dens_sp{sc}"] = (wood_dens * 0.8, wood_dens * 1.2)

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
    base_tree_species_data: Dict[int, Dict],
    base_crop_species_data: Dict[int, Dict],
    base_pool_species_data: Dict[int, Dict],
    base_soil_model_params: SoilModelParams = RothCParams(),
) -> MorrisDesignRowResult:
    """Map one row of the SALib design matrix to SHAMBA typed inputs.

    Soil Ceq and iom are recomputed from drawn Cy0, matching SoilParams.create() logic.
    Rain and evaporation scales are clamped to zero from below. Thinning and
    mortality proportions are clamped to [0, 1] after scaling.

    Species/pool/RothC data is shallow-copied per family, not mutating the
    base_* arguments in place, mirroring monte_carlo/sampler.py. 
    Bare parameter names (e.g. "cy0", "tree_wood_dens_sp2", "roth_c_temp_a1") 
    are substitutions: the drawn value becomes the new value of that quantity.
    "_scale"-suffixed names are multiplicative scale factors applied to every
    matching base value instead.
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

    # --- RothC ---
    roth_c_updates = {
        field: vals[f"roth_c_{field}"]
        for field in base_soil_model_params._fields
        if f"roth_c_{field}" in vals
    }
    soil_model_params = (
        base_soil_model_params._replace(**roth_c_updates) if roth_c_updates else base_soil_model_params
    )

    # --- Tree species: scalars + nitrogen (whole-vector and per-pool element) ---
    tree_species_data = {sc: dict(species) for sc, species in base_tree_species_data.items()}
    for name, v in vals.items():
        match = TREE_SPECIES_DIST_KEY_PATTERN.match(name)
        if match:
            field, sc = match.group(1), int(match.group(2))
            if field == "nitrogen":
                # Whole-vector: one drawn value sets every pool to the same level.
                tree_species_data[sc]["nitrogen"] = np.full(len(_BIOMASS_POOLS), v)
            else:
                tree_species_data[sc][field] = v
            continue
        match = TREE_SPECIES_ELEMENT_DIST_KEY_PATTERN.match(name)
        if match:
            pool, sc = match.group(1), int(match.group(2))
            nitrogen = tree_species_data[sc]["nitrogen"].copy()
            nitrogen[_BIOMASS_POOL_INDEX[pool]] = v
            tree_species_data[sc]["nitrogen"] = nitrogen

    # --- Crop species: scalars ---
    crop_species_data = {sc: dict(species) for sc, species in base_crop_species_data.items()}
    for name, v in vals.items():
        match = CROP_SPECIES_DIST_KEY_PATTERN.match(name)
        if match:
            field, sc = match.group(1), int(match.group(2))
            crop_species_data[sc][field] = v

    # --- Biomass pool species: turnover/alloc (whole-vector and per-pool element) ---
    pool_species_data = {sc: dict(species) for sc, species in base_pool_species_data.items()}
    for name, v in vals.items():
        match = BIOMASS_POOL_DIST_KEY_PATTERN.match(name)
        if match:
            field, sc = match.group(1), int(match.group(2))
            if field in _POOL_ELEMENT_FIELDS:
                pool_species_data[sc][field] = np.full(len(_BIOMASS_POOLS), v)
            continue
        match = BIOMASS_POOL_ELEMENT_DIST_KEY_PATTERN.match(name)
        if match:
            field, pool, sc = match.group(1), match.group(2), int(match.group(3))
            if field in _POOL_ELEMENT_FIELDS:
                arr = pool_species_data[sc][field].copy()
                arr[_BIOMASS_POOL_INDEX[pool]] = v
                pool_species_data[sc][field] = arr

    # --- Management: synthetic fertiliser N fraction scale (all cohort indices) ---
    # A multiplicative scale applied to every base_sf_n{i}/ proj_sf_n{i} key present.
    for scale_key, key_pattern in (
        ("base_sf_n_scale", r"^base_sf_n\d+$"),
        ("proj_sf_n_scale", r"^proj_sf_n\d+$"),
    ):
        if scale_key in vals:
            s = vals[scale_key]
            for k in list(input_dict.keys()):
                if re.match(key_pattern, k):
                    base_arr = np.asarray(base_input[k], dtype=float)
                    input_dict[k] = np.clip(base_arr * s, 0.0, 1.0)

    # --- Management: quantity scale multipliers (all cohort/event indices) ---
    for scale_key, key_pattern in (
        ("base_sf_qty_scale", r"^base_sf_qty\d+$"),
        ("proj_sf_qty_scale", r"^proj_sf_qty\d+$"),
        ("base_lit_qty_scale", r"^base_lit_qty\d+$"),
        ("proj_lit_qty_scale", r"^proj_lit_qty\d+$"),
    ):
        if scale_key in vals:
            s = vals[scale_key]
            for k in list(input_dict.keys()):
                if re.match(key_pattern, k):
                    base_arr = np.asarray(base_input[k], dtype=float)
                    input_dict[k] = np.clip(base_arr * s, 0.0, None)

    # --- Management: thinning/mortality REGIME scale multipliers ---
    # This is the per-cohort thinning/mortality regime.
    for scale_key, key_pattern in (
        ("proj_thinning_scale", r"^thin_proj_cohort\d+$"),
        ("base_thinning_scale", r"^thin_base_cohort\d+$"),
        ("proj_mortality_scale", r"^mort_proj_cohort\d+$"),
        ("base_mortality_scale", r"^mort_base_cohort\d+$"),
    ):
        if scale_key in vals:
            s = vals[scale_key]
            for k in list(input_dict.keys()):
                if re.match(key_pattern, k):
                    base_arr = np.asarray(base_input[k], dtype=float)
                    input_dict[k] = np.clip(base_arr * s, 0.0, 1.0)

    # --- Management: thinning/mortality-fraction-per-pool scale ---
    # A cohort's effective thinning/mortality pool fraction may come from
    # either the mgmt-input override column or the pool_species_data species
    # default, depending on the plot — so both sources are scaled together,
    # rather than perturbing only one and risking zero effect on plots that
    # read the other.
    for field, mgmt_prefix in _FRACTION_MGMT_PREFIX.items():
        for pool in _BIOMASS_POOLS:
            scale_key = f"{field}_{pool}_scale"
            if scale_key not in vals:
                continue
            s = vals[scale_key]
            token = _POOL_COLUMN_TOKEN[pool]
            key_pattern = rf"^{mgmt_prefix}_(base|proj)_{token}_cohort\d+$"
            for k in list(input_dict.keys()):
                if re.match(key_pattern, k):
                    base_arr = np.asarray(base_input[k], dtype=float)
                    input_dict[k] = np.clip(base_arr * s, 0.0, 1.0)

            pool_idx = _BIOMASS_POOL_INDEX[pool]
            for sc in pool_species_data:
                arr = pool_species_data[sc][field].copy()
                arr[pool_idx] = np.clip(arr[pool_idx] * s, 0.0, 1.0)
                pool_species_data[sc][field] = arr

    # --- Tree stand density: scale planting density, base/proj independently ---
    # Replaces the legacy combined tree_biomass_scale, which scaled both sides
    # together — each side now has its own multiplier.
    for scale_key, key_pattern in (
        ("base_stand_density_scale", r"^base_plant_dens\d+$"),
        ("proj_stand_density_scale", r"^proj_plant_dens\d+$"),
    ):
        if scale_key in vals:
            s = vals[scale_key]
            for k in list(input_dict.keys()):
                if re.match(key_pattern, k):
                    base_val = np.atleast_1d(np.asarray(base_input[k], dtype=float))
                    input_dict[k] = np.clip(base_val * s, 0.0, None)

    # --- Fire on/off scale ---
    # Confirmed meaningful: emit.fire_emit() uses the fire array as a genuine
    # multiplier on burnable biomass, not a boolean gate.
    for scale_key, data_key in (
        ("base_fire_on_scale", "fire_on_base"),
        ("proj_fire_on_scale", "fire_on_proj"),
        ("base_fire_off_scale", "fire_off_base"),
        ("proj_fire_off_scale", "fire_off_proj"),
    ):
        if scale_key in vals and data_key in input_dict:
            base_arr = np.asarray(base_input[data_key], dtype=float)
            input_dict[data_key] = np.clip(base_arr * vals[scale_key], 0.0, 1.0)

    # --- Crop yield/residue-left scale (all cohort indices) ---
    for scale_key, key_pattern, clip_hi in (
        ("crop_base_yield_scale", r"^crop_base_yd\d+$", None),
        ("crop_proj_yield_scale", r"^crop_proj_yd\d+$", None),
        ("crop_base_left_scale", r"^crop_base_left\d+$", 1.0),
        ("crop_proj_left_scale", r"^crop_proj_left\d+$", 1.0),
    ):
        if scale_key in vals:
            s = vals[scale_key]
            for k in list(input_dict.keys()):
                if re.match(key_pattern, k):
                    base_arr = np.asarray(base_input[k], dtype=float)
                    input_dict[k] = np.clip(base_arr * s, 0.0, clip_hi)

    # --- Tree/crop root-in-top-30 (global scalars, no base object to copy) ---
    tree_root_in_top_30 = vals.get("tree_root_in_top_30", TREE_ROOT_IN_TOP_30)
    crop_root_in_top_30 = vals.get("crop_root_in_top_30", CROP_ROOT_IN_TOP_30)

    return MorrisDesignRowResult(
        input_dict=input_dict,
        soil=soil,
        climate=climate,
        emission_factors=emission_factors,
        soil_model_params=soil_model_params,
        tree_species_data=tree_species_data,
        crop_species_data=crop_species_data,
        pool_species_data=pool_species_data,
        tree_root_in_top_30=tree_root_in_top_30,
        crop_root_in_top_30=crop_root_in_top_30,
    )
