"""Shared site-loading logic for Morris-family sensitivity analysis entry points.

Loads a project's split-file inputs, resolves climate/soil, builds the Morris
parameter bounds dictionary, and prunes parameters that don't apply to this
site (no trees / no fertiliser / no fire / no matching cohort columns).

Used by both run_morris_direct.py (SALib trajectory sampling) and
run_oat_direct.py (one-at-a-time min/max sweep) so the two always agree on
which parameters are in scope for a given site.
"""

import os
import re
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from model.common import data_handler
from model.common.calculate_emissions import get_location
import model.climate as Climate
import model.soil_params as SoilParams
import model.soil_models.forward_soil_model as ForwardSoilModule
import model.soil_models.inverse_soil_model as InverseSoilModule
import model.tree_params as TreeParams
import model.crop_params as CropParams
import model.tree_model as TreeModel
from model.climate import ClimateData
from model.soil_params import SoilParamsData
from model.morris.parameter_space import default_bounds, load_bounds_override
from model.morris.parameter_registry import MorrisSpeciesContext, validate_bounds_parameter_names


class SiteContext(NamedTuple):
    vector_input_data: Dict
    soil_params: SoilParamsData
    climate: ClimateData
    species_ctx: MorrisSpeciesContext
    tree_species_data: Dict[int, Dict]
    crop_species_data: Dict[int, Dict]
    pool_species_data: Dict[int, Dict]
    bounds_dict: Dict[str, Tuple[float, float]]
    param_names: List[str]
    n_years: int
    ForwardSoilModel: object
    InverseSoilModel: object


def load_site_context(
    input_dir: str,
    prefix: str,
    bounds_file: Optional[str],
    use_climate_api: bool,
    use_soil_api: bool,
) -> SiteContext:
    """Load one site's split-file inputs and build its Morris parameter bounds.

    Callers are responsible for setting configuration.SAVE_DIR/INPUT_DIR/
    OUTPUT_DIR before calling this (input_dir is passed explicitly rather than
    read from configuration here, so this function has no global side effects).
    """
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

    # Load the three species-lookup tables once here, at the shared import/validate
    # step, rather than lazily inside calculate_emissions.py. This ensures a malformed
    # tree_params.csv/crop_params.csv/biomass_pool_params.csv is reported alongside the
    # other input errors above, instead of surfacing as a mid-calculation crash.
    tree_species_data = crop_species_data = pool_species_data = None
    try:
        tree_species_data = TreeParams.load_tree_species_data()
    except ValueError as e:
        validation_errors.append(str(e))
    try:
        crop_species_data = CropParams.load_crop_species_data()
    except ValueError as e:
        validation_errors.append(str(e))
    try:
        pool_species_data = TreeModel.load_biomass_pool_species_data()
    except ValueError as e:
        validation_errors.append(str(e))

    if validation_errors:
        raise ValueError("\n".join(validation_errors))

    species_ctx = MorrisSpeciesContext(
        tree_species_data=tree_species_data,
        crop_species_data=crop_species_data,
        pool_species_data=pool_species_data,
    )

    # --- Resolve climate and soil ---
    climate_vectors = None
    if "temp" in vector_input_data:
        climate_vectors = (
            vector_input_data["temp"],
            vector_input_data["rain"],
            vector_input_data["evap"],
        )

    location = get_location(vector_input_data)
    climate = Climate.from_location(
        location=location,
        use_climate_api=use_climate_api,
        climate_vectors=climate_vectors,
    )
    # from_location()/from_vectors() only ever see a plain (temp, rain, evap)
    # tuple, so they recompute std from that (always zero for a tiled 12-row
    # split file). If the site's climate_cover_data.csv carries its own real
    # temp_std/rain_std/evap_std, use those: they're already monthly
    # climatology (12 values, tiled to 12*n_years by the reader above),
    # so the first 12 elements are enough.
    if "temp_std" in vector_input_data:
        climate.temperature_std = np.asarray(vector_input_data["temp_std"][:12])
        climate.rain_std = np.asarray(vector_input_data["rain_std"][:12])
        climate.evaporation_std = np.asarray(vector_input_data["evap_std"][:12])
    plot_id = vector_input_data.get("plot_name", None)
    soil_params = SoilParams.get_soil_params(
        location=location,
        use_soil_api=use_soil_api,
        plot_id=plot_id,
        plot_index=0,
    )

    # --- Build parameter bounds ---
    bounds_dict = default_bounds(vector_input_data, soil_params, species_ctx)

    if bounds_file is not None:
        overrides = load_bounds_override(bounds_file)
        validate_bounds_parameter_names(overrides, species_ctx)
        bounds_dict.update(overrides)
    else:
        print(
            "WARNING: no bounds file given. Running with the illustrative default "
            "bounds only (soil, climate, emission-factor, and management scales) — "
            "RothC, tree, crop, and biomass-pool parameters are recognised but not "
            "included unless supplied via a bounds file. This default set is not a "
            "validated sensitivity configuration."
        )

    # The next few blocks remove parameters from bounds_dict if they are not relevant to the input data.

    # Stand density scales: skip if no trees are present (i.e. no plant_dens keys in the input).
    has_trees = any(
        k.startswith("proj_plant_dens") or k.startswith("base_plant_dens")
        for k in vector_input_data
        )
    if not has_trees:
        bounds_dict.pop("base_stand_density_scale", None)
        bounds_dict.pop("proj_stand_density_scale", None)

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
        "base_sf_n_delta": ("base_sf_qty1",),
        "proj_sf_n_delta": ("proj_sf_qty1",),
        "base_lit_qty_scale": ("base_lit_qty1",),
        "proj_lit_qty_scale": ("proj_lit_qty1",),
        # Emission factors below are shared across base/project (not split
        # like the params above), so they're only removed if neither side
        # ever applies the relevant input — see emit.py:fert_emit().
        "volatile_frac_organic_fertiliser": ("base_lit_qty1", "proj_lit_qty1"),
        "volatile_frac_synthetic_fertiliser": ("base_sf_qty1", "proj_sf_qty1"),
    }
    for param_key, data_keys in _qty_dependent.items():
        if not _any_nonzero(*data_keys):
            bounds_dict.pop(param_key, None)

    # Fire-related emission factors: remove only if fire can never occur.
    # base_fire_on/proj_fire_on/base_fire_off/proj_fire_off are "direct"
    # Morris parameters (see parameter_registry.py). apply_design_row()
    # replaces the fire_on/off arrays outright with the drawn value, regardless
    # of what the static mgmt CSV says. Crop residues can be
    # burned on-farm (fire_on) or off-farm (fire_off via burn_off); trees are
    # only burned on-farm — see emit.py:fire_emit().
    _fire_toggles = ("base_fire_on", "proj_fire_on", "base_fire_off", "proj_fire_off")
    _crop_fire_possible = _any_nonzero(
        "fire_on_base", "fire_on_proj", "fire_off_base", "fire_off_proj"
    ) or any(k in bounds_dict for k in _fire_toggles)
    if not _crop_fire_possible:
        bounds_dict.pop("ef_burn_crop_N2O_scale", None)
        bounds_dict.pop("ef_burn_crop_CH4_scale", None)
        bounds_dict.pop("combustion_factor_crop", None)
    _tree_fire_possible = _any_nonzero("fire_on_base", "fire_on_proj") or any(
        k in bounds_dict for k in ("base_fire_on", "proj_fire_on")
    )
    if not _tree_fire_possible:
        bounds_dict.pop("ef_burn_tree_N2O_scale", None)
        bounds_dict.pop("ef_burn_tree_CH4_scale", None)
        bounds_dict.pop("combustion_factor_tree", None)

    # Remove thinning/mortality parameters only if there's no matching cohort
    # column at all (i.e. no tree cohorts of that kind exist for this plot).
    # These are "delta" (thinning: base + x, clamped) or "direct" (mortality:
    # x replaces the value outright) Morris parameters. Unlike the "scale"
    # (base * x) parameters above, a zero baseline does NOT guarantee zero
    # effect so the only case where these truly have no effect is when the
    # cohort column they'd modify doesn't exist in the data at all.
    for param_key, pattern in (
        ("proj_thinning_delta", r"^thin_proj_cohort\d+$"),
        ("base_thinning_delta", r"^thin_base_cohort\d+$"),
        ("proj_mortality", r"^mort_proj_cohort\d+$"),
        ("base_mortality", r"^mort_base_cohort\d+$"),
    ):
        matching = [k for k in vector_input_data if re.match(pattern, k)]
        if not matching:
            bounds_dict.pop(param_key, None)

    param_names = list(bounds_dict.keys())

    return SiteContext(
        vector_input_data=vector_input_data,
        soil_params=soil_params,
        climate=climate,
        species_ctx=species_ctx,
        tree_species_data=tree_species_data,
        crop_species_data=crop_species_data,
        pool_species_data=pool_species_data,
        bounds_dict=bounds_dict,
        param_names=param_names,
        n_years=n_years,
        ForwardSoilModel=ForwardSoilModel,
        InverseSoilModel=InverseSoilModel,
    )
