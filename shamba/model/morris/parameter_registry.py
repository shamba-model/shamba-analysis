"""Recognised-parameter vocabulary and validation for Morris bounds files.

Morris reuses Monte Carlo's existing per-species/RothC field-list constants
and regex patterns (imported below) as the shared vocabulary for valid names in 
an MC distributions.csv or a Morris bounds.csv. Per-element vector
granularity (one elementary-effect axis per pool, rather than MC's one
axis per whole vector) is currently Morris-only functionality, defined in this file, 
with no footprint on tree_params.py/tree_model.py/crop_params.py.
"""

import re
from typing import Dict, FrozenSet, NamedTuple, Optional, Tuple

from model.soil_models.soil_model_params import ROTH_C_DIST_KEYS
from model.tree_params import TREE_SPECIES_DIST_KEY_PATTERN
from model.crop_params import CROP_SPECIES_DIST_KEY_PATTERN
from model.tree_model import BIOMASS_POOL_DIST_KEY_PATTERN, BIOMASS_POOL_PARAM_FIELDS


class MorrisSpeciesContext(NamedTuple):
    """Bundles the three species-lookup tables a Morris run needs, so callers
    pass one object instead of three."""
    tree_species_data: Dict[int, Dict]
    crop_species_data: Dict[int, Dict]
    pool_species_data: Dict[int, Dict]


# ---------------------------------------------------------------------------
# Per-element vector-field patterns — Morris only. 
# Morris' apply_design_row() needs one elementary-effect axis per value.
# MC's sampler perturbs a whole vector from a single distribution. 
# This means the two approaches are inconsistent in some places where values 
# are defined per pool- e.g. biomass allocation, turnover and N content.
# For Morris, pool tuples are duplicated here rather than imported from 
# tree_params.py/tree_model.py, so this per-element granularity has no 
# footprint on those files. Order must stay in sync with tree_params.csv's 
# N_leaf..N_froot columns and tree_model.py's biomass-pool ordering.
# ---------------------------------------------------------------------------
_TREE_NITROGEN_POOLS = ("leaf", "branch", "stem", "croot", "froot")
_BIOMASS_POOLS = ("leaf", "branch", "stem", "croot", "froot")

TREE_SPECIES_ELEMENT_DIST_KEY_PATTERN = re.compile(
    rf"^tree_nitrogen_({'|'.join(_TREE_NITROGEN_POOLS)})_sp(\d+)$"
)
BIOMASS_POOL_ELEMENT_DIST_KEY_PATTERN = re.compile(
    rf"^pool_({'|'.join(BIOMASS_POOL_PARAM_FIELDS)})_({'|'.join(_BIOMASS_POOLS)})_sp(\d+)$"
)

# Which pools each per-species biomass-pool field may be perturbed at,
# element-by-element. Only turnover/alloc are handled via pool_species_data.
# alloc[branch]/alloc[croot] are always recomputed from alloc[stem]
# (see tree_model.py:341-349, from_defaults()), so they're excluded.
#
# thinning_fraction/mortality_fraction are NOT handled via pool_species_data.
# A given cohort's effective value may come from pool_species_data (the species 
# default) or from the plot's own mgmt-input column, if present,
# whilst branch/stem are a required mgmt-input column.
# These two fields are perturbed by scaling every value wherever it's found.
# See the flat thinning_fraction_{pool}_scale/mortality_fraction_{pool}_scale 
# names in SCALAR_PARAMETER_NAMES below, applied in apply_design_row().
_POOL_FIELD_ALLOWED_ELEMENT_POOLS: Dict[str, FrozenSet[str]] = {
    "turnover": frozenset(_BIOMASS_POOLS),
    "alloc": frozenset({"leaf", "stem", "froot"}),
}

# thinning_fraction/mortality_fraction are recognised names (via
# BIOMASS_POOL_DIST_KEY_PATTERN/BIOMASS_POOL_ELEMENT_DIST_KEY_PATTERN, MC's
# own per-species vocabulary) but Morris explicitly redirects them to the
# flat scale parameters above rather than accepting them as per-species
# pool_species_data perturbations.
_POOL_FIELDS_REDIRECTED_TO_SCALE = frozenset({"thinning_fraction", "mortality_fraction"})


# ---------------------------------------------------------------------------
# Flat scalar parameter names — no species/pool indexing.
# ---------------------------------------------------------------------------
SCALAR_PARAMETER_NAMES: FrozenSet[str] = frozenset({ ##
    # Soil
    "cy0", "clay",
    # Climate: dimensionless CI-fraction in [-1, 1]. At x, month m moves by
    # x * 1.96 * that month's own std (see apply_design_row()) — one axis per
    # variable, but the per-month magnitude still reflects real site data.
    "temp_ci_scale", "rain_ci_scale", "evap_ci_scale",
    # Emission factors
    "ef_burn_crop_N2O", "ef_burn_crop_CH4",
    "ef_burn_tree_N2O", "ef_burn_tree_CH4",
    "ef_N_inputs",
    "combustion_factor_crop", "combustion_factor_tree",
    "volatile_frac_organic_fertiliser", "volatile_frac_synthetic_fertiliser",
    # Management: multiplicative scales
    "base_sf_n_scale", "proj_sf_n_scale",
    "base_sf_qty_scale", "proj_sf_qty_scale",
    "base_lit_qty_scale", "proj_lit_qty_scale",
    "base_thinning_scale", "proj_thinning_scale",
    "base_mortality_scale", "proj_mortality_scale",
    "base_stand_density_scale", "proj_stand_density_scale",
    "base_fire_on_scale", "proj_fire_on_scale",
    "base_fire_off_scale", "proj_fire_off_scale",
    "crop_base_yield_scale", "crop_proj_yield_scale",
    "crop_base_left_scale", "crop_proj_left_scale",
    # Thinning/mortality pool-allocation fractions: one scale per pool,
    # shared across base and proj. Scales the effective value wherever it's
    # found — the plot's own mgmt-input override column if present, and/or
    # the pool_species_data species default — so it has an effect regardless
    # of which source a given cohort actually reads from. See
    # apply_design_row() and _POOL_FIELDS_REDIRECTED_TO_SCALE above.
    "thinning_fraction_leaf_scale", "thinning_fraction_branch_scale",
    "thinning_fraction_stem_scale", "thinning_fraction_croot_scale",
    "thinning_fraction_froot_scale",
    "mortality_fraction_leaf_scale", "mortality_fraction_branch_scale",
    "mortality_fraction_stem_scale", "mortality_fraction_croot_scale",
    "mortality_fraction_froot_scale",
    # Global scalars with a real injection point — see parameter_space.py's
    # apply_design_row() and tree_model.py/crop_model.py's get_inputs().
    "tree_root_in_top_30", "crop_root_in_top_30",
})


def unrecognised_reason(
    name: str,
    species_ctx: MorrisSpeciesContext,
) -> Optional[str]:
    """Plain-language reason `name` is not a valid Morris bounds parameter,
    or None if it is valid.
    """
    if name in SCALAR_PARAMETER_NAMES or name in ROTH_C_DIST_KEYS:
        return None

    match = TREE_SPECIES_DIST_KEY_PATTERN.match(name)
    if match:
        sc = int(match.group(2))
        if sc not in species_ctx.tree_species_data:
            return (
                f"'{name}': species code {sc} not found in tree_params.csv "
                f"(available: {sorted(species_ctx.tree_species_data)})."
            )
        return None

    match = TREE_SPECIES_ELEMENT_DIST_KEY_PATTERN.match(name)
    if match:
        sc = int(match.group(2))
        if sc not in species_ctx.tree_species_data:
            return (
                f"'{name}': species code {sc} not found in tree_params.csv "
                f"(available: {sorted(species_ctx.tree_species_data)})."
            )
        return None

    match = CROP_SPECIES_DIST_KEY_PATTERN.match(name)
    if match:
        sc = int(match.group(2))
        if sc not in species_ctx.crop_species_data:
            return (
                f"'{name}': species code {sc} not found in crop_params.csv "
                f"(available: {sorted(species_ctx.crop_species_data)})."
            )
        return None

    match = BIOMASS_POOL_DIST_KEY_PATTERN.match(name)
    if match:
        # Whole-vector. thinning_fraction/mortality_fraction are redirected
        # to the flat scale parameters instead (see
        # _POOL_FIELDS_REDIRECTED_TO_SCALE); turnover/alloc are recognised
        # for every pool here (the branch/croot-derived-alloc caveat only
        # applies to the per-element pattern below, since a whole-vector
        # draw still perturbs the elements that do take effect).
        field, sc = match.group(1), int(match.group(2))
        if field in _POOL_FIELDS_REDIRECTED_TO_SCALE:
            return (
                f"'{name}': {field} can't be perturbed via species data alone — "
                f"a cohort's effective {field} may instead come from the plot's "
                f"own mgmt-input override column, which this name wouldn't touch. "
                f"Use '{field}_{{pool}}_scale' instead (e.g. '{field}_leaf_scale'), "
                f"which scales the effective value wherever it's actually sourced "
                f"from (mgmt-input override or species default)."
            )
        if sc not in species_ctx.pool_species_data:
            return (
                f"'{name}': species code {sc} not found in "
                f"biomass_pool_params.csv (available: {sorted(species_ctx.pool_species_data)})."
            )
        return None

    match = BIOMASS_POOL_ELEMENT_DIST_KEY_PATTERN.match(name)
    if match:
        field, pool, sc = match.group(1), match.group(2), int(match.group(3))
        if field in _POOL_FIELDS_REDIRECTED_TO_SCALE:
            return (
                f"'{name}': {field} can't be perturbed via species data alone — "
                f"a cohort's effective {field} may instead come from the plot's "
                f"own mgmt-input override column, which this name wouldn't touch. "
                f"Use '{field}_{pool}_scale' instead, which scales the effective "
                f"value wherever it's actually sourced from (mgmt-input override "
                f"or species default)."
            )
        if sc not in species_ctx.pool_species_data:
            return (
                f"'{name}': species code {sc} not found in "
                f"biomass_pool_params.csv (available: {sorted(species_ctx.pool_species_data)})."
            )
        if pool not in _POOL_FIELD_ALLOWED_ELEMENT_POOLS[field]:
            # Only alloc has a restricted set — turnover allows every pool.
            return (
                f"'{name}': '{pool}' alloc is derived from 'stem' "
                f"(tree_model.py's from_defaults()), not read independently "
                f"— it cannot be varied on its own."
            )
        return None

    return f"'{name}' is not a recognised Morris parameter."


def validate_bounds_parameter_names(
    overrides: Dict[str, Tuple[float, float]],
    species_ctx: MorrisSpeciesContext,
) -> None:
    """Validate a bounds-file override dict before any Morris sampling starts.

    Collects every error across all rows and raises once, rather than
    failing on the first bad row — matching
    distribution_handler.load_distributions()'s convention.
    """
    all_errors = []
    names = set(overrides)

    for name, bounds in overrides.items():
        reason = unrecognised_reason(name, species_ctx)
        if reason is not None:
            all_errors.append(reason)
            continue

        lo, hi = bounds
        if lo >= hi:
            all_errors.append(f"'{name}': min ({lo}) must be less than max ({hi}).")

    # Reject specifying both a whole-vector key and one of its own element
    # keys for the same species — ambiguous which should apply. Only
    # turnover/alloc have an element form (thinning_fraction/
    # mortality_fraction are redirected to flat scale names, with no
    # per-species element form to collide with).
    for name in names:
        match = TREE_SPECIES_DIST_KEY_PATTERN.match(name)
        if match and match.group(1) == "nitrogen":
            sc = match.group(2)
            element_names = sorted(
                n for n in names
                if (m := TREE_SPECIES_ELEMENT_DIST_KEY_PATTERN.match(n)) and m.group(2) == sc
            )
            if element_names:
                all_errors.append(
                    f"'{name}' (whole-vector) and {element_names} (per-pool) "
                    f"both perturb tree_nitrogen for species {sc} — specify one "
                    f"or the other, not both."
                )

        match = BIOMASS_POOL_DIST_KEY_PATTERN.match(name)
        if match and match.group(1) in _POOL_FIELD_ALLOWED_ELEMENT_POOLS:
            field, sc = match.group(1), match.group(2)
            element_names = sorted(
                n for n in names
                if (m := BIOMASS_POOL_ELEMENT_DIST_KEY_PATTERN.match(n))
                and m.group(1) == field and m.group(3) == sc
            )
            if element_names:
                all_errors.append(
                    f"'{name}' (whole-vector) and {element_names} (per-pool) "
                    f"both perturb pool_{field} for species {sc} — specify one "
                    f"or the other, not both."
                )

    if all_errors:
        raise ValueError("Errors in bounds file:\n" + "\n".join(f"  - {e}" for e in all_errors))
