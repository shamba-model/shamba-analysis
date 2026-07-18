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
# These two fields are perturbed by shifting every value wherever it's found.
# See the flat thinning_fraction_{pool}_delta/mortality_fraction_{pool}_delta
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
#
# Bound categories, per how apply_design_row() actually applies the drawn
# value x. Three categories:
#   direct — x becomes the new value outright, no reference to base
#   scale  — multiplier, centred on ~1 (base * x)
#   delta  — additive, centred on ~0 (base + x)
# A name's own suffix (_scale/_delta) always matches its real category here —
# the only exceptions are the species-indexed scale fields (see
# TREE_SPECIES_SCALE_FIELDS/CROP_SPECIES_SCALE_FIELDS below), which can't
# carry a suffix at all since their name is shared with Monte Carlo's own
# per-species vocabulary.
# ---------------------------------------------------------------------------
SCALAR_PARAMETER_NAMES: FrozenSet[str] = frozenset({ ##
    # Soil — direct
    "cy0", "clay",
    # Soil — direct. Ceq = cy0_to_ceq_multiplier * cy0 (default 1.25, the
    # same literal SoilParams.create()/monte_carlo/sampler.py use — see
    # apply_design_row(), which recomputes Ceq/iom from drawn cy0 already).
    # Morris-local only: does not touch soil_params.py or sampler.py.
    "cy0_to_ceq_multiplier",
    # Litter — direct. The carbon/nitrogen-content fractions applied to
    # every external litter addition (LitterModel.from_defaults()); does not
    # affect synthetic fertiliser, which hardcodes carbon=0 and sources
    # nitrogen from its own per-cohort mgmt-input vector instead.
    "litter_carbon", "litter_nitrogen",
    # Climate — delta. Dimensionless CI-fraction in [-1, 1]. At x, month m
    # moves by x * 1.96 * that month's own std — one axis per variable, but
    # the per-month magnitude still reflects real site data.
    "temp_ci_delta", "rain_ci_delta", "evap_ci_delta",
    # Emission factors — scale for the burn EFs (multiplier on the fixed
    # global constant, centred on 1); direct for the rest.
    "ef_burn_crop_N2O_scale", "ef_burn_crop_CH4_scale",
    "ef_burn_tree_N2O_scale", "ef_burn_tree_CH4_scale",
    "ef_N_inputs",
    "combustion_factor_crop", "combustion_factor_tree",
    "volatile_frac_organic_fertiliser", "volatile_frac_synthetic_fertiliser",
    # Management — scale (base * x, centred on 1)
    "base_sf_qty_scale", "proj_sf_qty_scale",
    "base_lit_qty_scale", "proj_lit_qty_scale",
    "base_stand_density_scale", "proj_stand_density_scale",
    # Management — delta (base + x, centred on 0). These are all [0,1]-clamped
    "base_sf_n_delta", "proj_sf_n_delta",
    "base_thinning_delta", "proj_thinning_delta",
    # Management — direct (x replaces every matching value outright)
    "base_mortality", "proj_mortality",
    "base_fire_on", "proj_fire_on",
    "base_fire_off", "proj_fire_off",
    # Soil cover — direct, applied as a fraction of the year covered:
    # RothC's cover_year == 1 test (roth_c.py's get_rmf()/get_acc_tsmd()) is
    # an exact-equality check against the integer 1, not a continuous
    # multiplier like fire's array (see emit.fire_emit()) — cover is a
    # per-month "crop present"/bare flag, not a scalable magnitude. So
    # apply_design_row() sets round(x * 12) of the 12 calendar months to
    # covered (still a real 0/1 each) rather than assigning the drawn value
    # itself to every month.
    "base_cover", "proj_cover",
    # Crop yield/residue-left — delta. Yield is an absolute per-site/per-crop
    # quantity (kg/ha), not a fraction, so this is additive in whatever units
    # the base yield is in rather than a percentage; residue-left is a [0,1]
    # fraction, additive for the same floor-at-zero reason as the pool
    # fractions below.
    "crop_base_yield_delta", "crop_proj_yield_delta",
    "crop_base_left_delta", "crop_proj_left_delta",
    # Thinning/mortality pool-allocation fractions: one delta per pool,
    # shared across base and proj. Shifts the effective value wherever it's
    # found — the plot's own mgmt-input override column if present, and/or
    # the pool_species_data species default — so it has an effect regardless
    # of which source a given cohort actually reads from, and (being
    # additive rather than multiplicative) can turn on a fraction that was
    # zero at baseline. See apply_design_row() and
    # _POOL_FIELDS_REDIRECTED_TO_SCALE above.
    "thinning_fraction_leaf_delta", "thinning_fraction_branch_delta",
    "thinning_fraction_stem_delta", "thinning_fraction_croot_delta",
    "thinning_fraction_froot_delta",
    "mortality_fraction_leaf_delta", "mortality_fraction_branch_delta",
    "mortality_fraction_stem_delta", "mortality_fraction_croot_delta",
    "mortality_fraction_froot_delta",
    # Global scalars with a real injection point — see parameter_space.py's
    # apply_design_row() and tree_model.py/crop_model.py's get_inputs().
    "tree_root_in_top_30", "crop_root_in_top_30",
})

# Species-indexed fields treated as "scale" (base * x, centred on 1) rather
# than the shared-vocabulary default of "direct". These names are NOT
# renamed with a "_scale" suffix — they're matched via
# TREE_SPECIES_DIST_KEY_PATTERN/CROP_SPECIES_DIST_KEY_PATTERN, imported
# directly from tree_params.py/crop_params.py and shared with Monte Carlo's
# own per-species vocabulary, so the name itself can't change without
# touching that shared field list. apply_design_row() special-cases these
# fields within the existing match block instead — the same approach already
# used there for "nitrogen" (whole-vector vs. per-pool).
TREE_SPECIES_SCALE_FIELDS: FrozenSet[str] = frozenset({"root_to_shoot"})
CROP_SPECIES_SCALE_FIELDS: FrozenSet[str] = frozenset({
    "slope", "root_to_shoot", "nitrogen_above", "nitrogen_below",
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
        # to the flat delta parameters instead (see
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
                f"Use '{field}_{{pool}}_delta' instead (e.g. '{field}_leaf_delta'), "
                f"which shifts the effective value wherever it's actually sourced "
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
                f"Use '{field}_{pool}_delta' instead, which shifts the effective "
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
