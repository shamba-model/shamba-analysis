"""Tests for model/morris/parameter_registry.py's recognised-parameter
vocabulary and bounds-file validation.
"""

import pytest

from model.morris.parameter_registry import (
    MorrisSpeciesContext,
    unrecognised_reason,
    validate_bounds_parameter_names,
)


def _species_ctx():
    return MorrisSpeciesContext(
        tree_species_data={1: {"species": 1}},
        crop_species_data={2: {"species": "maize"}},
        pool_species_data={1: {"turnover": None, "alloc": None}},
    )


@pytest.mark.parametrize("name", [
    "cy0", "temp_delta",                            # flat scalar
    "roth_c_temp_a1",                                # RothC field
    "tree_wood_dens_sp1", "tree_nitrogen_sp1",       # tree scalar / whole-vector
    "tree_nitrogen_leaf_sp1",                        # tree per-element
    "crop_slope_sp2",                                # crop scalar
    "pool_turnover_sp1", "pool_turnover_stem_sp1",   # pool whole-vector / element
    "pool_alloc_leaf_sp1",                           # pool alloc, allowed pool
])
def test_accepts_valid_names_across_all_families(name):
    assert unrecognised_reason(name, _species_ctx()) is None


@pytest.mark.parametrize("name, expected_in_reason", [
    ("not_a_real_parameter", None),
    ("tree_biomass_scale", None),                       # replaced by stand_density_scale
    ("base_sf_n1", None),                                # replaced by sf_n_scale
    ("base_cover_scale", None),                          # excluded — cover is an exact on/off flag, not scalable
    ("tree_wood_dens_sp99", "99"),                       # species not present
    ("pool_alloc_branch_sp1", "derived from 'stem'"),    # branch alloc not independent
    ("pool_thinning_fraction_sp1", "_scale"),            # redirected to the flat scale name
])
def test_rejects_invalid_names_with_a_plain_language_reason(name, expected_in_reason):
    reason = unrecognised_reason(name, _species_ctx())
    assert reason is not None
    if expected_in_reason:
        assert expected_in_reason in reason


def test_validate_accepts_a_well_formed_bounds_dict():
    overrides = {"cy0": (30.0, 70.0), "tree_wood_dens_sp1": (0.3, 0.7)}
    validate_bounds_parameter_names(overrides, _species_ctx())  # must not raise


def test_validate_rejects_whole_vector_and_element_collision():
    overrides = {"tree_nitrogen_sp1": (0.01, 0.03), "tree_nitrogen_leaf_sp1": (0.01, 0.02)}
    with pytest.raises(ValueError, match="both perturb tree_nitrogen"):
        validate_bounds_parameter_names(overrides, _species_ctx())


def test_validate_rejects_min_greater_than_or_equal_max():
    with pytest.raises(ValueError, match="must be less than"):
        validate_bounds_parameter_names({"cy0": (70.0, 50.0)}, _species_ctx())


def test_validate_collect_then_raise_reports_all_errors_together():
    """Three independent problems in one bounds dict produce one ValueError
    whose message contains all three reasons — not just the first."""
    overrides = {
        "not_a_real_parameter": (0.0, 1.0),
        "tree_wood_dens_sp99": (0.3, 0.7),
        "cy0": (70.0, 50.0),  # min >= max
    }
    with pytest.raises(ValueError) as exc_info:
        validate_bounds_parameter_names(overrides, _species_ctx())

    message = str(exc_info.value)
    assert "not_a_real_parameter" in message
    assert "99" in message
    assert "cy0" in message
