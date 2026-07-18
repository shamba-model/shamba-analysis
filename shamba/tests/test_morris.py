"""Tests for the Morris sensitivity analysis module.

Covers parameter_space.py (apply_design_row, default_bounds) and
runner.py (run_morris shape, compute_morris_indices columns).
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, create_autospec, patch

from model.soil_params import SoilParamsData
from model.climate import ClimateData
from model.emit import EmissionFactors
from model.soil_models.soil_model_params import RothCParams
from model.morris.parameter_registry import MorrisSpeciesContext
from model.common.calculate_emissions import handle_intervention
from model.morris.parameter_space import (
    apply_design_row,
    default_bounds,
    build_salib_problem,
)
from model.morris.runner import compute_morris_indices, run_morris


# ---------------------------------------------------------------------------
# Helpers — minimal fake objects
# ---------------------------------------------------------------------------

def _make_soil(cy0=50.0, clay=30.0, cy0_q05=40.0, cy0_q95=65.0,
               clay_q05=20.0, clay_q95=45.0) -> SoilParamsData:
    ceq = 1.25 * cy0
    return SoilParamsData(
        Cy0=cy0, clay=clay, depth=30.0,
        Ceq=ceq, iom=0.049 * ceq ** 1.139,
        Cy0_q05=cy0_q05, Cy0_q95=cy0_q95,
        clay_q05=clay_q05, clay_q95=clay_q95,
    )


def _make_climate(temperature_std=None, rain_std=None, evaporation_std=None) -> ClimateData:
    return ClimateData(
        temperature=np.full(12, 20.0),
        rain=np.full(12, 80.0),
        evaporation=np.full(12, 50.0),
        temperature_std=np.zeros(12) if temperature_std is None else temperature_std,
        rain_std=np.zeros(12) if rain_std is None else rain_std,
        evaporation_std=np.zeros(12) if evaporation_std is None else evaporation_std,
    )


def _make_base_input(n_proj=1, n_base=1):
    d = {
        "base_sf_n1": np.array([0.46]),
        "proj_sf_n1": np.array([0.46]),
        "base_sf_qty1": np.full(20, 10.0),
        "proj_sf_qty1": np.full(20, 10.0),
        "base_lit_qty1": np.full(20, 5.0),
        "proj_lit_qty1": np.full(20, 5.0),
    }
    for i in range(1, n_proj + 1):
        d[f"proj_plant_dens{i}"] = np.array([100.0])
        d[f"thin_proj_cohort{i}"] = np.zeros(21)
        d[f"mort_proj_cohort{i}"] = np.full(21, 0.02)
    for i in range(1, n_base + 1):
        d[f"base_plant_dens{i}"] = np.array([80.0])
        d[f"thin_base_cohort{i}"] = np.zeros(21)
        d[f"mort_base_cohort{i}"] = np.full(21, 0.01)
    return d


def _make_tree_species_data():
    return {
        1: {
            "species": 1,
            "name": "TestTree",
            "wood_dens": 0.5,
            "carbon": 0.47,
            "root_to_shoot": 0.24,
            "nitrogen": np.array([0.020, 0.010, 0.005, 0.010, 0.015]),
        },
    }


def _make_crop_species_data():
    return {
        2: {
            "species": "TestCrop",
            "species_code": 2,
            "slope": 1.0,
            "intercept": 0.5,
            "nitrogen_below": 0.01,
            "nitrogen_above": 0.02,
            "carbon_below": 0.40,
            "carbon_above": 0.45,
            "root_to_shoot": 0.20,
        },
    }


def _make_pool_species_data():
    return {
        1: {
            "turnover": np.array([0.30, 0.05, 0.02, 0.10, 0.40]),
            "alloc": np.array([0.10, 0.45, 0.45, 0.05, 0.10]),
            "thinning_fraction": np.array([0.9, 0.9, 0.9, 0.9, 0.9]),
            "mortality_fraction": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        },
    }


def _make_species_ctx():
    return MorrisSpeciesContext(
        tree_species_data=_make_tree_species_data(),
        crop_species_data=_make_crop_species_data(),
        pool_species_data=_make_pool_species_data(),
    )


def _apply(x, param_names, base_input=None, base_soil=None, base_climate=None,
           base_emission_factors=None, base_tree_species_data=None,
           base_crop_species_data=None, base_pool_species_data=None,
           base_soil_model_params=None):
    """Thin wrapper so most tests don't repeat every mandatory species arg."""
    kwargs = dict(
        x=np.asarray(x),
        param_names=param_names,
        base_input=base_input if base_input is not None else {},
        base_soil=base_soil if base_soil is not None else _make_soil(),
        base_climate=base_climate if base_climate is not None else _make_climate(),
        base_emission_factors=base_emission_factors if base_emission_factors is not None else EmissionFactors(),
        base_tree_species_data=base_tree_species_data if base_tree_species_data is not None else {},
        base_crop_species_data=base_crop_species_data if base_crop_species_data is not None else {},
        base_pool_species_data=base_pool_species_data if base_pool_species_data is not None else {},
    )
    if base_soil_model_params is not None:
        kwargs["base_soil_model_params"] = base_soil_model_params
    return apply_design_row(**kwargs)


# ---------------------------------------------------------------------------
# Tests: apply_design_row
# ---------------------------------------------------------------------------

def test_apply_design_row_soil_recomputes_ceq_iom():
    """Drawing a cy0 different from base recomputes Ceq and iom correctly."""
    base_soil = _make_soil(cy0=50.0)
    drawn_cy0 = 70.0
    result = _apply(x=[drawn_cy0, base_soil.clay], param_names=["cy0", "clay"], base_soil=base_soil)

    expected_ceq = 1.25 * drawn_cy0
    assert result.soil.Cy0 == pytest.approx(drawn_cy0)
    assert result.soil.Ceq == pytest.approx(expected_ceq)
    assert result.soil.iom == pytest.approx(0.049 * expected_ceq ** 1.139)


def test_apply_design_row_climate_perturbations():
    """temp_ci_delta/rain_ci_delta shift each month by x * 1.96 * that month's
    own std — a site with zero std is an unaffected no-op, and rain is clamped
    to zero from below (never negative)."""
    z_95 = 1.96
    temp_std = np.linspace(0.5, 2.0, 12)
    base_climate = _make_climate(temperature_std=temp_std)

    result = _apply(x=[0.5], param_names=["temp_ci_delta"], base_climate=base_climate)
    np.testing.assert_allclose(result.climate.temperature, base_climate.temperature + 0.5 * z_95 * temp_std)
    np.testing.assert_array_equal(result.climate.rain, base_climate.rain)  # rain_ci_delta unset, std zero

    result = _apply(x=[-100.0], param_names=["rain_ci_delta"], base_climate=_make_climate(rain_std=np.full(12, 1.0)))
    assert np.all(result.climate.rain == 0.0)


def test_apply_design_row_roth_c_field_applies():
    """roth_c_temp_a1 overrides only that field; other RothCParams fields keep defaults."""
    base = RothCParams()
    result = _apply(x=[99.0], param_names=["roth_c_temp_a1"], base_soil_model_params=base)
    assert result.soil_model_params.temp_a1 == pytest.approx(99.0)
    assert result.soil_model_params.dpm_frac_crop == base.dpm_frac_crop


def test_apply_design_row_tree_species_scalar_applies():
    """tree_wood_dens_sp1 (direct) replaces wood_dens outright; tree_root_to_shoot_sp1
    (scale — special-cased despite sharing MC's shared field-name vocabulary,
    see TREE_SPECIES_SCALE_FIELDS) multiplies the base value. Base dict untouched."""
    base_tree = _make_tree_species_data()
    result = _apply(
        x=[0.8, 1.5],
        param_names=["tree_wood_dens_sp1", "tree_root_to_shoot_sp1"],
        base_tree_species_data=base_tree,
    )
    assert result.tree_species_data[1]["wood_dens"] == pytest.approx(0.8)
    assert result.tree_species_data[1]["root_to_shoot"] == pytest.approx(0.24 * 1.5)
    assert base_tree[1]["wood_dens"] == pytest.approx(0.5)  # base untouched
    assert base_tree[1]["root_to_shoot"] == pytest.approx(0.24)  # base untouched


def test_apply_design_row_tree_nitrogen_whole_vector_and_element():
    """tree_nitrogen_sp1 (whole-vector) sets every pool; tree_nitrogen_leaf_sp1
    (element) changes only the leaf pool. Base dict is never mutated."""
    base_tree = _make_tree_species_data()

    result = _apply(x=[0.03], param_names=["tree_nitrogen_sp1"], base_tree_species_data=base_tree)
    np.testing.assert_allclose(result.tree_species_data[1]["nitrogen"], np.full(5, 0.03))
    np.testing.assert_allclose(base_tree[1]["nitrogen"], [0.020, 0.010, 0.005, 0.010, 0.015])

    result = _apply(x=[0.09], param_names=["tree_nitrogen_leaf_sp1"], base_tree_species_data=base_tree)
    nitrogen = result.tree_species_data[1]["nitrogen"]
    assert nitrogen[0] == pytest.approx(0.09)
    np.testing.assert_allclose(nitrogen[1:], base_tree[1]["nitrogen"][1:])


def test_apply_design_row_crop_species_scalar_applies():
    """crop_slope_sp2 (scale — see CROP_SPECIES_SCALE_FIELDS) multiplies the
    base value; crop_intercept_sp2 (direct) replaces it outright. Base dict
    untouched either way."""
    base_crop = _make_crop_species_data()
    result = _apply(
        x=[2.5, 0.9],
        param_names=["crop_slope_sp2", "crop_intercept_sp2"],
        base_crop_species_data=base_crop,
    )
    assert result.crop_species_data[2]["slope"] == pytest.approx(1.0 * 2.5)
    assert result.crop_species_data[2]["intercept"] == pytest.approx(0.9)
    assert base_crop[2]["slope"] == pytest.approx(1.0)  # base untouched
    assert base_crop[2]["intercept"] == pytest.approx(0.5)  # base untouched


def test_apply_design_row_pool_turnover_whole_vector_and_element():
    """pool_turnover_sp1 (whole-vector) sets every pool; pool_turnover_stem_sp1
    (element) changes only the stem pool (index 2)."""
    base_pool = _make_pool_species_data()

    result = _apply(x=[0.25], param_names=["pool_turnover_sp1"], base_pool_species_data=base_pool)
    np.testing.assert_allclose(result.pool_species_data[1]["turnover"], np.full(5, 0.25))

    result = _apply(x=[0.5], param_names=["pool_turnover_stem_sp1"], base_pool_species_data=base_pool)
    turnover = result.pool_species_data[1]["turnover"]
    assert turnover[2] == pytest.approx(0.5)
    np.testing.assert_allclose(np.delete(turnover, 2), np.delete(base_pool[1]["turnover"], 2))


def test_apply_design_row_pool_alloc_leaf_and_stem_apply():
    """pool_alloc_leaf_sp1 and pool_alloc_stem_sp1 each change only their own
    element; branch/croot (derived from stem elsewhere) are untouched here."""
    base_pool = _make_pool_species_data()
    result = _apply(
        x=[0.2, 0.6],
        param_names=["pool_alloc_leaf_sp1", "pool_alloc_stem_sp1"],
        base_pool_species_data=base_pool,
    )
    alloc = result.pool_species_data[1]["alloc"]
    assert alloc[0] == pytest.approx(0.2)   # leaf
    assert alloc[2] == pytest.approx(0.6)   # stem
    assert alloc[1] == pytest.approx(base_pool[1]["alloc"][1])  # branch untouched


def test_apply_design_row_thinning_fraction_delta_applies_to_mgmt_column_and_pool_default():
    """thinning_fraction_{pool}_delta shifts both the mgmt-input override
    column (clamped to [0,1]) and the species pool_species_data default
    additively — not multiplicatively, so a zero baseline can still move;
    the branch column is targeted via its 'br' abbreviation."""
    base_input = {
        "thin_base_leaf_cohort1": np.array([0.3]),
        "thin_proj_br_cohort1": np.array([0.0]),
    }
    base_pool = _make_pool_species_data()  # thinning_fraction all 0.9

    result = _apply(
        x=[0.2, 0.5],
        param_names=["thinning_fraction_leaf_delta", "thinning_fraction_branch_delta"],
        base_input=base_input, base_pool_species_data=base_pool,
    )

    np.testing.assert_allclose(result.input_dict["thin_base_leaf_cohort1"], [0.5])
    np.testing.assert_allclose(result.input_dict["thin_proj_br_cohort1"], [0.5])  # 0.0 + 0.5, was stuck at 0 under a scale
    assert result.pool_species_data[1]["thinning_fraction"][0] == pytest.approx(1.0)  # 0.9+0.2 clamped


def test_apply_design_row_stand_density_scale_applies_per_side():
    """base_stand_density_scale and proj_stand_density_scale each only touch
    their own side's plant_dens keys."""
    base_input = _make_base_input(n_proj=2, n_base=2)

    result = _apply(x=[2.0], param_names=["base_stand_density_scale"], base_input=base_input)
    for key in ("base_plant_dens1", "base_plant_dens2"):
        expected = np.atleast_1d(np.asarray(base_input[key], dtype=float)) * 2.0
        np.testing.assert_allclose(result.input_dict[key], expected)
    for key in ("proj_plant_dens1", "proj_plant_dens2"):
        np.testing.assert_allclose(result.input_dict[key], base_input[key])


def test_apply_design_row_ef_burn_scale_multiplies_combustion_factor_direct():
    """ef_burn_crop_N2O_scale multiplies the base EF constant (centred on 1);
    combustion_factor_crop is a plain direct substitution."""
    base_ef = EmissionFactors()
    result = _apply(
        x=[1.1, 0.5],
        param_names=["ef_burn_crop_N2O_scale", "combustion_factor_crop"],
        base_emission_factors=base_ef,
    )
    assert result.emission_factors.ef_burn["crop_N2O"] == pytest.approx(base_ef.ef_burn["crop_N2O"] * 1.1)
    assert result.emission_factors.combustion_factor["crop"] == pytest.approx(0.5)


def test_apply_design_row_sf_n_delta_shifts_and_clamps():
    """base_sf_n_delta shifts every base_sf_n{i} key additively (not
    multiplicatively, so a zero baseline can still move) and clamps to [0, 1]."""
    base_input = {"base_sf_n1": np.array([0.1])}
    result = _apply(x=[0.95], param_names=["base_sf_n_delta"], base_input=base_input)
    np.testing.assert_allclose(result.input_dict["base_sf_n1"], [1.0])  # 0.1+0.95 clamped


def test_apply_design_row_thinning_regime_delta_mortality_regime_direct():
    """base_thinning_delta shifts thin_base_cohort{i} additively; base_mortality
    replaces mort_base_cohort{i} outright, every element, unrelated to base."""
    base_input = {
        "thin_base_cohort1": np.array([0.0, 0.1]),
        "mort_base_cohort1": np.array([0.02, 0.03]),
    }
    result = _apply(
        x=[0.3, 0.15],
        param_names=["base_thinning_delta", "base_mortality"],
        base_input=base_input,
    )
    np.testing.assert_allclose(result.input_dict["thin_base_cohort1"], [0.3, 0.4])
    np.testing.assert_allclose(result.input_dict["mort_base_cohort1"], [0.15, 0.15])


def test_apply_design_row_crop_yield_left_delta():
    """crop_base_yield_delta shifts yield additively in absolute units (no
    [0,1] clamp, just >=0); crop_base_left_delta shifts the [0,1] residue
    fraction additively."""
    base_input = {"crop_base_yd1": np.array([2000.0]), "crop_base_left1": np.array([0.2])}
    result = _apply(
        x=[-500.0, 0.9],
        param_names=["crop_base_yield_delta", "crop_base_left_delta"],
        base_input=base_input,
    )
    np.testing.assert_allclose(result.input_dict["crop_base_yd1"], [1500.0])
    np.testing.assert_allclose(result.input_dict["crop_base_left1"], [1.0])  # 0.2+0.9 clamped


def test_apply_design_row_fire_on_off_direct_replaces_array_and_clamps():
    """base_fire_on/base_fire_off replace the whole fire array outright (not
    derived from base at all) and clamp to [0, 1]."""
    base_input = {"fire_on_base": np.array([0.6, 0.6]), "fire_off_base": np.array([0.6, 0.6])}
    result = _apply(
        x=[1.5, 0.3],
        param_names=["base_fire_on", "base_fire_off"],
        base_input=base_input,
    )
    np.testing.assert_allclose(result.input_dict["fire_on_base"], [1.0, 1.0])  # 1.5 clamped, every element
    np.testing.assert_allclose(result.input_dict["fire_off_base"], [0.3, 0.3])


def test_apply_design_row_cy0_to_ceq_multiplier_overrides_default():
    """cy0_to_ceq_multiplier defaults to 1.25 (matching SoilParams.create())
    when not drawn, and reflects the drawn value when it is — Ceq/iom are
    still recomputed from it exactly as for the untouched default."""
    base_soil = _make_soil(cy0=50.0)

    default_result = _apply(x=[70.0], param_names=["cy0"], base_soil=base_soil)
    assert default_result.soil.Ceq == pytest.approx(1.25 * 70.0)

    result = _apply(
        x=[70.0, 1.5], param_names=["cy0", "cy0_to_ceq_multiplier"], base_soil=base_soil,
    )
    expected_ceq = 1.5 * 70.0
    assert result.soil.Ceq == pytest.approx(expected_ceq)
    assert result.soil.iom == pytest.approx(0.049 * expected_ceq ** 1.139)


def test_apply_design_row_litter_carbon_nitrogen_override_constants():
    """litter_carbon/litter_nitrogen default to CONSTANTS.ORGANIC_INPUT_C/N
    when not drawn, and reflect the drawn value when they are."""
    import model.common.constants as CONSTANTS

    default_result = _apply(x=[50.0], param_names=["cy0"])
    assert default_result.litter_carbon == CONSTANTS.ORGANIC_INPUT_C
    assert default_result.litter_nitrogen == CONSTANTS.ORGANIC_INPUT_N

    result = _apply(x=[0.4, 0.03], param_names=["litter_carbon", "litter_nitrogen"])
    assert result.litter_carbon == pytest.approx(0.4)
    assert result.litter_nitrogen == pytest.approx(0.03)


def test_apply_design_row_soil_cover_sets_fraction_of_months_covered():
    """base_cover/proj_cover set round(x * 12) of the 12 calendar months to
    covered (still an exact 0/1 each, as RothC's cover_year == 1 test
    requires), tiled across however many years the base array spans —
    rather than assigning the drawn value itself to every element."""
    base_input = {
        "base_cover": np.ones(24),   # 2 years, 12 months each
        "proj_cover": np.ones(24),
    }

    result = _apply(x=[0.25, 1.0], param_names=["base_cover", "proj_cover"], base_input=base_input)

    expected_base_year = np.array([1.0] * 3 + [0.0] * 9)  # round(0.25*12) = 3 months covered
    np.testing.assert_allclose(result.input_dict["base_cover"], np.tile(expected_base_year, 2))
    np.testing.assert_allclose(result.input_dict["proj_cover"], np.ones(24))  # x=1.0 -> all 12 months

    # x=0.0 -> no months covered
    result = _apply(x=[0.0], param_names=["base_cover"], base_input=base_input)
    np.testing.assert_allclose(result.input_dict["base_cover"], np.zeros(24))


def test_apply_design_row_tree_root_in_top_30_overrides_constant():
    """tree_root_in_top_30/crop_root_in_top_30 default to the CONSTANTS values
    when not drawn, and reflect the drawn value when they are."""
    import model.common.constants as CONSTANTS

    default_result = _apply(x=[50.0], param_names=["cy0"])
    assert default_result.tree_root_in_top_30 == CONSTANTS.TREE_ROOT_IN_TOP_30

    overridden = _apply(x=[0.5, 0.6], param_names=["tree_root_in_top_30", "crop_root_in_top_30"])
    assert overridden.tree_root_in_top_30 == pytest.approx(0.5)
    assert overridden.crop_root_in_top_30 == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Tests: run_morris
# ---------------------------------------------------------------------------

class _InProcessExecutor:
    """ProcessPoolExecutor stand-in that runs in the same process (keeps mocks visible)."""
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def map(self, fn, iterable): return [fn(item) for item in iterable]


def test_run_morris_returns_correct_shape():
    """run_morris returns Y of shape (n_rows, n_years)."""
    N_YEARS = 5
    k = 3
    n_traj = 4
    n_rows = n_traj * (k + 1)

    fake_result = MagicMock()
    fake_result.emit_project_emissions = np.full(N_YEARS, 12.0)
    fake_result.emit_base_emissions = np.full(N_YEARS, 4.0)

    X = np.zeros((n_rows, k))
    param_names = ["cy0", "clay", "temp_ci_delta"]

    # create_autospec (not a bare MagicMock) so a call missing a required
    # handle_intervention() kwarg raises TypeError here, instead of silently
    # succeeding and hiding a wiring regression.
    mock_handle_intervention = create_autospec(handle_intervention, return_value=fake_result)

    with patch("model.morris.runner.handle_intervention", mock_handle_intervention), \
         patch("model.morris.runner.concurrent.futures.ProcessPoolExecutor", _InProcessExecutor):
        Y = run_morris(
            X=X,
            param_names=param_names,
            base_input={},
            base_soil=_make_soil(),
            base_climate=_make_climate(),
            tree_species_data={},
            crop_species_data={},
            pool_species_data={},
            create_forward_soil_model=MagicMock(),
            create_inverse_soil_model=MagicMock(),
            n_proj_cohorts=1,
            n_base_cohorts=1,
            plot_index=0,
        )

    assert Y.shape == (n_rows, N_YEARS)


def test_run_morris_threads_species_data_and_root_in_top_30_into_handle_intervention():
    """run_morris passes species data through unperturbed and threads a drawn
    root-in-top-30 value into handle_intervention() — regression test for the
    bug where these arguments were silently omitted."""
    N_YEARS = 2
    X = np.array([[0.55]])
    param_names = ["tree_root_in_top_30"]

    fake_result = MagicMock()
    fake_result.emit_project_emissions = np.full(N_YEARS, 1.0)
    fake_result.emit_base_emissions = np.full(N_YEARS, 0.0)

    tree_species_data = {1: {"species": 1}}
    crop_species_data = {2: {"species": "maize"}}
    pool_species_data = {1: {"turnover": np.zeros(5)}}

    mock_handle_intervention = create_autospec(handle_intervention, return_value=fake_result)

    with patch("model.morris.runner.handle_intervention", mock_handle_intervention), \
         patch("model.morris.runner.concurrent.futures.ProcessPoolExecutor", _InProcessExecutor):
        run_morris(
            X=X,
            param_names=param_names,
            base_input={},
            base_soil=_make_soil(),
            base_climate=_make_climate(),
            tree_species_data=tree_species_data,
            crop_species_data=crop_species_data,
            pool_species_data=pool_species_data,
            create_forward_soil_model=MagicMock(),
            create_inverse_soil_model=MagicMock(),
            n_proj_cohorts=1,
            n_base_cohorts=1,
            plot_index=0,
        )

    _, call_kwargs = mock_handle_intervention.call_args
    assert call_kwargs["tree_species_data"] == tree_species_data
    assert call_kwargs["crop_species_data"] == crop_species_data
    assert call_kwargs["pool_species_data"] == pool_species_data
    assert call_kwargs["tree_root_in_top_30"] == pytest.approx(0.55)


def test_compute_morris_indices_column_names():
    """Returned DataFrame has expected columns and one row per (parameter, year)."""
    from SALib.sample import morris as morris_sample

    N_YEARS = 3
    k = 2
    n_traj = 10

    param_names = ["cy0", "clay"]
    problem = build_salib_problem(param_names, [(30.0, 70.0), (5.0, 60.0)])
    X = morris_sample.sample(problem, N=n_traj, num_levels=4, seed=0)
    Y = np.random.default_rng(0).random((X.shape[0], N_YEARS))

    df = compute_morris_indices(problem, X, Y)

    expected_cols = {"parameter", "year", "mu", "mu_star", "sigma", "mu_star_conf"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == k * N_YEARS
    assert set(df["parameter"].unique()) == set(param_names)


# ---------------------------------------------------------------------------
# Tests: default_bounds
# ---------------------------------------------------------------------------

def test_default_bounds_soil_bounds():
    """When Q05/Q95 are distinct, cy0/clay bounds are a 95% CI built from
    sigma inferred from the quantile pair (not the raw quantiles themselves,
    which only span a 90% CI). When Q05==Q95==mean, a fixed fallback is used."""
    z_95, z_90_half = 1.96, 1.645

    soil = _make_soil(cy0=50.0, cy0_q05=35.0, cy0_q95=65.0, clay=30.0, clay_q05=15.0, clay_q95=50.0)
    bounds = default_bounds(_make_base_input(), soil, _make_species_ctx())
    cy0_sigma = (65.0 - 35.0) / (2.0 * z_90_half)
    assert bounds["cy0"] == pytest.approx((50.0 - z_95 * cy0_sigma, 50.0 + z_95 * cy0_sigma))

    soil = _make_soil(cy0=50.0, cy0_q05=50.0, cy0_q95=50.0, clay=30.0, clay_q05=30.0, clay_q95=30.0)
    bounds = default_bounds(_make_base_input(), soil, _make_species_ctx())
    assert bounds["cy0"] == (40.0, 60.0)
    assert bounds["clay"] == (5.0, 70.0)


def test_default_bounds_climate_bounds():
    """Climate bounds are a fixed (-1, 1) CI-fraction range, independent of the
    site's actual std values — apply_design_row() is what applies the real
    per-month magnitude, not the bound itself."""
    bounds = default_bounds(_make_base_input(), _make_soil(), _make_species_ctx())
    assert bounds["temp_ci_delta"] == (-1.0, 1.0)
    assert bounds["rain_ci_delta"] == (-1.0, 1.0)
    assert bounds["evap_ci_delta"] == (-1.0, 1.0)


def test_default_bounds_does_not_emit_removed_or_delta_only_names():
    """tree_biomass_scale/base_sf_n1 are long-removed legacy names, never
    recognised. sf_n/thinning (delta) and mortality (direct) have no generic
    default magnitude, so they're bounds-file-only and must not appear either
    — unlike stand_density_scale and the ef_burn_*_scale families, which do
    have a real default."""
    bounds = default_bounds(_make_base_input(), _make_soil(), _make_species_ctx())

    assert "tree_biomass_scale" not in bounds
    assert "base_sf_n1" not in bounds
    assert "base_sf_n_delta" not in bounds
    assert "base_thinning_delta" not in bounds
    assert "base_mortality" not in bounds
    assert "base_stand_density_scale" in bounds
    assert "ef_burn_crop_N2O_scale" in bounds
