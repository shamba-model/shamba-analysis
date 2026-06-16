"""Tests for the Morris sensitivity analysis module.

Covers parameter_space.py (apply_design_row, default_bounds) and
runner.py (run_morris shape, compute_morris_indices columns).
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from model.soil_params import SoilParamsData
from model.climate import ClimateData
from model.emit import EmissionFactors
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


def _make_climate() -> ClimateData:
    return ClimateData(
        temperature=np.full(12, 20.0),
        rain=np.full(12, 80.0),
        evaporation=np.full(12, 50.0),
        temperature_std=np.zeros(12),
        rain_std=np.zeros(12),
        evaporation_std=np.zeros(12),
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


# ---------------------------------------------------------------------------
# Tests: apply_design_row — soil
# ---------------------------------------------------------------------------

def test_apply_design_row_soil_recomputes_ceq_iom():
    """Drawing a cy0 different from base recomputes Ceq and iom correctly."""
    base_soil = _make_soil(cy0=50.0)
    base_climate = _make_climate()

    drawn_cy0 = 70.0
    x = np.array([drawn_cy0, base_soil.clay])
    param_names = ["cy0", "clay"]

    _, soil, _, _ = apply_design_row(
        x=x, param_names=param_names,
        base_input={}, base_soil=base_soil, base_climate=base_climate,
        base_emission_factors=EmissionFactors(),
    )

    expected_ceq = 1.25 * drawn_cy0
    expected_iom = 0.049 * expected_ceq ** 1.139

    assert soil.Cy0 == pytest.approx(drawn_cy0)
    assert soil.Ceq == pytest.approx(expected_ceq)
    assert soil.iom == pytest.approx(expected_iom)


# ---------------------------------------------------------------------------
# Tests: apply_design_row — climate
# ---------------------------------------------------------------------------

def test_apply_design_row_climate_delta():
    """temp_delta shifts all 12 temperature months; rain and evaporation unchanged."""
    base_soil = _make_soil()
    base_climate = _make_climate()

    delta = 1.5
    x = np.array([delta])
    param_names = ["temp_delta"]

    _, _, climate, _ = apply_design_row(
        x=x, param_names=param_names,
        base_input={}, base_soil=base_soil, base_climate=base_climate,
        base_emission_factors=EmissionFactors(),
    )

    np.testing.assert_allclose(climate.temperature, base_climate.temperature + delta)
    np.testing.assert_array_equal(climate.rain, base_climate.rain)
    np.testing.assert_array_equal(climate.evaporation, base_climate.evaporation)


def test_apply_design_row_rain_scale_clamps_to_zero():
    """A negative rain_scale produces a zero rain vector (not negative)."""
    base_climate = _make_climate()
    x = np.array([-0.1])
    _, _, climate, _ = apply_design_row(
        x=x, param_names=["rain_scale"],
        base_input={}, base_soil=_make_soil(), base_climate=base_climate,
        base_emission_factors=EmissionFactors(),
    )
    assert np.all(climate.rain == 0.0)


# ---------------------------------------------------------------------------
# Tests: apply_design_row — tree biomass scale
# ---------------------------------------------------------------------------

def test_apply_design_row_tree_biomass_scale_applies_to_all_cohorts():
    """tree_biomass_scale=2.0 doubles every base_plant_dens and proj_plant_dens."""
    base_input = _make_base_input(n_proj=2, n_base=2)
    base_soil = _make_soil()
    base_climate = _make_climate()

    x = np.array([2.0])
    param_names = ["tree_biomass_scale"]

    result_dict, _, _, _ = apply_design_row(
        x=x, param_names=param_names,
        base_input=base_input, base_soil=base_soil, base_climate=base_climate,
        base_emission_factors=EmissionFactors(),
    )

    for key in ("base_plant_dens1", "base_plant_dens2", "proj_plant_dens1", "proj_plant_dens2"):
        expected = np.atleast_1d(np.asarray(base_input[key], dtype=float)) * 2.0
        np.testing.assert_allclose(result_dict[key], expected, err_msg=f"Failed for {key}")


# ---------------------------------------------------------------------------
# Tests: run_morris — output shape
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

    # Fake SOC array: (N_YEARS+1, 4)
    fake_soc = np.ones((N_YEARS + 1, 4)) * 10.0
    fake_project_soil = MagicMock()
    fake_project_soil.SOC = fake_soc

    fake_result = MagicMock()
    fake_result.project_forward_soil_data = fake_project_soil

    X = np.zeros((n_rows, k))
    param_names = ["cy0", "clay", "temp_delta"]

    base_soil = _make_soil()
    base_climate = _make_climate()
    base_input = {}

    with patch("model.morris.runner.handle_intervention", return_value=fake_result), \
         patch("model.morris.runner.concurrent.futures.ProcessPoolExecutor", _InProcessExecutor):
        Y = run_morris(
            X=X,
            param_names=param_names,
            base_input=base_input,
            base_soil=base_soil,
            base_climate=base_climate,
            create_forward_soil_model=MagicMock(),
            create_inverse_soil_model=MagicMock(),
            n_proj_cohorts=1,
            n_base_cohorts=1,
            plot_index=0,
        )

    assert Y.shape == (n_rows, N_YEARS)


# ---------------------------------------------------------------------------
# Tests: compute_morris_indices — column names
# ---------------------------------------------------------------------------

def test_compute_morris_indices_column_names():
    """Returned DataFrame has expected columns and one row per (parameter, year)."""
    from SALib.sample import morris as morris_sample

    N_YEARS = 3
    k = 2
    n_traj = 10

    param_names = ["cy0", "clay"]
    problem = build_salib_problem(param_names, [(30.0, 70.0), (5.0, 60.0)])
    X = morris_sample.sample(problem, N=n_traj, num_levels=4, seed=0)
    n_rows = X.shape[0]
    Y = np.random.default_rng(0).random((n_rows, N_YEARS))

    df = compute_morris_indices(problem, X, Y)

    expected_cols = {"parameter", "year", "mu", "mu_star", "sigma", "mu_star_conf"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == k * N_YEARS
    assert set(df["parameter"].unique()) == set(param_names)
    assert set(df["year"].unique()) == set(range(1, N_YEARS + 1))


# ---------------------------------------------------------------------------
# Tests: default_bounds — soil quantiles
# ---------------------------------------------------------------------------

def test_default_bounds_soil_uses_quantiles():
    """When soil Q05/Q95 are distinct, cy0 and clay bounds match the stored quantiles."""
    soil = _make_soil(cy0=50.0, cy0_q05=35.0, cy0_q95=65.0,
                      clay=30.0, clay_q05=15.0, clay_q95=50.0)
    base_input = _make_base_input()
    bounds = default_bounds(base_input, soil)

    assert bounds["cy0"] == (35.0, 65.0)
    assert bounds["clay"] == (15.0, 50.0)


def test_default_bounds_soil_uses_fallback_when_quantiles_equal():
    """When Q05 == Q95 == mean, fallback bounds are applied."""
    soil = _make_soil(cy0=50.0, cy0_q05=50.0, cy0_q95=50.0,
                      clay=30.0, clay_q05=30.0, clay_q95=30.0)
    base_input = _make_base_input()
    bounds = default_bounds(base_input, soil)

    assert bounds["cy0"] == (max(50.0 - 10.0, 0.0), 60.0)
    assert bounds["clay"] == (5.0, 70.0)
