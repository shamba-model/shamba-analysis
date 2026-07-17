import os  # Add the parent directory to the Python path
import model.emit as Emit
import numpy as np
import pytest
from model import configuration
from model.crop_model import get_crop_bases, get_crop_projects, create as create_crop_model
from model.crop_params import load_crop_species_data, CropParamsData
from model.common.data_handler import expand_single_row_data_input

#-- Expected emissions arrays -- #
# WL: constant nitrogen emission from crop biomass every year (50-year project).
# These are verified in an excel worked example.
WL_expected_base_emissions = [0.22413] * 50
# WL project: same emission for first 5 years then zero (crop removed after year 5).
WL_expected_project_emissions = [0.22413] * 5 + [0.00000] * 45

testB_expected_base_emissions = [0.0000] + [0.24072] * 4 + [0.10581] * 35
# These are verified in an excel worked example.
testB_expected_project_emissions = [0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.21471,
0.19677,
0.19677,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.11064,
0.11064,
0.11064,
0.11064,
0.12334,
0.11064,
0.11064,
0.11064,
0.11064,
0.11064,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
0.00000,
]

#-- Test function -- #

@pytest.mark.parametrize("csv_input_file, expected_base_emissions, expected_project_emissions", [
    pytest.param("WL_input.csv", WL_expected_base_emissions, 
                WL_expected_project_emissions, id = "Test Case: WL"),
    pytest.param("testB_input.csv", testB_expected_base_emissions, testB_expected_project_emissions, id = "Test Case: testB"),
])
def test_crop_model(csv_input_file, expected_base_emissions, expected_project_emissions):
    file_path = os.path.join(configuration.TESTS_DIR, "fixtures", csv_input_file)
    scalar_input_data, _, mgmt_input_data, _ = expand_single_row_data_input(file_path)
    N_YEARS = int(scalar_input_data["yrs_proj"].item())
    input_data = {**scalar_input_data, **mgmt_input_data}

    # Crop slots with species code 0 (no crop) are omitted from input_data by
    # expand_single_row_data_input, so the real cohort count must be discovered
    # from which spp keys are actually present, same as calculate_emissions.py does.
    n_crop_base = sum(1 for i in range(1, 100) if f"crop_base_spp{i}" in input_data)
    n_crop_proj = sum(1 for i in range(1, 100) if f"crop_proj_spp{i}" in input_data)

    species_data = load_crop_species_data()
    crop_base, _crop_par_base = get_crop_bases(
        input_data=input_data, no_of_years=N_YEARS, start_index=1, end_index=n_crop_base,
        species_data=species_data,
    )
    crop_project, _crop_par_project = get_crop_projects(
        input_data=input_data, no_of_years=N_YEARS, start_index=1, end_index=n_crop_proj,
        species_data=species_data,
    )

    crop_base_emissions = Emit.create(
        no_of_years=N_YEARS,
        crop=crop_base,
        fire=mgmt_input_data["fire_on_base"],
        burn_off=mgmt_input_data["fire_off_base"],
    )
    crop_project_emissions = Emit.create(
        no_of_years=N_YEARS,
        crop=crop_project,
        fire=mgmt_input_data["fire_on_proj"],
        burn_off=mgmt_input_data["fire_off_proj"],
    )
    assert crop_base_emissions == pytest.approx(
        expected_base_emissions, rel=1e-4
    )
    assert crop_project_emissions == pytest.approx(
        expected_project_emissions, rel=1e-4
    )


def test_create_crop_root_in_top_30_overrides_below_ground_only():
    """No override matches CONSTANTS.CROP_ROOT_IN_TOP_30 exactly (the default
    preserves current behavior); an explicit override scales only the
    below-ground output, leaving above-ground untouched."""
    import model.common.constants as CONSTANTS

    crop_params = CropParamsData(
        species="TestCrop", species_code=1, slope=1.0, intercept=0.5,
        nitrogen_below=0.01, nitrogen_above=0.02,
        carbon_below=0.40, carbon_above=0.45, root_to_shoot=0.20,
    )
    crop_yield = np.array([5.0, 6.0, 7.0])

    default_crop = create_crop_model(crop_params, no_of_years=3, crop_yield=crop_yield, left_in_field=0.5)
    explicit_default_crop = create_crop_model(
        crop_params, no_of_years=3, crop_yield=crop_yield, left_in_field=0.5,
        crop_root_in_top_30=CONSTANTS.CROP_ROOT_IN_TOP_30,
    )
    np.testing.assert_array_equal(
        default_crop.output["below"]["carbon"], explicit_default_crop.output["below"]["carbon"]
    )

    overridden_value = 0.5
    overridden_crop = create_crop_model(
        crop_params, no_of_years=3, crop_yield=crop_yield, left_in_field=0.5,
        crop_root_in_top_30=overridden_value,
    )
    np.testing.assert_array_equal(
        default_crop.output["above"]["carbon"], overridden_crop.output["above"]["carbon"]
    )
    ratio = overridden_value / CONSTANTS.CROP_ROOT_IN_TOP_30
    np.testing.assert_allclose(
        overridden_crop.output["below"]["carbon"],
        np.array(default_crop.output["below"]["carbon"]) * ratio,
    )
