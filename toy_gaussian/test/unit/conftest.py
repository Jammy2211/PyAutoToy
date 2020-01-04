from os import path

import numpy as np
import pytest

import autofit as af
import toy_gaussian
from autoarray.mask import mask
from test_autoarray.unit.conftest import *
from toy_gaussian.test.mock import mock_pipeline

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path():
    af.conf.instance = af.conf.Config(
        path.join(directory, "test_files/config"), path.join(directory, "output")
    )


#
# MODEL #
#


@pytest.fixture(name="gaussian")
def make_gaussian():
    # noinspection PyTypeChecker
    return toy_gaussian.SphericalGaussian(centre=(0.0, 0.0), intensity=1.0, sigma=0.5)


@pytest.fixture(name="mask_function_7x7_1_pix")
def make_mask_function_7x7_1_pix():
    # noinspection PyUnusedLocal
    def mask_function_7x7_1_pix(shape_2d, pixel_scales):
        array = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, False, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        return mask.Mask(mask_2d=array, pixel_scales=1.0)

    return mask_function_7x7_1_pix


@pytest.fixture(name="mask_function_7x7")
def make_mask_function_7x7():
    # noinspection PyUnusedLocal
    def mask_function_7x7(shape_2d, pixel_scales):
        array = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, False, False, False, True, True],
                [True, True, False, False, False, True, True],
                [True, True, False, False, False, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        return aa.mask.manual(mask_2d=array, pixel_scales=1.0)

    return mask_function_7x7


@pytest.fixture(name="phase_dataset_7x7")
def make_phase_data(mask_function_7x7):
    return toy_gaussian.PhaseDataset(
        optimizer_class=mock_pipeline.MockNLO, phase_tag="", phase_name="test_phase"
    )


@pytest.fixture(name="phase_imaging_7x7")
def make_phase_imaging_7x7(mask_function_7x7):
    return toy_gaussian.PhaseImaging(
        optimizer_class=mock_pipeline.MockNLO,
        mask_function=mask_function_7x7,
        phase_name="test_phase",
    )


@pytest.fixture(name="results_7x7")
def make_results(mask_7x7,):
    return mock_pipeline.MockResults(mask=mask_7x7)


@pytest.fixture(name="results_collection_7x7")
def make_results_collection(results_7x7):
    results_collection = af.ResultsCollection()
    results_collection.add("phase", results_7x7)
    return results_collection
