import numpy as np
import pytest

import time_series as ts


@pytest.fixture(name="single_time_fit")
def make_single_time_fit():
    return ts.SingleTimeFit(np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0]))


@pytest.fixture(name="multi_time_fit")
def make_multi_time_fit(single_time_fit):
    return ts.MultiTimeFit([single_time_fit, single_time_fit])


class TestSingleTimeFit:
    def test_residuals(self, single_time_fit):
        assert (single_time_fit.residuals == np.array([1.0, 0.0, -1.0])).all()

    def test_chi_squared_list(self, single_time_fit):
        assert (single_time_fit.chi_squared_list == np.array([1.0, 0.0, 1.0])).all()

    def test_chi_squared(self, single_time_fit):
        assert single_time_fit.chi_squared == 2.0


class TestMultiTimeFit:
    def test_chi_squared(self, multi_time_fit):
        assert multi_time_fit.chi_squared == 4.0
