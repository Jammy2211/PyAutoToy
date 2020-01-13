import numpy as np
import pytest

from time_series.fit import Fit


@pytest.fixture(name="fit")
def make_fit():
    return Fit(
        np.array([
            0.0, 1.0, 2.0
        ]),
        np.array([
            1.0, 1.0, 1.0
        ])
    )


class TestFit:
    def test_residuals(self, fit):
        assert (fit.residuals == np.array([
            1.0, 0.0, -1.0
        ])).all()

    def test_chi_squared_list(self, fit):
        assert (fit.chi_squared_list == np.array([
            1.0, 0.0, 1.0
        ])).all()

    def test_chi_squared(self, fit):
        assert fit.chi_squared == 2.0
