import numpy as np

from time_series.fit import Fit


class TestFit:
    def test_residuals(self):
        fit = Fit(
            np.array([
                0.0, 1.0, 2.0
            ]),
            np.array([
                1.0, 1.0, 1.0
            ])
        )

        assert (fit.residuals == np.array([
            1.0, 0.0, -1.0
        ])).all()
