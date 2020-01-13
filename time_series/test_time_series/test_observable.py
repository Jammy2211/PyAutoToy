import numpy as np

from time_series.observable import Observable


class TestObservable:
    def test_distribution(self):
        observable = Observable(
            mean=1.0,
            deviation=0.5
        )

        assert np.allclose(
            observable.pdf(0, 2, 3),
            np.array([
                [0.10798193],
                [0.79788456],
                [0.10798193],
            ])
        )
