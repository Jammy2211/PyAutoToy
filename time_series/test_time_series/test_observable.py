import numpy as np

from time_series.observable import Observable
import autofit as af


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

    def test_prior_model(self):
        model = af.PriorModel(Observable)

        assert model.prior_count == 2

        instance = model.instance_from_prior_medians()
        assert isinstance(instance, Observable)
