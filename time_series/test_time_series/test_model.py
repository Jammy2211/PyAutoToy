import numpy as np


from time_series import lotka_voltera as lv
from time_series import species as s


class TestLotkaVoltera:
    def test_static(self):
        collection = s.SpeciesCollection(
            s.Species(),
            s.Species()
        )

        for i in range(len(collection)):
            collection[i, i] = 1.0

        model = lv.LotkaVolteraModel(
            collection,
            capacity=1.0
        )

        population = np.array(
            [1.0, 1.0]
        )

        assert (model.step(
            population
        ) == population).all()
