import numpy as np
import pytest

from time_series import lotka_voltera as lv
from time_series import species as s


@pytest.fixture(name="lotka_voltera_model")
def make_lotka_voltera_model():
    collection = s.SpeciesCollection([s.Species(), s.Species()])

    for i in range(len(collection)):
        collection[i, i] = 1.0

    return lv.LotkaVolteraModel(collection, capacity=1.0)


class TestLotkaVoltera:
    def test_static(self, lotka_voltera_model):
        population = np.array([1.0, 1.0])

        assert (lotka_voltera_model.step(population) == population).all()

    def test_convergence(self, lotka_voltera_model):
        population = np.array([1.1, 1.1])

        assert np.allclose(
            lotka_voltera_model.change(population), np.array([-0.11, -0.11])
        )

        for _ in range(10):
            population = lotka_voltera_model.step(population)
            change = lotka_voltera_model.change(population)
            sign = change * (population - np.array([1.0, 1.0]))
            assert (sign <= 0).all()
