import pytest

import autofit as af
from time_series.observable import Observable
from time_series.species import Species, SpeciesObservables


class Data:
    def __init__(self, **observables):
        self.observables = observables


class Analysis(af.Analysis):
    def fit(self, instance):
        pass

    def visualize(self, instance, during_analysis):
        pass

    def __init__(self, data):
        self.data = data


LOWER_LIMIT = 0
UPPER_LIMIT = 20
NUMBER_OF_POINTS = 400


@pytest.fixture(name="a_0")
def make_a_0():
    return Observable(
        3,
        2
    )


@pytest.fixture(name="a_1")
def make_a_1():
    return Observable(
        15,
        4
    )


@pytest.fixture(name="b_0")
def make_b_0():
    return Observable(
        13,
        2
    )


@pytest.fixture(name="b_1")
def make_b_1():
    return Observable(
        7,
        1
    )


@pytest.fixture(name="species_0")
def make_species_0(a_0, b_0):
    return Species(
        observables=dict(
            a=a_0,
            b=b_0
        )
    )


@pytest.fixture(name="species_1")
def make_species_1(a_1, b_1):
    return Species(
        observables=dict(
            a=a_1,
            b=b_1
        )
    )


def pdf(observable):
    return observable.pdf(
        0,
        20,
        400
    )


@pytest.fixture(name="data")
def make_data(a_0, a_1, b_0, b_1):
    Data(
        a=pdf(a_0) + pdf(a_1),
        b=pdf(b_0) + pdf(b_1)
    )


class TestAnalysis:
    def test_species_observables(
            self,
            species_0,
            species_1,
            a_0,
            a_1
    ):
        species_observables = SpeciesObservables(
            abundances=[1.0, 1.0],
            species=[species_0, species_1]
        )
        assert species_observables.observable_names == {"a", "b"}

        compound_observable_a = species_observables["a"]
        compound_result = pdf(compound_observable_a)
        addition_result = pdf(a_0) + pdf(a_1)
        # noinspection PyUnresolvedReferences
        assert (compound_result == addition_result).all()

    # def test_analysis(self, species_0, species_1, data):
    #     instance = af.ModelInstance()
    #     instance.populations = [1.0, 1.0]
    #     instance.species = [
    #         species_0,
    #         species_1
    #     ]
    #
    #     analysis = Analysis(
    #         data
    #     )
    #
    #     assert analysis.fit(
    #         instance
    #     ) == 0.0
