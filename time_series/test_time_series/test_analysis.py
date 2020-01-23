import pytest

import autofit as af
from time_series.fit import Fit
from time_series.observable import Observable
from time_series.species import Species, SpeciesObservables


class Data:
    def __init__(
            self,
            **observables
    ):
        self.observables = observables

    @property
    def observable_names(self):
        return set(self.observables.keys())

    def __getitem__(self, observable_name):
        return self.observables[observable_name]


class Analysis(af.Analysis):
    def fit(self, instance):
        fitness = 0
        species_observables = instance.species_observables
        for observable_name in self.data.observable_names:
            fitness -= Fit(
                self.data[observable_name],
                pdf(species_observables[observable_name])
            ).chi_squared
        return fitness

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
        LOWER_LIMIT,
        UPPER_LIMIT,
        NUMBER_OF_POINTS
    )


@pytest.fixture(name="data")
def make_data(a_0, a_1, b_0, b_1):
    return Data(
        a=pdf(a_0) + pdf(a_1),
        b=pdf(b_0) + pdf(b_1)
    )


@pytest.fixture(
    name="species_observables"
)
def make_species_observables(
        species_0,
        species_1
):
    return SpeciesObservables(
        abundances=[1.0, 1.0],
        species=[species_0, species_1]
    )


class TestAnalysis:
    def test_species_observables(
            self,
            species_observables,
            a_0,
            a_1
    ):
        assert species_observables.observable_names == {"a", "b"}

        compound_observable_a = species_observables["a"]
        compound_result = pdf(compound_observable_a)
        addition_result = pdf(a_0) + pdf(a_1)
        # noinspection PyUnresolvedReferences
        assert (compound_result == addition_result).all()

    def test_analysis(
            self,
            species_observables,
            data
    ):
        instance = af.ModelInstance()
        instance.species_observables = species_observables

        analysis = Analysis(
            data
        )

        # noinspection PyTypeChecker
        assert analysis.fit(
            instance
        ) == 0.0

        instance.species_observables.abundances = [
            0.5, 0.5
        ]
        # noinspection PyTypeChecker
        assert analysis.fit(
            instance
        ) < 0.0
