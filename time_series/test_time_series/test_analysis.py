import pytest

import autofit as af
from time_series.analysis import Analysis
from time_series.data import Data
from time_series.data import pdf
from time_series.observable import Observable
from time_series.species import Species, SpeciesObservables


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


@pytest.fixture(name="data")
def make_data(a_0, a_1, b_0, b_1):
    return Data(
        a=a_0 + a_1,
        b=b_0 + b_1
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

    def test_analysis(
            self,
            species_0,
            species_1,
            data
    ):
        instance = af.ModelInstance()
        instance.abundances = [1.0, 1.0]
        instance.species = [
            species_0,
            species_1
        ]

        analysis = Analysis(
            data
        )

        # noinspection PyTypeChecker
        assert analysis.fit(
            instance
        ) == 0.0

        instance.abundances = [
            0.5, 0.5
        ]
        # noinspection PyTypeChecker
        assert analysis.fit(
            instance
        ) < 0.0
