from functools import wraps

import numpy as np
import pytest

import autofit as af
from time_series.observable import Observable
from time_series.species import Species


class Data:
    def __init__(self, **observables):
        self.observables = observables


def assert_lengths_match(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        self, first, second = args + tuple(kwargs.values())
        if len(first) != len(second):
            raise AssertionError(
                f"Length of lists passed to {func.__name__} do not match"
            )
        return func(self, first, second)

    return wrapper


class CompoundObservable:
    @assert_lengths_match
    def __init__(self, abundances, observables):
        self.abundances = abundances
        self.observables = observables

    def pdf(
            self,
            lower_limit: int = -2,
            upper_limit: int = 2,
            number_of_points: int = 1000
    ):
        pdfs = [
            abundance * observable.pdf(
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                number_points=number_of_points
            )
            for abundance, observable
            in zip(
                self.abundances,
                self.observables
            )
        ]
        return np.add(*pdfs)


class SpeciesObservables:
    @assert_lengths_match
    def __init__(
            self,
            abundances,
            species
    ):
        self.abundances = abundances
        self.species = species

    @property
    def observable_names(self):
        return {
            key for species
            in self.species
            for key in species.observables.keys()
        }

    def __getitem__(self, item):
        observables = [
            species.observables[item]
            for species in self.species
        ]
        return CompoundObservable(
            self.abundances,
            observables
        )


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

    def test_analysis(self, species_0, species_1, data):
        instance = af.ModelInstance()
        instance.populations = [1.0, 1.0]
        instance.species = [
            species_0,
            species_1
        ]

        analysis = Analysis(
            data
        )

        assert analysis.fit(
            instance
        ) == 0.0
