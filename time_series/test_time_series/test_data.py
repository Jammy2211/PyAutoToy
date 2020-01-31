import numpy as np
import pytest

import time_series as ts


@pytest.fixture(
    name="data_list"
)
def make_data_list():
    return ts.data.generate_data_at_timesteps(
        number_of_observables=3,
        number_of_species=4,
        timesteps=[5, 7, 13]
    )


@pytest.fixture(
    name="data"
)
def make_data(data_list):
    return data_list[0]


def test_timesteps(data_list):
    assert [
               data.timestep
               for data
               in data_list
           ] == [
               5, 7, 13
           ]


def test_observable_distributions(data_list):
    observables = list(data_list[0].observables.values())[0].observables
    for data in data_list:
        pairs = zip(list(data.observables.values())[0].observables, observables)
        for one, two in pairs:
            assert one == two


def test_abundances(data_list):
    abundances = list(data_list[0].observables.values())[0].abundances
    for data in data_list[1:]:
        assert abundances != list(data.observables.values())[0].abundances


def test_type(data):
    assert isinstance(data["1"], np.ndarray)
