import pytest

from time_series import species as s


@pytest.fixture(name="species_a")
def make_species_a():
    return s.Species(1.0)


@pytest.fixture(name="species_b")
def make_species_b():
    return s.Species(2.0)


@pytest.fixture(name="species_collection")
def make_species_collection(
        species_a,
        species_b
):
    return s.SpeciesCollection(
        species_a,
        species_b
    )
