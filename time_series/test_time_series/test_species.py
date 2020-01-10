import numpy as np
import pytest

from time_series import species as s


@pytest.fixture(name="species")
def make_species():
    return s.Species()


class TestSpecies:
    def test_interaction(self):
        species = s.Species()
        assert species.interactions[species] == 1.0

        species.interactions[species] = 0.5
        assert species.interactions[species] == 0.5


class TestSpeciesCollection:
    def test_interaction_matrix(self, species):
        collection = s.SpeciesCollection(species)

        assert collection.species == (species,)
        assert collection.interaction_matrix == np.array([[1.0]])
