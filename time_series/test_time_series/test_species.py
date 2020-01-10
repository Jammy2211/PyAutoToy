import numpy as np
import pytest

from time_series import species as s


@pytest.fixture(name="species_a")
def make_species_a():
    return s.Species()


@pytest.fixture(name="species_b")
def make_species_b():
    return s.Species()


class TestSpecies:
    def test_interaction(self, species_a):
        assert species_a.interactions[species_a] == 1.0

        species_a.interactions[species_a] = 0.5
        assert species_a.interactions[species_a] == 0.5


class TestSpeciesCollection:
    def test_trivial_interaction_matrix(self, species_a):
        collection = s.SpeciesCollection(species_a)

        assert collection.species == (species_a,)
        assert collection.interaction_matrix == np.array([[1.0]])

    def test_interaction_matrix(self, species_a, species_b):
        species_a.interactions[species_b] = 0.5
        species_b.interactions[species_a] = 0.7

        collection = s.SpeciesCollection(species_a, species_b)
        assert (collection.interaction_matrix == np.array(
            [
                [1.0, 0.5],
                [0.7, 1.0]
            ]
        )).all()
